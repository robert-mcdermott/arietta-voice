from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import ExitStack
import logging
from pathlib import Path
import re
import sys
import time
import threading

import numpy as np

from arietta_voice.audio import (
    ChimePlayer,
    MicrophoneStream,
    RawKeypressMonitor,
    UtteranceBuffer,
    AudioReferenceBuffer,
    RecordingWriter,
    discard_capture,
    capture_phase,
    play_tts_stream,
    play_tts_stream_passive,
)
from arietta_voice.config import AppConfig
from arietta_voice.diagnostics import run_startup_diagnostics
from arietta_voice.logging_utils import EventLogger
from arietta_voice.models import AssistantState, ConversationTurn, SessionContext, UtteranceResult
from arietta_voice.prompts import sanitize_for_speech
from arietta_voice.runtime_status import RuntimeStatusPublisher
from arietta_voice.turn_detection import SileroVoiceActivityDetector
from arietta_voice.worker import InferenceWorker, WorkerTimeout, WorkerCancelled
from arietta_voice.turns import CancellationToken
from arietta_voice.streaming import speech_stream
from arietta_voice.memory_jobs import MemoryJobs
from arietta_voice.wake import WakeWordMatch, build_wake_word_detector
from arietta_voice.devices import AudioDeviceError, DEVICE_ERRORS, select_devices, refresh_audio_backend


GOODBYE_PATTERN = re.compile(
    r"(?:ok(?:ay)?[,.! ]+)?(?:thanks?[,.! ]+|thank you[,.! ]+)?"
    r"(?:goodbye|bye(?: bye)?|see you(?: later)?|talk to you later|"
    r"(?:that's|that is) all(?: for now)?|(?:i'm|i am) done)"
    r"(?:[,.! ]+(?:thanks?|thank you))?[.!\s]*",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RuntimeOptions:
    record_path: Path | None = None


class AriettaVoiceRuntime:
    def __init__(self, config: AppConfig, options: RuntimeOptions | None = None, *, worker=None) -> None:
        self.config = config
        self.options = options or RuntimeOptions()
        self.started_at = time.monotonic()
        self.logger = logging.getLogger("arietta_voice.runtime")
        self.event_logger = EventLogger.from_config(config.logging)
        self.worker = worker if worker is not None else InferenceWorker(config)
        self.memory_jobs = MemoryJobs(self.worker, timeout_s=config.runtime.turn_timeout_s, log=self.event_logger.log)
        self.vad = SileroVoiceActivityDetector(config.audio.sample_rate)
        self.microphone = None
        self._last_audio_statistics = {}
        self.turn_stage = "idle"
        self.turn_started_at: float | None = None
        self._stage_started_at: float | None = None
        self._turn_deadline: float | None = None
        self._turn_token = None
        self._turn_keys = None
        self._turn_thread_ident = None
        self._turn_microphone = None
        self.turn_cancellations = 0
        self.turn_failures = 0
        self.turn_timeouts = 0
        self.utterances_discarded = 0
        self.last_turn_duration_ms: int | None = None
        self.chime_player = ChimePlayer(config.audio.chime_enabled)
        self.wake_detector = build_wake_word_detector(config.wake_word)
        self.session = SessionContext()
        self.state = AssistantState.BOOTING
        self.startup_diagnostics: dict | None = None
        self.last_error: str | None = None
        self.last_user_transcript = ""
        self.last_assistant_text = "Listening for the wake word."
        self.display_speech_level = 0.0
        self.display_speaking_active = False
        self._greeting_index = 0
        self._goodbye_index = 0
        self.device_connection = "starting"
        self.audio_reconnect_attempts = 0
        self.audio_recoveries = 0
        self.audio_device_error = None
        self.selected_input = self.selected_output = None
        self.output_device = None
        self.status_publisher = RuntimeStatusPublisher(config)

    def set_state(self, state: AssistantState) -> None:
        if self.state == state:
            return
        previous = self.state
        self.state = state
        self.logger.info("State: %s -> %s", previous.value, state.value)
        self.event_logger.log(
            "state_change",
            from_state=previous.value,
            to_state=state.value,
        )
        self._publish_status(force=True)

    def _publish_status(self, *, lifecycle: str = "running", force: bool = False) -> None:
        try:
            if force:
                self.status_publisher.publish(self, lifecycle=lifecycle)
            else:
                self.status_publisher.publish_if_due(self, lifecycle=lifecycle)
        except OSError as exc:
            self.logger.warning("Runtime status update failed: %s", exc)

    def next_greeting(self) -> str:
        variants = self.config.assistant.greeting_variants
        greeting = variants[self._greeting_index % len(variants)]
        self._greeting_index += 1
        return greeting

    def next_goodbye(self) -> str:
        variants = self.config.assistant.goodbye_variants
        goodbye = variants[self._goodbye_index % len(variants)]
        self._goodbye_index += 1
        return goodbye

    def warmup(self) -> None:
        self.startup_diagnostics = run_startup_diagnostics(self.config)
        self.event_logger.log("startup_diagnostics", overall_status=self.startup_diagnostics["overall_status"])

        self.logger.info("Loading VAD")
        self._publish_status(force=True)
        self.vad.ensure_loaded()
        self.turn_stage = "loading_models"
        self._stage_started_at = time.monotonic()
        self.worker.call("warmup", timeout_s=self.config.runtime.startup_timeout_s,
                         on_wait=self._publish_status)
        self.turn_stage = "idle"
        self.set_state(
            AssistantState.WAITING_FOR_WAKE if self.config.wake_word.enabled else AssistantState.LISTENING
        )

    def reset_session(self) -> None:
        self.session.active = False
        self.session.history.clear()
        self.session.turn_count = 0
        self.session.activated_at = 0.0
        self.session.last_activity_at = 0.0
        self.session.last_wake_phrase = None
        self.last_user_transcript = ""
        self.last_assistant_text = ""

    def _open_session(self, wake_match: WakeWordMatch | None) -> None:
        self.session.active = True
        self.session.history.clear()
        self.session.turn_count = 0
        now = time.monotonic()
        self.session.activated_at = now
        self.session.last_activity_at = now
        self.session.last_wake_phrase = None if wake_match is None else wake_match.phrase
        self.event_logger.log(
            "session_started",
            wake_phrase=None if wake_match is None else wake_match.phrase,
        )
        self.set_state(AssistantState.LISTENING)

    def _end_session(self, *, reason: str, microphone: MicrophoneStream, speak_goodbye: bool) -> None:
        goodbye = self.next_goodbye() if speak_goodbye else None
        if goodbye:
            self.last_assistant_text = goodbye
            self._print_response(goodbye)
            self._speak_blocking(goodbye)

        self.event_logger.log(
            "session_ended",
            reason=reason,
            content={"goodbye": goodbye},
        )
        self._update_display_speech(0.0, False)
        microphone.drain()
        self.reset_session()

        if self.config.wake_word.enabled:
            self.last_user_transcript = ""
            self.set_state(AssistantState.WAITING_FOR_WAKE)
        else:
            self._open_session(None)

    def _session_timed_out(self, now: float) -> bool:
        if not self.session.active:
            return False
        return (now - self.session.last_activity_at) > self.config.wake_word.session_timeout_s

    def _touch_session(self) -> None:
        self.session.last_activity_at = time.monotonic()

    def _update_display_speech(self, level: float, active: bool) -> None:
        self.display_speech_level = max(0.0, min(1.0, float(level)))
        self.display_speaking_active = bool(active)
        self._publish_status(force=False)

    def _remaining_time(self) -> float:
        if self._turn_deadline is None:
            return self.config.runtime.turn_timeout_s
        remaining = self._turn_deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerTimeout("Turn exceeded its time limit")
        return remaining

    def cancel_turn(self):
        token = self._turn_token
        if token is not None:
            token.cancel()

    def _inference_wait(self):
        if self._turn_thread_ident is None or threading.get_ident() == self._turn_thread_ident:
            self._check_audio_health()
            if self.turn_stage == "transcribe" and self._turn_microphone is not None:
                discard_capture(self._turn_microphone, "transcribing")
        self._publish_status()
        if (self._turn_keys is not None and threading.get_ident() == self._turn_thread_ident
                and self._turn_keys.pressed()):
            self.cancel_turn()

    def _check_audio_health(self):
        microphone = self.microphone
        if microphone is not None and hasattr(type(microphone), 'check_health'):
            microphone.check_health()

    def _playback_wait(self):
        self._check_audio_health()
        self._publish_status()

    def _infer(self, operation: str, payload=None, *, cancellation=None):
        token = cancellation or self._turn_token
        if token is not None:
            token.check()
        self.turn_stage = operation
        self._publish_status(force=True)
        foreground = self._turn_thread_ident is None or threading.get_ident() == self._turn_thread_ident
        if foreground:
            self._check_audio_health()
            if operation == "transcribe":
                capture_phase(self._turn_microphone, "transcribe")
        result = self.worker.call(operation, payload, timeout_s=self._remaining_time(),
                                  on_wait=self._inference_wait, cancellation=token)
        if foreground:
            self._check_audio_health()
        return result

    def _chat_request(self, user_text):
        return {"message": user_text,
            "history": [asdict(turn) for turn in self.session.history], "use_tools": True,
            "enabled_tools": self.config.tools.enabled,
            "enabled_model_tools": self.config.model_tools.enabled}

    def _respond(self, user_text: str) -> str:
        result = self._infer("chat", self._chat_request(user_text))
        self._log_tool_result(result)
        return result["response"]

    def _log_tool_result(self, result):
        tool = result.get("model_tool") or result.get("used_tool")
        if tool:
            self.event_logger.log("tool_call", tool=tool,
                                  mode="model_requested" if result.get("model_tool") else result["source"])

    async def _tts_stream(self, text: str):
        stream = speech_stream(text, infer=self._infer, worker=self.worker,
                               cancellation=self._turn_token or CancellationToken(),
                               deadline=self._turn_deadline, on_wait=self._playback_wait,
                               playback_events=True)
        try:
            async for frame in stream:
                yield frame
        finally:
            await stream.aclose()

    def _speak_blocking(self, text: str) -> None:
        if not self.config.audio.tts_enabled:
            return

        play_tts_stream_passive(
            tts_stream=self._tts_stream(text),
            cancellation=self._turn_token,
            deadline=self._turn_deadline,
            chime_player=self.chime_player,
            keypress_monitor=self._turn_keys,
            output_device=self.output_device,
            health_check=self._check_audio_health,
            playback_level_callback=self._update_display_speech,
        )
        self.event_logger.log("assistant_audio_complete", interrupted=False)
        self._update_display_speech(0.0, False)

    def _deliver_response(
        self,
        response: str,
        *,
        microphone: MicrophoneStream,
        keypress_monitor: RawKeypressMonitor,
        next_state: AssistantState,
        stream=None,
    ) -> list[np.ndarray]:
        if stream is None:
            self._record_response(response)

        if not self.config.audio.tts_enabled:
            self.chime_player.stop_at_gap()
            self._print_response(response)
            self._update_display_speech(0.0, False)
            self.set_state(next_state)
            return []

        self.set_state(AssistantState.SPEAKING)
        playback = play_tts_stream(
            tts_stream=stream if stream is not None else self._tts_stream(response),
            cancellation=self._turn_token,
            deadline=self._turn_deadline,
            reference_seconds=self.config.audio.playback_reference_s,
            pre_roll_ms=self.config.audio.pre_roll_ms,
            output_device=self.output_device,
            health_check=self._check_audio_health,
            aec_delay_ms=self.config.audio.aec_delay_ms,
            microphone=microphone,
            vad=self.vad,
            keypress_monitor=keypress_monitor,
            chime_player=self.chime_player,
            sample_rate=self.config.audio.sample_rate,
            use_aec=self.config.audio.aec_enabled,
            barge_in_threshold_chunks=self.config.audio.barge_in_threshold_chunks,
            playback_level_callback=self._update_display_speech,
        )
        self.event_logger.log("assistant_audio_complete", interrupted=playback.interrupted,
                              alignment_resets=playback.alignment_resets,
                              alignment_reset_reasons=playback.alignment_reset_reasons,
                              capture_discard_reasons=playback.capture_discard_reasons,
                              output_underflows=playback.output_underflows)
        self._update_display_speech(0.0, False)
        self.set_state(next_state)
        return list(playback.buffered_chunks)

    def _record_response(self, response):
        if self.config.logging.content_enabled:
            self.logger.info("Assistant: %s", response)
        self.last_assistant_text = response
        self.event_logger.log("assistant_response", content={"text": response}, chars=len(response))

    def _respond_with_speech(self, transcript, microphone, keypress_monitor):
        result = {}
        stream = speech_stream(infer=self._infer, worker=self.worker,
                               cancellation=self._turn_token or CancellationToken(),
                               deadline=self._turn_deadline, on_wait=self._playback_wait,
                               playback_events=True,
                               request=self._chat_request(transcript), on_result=result.update)
        buffered = self._deliver_response('', microphone=microphone,
                                          keypress_monitor=keypress_monitor,
                                          next_state=AssistantState.LISTENING, stream=stream)
        if self._turn_token is not None and self._turn_token.cancelled:
            return '', buffered
        if 'response' not in result:
            raise RuntimeError('Speech ended without a completed answer')
        self._log_tool_result(result)
        self._record_response(result['response'])
        return result['response'], buffered

    def _print_response(self, response: str) -> None:
        # Interactive text-only use still needs an answer. Redirected output
        # (including launchd/managed runtime logs) must respect content opt-in.
        if sys.stdout.isatty() or self.config.logging.content_enabled:
            print(f"\n> {response}\n", flush=True)

    def _store_turn(self, transcript: str, response: str) -> None:
        self.session.history.extend(
            [
                ConversationTurn(role="user", content=transcript),
                ConversationTurn(role="assistant", content=response),
            ]
        )
        max_turns = self.config.assistant.max_history_turns * 2
        if max_turns == 0:
            self.session.history.clear()
        elif len(self.session.history) > max_turns:
            self.session.history = self.session.history[-max_turns:]
        self.session.turn_count += 1

    def _remember_turn(self, transcript: str, response: str) -> None:
        if self.config.memory.enabled:
            self.memory_jobs.submit(transcript, response, self.session.turn_count)

    def _handle_active_turn(
        self,
        transcript: str,
        microphone: MicrophoneStream,
        keypress_monitor: RawKeypressMonitor,
    ) -> UtteranceResult:
        self._touch_session()
        self.set_state(AssistantState.THINKING)
        self.chime_player.start()
        self.last_user_transcript = transcript

        end_session_requested = False
        delivered = False
        if not transcript:
            response = "I didn't catch that. Could you repeat it?"
        elif GOODBYE_PATTERN.fullmatch(transcript.strip().replace("’", "'")):
            response = self.next_goodbye()
            end_session_requested = True
        elif self.config.audio.tts_enabled:
            response, buffered_chunks = self._respond_with_speech(transcript, microphone, keypress_monitor)
            delivered = True
        else:
            response = self._respond(transcript)
            if not response:
                response = "I don't have a good answer for that yet."
            response = sanitize_for_speech(response)

        if not delivered:
            buffered_chunks = self._deliver_response(
                response,
                microphone=microphone,
                keypress_monitor=keypress_monitor,
                next_state=AssistantState.LISTENING,
            )
        if (transcript and not end_session_requested
                and not (self._turn_token is not None and self._turn_token.cancelled)):
            self._store_turn(transcript, response)
            if not buffered_chunks:
                self._remember_turn(transcript, response)
        self._touch_session()
        return UtteranceResult(
            buffered_chunks=buffered_chunks,
            end_session_requested=end_session_requested,
        )

    def _handle_wake_match(
        self,
        wake_match: WakeWordMatch,
        microphone: MicrophoneStream,
        keypress_monitor: RawKeypressMonitor,
    ) -> UtteranceResult:
        self.event_logger.log(
            "wake_detected",
            phrase=wake_match.phrase,
            inline_command=bool(wake_match.prompt_text),
        )
        self._open_session(wake_match)

        if wake_match.prompt_text and self.config.wake_word.allow_inline_commands:
            self.last_user_transcript = wake_match.prompt_text
            return self._handle_active_turn(wake_match.prompt_text, microphone, keypress_monitor)

        acknowledgement = self.config.wake_word.acknowledgement.strip() or self.next_greeting()
        buffered_chunks = self._deliver_response(
            acknowledgement,
            microphone=microphone,
            keypress_monitor=keypress_monitor,
            next_state=AssistantState.LISTENING,
        )
        self._touch_session()
        return UtteranceResult(buffered_chunks=buffered_chunks)

    def _process_utterance(
        self,
        audio: np.ndarray,
        microphone: MicrophoneStream,
        keypress_monitor: RawKeypressMonitor,
    ) -> UtteranceResult:
        started_at = time.monotonic()
        transcript = self._infer("transcribe", audio).strip()
        stt_latency_ms = int((time.monotonic() - started_at) * 1000)

        if not self.session.active and self.config.wake_word.enabled:
            if not transcript:
                self.set_state(AssistantState.WAITING_FOR_WAKE)
                return UtteranceResult(buffered_chunks=[])
            wake_match = self.wake_detector.match(transcript)
            if wake_match is None:
                self.last_user_transcript = ""
                self.event_logger.log("wake_miss", latency_ms=stt_latency_ms)
                self.set_state(AssistantState.WAITING_FOR_WAKE)
                return UtteranceResult(buffered_chunks=[])
            self._log_transcript(wake_match.prompt_text or "", stt_latency_ms)
            return self._handle_wake_match(wake_match, microphone, keypress_monitor)

        self._log_transcript(transcript, stt_latency_ms)
        if not self.session.active:
            self._open_session(None)
        return self._handle_active_turn(transcript, microphone, keypress_monitor)

    def _log_transcript(self, transcript: str, latency_ms: int) -> None:
        if self.config.logging.content_enabled:
            self.logger.info("Heard: %s", transcript or "<empty>")
        self.event_logger.log("user_transcript", content={"text": transcript},
                              chars=len(transcript), latency_ms=latency_ms)

    def process_utterance(self, audio: np.ndarray, microphone: MicrophoneStream,
                          keypress_monitor: RawKeypressMonitor) -> UtteranceResult:
        self._turn_token = CancellationToken()
        self._turn_keys = keypress_monitor
        self._turn_thread_ident = threading.get_ident()
        self._turn_microphone = microphone
        self.turn_started_at = time.monotonic()
        self._turn_deadline = self.turn_started_at + self.config.runtime.turn_timeout_s
        self.last_error = None
        try:
            self.memory_jobs.suspend()
            if len(audio) > self.config.audio.max_utterance_s * self.config.audio.sample_rate:
                self.utterances_discarded += 1
                raise ValueError("Utterance exceeds the audio limit")
            return self._process_utterance(audio, microphone, keypress_monitor)
        except DEVICE_ERRORS:
            self.cancel_turn()
            raise
        except WorkerCancelled:
            self.chime_player.stop()
            microphone.drain()
            self.vad.reset()
            self._update_display_speech(0, False)
            self._touch_session()
            self.set_state(AssistantState.LISTENING if self.session.active or not self.config.wake_word.enabled
                           else AssistantState.WAITING_FOR_WAKE)
            return UtteranceResult(buffered_chunks=[])
        except Exception as exc:
            self._recover(exc, microphone)
            return UtteranceResult(buffered_chunks=[])
        finally:
            if self._turn_token.cancelled:
                self.turn_cancellations += 1
                self.event_logger.log("turn_cancelled")
            self._turn_token = self._turn_keys = self._turn_thread_ident = None
            self._turn_microphone = None
            capture_phase(microphone, "listening")
            self.last_turn_duration_ms = round((time.monotonic() - self.turn_started_at) * 1000)
            self.turn_started_at = self._turn_deadline = None
            self.turn_stage = "idle"
            self._publish_status(force=True)

    def _recover(self, exc: Exception, microphone) -> None:
        self.turn_failures += 1
        timed_out = isinstance(exc, (WorkerTimeout, TimeoutError))
        self.turn_timeouts += int(timed_out)
        self.last_error = "TurnTimeout" if timed_out else type(exc).__name__
        self.event_logger.log("turn_failed", stage=self.turn_stage, error_type=self.last_error,
                              content={"error": str(exc)})
        self.logger.warning("Turn failed during %s (%s); listening will resume", self.turn_stage, self.last_error)
        # Do not replay a possibly completed external action or send partial
        # audio to STT. A new user turn is required after a failure.
        try:
            self.chime_player.stop()
        except Exception:
            self.logger.warning("Could not stop the thinking chime")
        self._update_display_speech(0, False)
        microphone.drain()
        try:
            self.vad.reset()
        except Exception:
            self.logger.warning("Could not reset voice activity detection")
        if self.session.active:
            self._print_response("That turn failed. Please check any external action before trying again.")
        self._touch_session()
        self.set_state(AssistantState.LISTENING if self.session.active or not self.config.wake_word.enabled
                       else AssistantState.WAITING_FOR_WAKE)

    def reliability_snapshot(self) -> dict:
        started_at = self.turn_started_at
        microphone = self.microphone
        stage_limit = self.config.runtime.turn_timeout_s
        if self.turn_stage in {"endpoint", "loading_models"}:
            started_at = self._stage_started_at
            stage_limit = (self.config.runtime.endpoint_timeout_s if self.turn_stage == "endpoint"
                           else self.config.runtime.startup_timeout_s)
        return {"turn_stage": self.turn_stage,
                "turn_elapsed_s": round(time.monotonic() - started_at, 1) if started_at else None,
                "turn_timeout_s": self.config.runtime.turn_timeout_s,
                "stage_timeout_s": stage_limit,
                "turn_failures": self.turn_failures, "turn_cancellations": self.turn_cancellations, "turn_timeouts": self.turn_timeouts,
                "utterances_discarded": self.utterances_discarded,
                "last_turn_duration_ms": self.last_turn_duration_ms,
                "worker_pid": self.worker.pid, "worker_restarts": self.worker.restarts,
                "memory": self.memory_jobs.snapshot(),
                "devices": {"connection": self.device_connection,
                            "reconnect_attempts": self.audio_reconnect_attempts,
                            "recoveries": self.audio_recoveries,
                            "last_error": self.audio_device_error,
                            "input": self.selected_input, "output": self.selected_output},
                "audio": microphone.statistics() if microphone is not None else self._last_audio_statistics}

    def run(self) -> None:
        self.status_publisher.start(self)
        try:
            self.warmup()
            with RawKeypressMonitor() as keypress_monitor:
                self._run_audio(keypress_monitor)
        except KeyboardInterrupt:
            self.set_state(AssistantState.STOPPING)
        except Exception as exc:
            self.last_error = type(exc).__name__
            raise
        finally:
            if self.microphone is not None:
                self._last_audio_statistics = self.microphone.statistics()
            self.microphone = None
            self.device_connection = "stopped"
            self.turn_stage = "stopped"
            with ExitStack() as cleanup:
                cleanup.callback(self.status_publisher.stop, self)
                cleanup.callback(self.worker.close)
                cleanup.callback(self.memory_jobs.close)
                cleanup.callback(self.chime_player.stop)

    def _run_audio(self, keypress_monitor):
        # Recording has one owner for this invocation, including reconnects.
        # Reopening a microphone must never truncate the requested WAV file.
        recorder = None
        if self.options.record_path:
            recorder = RecordingWriter(self.options.record_path, self.config.audio.sample_rate,
                                       self.config.audio.recording_queue_max_chunks,
                                       self.config.audio.max_recording_s)
            recorder.start()
        failures = 0
        try:
            while True:
                try:
                    if failures:
                        refresh_audio_backend()
                    source, output = select_devices(self.config.audio)
                    self.selected_input, self.selected_output = source['selector'], output['selector'] if output else None
                    self.output_device = output['index'] if output else None
                    self.chime_player.device = self.output_device
                    with MicrophoneStream(self.config.audio.sample_rate, self.config.audio.chunk_samples,
                            device=source['index'], stall_s=self.config.audio.device_stall_s,
                            max_queue_chunks=self.config.audio.queue_max_chunks, recorder=recorder) as microphone:
                        self.microphone = microphone
                        # Require a fresh callback before reporting recovery.
                        while microphone.get(timeout=0.1) is None:
                            self._publish_status()
                        self.reset_session()
                        self.vad.reset()
                        if failures:
                            self.audio_recoveries += 1
                        recovered = bool(failures)
                        failures = 0
                        self.device_connection = "connected"
                        self.audio_device_error = self.last_error = None
                        self.turn_stage = "idle"
                        self.set_state(AssistantState.WAITING_FOR_WAKE if self.config.wake_word.enabled else AssistantState.LISTENING)
                        if not self.config.wake_word.enabled:
                            self._open_session(None)
                        self.event_logger.log("audio_connected", recovery=recovered)
                        try:
                            self._listen(microphone, keypress_monitor)
                            return
                        finally:
                            self._last_audio_statistics = microphone.statistics()
                except DEVICE_ERRORS as exc:
                    self.cancel_turn()
                    self.memory_jobs.suspend()
                    # All microphone/output contexts have unwound before refresh.
                    try:
                        self.chime_player.stop()
                    except DEVICE_ERRORS:
                        pass
                    self.reset_session()
                    self.vad.reset()
                    self._update_display_speech(0, False)
                    failures += 1
                    self.audio_reconnect_attempts += 1
                    self.device_connection = "waiting"
                    self.audio_device_error = (str(exc) if isinstance(exc, AudioDeviceError)
                                               else type(exc).__name__)
                    self.last_error = self.audio_device_error
                    self.turn_stage = "waiting_for_audio"
                    self.set_state(AssistantState.WAITING_FOR_AUDIO)
                    self.event_logger.log("audio_disconnected", error_type=self.audio_device_error,
                                          attempt=self.audio_reconnect_attempts)
                    self._publish_status(force=True)
                    delay = min(30., self.config.audio.reconnect_delay_s * 2 ** min(failures-1, 5))
                    time.sleep(delay)
                finally:
                    self.microphone = None
        finally:
            if recorder is not None:
                recorder.close()

    def _listen(self, microphone, keypress_monitor) -> None:
        silence_limit = max(1, int(self.config.audio.silence_ms * self.config.audio.sample_rate /
                                   (self.config.audio.chunk_samples * 1000)))
        buffer = UtteranceBuffer(int(self.config.audio.max_utterance_s * self.config.audio.sample_rate))
        pre_roll = AudioReferenceBuffer(max(1, int(self.config.audio.pre_roll_ms * self.config.audio.sample_rate / 1000)))
        def clear_pre_roll():
            pre_roll.samples = np.zeros(0, dtype=np.float32)
            pre_roll.end = 0
        silent_chunks = 0
        discard_until_quiet = False
        observed_drops = microphone.dropped_chunks + microphone.input_overflows
        while True:
            capture_phase(microphone, "listening")
            if self.config.wake_word.enabled and self.session.active and self._session_timed_out(time.monotonic()):
                self.memory_jobs.suspend()
                self._end_session(reason="idle_timeout", microphone=microphone, speak_goodbye=False)
                buffer.clear()
                clear_pre_roll()
                silent_chunks = 0
            chunk = microphone.get(timeout=0.1)
            if chunk is None or len(chunk) != self.config.audio.chunk_samples:
                if chunk is not None:
                    buffer.clear()
                    clear_pre_roll()
                    discard_until_quiet = True
                    silent_chunks = 0
                    self.vad.reset()
                continue
            drops = microphone.dropped_chunks + microphone.input_overflows
            if drops != observed_drops:
                observed_drops = drops
                buffer.clear()
                clear_pre_roll()
                discard_until_quiet = True
                silent_chunks = 0
                self.event_logger.log("audio_gap", dropped_chunks=drops)
            try:
                is_speech = self.vad.probability(chunk) > 0.5
                if is_speech:
                    self.memory_jobs.suspend(wait=False)
                elif not buffer.samples and not discard_until_quiet:
                    self.memory_jobs.allow_idle()
                silent_chunks = 0 if is_speech else silent_chunks + 1
                if discard_until_quiet:
                    if silent_chunks >= silence_limit:
                        discard_until_quiet = False
                        self.vad.reset()
                    continue
                if not buffer.samples and not is_speech:
                    if self.config.audio.pre_roll_ms:
                        pre_roll.append(chunk)
                    continue
                if not buffer.samples and len(pre_roll.samples):
                    buffer.append(pre_roll.samples.copy())
                    clear_pre_roll()
                if not buffer.append(chunk):
                    self.utterances_discarded += 1
                    self.event_logger.log("utterance_limit", max_seconds=self.config.audio.max_utterance_s)
                    buffer.clear()
                    discard_until_quiet = True
                    self.vad.reset()
                    continue
                if is_speech or silent_chunks < silence_limit:
                    continue
                utterance = buffer.audio()
                if self.config.audio.smart_turn_enabled:
                    self.memory_jobs.suspend()
                    self.turn_stage = "endpoint"
                    capture_phase(microphone, "endpoint")
                    self._stage_started_at = time.monotonic()
                    endpoint = self.worker.call("endpoint", utterance,
                        timeout_s=self.config.runtime.endpoint_timeout_s, on_wait=self._playback_wait)
                    self._check_audio_health()
                    self.event_logger.log("smart_turn_score", probability=round(endpoint, 4))
                    self.turn_stage = "idle"
                    capture_phase(microphone, "listening")
                    if endpoint < 0.5:
                        silent_chunks = 0
                        continue
                result = self.process_utterance(utterance, microphone, keypress_monitor)
                if not result.buffered_chunks:
                    microphone.drain()
                buffer.clear()
                clear_pre_roll()
                if result.end_session_requested:
                    self._end_session(reason="user_goodbye", microphone=microphone, speak_goodbye=False)
                else:
                    for buffered in result.buffered_chunks:
                        if not buffer.append(buffered):
                            buffer.clear()
                            discard_until_quiet = True
                            self.utterances_discarded += 1
                            break
                observed_drops = microphone.dropped_chunks + microphone.input_overflows
                silent_chunks = 0
                self.vad.reset()
            except DEVICE_ERRORS:
                raise
            except Exception as exc:
                buffer.clear()
                clear_pre_roll()
                silent_chunks = 0
                discard_until_quiet = True
                self._recover(exc, microphone)
                self.turn_stage = "idle"
                observed_drops = microphone.dropped_chunks + microphone.input_overflows
