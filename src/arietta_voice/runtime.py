from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import time

import numpy as np

from arietta_voice.audio import (
    ChimePlayer,
    MicrophoneStream,
    RawKeypressMonitor,
    play_tts_stream,
    play_tts_stream_passive,
)
from arietta_voice.config import AppConfig
from arietta_voice.diagnostics import run_startup_diagnostics
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.llm import build_chat_model
from arietta_voice.logging_utils import EventLogger
from arietta_voice.memory import MemoryStore
from arietta_voice.model_tools import ModelToolRegistry
from arietta_voice.model_tooling import maybe_generate_with_model_tools
from arietta_voice.models import AssistantState, ConversationTurn, SessionContext, UtteranceResult
from arietta_voice.prompts import PromptManager, sanitize_for_speech
from arietta_voice.stt import MoonshineSpeechToText
from arietta_voice.tools import ToolRegistry
from arietta_voice.tts import KokoroTextToSpeech
from arietta_voice.turn_detection import SileroVoiceActivityDetector, SmartTurnEndpointDetector
from arietta_voice.wake import WakeWordMatch, build_wake_word_detector


GOODBYE_PATTERN = re.compile(
    r"\b(?:goodbye|bye|see you|talk to you later|that's all|that is all|i'm done|i am done)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class RuntimeOptions:
    record_path: Path | None = None


class AriettaVoiceRuntime:
    def __init__(self, config: AppConfig, options: RuntimeOptions | None = None) -> None:
        self.config = config
        self.options = options or RuntimeOptions()
        self.started_at = time.monotonic()
        self.logger = logging.getLogger("arietta_voice.runtime")
        self.event_logger = EventLogger(config.logging.events_file)
        self.prompts = PromptManager(config)
        self.memory = MemoryStore(
            config.prompts.memory_path,
            enabled=config.memory.enabled,
            consolidate_every_turns=config.memory.consolidate_every_turns,
        )
        self.knowledge = KnowledgeRetriever(config.knowledge)
        self.tools = ToolRegistry(
            config,
            knowledge_retriever=self.knowledge if config.knowledge.enabled else None,
        )
        self.model_tools = ModelToolRegistry(
            config,
            knowledge_retriever=self.knowledge if config.knowledge.enabled else None,
        )
        self.vad = SileroVoiceActivityDetector(config.audio.sample_rate)
        self.smart_turn = (
            SmartTurnEndpointDetector(config.audio.sample_rate)
            if config.audio.smart_turn_enabled
            else None
        )
        self.stt = MoonshineSpeechToText(config.models.stt_language)
        self.llm = build_chat_model(config.models)
        self.speaker = (
            KokoroTextToSpeech(config.audio.voice, speed=config.audio.tts_speed)
            if config.audio.tts_enabled
            else None
        )
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
        self.vad.ensure_loaded()
        self.logger.info("Loading STT")
        self.stt.ensure_loaded()
        self.logger.info("Loading LLM")
        self.llm.ensure_loaded()
        if self.speaker is not None:
            self.logger.info("Loading TTS")
            self.speaker.ensure_loaded()
        if self.smart_turn is not None:
            self.logger.info("Loading Smart Turn")
            self.smart_turn.ensure_loaded()
        self.memory.ensure_exists()
        self.knowledge.preload()
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
            print(f"\n> {goodbye}\n", flush=True)
            self._speak_blocking(goodbye)

        self.event_logger.log(
            "session_ended",
            reason=reason,
            goodbye=goodbye,
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

    def _generate_chat_response(self, messages: list[dict[str, str]]) -> str:
        return self.llm.generate(
            messages,
            max_tokens=self.config.models.max_tokens,
            temperature=self.config.models.temperature,
        )

    def _respond(self, user_text: str) -> str:
        tool_response = self.tools.maybe_handle(user_text)
        if tool_response and tool_response.bypass_model:
            self.event_logger.log("tool_call", tool=tool_response.name, mode="direct")
            return tool_response.content
        if tool_response:
            self.event_logger.log("tool_call", tool=tool_response.name, mode="context")

        messages = self.prompts.build_messages(self.session.history, user_text, tool_response)
        response, model_tool_use = maybe_generate_with_model_tools(
            messages,
            model_tool_registry=self.model_tools,
            llm_generate=self._generate_chat_response,
        )
        if model_tool_use is not None:
            self.event_logger.log(
                "tool_call",
                tool=model_tool_use.name,
                mode="model_requested",
                **model_tool_use.details,
            )
        return sanitize_for_speech(response)

    def _speak_blocking(self, text: str) -> None:
        if self.speaker is None:
            return

        play_tts_stream_passive(
            tts_stream=self.speaker.create_stream(text),
            chime_player=self.chime_player,
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
    ) -> list[np.ndarray]:
        self.logger.info("Assistant: %s", response)
        self.last_assistant_text = response
        self.event_logger.log("assistant_response", text=response)

        if self.speaker is None:
            self.chime_player.stop_at_gap()
            print(f"\n> {response}\n", flush=True)
            self._update_display_speech(0.0, False)
            self.set_state(next_state)
            return []

        self.set_state(AssistantState.SPEAKING)
        playback = play_tts_stream(
            tts_stream=self.speaker.create_stream(response),
            microphone=microphone,
            vad=self.vad,
            keypress_monitor=keypress_monitor,
            chime_player=self.chime_player,
            sample_rate=self.config.audio.sample_rate,
            use_aec=self.config.audio.aec_enabled,
            barge_in_threshold_chunks=self.config.audio.barge_in_threshold_chunks,
            playback_level_callback=self._update_display_speech,
        )
        self.event_logger.log("assistant_audio_complete", interrupted=playback.interrupted)
        self._update_display_speech(0.0, False)
        self.set_state(next_state)
        return list(playback.buffered_chunks)

    def _store_turn(self, transcript: str, response: str) -> None:
        self.session.history.extend(
            [
                ConversationTurn(role="user", content=transcript),
                ConversationTurn(role="assistant", content=response),
            ]
        )
        max_turns = self.config.assistant.max_history_turns * 2
        if len(self.session.history) > max_turns:
            self.session.history = self.session.history[-max_turns:]
        self.session.turn_count += 1

        memory_updates = self.memory.update_from_turn(self.llm, transcript, response)
        if memory_updates:
            self.event_logger.log("memory_updated", facts_added=memory_updates)
        if self.memory.maybe_consolidate(self.llm, self.session.turn_count):
            self.event_logger.log("memory_consolidated", turn_count=self.session.turn_count)

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
        if not transcript:
            response = "I didn't catch that. Could you repeat it?"
        elif GOODBYE_PATTERN.search(transcript):
            response = self.next_goodbye()
            end_session_requested = True
        else:
            response = self._respond(transcript)
            if not response:
                response = "I don't have a good answer for that yet."
            response = sanitize_for_speech(response)
            self._store_turn(transcript, response)

        buffered_chunks = self._deliver_response(
            response,
            microphone=microphone,
            keypress_monitor=keypress_monitor,
            next_state=AssistantState.LISTENING,
        )
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

    def process_utterance(
        self,
        audio: np.ndarray,
        microphone: MicrophoneStream,
        keypress_monitor: RawKeypressMonitor,
    ) -> UtteranceResult:
        started_at = time.monotonic()
        transcript = sanitize_for_speech(self.stt.transcribe(audio, self.config.audio.sample_rate))
        stt_latency_ms = int((time.monotonic() - started_at) * 1000)
        self.logger.info("Heard: %s", transcript or "<empty>")
        self.event_logger.log("user_transcript", text=transcript, latency_ms=stt_latency_ms)

        if not self.session.active and self.config.wake_word.enabled:
            if not transcript:
                self.set_state(AssistantState.WAITING_FOR_WAKE)
                return UtteranceResult(buffered_chunks=[])
            wake_match = self.wake_detector.match(transcript)
            if wake_match is None:
                self.last_user_transcript = transcript
                self.event_logger.log("wake_miss", text=transcript)
                self.set_state(AssistantState.WAITING_FOR_WAKE)
                return UtteranceResult(buffered_chunks=[])
            return self._handle_wake_match(wake_match, microphone, keypress_monitor)

        if not self.session.active:
            self._open_session(None)
        return self._handle_active_turn(transcript, microphone, keypress_monitor)

    def run(self) -> None:
        self.warmup()

        silence_limit = max(
            1,
            int(
                self.config.audio.silence_ms
                / (self.config.audio.chunk_samples / self.config.audio.sample_rate * 1000)
            ),
        )

        speech_buffer: list[np.ndarray] = []
        speaking = False
        silent_chunks = 0

        with RawKeypressMonitor() as keypress_monitor, MicrophoneStream(
            self.config.audio.sample_rate,
            self.config.audio.chunk_samples,
            record_path=self.options.record_path,
        ) as microphone:
            try:
                if self.config.wake_word.enabled:
                    wake_words = ", ".join(self.config.wake_word.phrases)
                    print(f"Listening for wake word(s): {wake_words}. Press Ctrl+C to quit.\n", flush=True)
                else:
                    self._open_session(None)
                    print("Wake word disabled. Listening. Press Ctrl+C to quit.\n", flush=True)

                while True:
                    now = time.monotonic()
                    if (
                        self.config.wake_word.enabled
                        and self.session.active
                        and self.state not in {AssistantState.THINKING, AssistantState.SPEAKING}
                        and self._session_timed_out(now)
                    ):
                        self._end_session(
                            reason="idle_timeout",
                            microphone=microphone,
                            speak_goodbye=False,
                        )
                        speech_buffer.clear()
                        speaking = False
                        silent_chunks = 0

                    chunk = microphone.get(timeout=0.1)
                    if chunk is None:
                        continue
                    if len(chunk) < self.config.audio.chunk_samples:
                        continue

                    speech_probability = self.vad.probability(chunk)
                    if speech_probability > 0.5:
                        speaking = True
                        silent_chunks = 0
                        speech_buffer.append(chunk)
                        continue

                    if not speaking:
                        continue

                    silent_chunks += 1
                    speech_buffer.append(chunk)
                    if silent_chunks < silence_limit:
                        continue

                    utterance = np.concatenate(speech_buffer)
                    if self.smart_turn is not None:
                        endpoint_probability = self.smart_turn.probability(utterance)
                        self.event_logger.log(
                            "smart_turn_score",
                            probability=round(endpoint_probability, 4),
                        )
                        if endpoint_probability < 0.5:
                            silent_chunks = 0
                            continue

                    result = self.process_utterance(utterance, microphone, keypress_monitor)
                    if result.end_session_requested:
                        self._end_session(
                            reason="user_goodbye",
                            microphone=microphone,
                            speak_goodbye=False,
                        )
                        speech_buffer.clear()
                        speaking = False
                        silent_chunks = 0
                    else:
                        speech_buffer = list(result.buffered_chunks)
                        speaking = bool(speech_buffer)
                        silent_chunks = 0
                    self.vad.reset()
            except KeyboardInterrupt:
                self.set_state(AssistantState.STOPPING)
                if self.session.active:
                    goodbye = self.next_goodbye()
                    print(f"\n> {goodbye}\n", flush=True)
                    self._speak_blocking(goodbye)
            except Exception as exc:
                self.last_error = type(exc).__name__
                raise
