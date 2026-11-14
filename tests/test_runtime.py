import logging
import time
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice import runtime as runtime_module
from arietta_voice.config import default_config
from arietta_voice.models import AssistantState
from arietta_voice.prompts import PromptManager
from arietta_voice.runtime_status import runtime_status_path


@pytest.mark.parametrize("wake_enabled", [False, True])
def test_device_failure_closes_before_refresh_clears_session_and_never_replays(
    runtime, monkeypatch, wake_enabled
):
    from arietta_voice.devices import AudioDeviceError
    from arietta_voice.turns import CancellationToken

    runtime.config.wake_word.enabled = wake_enabled
    order, waits = [], []
    opened = []
    token = CancellationToken()
    monkeypatch.setattr(runtime, "cancel_turn", token.cancel)
    runtime.memory_jobs = Mock(snapshot=Mock(return_value={}))

    def select(_):
        return {"selector": "CoreAudio::Mic", "index": len(opened) + 1}, {
            "selector": "CoreAudio::Speaker",
            "index": 9,
        }

    monkeypatch.setattr(runtime_module, "select_devices", select)

    class Mic:
        def __init__(self, *a, **kw):
            self.device = kw["device"]

        def __enter__(self):
            opened.append(self.device)
            order.append("open")
            return self

        def __exit__(self, *a):
            order.append("close")

        def get(self, **kw):
            return np.zeros(512)

        def statistics(self):
            return {}

    monkeypatch.setattr(runtime_module, "MicrophoneStream", Mic)
    monkeypatch.setattr(runtime_module, "refresh_audio_backend", lambda: order.append("refresh"))

    def backoff(delay):
        assert runtime.state == AssistantState.WAITING_FOR_AUDIO
        assert not runtime.session.active and not runtime.session.history
        assert runtime.last_user_transcript == ""
        assert token.cancelled
        waits.append(delay)

    monkeypatch.setattr(runtime_module.time, "sleep", backoff)

    def listen(mic, keys):
        assert runtime.output_device == runtime.chime_player.device == 9
        assert runtime.session.active is (not wake_enabled)
        assert not runtime.session.history
        if len(opened) == 1:
            runtime._store_turn("old question", "old answer")
            runtime.last_user_transcript = "old"
            raise AudioDeviceError("InputStreamInactive")

    monkeypatch.setattr(runtime, "_listen", listen)
    runtime._run_audio(Mock())
    assert order == ["open", "close", "refresh", "open", "close"]
    assert opened == [1, 2] and waits == [1.0]
    assert runtime.audio_reconnect_attempts == runtime.audio_recoveries == 1
    assert runtime.device_connection == "connected"
    assert runtime.microphone is None
    runtime.memory_jobs.suspend.assert_called_once()


def test_missing_device_backs_off_without_falling_back(runtime, monkeypatch):
    from arietta_voice.devices import AudioDeviceError

    select = Mock(side_effect=AudioDeviceError("InputUnavailable"))
    mic = Mock()
    monkeypatch.setattr(runtime_module, "select_devices", select)
    monkeypatch.setattr(runtime_module, "MicrophoneStream", mic)
    monkeypatch.setattr(runtime_module, "refresh_audio_backend", lambda: None)
    delays = []

    def sleep(delay):
        delays.append(delay)
        if len(delays) == 8:
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime_module.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        runtime._run_audio(Mock())
    assert delays == [1, 2, 4, 8, 16, 30, 30, 30]
    mic.assert_not_called()
    assert runtime.audio_recoveries == 0


def test_open_stream_without_callback_is_not_reported_as_recovered(runtime, monkeypatch):
    from arietta_voice.devices import AudioDeviceError

    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    stream.get.side_effect = [None, AudioDeviceError("CaptureStalled")]
    stream.statistics.return_value = {}
    monkeypatch.setattr(
        runtime_module, "select_devices", lambda _: ({"selector": "Mic", "index": 1}, None)
    )
    monkeypatch.setattr(runtime_module, "MicrophoneStream", lambda *a, **kw: stream)
    monkeypatch.setattr(runtime_module.time, "sleep", Mock(side_effect=KeyboardInterrupt))
    listen = Mock()
    monkeypatch.setattr(runtime, "_listen", listen)
    with pytest.raises(KeyboardInterrupt):
        runtime._run_audio(Mock())
    listen.assert_not_called()
    stream.__exit__.assert_called_once()
    assert runtime.device_connection == "waiting"
    assert runtime.audio_recoveries == 0


def test_device_error_escapes_turn_recovery(runtime, monkeypatch):
    from arietta_voice.devices import AudioDeviceError

    monkeypatch.setattr(runtime, "_infer", Mock(side_effect=AudioDeviceError("CaptureStalled")))
    recover = Mock()
    monkeypatch.setattr(runtime, "_recover", recover)
    with pytest.raises(AudioDeviceError):
        runtime.process_utterance(np.zeros(512), Mock(), Mock())
    recover.assert_not_called()
    assert not runtime.session.history


class FakeWorker:
    pid = None
    restarts = 0
    transcript = ""
    response = "Answer"

    def call(self, operation, payload=None, **kwargs):
        if operation == "transcribe":
            return self.transcript
        if operation == "chat":
            return {"response": self.response, "source": "model"}
        if operation == "memory":
            return {"facts_added": 0, "consolidated": False}
        if operation == "endpoint":
            return 1.0

    def close(self):
        pass


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    config.audio.tts_enabled = False
    config.audio.smart_turn_enabled = False
    config.tools.enabled = ()
    config.model_tools.enabled = ()
    config.knowledge.enabled = False
    monkeypatch.setattr(runtime_module, "SileroVoiceActivityDetector", lambda *args: Mock())
    monkeypatch.setattr(runtime_module, "ChimePlayer", lambda *args: Mock())
    return runtime_module.AriettaVoiceRuntime(config, worker=FakeWorker())


@pytest.mark.parametrize("content_enabled", [False, True])
def test_passive_speech_never_reaches_logs_or_status(
    runtime, content_enabled, caplog, capsys
) -> None:
    runtime.config.logging.content_enabled = content_enabled
    runtime.event_logger.content_enabled = content_enabled
    runtime.worker.transcript = "PRIVATE ambient conversation"
    with caplog.at_level(logging.INFO):
        runtime.process_utterance(np.zeros(512), Mock(), Mock())
    runtime._publish_status(force=True)
    assert not runtime.session.active
    assert runtime.last_user_transcript == ""
    assert "PRIVATE" not in caplog.text
    assert "PRIVATE" not in capsys.readouterr().out
    assert "PRIVATE" not in runtime.config.logging.events_file.read_text()
    assert "PRIVATE" not in runtime_status_path(runtime.config).read_text()


@pytest.mark.parametrize(
    "stage,foreground,discarded",
    [
        ("transcribe", True, True),
        ("endpoint", True, False),
        ("chat", True, False),
        ("transcribe", False, False),
    ],
)
def test_only_foreground_transcription_wait_retires_capture(runtime, stage, foreground, discarded):
    import threading

    from arietta_voice.audio import MicrophoneStream

    mic = MicrophoneStream(16000, 512)
    mic._callback(np.ones((512, 1)), 512, None, False)
    runtime._turn_microphone = mic
    runtime._turn_thread_ident = threading.get_ident() if foreground else -1
    runtime.turn_stage = stage
    runtime._inference_wait()
    assert mic.empty() == discarded
    assert mic.dropped_chunks == 0
    assert mic.statistics()["discarded_by_reason"] == ({"transcribing": 1} if discarded else {})


@pytest.mark.parametrize("content_enabled", [False, True])
def test_active_content_respects_opt_in_everywhere(
    runtime, content_enabled, caplog, capsys, monkeypatch
) -> None:
    runtime.config.logging.content_enabled = content_enabled
    runtime.event_logger.content_enabled = content_enabled
    runtime.worker.transcript = "Bridget, PRIVATE_CMD -5.2*7!"
    respond = Mock(return_value="PRIVATE ANSWER")
    monkeypatch.setattr(runtime, "_respond", respond)
    with caplog.at_level(logging.INFO):
        runtime.process_utterance(np.zeros(512), Mock(), Mock())
    runtime._publish_status(force=True)
    respond.assert_called_once_with("PRIVATE_CMD -5.2*7!")
    for output in (
        caplog.text,
        capsys.readouterr().out,
        runtime.config.logging.events_file.read_text(),
        runtime_status_path(runtime.config).read_text(),
    ):
        assert ("PRIVATE" in output) is content_enabled


def test_text_only_response_is_visible_in_interactive_terminal(
    runtime, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(runtime_module.sys.stdout, "isatty", lambda: True)
    runtime._deliver_response(
        "Visible answer",
        microphone=Mock(),
        keypress_monitor=Mock(),
        next_state=AssistantState.LISTENING,
    )
    assert "Visible answer" in capsys.readouterr().out
    assert "Visible answer" not in runtime.config.logging.events_file.read_text()


@pytest.mark.parametrize("limit", [0, 1, 8])
def test_runtime_and_prompt_history_limits_agree(runtime, limit) -> None:
    runtime.config.assistant.max_history_turns = limit
    for index in range(10):
        runtime._store_turn(f"Question {index}", f"Answer {index}")
    assert len(runtime.session.history) == limit * 2
    messages = PromptManager(runtime.config).build_messages(runtime.session.history, "Next")
    assert len(messages) == limit * 2 + 1
    assert runtime.session.turn_count == 10


@pytest.mark.parametrize(
    "text, should_end",
    [
        ("Goodbye.", True),
        ("Thanks, goodbye!", True),
        ("I'm done.", True),
        ("That’s all for now.", True),
        ("Okay, see you later!", True),
        ("Bye bye", True),
        ("Talk to you later, thank you.", True),
        ("What does goodbye mean?", False),
        ("Don't say goodbye yet", False),
        ("I'm done with my homework; what next?", False),
        ("Write a poem called Goodbye", False),
        ("That is all the information I have", False),
    ],
)
def test_only_direct_farewells_end_session(runtime, text, should_end, monkeypatch) -> None:
    respond = Mock(return_value="Answer")
    monkeypatch.setattr(runtime, "_respond", respond)
    result = runtime._handle_active_turn(text, Mock(), Mock())
    assert result.end_session_requested is should_end
    assert respond.called is not should_end


def test_ending_session_clears_private_display_state(runtime) -> None:
    runtime.last_user_transcript = "PRIVATE"
    runtime.last_assistant_text = "PRIVATE"
    runtime.reset_session()
    assert runtime.last_user_transcript == runtime.last_assistant_text == ""


@pytest.mark.parametrize("stage", ["transcribe", "chat", "tts_next"])
@pytest.mark.parametrize("timeout", [False, True])
def test_failed_turn_is_not_stored_and_next_turn_recovers(runtime, monkeypatch, stage, timeout):
    from arietta_voice.worker import WorkerFailure, WorkerTimeout

    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = "Test command"
    original = runtime.worker.call
    budgets = []

    def fail(operation, payload=None, **kwargs):
        budgets.append(kwargs["timeout_s"])
        if operation == stage:
            raise (WorkerTimeout if timeout else WorkerFailure)("PRIVATE failure")
        return original(operation, payload, **kwargs)

    runtime.worker.call = fail
    if stage == "tts_next":

        def deliver(*args, **kwargs):
            runtime._infer("tts_next")
            return []

        monkeypatch.setattr(runtime, "_deliver_response", deliver)
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.turn_failures == 1
    assert runtime.turn_timeouts == int(timeout)
    assert not runtime.session.history
    assert runtime.state == AssistantState.LISTENING
    assert "PRIVATE" not in runtime.config.logging.events_file.read_text()
    assert budgets == sorted(budgets, reverse=True)
    runtime.worker.call = original
    monkeypatch.setattr(runtime, "_deliver_response", lambda *args, **kwargs: [])
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.session.turn_count == 1
    assert runtime.last_error is None
    assert runtime.turn_stage == "idle"


def test_delivered_turn_queues_memory_without_calling_model(runtime):
    runtime.config.wake_word.enabled = False
    runtime.config.memory.enabled = True
    runtime.worker.transcript = "Remember a synthetic fact"
    original = runtime.worker.call
    calls = []

    def record(operation, payload=None, **kwargs):
        calls.append(operation)
        assert not operation.startswith("memory")
        return original(operation, payload, **kwargs)

    runtime.worker.call = record
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.session.turn_count == 1
    assert runtime.memory_jobs.snapshot()["queued"] == 1
    assert runtime.last_error is None
    assert runtime.turn_failures == 0
    assert runtime.turn_stage == "idle"
    runtime.memory_jobs.close()


def test_disabled_memory_never_queues_private_text(runtime):
    runtime.config.memory.enabled = False
    runtime._remember_turn("Synthetic fact", "Answer")
    assert runtime.memory_jobs.snapshot()["queued"] == 0


class ScriptedMicrophone:
    dropped_chunks = 0
    input_overflows = 0

    def __init__(self, samples):
        self.samples = iter(samples)

    def get(self, **kwargs):
        try:
            value = next(self.samples)
        except StopIteration:
            raise KeyboardInterrupt()
        if value == 2:
            self.dropped_chunks += 1
        return np.full(512, value, dtype=np.float32)

    def drain(self):
        pass


@pytest.mark.parametrize("pre_roll_ms,expected", [(0, [1, 0]), (64, [0.2, 0.3, 1, 0])])
def test_listening_keeps_only_bounded_pre_roll(runtime, pre_roll_ms, expected):
    from arietta_voice.models import UtteranceResult

    runtime.config.audio.pre_roll_ms = pre_roll_ms
    runtime.config.audio.silence_ms = 32
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] >= 1)
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([0.1, 0.2, 0.3, 1, 0]), Mock())
    recorded = runtime.process_utterance.call_args.args[0]
    np.testing.assert_allclose(recorded[::512], expected)


@pytest.mark.parametrize("prefix", [[1, 1, 1, 1, 0], [1, 2, 1, 0]])
def test_discard_long_or_gapped_speech_until_quiet_then_accept_new_turn(runtime, prefix):
    from arietta_voice.models import UtteranceResult

    runtime.config.audio.max_utterance_s = 1536 / 16000
    runtime.config.audio.silence_ms = 32
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone(prefix + [1, 0]), Mock())
    assert runtime.process_utterance.call_count == 1
    assert len(runtime.process_utterance.call_args.args[0]) == 1024


def test_negative_endpoint_cannot_accumulate_unbounded_audio(runtime):
    runtime.config.audio.smart_turn_enabled = True
    runtime.config.audio.silence_ms = 32
    runtime.config.audio.max_utterance_s = 1536 / 16000
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.worker.call = Mock(return_value=0)
    runtime.process_utterance = Mock()
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1] + [0] * 20), Mock())
    runtime.process_utterance.assert_not_called()
    assert runtime.utterances_discarded == 1
    assert all(len(call.args[1]) <= 1536 for call in runtime.worker.call.call_args_list)


def test_endpoint_timeout_recovers_and_accepts_next_turn(runtime):
    from arietta_voice.models import UtteranceResult
    from arietta_voice.worker import WorkerTimeout

    runtime.config.audio.smart_turn_enabled = True
    runtime.config.audio.silence_ms = 32
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.worker.call = Mock(side_effect=[WorkerTimeout(), 1.0])
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1, 0, 0, 1, 0]), Mock())
    assert runtime.turn_timeouts == 1
    runtime.process_utterance.assert_called_once()


def _early_endpoint_runtime(runtime, scores, *, early_ms=64, silence_ms=320):
    """Two silent chunks trigger the early check; ten trigger the fallback."""
    from arietta_voice.models import UtteranceResult

    runtime.config.audio.smart_turn_enabled = True
    runtime.config.audio.silence_ms = silence_ms
    runtime.config.audio.endpoint_early_ms = early_ms
    runtime.config.audio.endpoint_early_threshold = 0.7
    runtime.config.audio.pre_roll_ms = 0
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.worker.call = Mock(side_effect=scores)
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    runtime.event_logger = Mock()
    return runtime


def _endpoint_events(runtime):
    return [
        call.kwargs
        for call in runtime.event_logger.log.call_args_list
        if call.args[0] == "smart_turn_score"
    ]


def test_confident_early_endpoint_answers_without_waiting_for_full_silence(runtime):
    _early_endpoint_runtime(runtime, [0.9])
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1, 0, 0]), Mock())
    runtime.process_utterance.assert_called_once()
    call = runtime.process_utterance.call_args
    assert len(call.args[0]) == 3 * 512  # One speech chunk and only two of silence.
    assert call.kwargs["timing"]["early_endpoint"] is True
    assert call.kwargs["timing"]["decided_at"] >= call.kwargs["timing"]["speech_end_at"]
    assert _endpoint_events(runtime) == [{"probability": 0.9, "early": True}]


def test_unsure_early_endpoint_waits_for_the_full_silence_window(runtime):
    # 0.6 would pass the normal 0.5 bar but not the stricter early one.
    _early_endpoint_runtime(runtime, [0.6, 0.6])
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1] + [0] * 9), Mock())
    runtime.process_utterance.assert_not_called()  # Nine silent chunks: still waiting.
    assert runtime.worker.call.call_count == 1

    _early_endpoint_runtime(runtime, [0.6, 0.6])
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1] + [0] * 10), Mock())
    runtime.process_utterance.assert_called_once()
    call = runtime.process_utterance.call_args
    assert len(call.args[0]) == 11 * 512
    assert call.kwargs["timing"]["early_endpoint"] is False
    assert [event["early"] for event in _endpoint_events(runtime)] == [True, False]


def test_speech_resuming_after_an_unsure_early_check_stays_in_the_same_turn(runtime):
    _early_endpoint_runtime(runtime, [0.2, 0.95])
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1, 0, 0, 1, 1, 0, 0]), Mock())
    runtime.process_utterance.assert_called_once()
    assert len(runtime.process_utterance.call_args.args[0]) == 7 * 512
    assert runtime.worker.call.call_count == 2


def test_early_endpoint_can_be_turned_off(runtime):
    _early_endpoint_runtime(runtime, [0.9], early_ms=0)
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1] + [0] * 10), Mock())
    runtime.process_utterance.assert_called_once()
    assert len(runtime.process_utterance.call_args.args[0]) == 11 * 512
    assert _endpoint_events(runtime) == [{"probability": 0.9, "early": False}]


def test_early_endpoint_needs_smart_turn_and_a_longer_fallback_window(runtime):
    # Without the endpoint model there is nothing to ask early.
    _early_endpoint_runtime(runtime, [])
    runtime.config.audio.smart_turn_enabled = False
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1] + [0] * 10), Mock())
    assert len(runtime.process_utterance.call_args.args[0]) == 11 * 512
    runtime.worker.call.assert_not_called()

    # An early window at or beyond the fallback would never fire first.
    _early_endpoint_runtime(runtime, [0.9], early_ms=320, silence_ms=320)
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1] + [0] * 10), Mock())
    assert _endpoint_events(runtime) == [{"probability": 0.9, "early": False}]


def test_turn_timing_is_logged_only_for_spoken_microphone_turns(runtime, monkeypatch):
    runtime.event_logger = Mock()
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: 12.5)

    runtime._turn_marks = {"speech_end_at": 10.0, "decided_at": 10.25, "stt_ms": 180}
    runtime._log_turn_timing()  # No audio was produced.
    runtime._turn_marks = {"first_audio_at": 11.0}
    runtime._log_turn_timing()  # Not a microphone turn.
    runtime.event_logger.log.assert_not_called()
    assert runtime.last_turn_timing is None

    runtime._turn_marks = {
        "speech_end_at": 10.0,
        "decided_at": 10.25,
        "early_endpoint": True,
        "stt_ms": 180,
        "first_audio_at": 11.1,
    }
    runtime._log_turn_timing()
    expected = {
        "endpoint_wait_ms": 250,
        "early_endpoint": True,
        "stt_ms": 180,
        "first_audio_ms": 1100,
        "total_ms": 2500,
    }
    runtime.event_logger.log.assert_called_once_with("turn_timing", **expected)
    assert runtime.last_turn_timing == expected
    assert runtime.reliability_snapshot()["last_turn_timing"] == expected
    assert runtime._turn_marks == {}


def test_first_playback_of_a_turn_marks_first_audio(runtime):
    from arietta_voice.audio import PlaybackResult

    runtime.event_logger = Mock()
    runtime._log_playback(PlaybackResult(interrupted=False, started_at=None))
    assert "first_audio_at" not in runtime._turn_marks
    runtime._log_playback(PlaybackResult(interrupted=False, started_at=5.0))
    runtime._log_playback(PlaybackResult(interrupted=False, started_at=9.0))
    assert runtime._turn_marks["first_audio_at"] == 5.0


def _provisional_turn(runtime, microphone, *, on_stage):
    """Run one early-decision turn; on_stage(operation) may add speech mid-call."""
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.event_logger = Mock()
    operations = []

    def call(operation, payload=None, **options):
        operations.append(operation)
        on_stage(operation)
        if options.get("on_wait") is not None:
            options["on_wait"]()  # The runtime polls while it waits on inference.
        if operation == "transcribe":
            return "Why does ice float"
        if operation == "chat":
            return {"response": "Answer", "source": "model"}

    runtime.worker.call = call
    now = time.monotonic()
    result = runtime.process_utterance(
        np.ones(512, dtype=np.float32),
        microphone,
        Mock(pressed=Mock(return_value=False)),
        timing={"speech_end_at": now, "decided_at": now, "early_endpoint": True},
    )
    return result, operations


def _logged(runtime, event):
    return [call for call in runtime.event_logger.log.call_args_list if call.args[0] == event]


@pytest.mark.parametrize("stage", ["transcribe", "chat"])
def test_speech_during_a_provisional_turn_discards_it_and_keeps_the_audio(runtime, stage):
    from tests.test_resume import QueuedMicrophone

    runtime.session.active = True
    runtime.config.audio.silence_ms = 60_000  # Keep the turn provisional throughout.
    microphone = QueuedMicrophone([0])

    def on_stage(operation):
        if operation == stage:
            microphone.push(1, 1, 0)

    result, operations = _provisional_turn(runtime, microphone, on_stage=on_stage)
    assert result.resumed and not result.end_session_requested
    assert [chunk[0] for chunk in result.buffered_chunks] == [0, 1, 1]
    assert microphone.pending == [0]  # Left for the listening loop, never drained.
    microphone.drain.assert_not_called()
    runtime.vad.reset.assert_not_called()
    assert operations == (["transcribe"] if stage == "transcribe" else ["transcribe", "chat"])
    if stage == "transcribe":
        assert not _logged(runtime, "user_transcript")  # Discarded before being logged.
    assert len(_logged(runtime, "turn_resumed")) == 1
    assert not _logged(runtime, "turn_cancelled") and not _logged(runtime, "turn_failed")
    assert (runtime.turn_resumes, runtime.turn_cancellations, runtime.turn_failures) == (1, 0, 0)
    assert runtime.session.history == [] and runtime._resume_watch is None
    assert runtime.state.value == "listening"
    assert runtime.reliability_snapshot()["turn_resumes"] == 1
    runtime.chime_player.start.assert_not_called()


def test_quiet_provisional_turn_completes_and_starts_the_chime_after_the_window(runtime):
    from tests.test_resume import QueuedMicrophone

    runtime.session.active = True
    runtime.config.audio.silence_ms = 0  # The window is already over at the first poll.
    microphone = QueuedMicrophone([0, 0])
    result, operations = _provisional_turn(runtime, microphone, on_stage=lambda operation: None)
    assert not result.resumed
    assert operations == ["transcribe", "chat"]
    assert len(runtime.session.history) == 2
    assert runtime.turn_resumes == 0
    runtime.chime_player.start.assert_called_once()


def test_only_early_decisions_are_provisional(runtime):
    runtime.config.audio.silence_ms = 700
    assert runtime._watch_for_resume({}) is None
    assert runtime._watch_for_resume({"early_endpoint": False, "speech_end_at": 5.0}) is None
    assert runtime._watch_for_resume({"early_endpoint": True}) is None
    watch = runtime._watch_for_resume({"early_endpoint": True, "speech_end_at": 5.0})
    assert watch.active and watch.window_end == pytest.approx(5.7)

    runtime._resume_watch = None
    assert runtime._discard_event() is None
    runtime._start_thinking_chime()
    runtime.chime_player.start.assert_called_once()  # Nothing provisional: chime at once.


def test_resumed_turn_continues_the_same_utterance_in_the_listening_loop(runtime):
    from arietta_voice.models import UtteranceResult

    _early_endpoint_runtime(runtime, [0.9, 0.9])
    heard_meanwhile = [np.full(512, value, dtype=np.float32) for value in (0, 1, 1)]
    runtime.process_utterance = Mock(
        side_effect=[
            UtteranceResult(buffered_chunks=heard_meanwhile, resumed=True),
            UtteranceResult(buffered_chunks=[]),
        ]
    )
    microphone = ScriptedMicrophone([1, 0, 0, 1, 0, 0])
    microphone.drain = Mock()
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(microphone, Mock())
    first, second = runtime.process_utterance.call_args_list
    assert len(first.args[0]) == 3 * 512
    # First attempt, what was heard while it ran, then the rest of the sentence.
    np.testing.assert_array_equal(second.args[0][::512], [1, 0, 0, 0, 1, 1, 1, 0, 0])
    assert second.kwargs["timing"]["early_endpoint"] is True
    assert microphone.drain.call_count == 1  # Only after the completed turn.
    assert runtime.vad.reset.call_count == 1


def test_resumed_audio_beyond_the_utterance_limit_is_discarded_not_spliced(runtime):
    from arietta_voice.models import UtteranceResult

    _early_endpoint_runtime(runtime, [0.9])
    runtime.config.audio.max_utterance_s = 4 * 512 / 16000
    too_much = [np.ones(512, dtype=np.float32)] * 2
    runtime.process_utterance = Mock(
        return_value=UtteranceResult(buffered_chunks=too_much, resumed=True)
    )
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1, 0, 0, 1]), Mock())
    assert runtime.process_utterance.call_count == 1
    assert runtime.utterances_discarded == 1


def test_startup_failure_stops_worker_and_publishes_final_status(runtime):
    import json

    runtime.warmup = Mock(side_effect=RuntimeError("failed warmup"))
    runtime.worker.close = Mock()
    runtime.chime_player.stop.side_effect = RuntimeError("failed chime cleanup")
    with pytest.raises(RuntimeError):
        runtime.run()
    runtime.worker.close.assert_called_once()
    payload = json.loads(runtime_status_path(runtime.config).read_text())
    assert payload["lifecycle"] == "stopped"
    assert payload["reliability"]["turn_stage"] == "stopped"
    assert runtime.status_publisher._thread is None


def test_cancelled_generation_is_not_a_failure_or_history_entry_and_next_turn_works(runtime):
    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = "Synthetic question"
    original = runtime.worker.call

    def cancel(operation, payload=None, **kwargs):
        if operation == "chat":
            runtime.cancel_turn()
            kwargs["cancellation"].check()
        return original(operation, payload, **kwargs)

    runtime.worker.call = cancel
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.turn_cancellations == 1
    assert runtime.turn_failures == runtime.turn_timeouts == 0
    assert not runtime.session.history
    assert runtime.state == AssistantState.LISTENING
    runtime.worker.call = original
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.session.turn_count == 1


def test_interrupted_reply_is_not_stored_or_memorized(runtime, monkeypatch):
    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = "Synthetic question"
    followup = [np.ones(512)]

    def deliver(*args, **kwargs):
        runtime.cancel_turn()
        return followup

    monkeypatch.setattr(runtime, "_deliver_response", deliver)
    remember = Mock()
    monkeypatch.setattr(runtime, "_remember_turn", remember)
    result = runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert result.buffered_chunks is followup
    assert runtime.turn_cancellations == 1
    assert not runtime.session.history
    remember.assert_not_called()


@pytest.mark.parametrize("outcome", ["complete", "cancel", "failure"])
def test_streamed_voice_only_commits_fully_delivered_answers(runtime, monkeypatch, outcome):
    import asyncio

    runtime.config.audio.tts_enabled = True
    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = "Synthetic question"
    original = runtime.worker.call
    events = iter(
        [
            {"kind": "clause", "text": "First sentence."},
            {
                "kind": "done",
                "result": {
                    "response": "First sentence.",
                    "source": "model",
                    "used_tool": None,
                    "model_tool": None,
                },
            },
        ]
    )
    frames = []

    def infer(operation, payload=None, **kwargs):
        kwargs["cancellation"].check()
        if operation == "chat_next":
            event = next(events)
            if outcome == "failure" and event["kind"] == "done":
                raise RuntimeError("Synthetic streaming failure")
            return event
        if operation == "tts_start":
            frames[:] = [(np.zeros(4096, dtype=np.float32), 24000), None]
        if operation == "tts_next":
            return frames.pop(0)
        return original(operation, payload, **kwargs)

    runtime.worker.call = infer

    def play(*, tts_stream, **kwargs):
        async def consume():
            try:
                async for samples, rate in tts_stream:
                    if samples is not None and outcome == "cancel":
                        runtime.cancel_turn()
                        break
            finally:
                await tts_stream.aclose()

        asyncio.run(consume())
        return PlaybackResult(interrupted=outcome == "cancel", buffered_chunks=[])

    from arietta_voice.audio import PlaybackResult

    monkeypatch.setattr(runtime_module, "play_tts_stream", play)
    runtime.process_utterance(np.zeros(512), Mock(), Mock(pressed=lambda: False))
    assert runtime.session.turn_count == (1 if outcome == "complete" else 0)
    assert runtime.turn_failures == (1 if outcome == "failure" else 0)
    assert runtime.turn_cancellations == (1 if outcome == "cancel" else 0)
    if outcome == "complete":
        assert runtime.session.history[-1].content == "First sentence."
