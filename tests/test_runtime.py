import logging
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice import runtime as runtime_module
from arietta_voice.config import default_config
from arietta_voice.models import AssistantState
from arietta_voice.prompts import PromptManager
from arietta_voice.runtime_status import runtime_status_path


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
def test_passive_speech_never_reaches_logs_or_status(runtime, content_enabled, caplog, capsys) -> None:
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


@pytest.mark.parametrize("content_enabled", [False, True])
def test_active_content_respects_opt_in_everywhere(runtime, content_enabled, caplog, capsys, monkeypatch) -> None:
    runtime.config.logging.content_enabled = content_enabled
    runtime.event_logger.content_enabled = content_enabled
    runtime.worker.transcript = "Bridget, PRIVATE_CMD -5.2*7!"
    respond = Mock(return_value="PRIVATE ANSWER")
    monkeypatch.setattr(runtime, "_respond", respond)
    with caplog.at_level(logging.INFO):
        runtime.process_utterance(np.zeros(512), Mock(), Mock())
    runtime._publish_status(force=True)
    respond.assert_called_once_with("PRIVATE_CMD -5.2*7!")
    for output in (caplog.text, capsys.readouterr().out,
                   runtime.config.logging.events_file.read_text(),
                   runtime_status_path(runtime.config).read_text()):
        assert ("PRIVATE" in output) is content_enabled


def test_text_only_response_is_visible_in_interactive_terminal(runtime, monkeypatch, capsys) -> None:
    monkeypatch.setattr(runtime_module.sys.stdout, "isatty", lambda: True)
    runtime._deliver_response("Visible answer", microphone=Mock(), keypress_monitor=Mock(),
                              next_state=AssistantState.LISTENING)
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


@pytest.mark.parametrize("text, should_end", [
    ("Goodbye.", True), ("Thanks, goodbye!", True), ("I'm done.", True),
    ("That’s all for now.", True), ("Okay, see you later!", True),
    ("Bye bye", True), ("Talk to you later, thank you.", True),
    ("What does goodbye mean?", False), ("Don't say goodbye yet", False),
    ("I'm done with my homework; what next?", False),
    ("Write a poem called Goodbye", False), ("That is all the information I have", False),
])
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


def test_memory_failure_does_not_erase_delivered_turn(runtime):
    from arietta_voice.worker import WorkerTimeout
    runtime.config.wake_word.enabled = False
    runtime.config.memory.enabled = True
    runtime.worker.transcript = "Remember a fact"
    original = runtime.worker.call
    def fail(operation, payload=None, **kwargs):
        if operation == "memory":
            raise WorkerTimeout()
        return original(operation, payload, **kwargs)
    runtime.worker.call = fail
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.session.turn_count == 1
    assert runtime.last_error == "MemoryUpdateFailed"
    assert runtime.turn_failures == 0


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
    from arietta_voice.worker import WorkerTimeout
    from arietta_voice.models import UtteranceResult
    runtime.config.audio.smart_turn_enabled = True
    runtime.config.audio.silence_ms = 32
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.worker.call = Mock(side_effect=[WorkerTimeout(), 1.0])
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1, 0, 0, 1, 0]), Mock())
    assert runtime.turn_timeouts == 1
    runtime.process_utterance.assert_called_once()


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
