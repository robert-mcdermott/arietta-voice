import logging
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice import runtime as runtime_module
from arietta_voice.config import default_config
from arietta_voice.models import AssistantState
from arietta_voice.prompts import PromptManager
from arietta_voice.runtime_status import runtime_status_path


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    config.audio.tts_enabled = False
    config.audio.smart_turn_enabled = False
    config.tools.enabled = ()
    config.model_tools.enabled = ()
    config.knowledge.enabled = False
    monkeypatch.setattr(runtime_module, "SileroVoiceActivityDetector", lambda *args: Mock())
    monkeypatch.setattr(runtime_module, "MoonshineSpeechToText", lambda *args: Mock())
    monkeypatch.setattr(runtime_module, "build_chat_model", lambda *args: Mock())
    monkeypatch.setattr(runtime_module, "ChimePlayer", lambda *args: Mock())
    return runtime_module.AriettaVoiceRuntime(config)


@pytest.mark.parametrize("content_enabled", [False, True])
def test_passive_speech_never_reaches_logs_or_status(runtime, content_enabled, caplog, capsys) -> None:
    runtime.config.logging.content_enabled = content_enabled
    runtime.event_logger.content_enabled = content_enabled
    runtime.stt.transcribe.return_value = "PRIVATE ambient conversation"
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
    runtime.stt.transcribe.return_value = "Bridget, PRIVATE_CMD -5.2*7!"
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
