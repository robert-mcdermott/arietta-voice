"""Deterministic end-to-end contracts; this is not a model-quality benchmark."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice.config import default_config
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.memory import MemoryStore
from arietta_voice.models import ConversationTurn, ToolResponse
from arietta_voice.prompts import PromptManager, sanitize_for_speech
from arietta_voice.runtime import AriettaVoiceRuntime
from arietta_voice.wake import build_wake_word_detector
from arietta_voice.worker import WorkerFailure

CASES = json.loads(Path(__file__).with_name("cases.json").read_text())
assert CASES["schema_version"] == 1


def cases(section):
    return pytest.mark.parametrize("case", CASES[section], ids=lambda case: case["id"])


@cases("wake")
def test_wake_contract(case, tmp_path):
    config = default_config(tmp_path)
    result = build_wake_word_detector(config.wake_word).match(case["text"])
    if case["command"] is None:
        assert result is None
    else:
        assert result is not None
        assert (result.prompt_text or "") == case["command"]


@cases("prompts")
def test_prompt_contract(case, tmp_path):
    config = default_config(tmp_path)
    config.prompts.soul_path.parent.mkdir(parents=True)
    config.prompts.soul_path.write_text(case["soul"])
    config.prompts.memory_path.write_text(case["memory"])
    config.memory.enabled = case["enabled"]
    config.assistant.max_history_turns = case["limit"]
    history = [
        ConversationTurn(role=role, content=f"{role} {i}")
        for i in range(3)
        for role in ("user", "assistant")
    ]
    original = list(history)
    context = ToolResponse("synthetic", "", False, case["context"])
    messages = PromptManager(config).build_messages(history, "Current synthetic question", context)
    expected = []
    system = "\n\n".join(filter(None, [case["soul"], case["memory"] if case["enabled"] else ""]))
    if system:
        expected.append({"role": "system", "content": system})
    if case["context"]:
        expected.append({"role": "system", "content": case["context"]})
    retained = original[-case["limit"] * 2 :] if case["limit"] else []
    expected.extend({"role": turn.role, "content": turn.content} for turn in retained)
    expected.append({"role": "user", "content": "Current synthetic question"})
    assert messages == expected
    assert history == original


@cases("memory")
def test_memory_contract(case, tmp_path):
    path = tmp_path / "MEMORY.md"
    store = MemoryStore(path, case["enabled"], 3)
    generator = Mock(generate=Mock(return_value=case["output"]))
    store.ensure_exists()
    count = store.update_from_turn(generator, "Synthetic user fact", "Synthetic answer")
    assert count == len(case["facts"])
    if not case["enabled"]:
        assert not path.exists()
        generator.generate.assert_not_called()
    else:
        assert path.read_text() == "# Memory\n" + (
            "\n" + "\n".join(case["facts"]) + "\n" if case["facts"] else ""
        )


@cases("tool_policy")
def test_tool_policy_contract(case, tmp_path, monkeypatch):
    from arietta_voice.conversation import ConversationService
    from arietta_voice.tools import BaseTool, ToolResult
    from arietta_voice.tools import registry as tool_registry

    calls = []

    class Counter(BaseTool):
        name = "synthetic_counter"
        description = "read a synthetic counter"

        def run(self, arguments):
            calls.append(arguments)
            if case.get("fails"):
                raise RuntimeError("synthetic tool failure")
            return ToolResult(content="Synthetic counter result")

    module = SimpleNamespace(create_tool=lambda context: Counter(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ("synthetic_counter",) if case["enabled"] else ()
    config.model_tools.enabled = ()
    generate = Mock(side_effect=[case["output"], case.get("followup", "Synthetic answer")])
    service = ConversationService(config, generate=generate)
    if case["outcome"] == "error":
        with pytest.raises(ValueError, match="planning"):
            service.chat(message="Read the synthetic counter")
        assert generate.call_count == 1
    else:
        result = service.chat(message="Read the synthetic counter")
        planned = case["enabled"] and case["output"].lstrip().startswith(("{", "`"))
        assert generate.call_count == 1 + planned
        expected = case.get("followup", "Synthetic answer") if planned else case["output"]
        assert result.response == sanitize_for_speech(expected)
        if planned:
            followup = generate.call_args_list[1].args[0][-1]["content"]
            assert case.get("followup_note", "Synthetic counter result") in followup
    # One call at most, never retried, and never when the tool is disabled.
    assert len(calls) == case["calls"]


class ScriptedWorker:
    pid = None
    restarts = 0

    def __init__(self):
        self.step = {}
        self.commands = []

    def call(self, operation, payload=None, **kwargs):
        if operation == "chat":
            self.commands.append(payload["message"])
        if operation == self.step.get("fail"):
            raise WorkerFailure("synthetic model failure")
        if operation == "transcribe":
            return self.step["text"]
        if operation == "chat":
            return {"response": "Synthetic answer", "source": "model"}
        raise AssertionError(f"Unexpected operation: {operation}")


@cases("sessions")
def test_runtime_session_contract(case, tmp_path, monkeypatch):
    config = default_config(tmp_path)
    config.audio.tts_enabled = False
    config.audio.smart_turn_enabled = False
    config.memory.enabled = False
    config.wake_word.enabled = case.get("wake_enabled", True)
    monkeypatch.setattr("arietta_voice.runtime.SileroVoiceActivityDetector", lambda *args: Mock())
    monkeypatch.setattr("arietta_voice.runtime.ChimePlayer", lambda *args: Mock())
    worker = ScriptedWorker()
    runtime = AriettaVoiceRuntime(config, worker=worker)
    failures = 0
    for step in case["steps"]:
        worker.step = step
        previous_calls = len(worker.commands)
        result = runtime.process_utterance(np.zeros(512), Mock(), Mock())
        assert result.end_session_requested == step.get("end", False)
        if result.end_session_requested:
            runtime._end_session(reason="user_goodbye", microphone=Mock(), speak_goodbye=False)
        assert runtime.state.value == step["state"]
        assert runtime.session.active == step["active"]
        assert runtime.session.turn_count == step["turns"]
        assert len(runtime.session.history) == step["turns"] * 2
        assert worker.commands[previous_calls:] == ([step["command"]] if "command" in step else [])
        failures += int("fail" in step)
        assert runtime.turn_failures == failures


@pytest.mark.parametrize("case", CASES["retrieval"]["queries"], ids=lambda case: case["id"])
def test_keyword_retrieval_contract(case, tmp_path):
    config = default_config(tmp_path).knowledge
    config.backend = "keyword"
    config.directory.mkdir()
    for name, text in CASES["retrieval"]["documents"].items():
        (config.directory / name).write_text(text)
    result = KnowledgeRetriever(config).search(case["text"])
    if case["source"] is None:
        assert result is None
    else:
        assert result is not None
        assert result.matches[0].source_path.as_posix() == case["source"]
