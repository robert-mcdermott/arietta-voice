from unittest.mock import Mock

from arietta_voice.config import default_config
from arietta_voice.conversation import ConversationService
from arietta_voice.inference import InferenceEngine
from arietta_voice.models import ToolResponse


def test_voice_and_console_share_policy_and_request_is_not_mutated(tmp_path):
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    generated = []

    def generate(messages, **options):
        generated.append((messages, options))
        return "**Synthetic** answer"

    engine = InferenceEngine(config)
    engine._service = ConversationService(config, generate=generate)
    payload = dict(
        message="  Next  ",
        history=[{"role": "user", "content": " Previous "}],
        use_tools=False,
        max_tokens=17,
        temperature=0.0,
        enabled_tools=[],
        enabled_model_tools=[],
    )
    try:
        console = engine.handle("chat", payload)
        voice = engine.handle("chat", {**payload, "voice_turn": True})
        assert console == voice
        assert generated[0] == generated[1]
        assert generated[0][1] == {"max_tokens": 17, "temperature": 0.0}
        assert generated[0][0][-1]["content"] == "Next"
        assert "enabled_tools" in payload
    finally:
        engine._loop.close()


def test_shared_service_retains_tools_but_drops_disabled_registry(tmp_path, monkeypatch):
    from arietta_voice import tools

    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ("synthetic",)
    config.model_tools.enabled = ()
    active = Mock(maybe_handle=Mock(return_value=ToolResponse("synthetic", "Tool answer")))
    empty = Mock(maybe_handle=Mock(return_value=None))
    factory = Mock(side_effect=[active, empty])
    monkeypatch.setattr(tools, "ToolRegistry", factory)
    service = ConversationService(config, generate=lambda *a, **kw: "Model answer")
    assert service.chat(message="First").source == "tool"
    assert service.chat(message="Second").source == "tool"
    assert factory.call_count == 1
    config.tools.enabled = ()
    assert service.chat(message="Third").source == "model"
    assert factory.call_count == 2
    assert active.maybe_handle.call_count == 2


def test_stream_yields_clause_before_model_finishes_and_closes_on_cancel(tmp_path):
    from arietta_voice.llm import ModelEvent

    config = default_config(tmp_path)
    config.knowledge.enabled = False
    progress = []

    def tokens(*a, **kw):
        try:
            yield ModelEvent("text", "First sentence. ")
            progress.append("continued")
            yield ModelEvent("text", "Second sentence.")
            yield ModelEvent("done")
        finally:
            progress.append("closed")

    service = ConversationService(config)
    service._llm = Mock(stream=tokens)
    stream = service.stream(message="Synthetic", use_tools=False)
    assert next(stream) == {"kind": "clause", "text": "First sentence."}
    assert not progress
    stream.close()
    assert progress == ["closed"]


def _planning_service(tmp_path, monkeypatch, streams, registry=None):
    """A service with one model tool on offer and scripted provider streams."""
    from arietta_voice import model_tools

    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ()
    registry = registry or Mock(instruction_text=lambda: "Plan privately")
    monkeypatch.setattr(model_tools, "ModelToolRegistry", lambda *a, **kw: registry)
    service = ConversationService(config)
    calls = []

    def stream(messages, **options):
        calls.append(messages)
        return streams[len(calls) - 1]()

    service._llm = Mock(stream=stream)
    return service, calls


def test_tool_planning_is_withheld_and_only_final_answer_streams(tmp_path, monkeypatch):
    from arietta_voice.llm import ModelEvent

    registry = Mock(
        instruction_text=lambda: "Plan privately",
        execute=Mock(return_value=Mock(followup_context="Synthetic tool result")),
    )
    # The request arrives in pieces, as a provider would stream it.
    request = ['{"tool_', 'call":{"name":"synth', 'etic","arguments":{}}}']
    service, calls = _planning_service(
        tmp_path,
        monkeypatch,
        [
            lambda: iter([*(ModelEvent("text", part) for part in request), ModelEvent("done")]),
            lambda: iter([ModelEvent("text", "Final answer. More."), ModelEvent("done")]),
        ],
        registry,
    )
    events = list(service.stream(message="Synthetic"))
    assert [e["text"] for e in events if e["kind"] == "clause"] == ["Final answer.", "More."]
    assert events[-1]["result"]["model_tool"] == "synthetic"
    registry.execute.assert_called_once_with("synthetic", {})
    assert calls[0][0] == {"role": "system", "content": "Plan privately"}
    assert calls[1][-1] == {"role": "system", "content": "Synthetic tool result"}


def test_fenced_tool_request_is_recognised_from_its_first_character(tmp_path, monkeypatch):
    from arietta_voice.llm import ModelEvent

    registry = Mock(
        instruction_text=lambda: "Plan privately",
        execute=Mock(return_value=Mock(followup_context="Result")),
    )
    fenced = ["\n ", "`", '``json\n{"tool_call":{"name":"synthetic","arguments":{"q":1}}}\n```']
    service, _ = _planning_service(
        tmp_path,
        monkeypatch,
        [
            lambda: iter([*(ModelEvent("text", part) for part in fenced), ModelEvent("done")]),
            lambda: iter([ModelEvent("text", "Done."), ModelEvent("done")]),
        ],
        registry,
    )
    events = list(service.stream(message="Synthetic"))
    assert [e["text"] for e in events if e["kind"] == "clause"] == ["Done."]
    registry.execute.assert_called_once_with("synthetic", {"q": 1})


def test_direct_answer_is_spoken_while_planning_response_still_generates(tmp_path, monkeypatch):
    from arietta_voice.llm import ModelEvent

    progress = []

    def answer():
        try:
            yield ModelEvent("text", " Four")
            yield ModelEvent("text", ". ")
            progress.append("continued")
            yield ModelEvent("text", "That is all.")
            yield ModelEvent("done")
        finally:
            progress.append("closed")

    service, calls = _planning_service(tmp_path, monkeypatch, [answer])
    stream = service.stream(message="Synthetic")
    assert next(stream) == {"kind": "clause", "text": "Four."}
    assert not progress  # The first clause left before the model finished.
    rest = list(stream)
    assert rest[0] == {"kind": "clause", "text": "That is all."}
    assert rest[-1]["result"] == {
        "response": "Four. That is all.",
        "source": "model",
        "used_tool": None,
        "model_tool": None,
    }
    assert progress == ["continued", "closed"]
    assert len(calls) == 1  # No second generation for a direct answer.


def test_direct_answer_stream_closes_when_the_turn_is_cancelled(tmp_path, monkeypatch):
    from arietta_voice.llm import ModelEvent

    closed = []

    def answer():
        try:
            yield ModelEvent("text", "First sentence. ")
            yield ModelEvent("text", "Second sentence.")
            yield ModelEvent("done")
        finally:
            closed.append(True)

    service, _ = _planning_service(tmp_path, monkeypatch, [answer])
    stream = service.stream(message="Synthetic")
    assert next(stream)["text"] == "First sentence."
    stream.close()
    assert closed == [True]


def test_tool_request_after_speech_started_stops_the_turn(tmp_path, monkeypatch):
    import pytest

    from arietta_voice.llm import ModelEvent

    closed = []

    def answer():
        try:
            yield ModelEvent("text", "Let me check. ")
            yield ModelEvent("text", '{"tool_')
            yield ModelEvent("text", 'call":{"name":"synthetic","arguments":{}}}')
            yield ModelEvent("done")
        finally:
            closed.append(True)

    registry = Mock(instruction_text=lambda: "Plan privately", execute=Mock())
    service, _ = _planning_service(tmp_path, monkeypatch, [answer], registry)
    stream = service.stream(message="Synthetic")
    assert next(stream)["text"] == "Let me check."
    with pytest.raises(ValueError, match="planning"):
        list(stream)
    registry.execute.assert_not_called()
    assert closed == [True]


def test_empty_planning_response_falls_back_to_a_spoken_apology(tmp_path, monkeypatch):
    from arietta_voice.llm import ModelEvent

    service, _ = _planning_service(
        tmp_path, monkeypatch, [lambda: iter([ModelEvent("text", "  "), ModelEvent("done")])]
    )
    events = list(service.stream(message="Synthetic"))
    assert events[0] == {"kind": "clause", "text": "I don't have a good answer for that yet."}
    assert events[-1]["kind"] == "done"


def test_truncated_or_oversized_tool_request_fails_without_running_a_tool(tmp_path, monkeypatch):
    import pytest

    from arietta_voice.llm import ModelEvent

    registry = Mock(instruction_text=lambda: "Plan privately", execute=Mock())
    service, _ = _planning_service(
        tmp_path,
        monkeypatch,
        [
            lambda: iter([ModelEvent("text", '{"tool_call":')]),
            lambda: iter([ModelEvent("text", "{" + " " * 12_001), ModelEvent("done")]),
        ],
        registry,
    )
    with pytest.raises(RuntimeError, match="Incomplete"):
        list(service.stream(message="Synthetic"))
    with pytest.raises(ValueError, match="limit"):
        list(service.stream(message="Synthetic"))
    registry.execute.assert_not_called()


def test_invalid_tool_payload_never_reaches_speech(tmp_path, monkeypatch):
    import pytest

    from arietta_voice import model_tools

    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ()
    monkeypatch.setattr(
        model_tools, "ModelToolRegistry", lambda *a, **kw: Mock(instruction_text=lambda: "Plan")
    )
    service = ConversationService(config, generate=lambda *a, **kw: '{"tool_call": broken')
    with pytest.raises(ValueError, match="planning"):
        next(service.stream(message="Synthetic"))
