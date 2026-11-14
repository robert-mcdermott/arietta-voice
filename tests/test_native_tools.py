from types import SimpleNamespace

import pytest

from arietta_voice.config import default_config
from arietta_voice.conversation import GAVE_UP, MAX_TOOL_STEPS, ConversationService
from arietta_voice.model_tooling import (
    ModelToolRequest,
    parse_native_tool_calls,
    tool_specs,
)
from arietta_voice.tools import BaseTool, ToolResult
from arietta_voice.tools import registry as tool_registry

Q = '<|"|>'


def call(name, **arguments):
    body = ",".join(f"{key}:{Q}{value}{Q}" for key, value in arguments.items())
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<|tool_call>call:system_info{}<tool_call|>", [("system_info", {})]),
        (
            call("set_timer", duration="3 minutes", label="tea"),
            [
                ("set_timer", {"duration": "3 minutes", "label": "tea"}),
            ],
        ),
        (
            f"<|tool_call>call:x{{n:3,ok:true,when:{Q}at 5:30, please{Q},"
            f"nested:{{a:{Q}b{Q}}},list:[1,{Q}two{Q}]}}<tool_call|>",
            [
                (
                    "x",
                    {
                        "n": 3,
                        "ok": True,
                        "when": "at 5:30, please",
                        "nested": {"a": "b"},
                        "list": [1, "two"],
                    },
                )
            ],
        ),
        (call("a") + "\n" + call("b"), [("a", {}), ("b", {})]),
    ],
)
def test_native_calls_are_parsed(text, expected):
    assert parse_native_tool_calls(text) == [ModelToolRequest(n, a) for n, a in expected]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The system_info tool says 32 GB.",
        "Sure. " + call("a"),
        "<|tool_call>call:a{x:" + Q + "open",
        "<|tool_call>call:a{x:1",
        "<|tool_call>{}<tool_call|>",
        call("a") + " and then some words",
    ],
)
def test_anything_else_is_not_a_native_call(text):
    assert parse_native_tool_calls(text) is None


def test_tool_specs_use_the_openai_function_shape():
    tool = SimpleNamespace(name="t", description="testing", parameters={"type": "object"})
    assert tool_specs([tool]) == [
        {
            "type": "function",
            "function": {
                "name": "t",
                "description": "Use when testing.",
                "parameters": {"type": "object"},
            },
        }
    ]


class Lookup(BaseTool):
    name = "lookup"
    description = "you need to look something up"
    parameters = {
        "type": "object",
        "properties": {"what": {"type": "string"}},
        "required": ["what"],
    }
    runs: list = []

    def run(self, arguments):
        self.runs.append(arguments)
        if arguments["what"] == "broken":
            raise RuntimeError("synthetic failure")
        return ToolResult(content=f"{arguments['what']} is 42")


class Scripted:
    """A model that answers from a script and records what it was sent."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    def __call__(self, messages, *, max_tokens=None, temperature=None, tools=None):
        self.requests.append({"messages": messages, "tools": tools})
        return self.outputs.pop(0)


@pytest.fixture
def service_for(tmp_path, monkeypatch):
    Lookup.runs = []
    module = SimpleNamespace(create_tool=lambda context: Lookup(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))

    def build(*outputs, native=True):
        config = default_config(tmp_path)
        config.knowledge.enabled = False
        config.tools.enabled = ("lookup",)
        model = Scripted(*outputs)
        return ConversationService(config, generate=model, native_tools=native), model

    return build


def spoken(service, message="What is the answer?"):
    events = list(service.stream(message=message))
    clauses = [event["text"] for event in events if event["kind"] == "clause"]
    return clauses, events[-1]["result"]


def test_a_direct_answer_is_spoken_with_tools_declared(service_for):
    service, model = service_for("Paris is the capital of France.")
    clauses, result = spoken(service)
    assert clauses == ["Paris is the capital of France."]
    assert result["model_tool"] is None
    declared = model.requests[0]["tools"]
    assert [spec["function"]["name"] for spec in declared] == ["lookup"]
    # The rules join the persona's system message, ahead of the declarations.
    system = model.requests[0]["messages"][0]
    assert system["role"] == "system" and "Never say you set" in system["content"]


def test_a_tool_call_runs_and_its_result_goes_back_as_a_tool_message(service_for):
    service, model = service_for(call("lookup", what="the answer"), "It is 42.")
    clauses, result = spoken(service)
    assert clauses == ["It is 42."]
    assert result["model_tool"] == "lookup" and result["source"] == "model"
    assert Lookup.runs == [{"what": "the answer"}]
    followup = model.requests[1]["messages"]
    assert followup[-2]["role"] == "assistant"
    assert followup[-2]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": {"what": "the answer"},
    }
    assert followup[-1] == {
        "role": "tool",
        "tool_call_id": followup[-2]["tool_calls"][0]["id"],
        "name": "lookup",
        "content": "the answer is 42",
    }
    # Same tools on every step, so the cached prompt prefix stays valid.
    assert model.requests[1]["tools"] == model.requests[0]["tools"]


def test_words_before_a_call_are_spoken_and_the_answer_continues(service_for):
    service, model = service_for("Let me check. " + call("lookup", what="x"), "It is 42.")
    clauses, result = spoken(service)
    assert "".join(clauses).replace(" ", "") == "Letmecheck.Itis42."
    assert "tool_call" not in result["response"]
    assert model.requests[1]["messages"][-2]["content"] == "Let me check. "


def test_calls_chain_up_to_the_limit_then_it_gives_up(service_for):
    outputs = [call("lookup", what=str(n)) for n in range(MAX_TOOL_STEPS + 1)]
    service, _ = service_for(*outputs)
    clauses, result = spoken(service)
    assert clauses == [GAVE_UP]
    assert len(Lookup.runs) == MAX_TOOL_STEPS
    assert result["model_tool"] == ", ".join(["lookup"] * MAX_TOOL_STEPS)


def test_a_failing_tool_is_reported_to_the_model(service_for):
    service, model = service_for(call("lookup", what="broken"), "I couldn't check that.")
    clauses, _ = spoken(service)
    assert clauses == ["I couldn't check that."]
    note = model.requests[1]["messages"][-1]["content"]
    assert "could not be used (failed: RuntimeError)" in note


def test_an_unknown_tool_is_reported_and_never_runs(service_for):
    service, model = service_for(call("delete_everything"), "I can't do that.")
    clauses, _ = spoken(service)
    assert clauses == ["I can't do that."]
    assert Lookup.runs == []
    assert "could not be used (unavailable" in model.requests[1]["messages"][-1]["content"]


def test_a_malformed_call_fails_the_turn_without_running_anything(service_for):
    service, _ = service_for("<|tool_call>call:lookup{what:" + Q + "open")
    with pytest.raises(ValueError, match="Invalid tool call"):
        spoken(service)
    assert Lookup.runs == []


def test_prompted_json_is_used_when_native_calls_are_off(service_for):
    service, model = service_for(
        '{"tool_call":{"name":"lookup","arguments":{"what":"x"}}}', "It is 42.", native=False
    )
    clauses, result = spoken(service)
    assert clauses == ["It is 42."] and result["model_tool"] == "lookup"
    assert model.requests[0]["tools"] is None
    assert "Available tools:" in model.requests[0]["messages"][0]["content"]


def test_the_setting_turns_native_calls_off(service_for):
    service, model = service_for("Fine.")
    service.config.models.native_tool_calls = False
    spoken(service)
    assert model.requests[0]["tools"] is None


def test_hidden_reasoning_is_never_spoken(service_for):
    service, _ = service_for(
        "<|channel>thought\nThe user wants a lookup.<channel|>" + call("lookup", what="x"),
        "<|channel>thought\nIt said 42.<channel|>It is 42.",
    )
    clauses, result = spoken(service)
    assert clauses == ["It is 42."]
    assert result["model_tool"] == "lookup"


def test_reasoning_split_across_chunks_is_dropped():
    from arietta_voice.conversation import _HiddenReasoning

    hidden = _HiddenReasoning()
    chunks = ["Hi ", "<|channel>", "secret", "<channel|>", " there", "<|channel>x<channel|>!"]
    assert "".join(hidden.visible(chunk) for chunk in chunks) == "Hi  there!"


class Confirming(Lookup):
    name = "lookup"
    speaks_for_itself = True

    def run(self, arguments):
        self.runs.append(arguments)
        return ToolResult(content="Timer started.", speak="Okay, ten minutes, starting now.")


@pytest.fixture
def confirming(monkeypatch):
    module = SimpleNamespace(create_tool=lambda context: Confirming(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))


@pytest.mark.parametrize("native", [True, False])
def test_a_tool_that_speaks_for_itself_ends_the_turn(service_for, confirming, native):
    output = (
        call("lookup", what="x")
        if native
        else ('{"tool_call":{"name":"lookup","arguments":{"what":"x"}}}')
    )
    service, model = service_for(output, native=native)
    clauses, result = spoken(service)
    assert clauses == ["Okay, ten minutes, starting now."]
    assert result["model_tool"] == "lookup"
    assert len(model.requests) == 1  # The model was not asked again.


def test_a_tool_s_own_answer_covers_a_silent_model(service_for, monkeypatch):
    class Quiet(Lookup):
        def run(self, arguments):
            return ToolResult(content="Done.", speak="It is done.")

    module = SimpleNamespace(create_tool=lambda context: Quiet(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))
    service, _ = service_for(call("lookup", what="x"), "")
    clauses, _ = spoken(service)
    assert clauses == ["It is done."]


class Slow(Lookup):
    triggers = (r"slow lookup",)
    delay = 0.2

    def match(self, text):
        return {"what": "x"} if "slow lookup" in text else None

    def run(self, arguments):
        import time

        time.sleep(self.delay)
        return ToolResult(content="It is 42.", speak="It is 42.")


@pytest.fixture
def slow(monkeypatch):
    from arietta_voice import conversation

    monkeypatch.setattr(conversation, "FILLER_AFTER_S", 0.05)
    module = SimpleNamespace(create_tool=lambda context: Slow(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))
    return Slow


def test_a_slow_instant_answer_is_preceded_by_a_filler(service_for, slow):
    from arietta_voice.conversation import FILLER

    service, model = service_for()
    clauses, result = spoken(service, "slow lookup please")
    assert clauses == [FILLER, "It is 42."]
    assert result["response"] == "It is 42."  # The filler is not part of the answer.
    assert model.requests == []


def test_a_slow_model_called_tool_gets_one_filler(service_for, slow, monkeypatch):
    from arietta_voice.conversation import FILLER

    service, _ = service_for(
        call("lookup", what="a"), call("lookup", what="b"), "Both are 42.", native=True
    )
    monkeypatch.setattr(slow, "speaks_for_itself", False)
    clauses, result = spoken(service, "Look two things up")
    assert clauses == [FILLER, "Both are 42."]
    assert result["response"] == "Both are 42."


def test_a_quick_tool_gets_no_filler(service_for, slow, monkeypatch):
    monkeypatch.setattr(slow, "delay", 0.0)
    service, _ = service_for()
    clauses, _ = spoken(service, "slow lookup please")
    assert clauses == ["It is 42."]
