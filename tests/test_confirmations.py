"""Tools that change something ask first, and only a clear spoken answer decides."""

from types import SimpleNamespace

import pytest

from arietta_voice import conversation
from arietta_voice.config import default_config
from arietta_voice.confirmations import confirmation_reply
from arietta_voice.conversation import CANCELLED, ConversationService
from arietta_voice.tools import BaseTool, ToolRegistry, ToolResult
from arietta_voice.tools import registry as tool_registry
from arietta_voice.tools.approvals import Approvals

Q = '<|"|>'


def switch(room="office", on="false"):
    return f"<|tool_call>call:lights{{room:{Q}{room}{Q},on:{on}}}<tool_call|>"


class Lights(BaseTool):
    name = "lights"
    description = "the user wants the lights in a room switched on or off"
    parameters = {
        "type": "object",
        "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}},
        "required": ["room", "on"],
    }
    triggers = (r"lights (?P<room>\w+) off",)
    consequential = True
    runs: list = []

    def run(self, arguments):
        Lights.runs.append(arguments)
        state = "on" if arguments["on"] else "off"
        return ToolResult(content=f"The {arguments['room']} lights are now {state}.")


class Scripted:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []

    def __call__(self, messages, *, max_tokens=None, temperature=None, tools=None):
        self.requests.append({"messages": messages, "tools": tools})
        return self.outputs.pop(0)


@pytest.fixture
def service_for(tmp_path, monkeypatch):
    Lights.runs = []
    module = SimpleNamespace(create_tool=lambda context: Lights(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))

    def build(*outputs, native=True, confirm="ask"):
        monkeypatch.setattr(Lights, "confirm", confirm)
        config = default_config(tmp_path)
        config.knowledge.enabled = False
        config.tools.enabled = ("lights",)
        model = Scripted(*outputs)
        return ConversationService(config, generate=model, native_tools=native), model

    return build


def turn(service, message, history=None):
    """One turn, keeping the conversation's history as the voice runtime does."""
    kept = _history(service)
    used = kept if history is None else history
    events = list(service.stream(message=message, history=list(used)))
    clauses = " ".join(event["text"] for event in events if event["kind"] == "clause")
    result = events[-1]["result"]
    if history is None:
        kept += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["response"]},
        ]
    return clauses, result


def _history(service):
    if not hasattr(service, "test_history"):
        service.test_history = []
    return service.test_history


QUESTION = "Should I turn off the office lights?"


def test_an_action_is_held_and_the_model_words_the_question(service_for):
    service, model = service_for(switch(), QUESTION)
    said, result = turn(service, "Turn off the office lights")
    assert said == QUESTION
    assert Lights.runs == []  # Nothing happens until the owner agrees.
    assert result["confirmation"] == {"tool": "lights", "question": QUESTION}
    assert result["model_tool"] is None
    # The question is worded from the held call alone, not the conversation.
    asked = model.requests[1]["messages"]
    assert [message["role"] for message in asked] == ["system", "user"]
    assert 'room="office", on=false' in asked[1]["content"]
    assert model.requests[1]["tools"] is None


def test_yes_runs_exactly_the_held_call_once(service_for):
    service, model = service_for(switch(), QUESTION, "Done, the office lights are off.")
    turn(service, "Turn off the office lights")
    said, result = turn(service, "Yeah, go ahead.")
    assert said == "Done, the office lights are off."
    assert Lights.runs == [{"room": "office", "on": False}]
    assert result["model_tool"] == "lights"
    answer = model.requests[2]["messages"]
    assert answer[-1]["role"] == "tool" and "now off" in answer[-1]["content"]
    # A second yes is just a word: the action never runs twice.
    model.outputs.append("Okay.")
    turn(service, "Yes")
    assert len(Lights.runs) == 1


def test_no_cancels_and_says_so(service_for):
    service, _ = service_for(switch(), QUESTION)
    turn(service, "Turn off the office lights")
    said, result = turn(service, "No, thanks.")
    assert said == CANCELLED
    assert result["source"] == "confirmation"
    assert Lights.runs == []


def test_anything_else_drops_the_action_and_is_answered(service_for):
    service, model = service_for(switch(), QUESTION, "It is sunny.")
    turn(service, "Turn off the office lights")
    said, _ = turn(service, "What's the weather like?")
    assert said == "It is sunny."
    assert Lights.runs == []
    # Dropped for good: a later yes does nothing.
    model.outputs.append("Okay.")
    turn(service, "Yes")
    assert Lights.runs == []


def test_a_late_yes_does_not_run_the_action(service_for, monkeypatch):
    service, model = service_for(switch(), QUESTION, "Okay.")
    turn(service, "Turn off the office lights")
    service._pending.expires_at = 0  # The minute has passed.
    turn(service, "Yes")
    assert Lights.runs == []


def test_always_runs_it_and_stops_asking(service_for, tmp_path):
    service, model = service_for(switch(), QUESTION, "Done.", switch("kitchen"), "Done too.")
    turn(service, "Turn off the office lights")
    turn(service, "Yes, and don't ask again.")
    assert Lights.runs == [{"room": "office", "on": False}]
    assert Approvals(tmp_path / "state" / "tool_approvals.json").all() == ["tool:lights"]
    said, result = turn(service, "Turn off the kitchen lights")
    assert said == "Done too."
    assert result["confirmation"] is None
    assert Lights.runs[-1] == {"room": "kitchen", "on": False}


def test_first_use_asks_once(service_for):
    service, _ = service_for(
        switch(), QUESTION, "Done.", switch("hall"), "Done.", confirm="first_use"
    )
    turn(service, "Turn off the office lights")
    turn(service, "Sure")
    said, _ = turn(service, "Turn off the hall lights")
    assert said == "Done."
    assert [run["room"] for run in Lights.runs] == ["office", "hall"]


def test_a_trusted_tool_never_asks(service_for):
    service, _ = service_for(switch(), "Done.", confirm="never")
    said, result = turn(service, "Turn off the office lights")
    assert said == "Done." and result["confirmation"] is None
    assert Lights.runs == [{"room": "office", "on": False}]


@pytest.mark.parametrize("question", ["Done, the lights are off.", switch(), ""])
def test_a_bad_question_falls_back_to_a_plain_one(service_for, question):
    service, _ = service_for(switch(), question)
    said, _ = turn(service, "Turn off the office lights")
    assert said == "Should I go ahead with lights, with room office and on off?"
    assert Lights.runs == []


def test_bad_arguments_go_back_to_the_model_instead_of_asking(service_for):
    service, model = service_for(
        f"<|tool_call>call:lights{{room:{Q}office{Q}}}<tool_call|>",
        switch(),
        QUESTION,
    )
    said, _ = turn(service, "Turn off the office lights")
    assert said == QUESTION
    assert "missing 'on'" in model.requests[1]["messages"][-1]["content"]


def test_a_trigger_never_runs_an_action_that_asks(service_for):
    service, _ = service_for("I can do that if you ask me to.")
    said, _ = turn(service, "lights office off")
    assert Lights.runs == []
    assert said == "I can do that if you ask me to."


def test_the_prompted_format_asks_too(service_for):
    request = '{"tool_call":{"name":"lights","arguments":{"room":"office","on":false}}}'
    service, model = service_for(request, QUESTION, "Done.", native=False)
    said, result = turn(service, "Turn off the office lights")
    assert said == QUESTION and result["confirmation"]["tool"] == "lights"
    assert 'room="office"' in model.requests[1]["messages"][-1]["content"]
    said, _ = turn(service, "Yes")
    assert Lights.runs == [{"room": "office", "on": False}]
    assert "Result from the lights tool" in model.requests[2]["messages"][-1]["content"]


def test_registry_rules(tmp_path, monkeypatch):
    module = SimpleNamespace(create_tool=lambda context: Lights(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))
    config = default_config(tmp_path)
    config.tools.enabled = ("lights",)
    registry = ToolRegistry(config)
    assert registry.needs_confirmation("lights")
    assert not registry.needs_confirmation("missing")
    registry.remember_approval("lights", always=False)
    assert registry.needs_confirmation("lights")  # A plain yes under "ask" is not kept.
    registry.remember_approval("lights", always=True)
    assert not registry.needs_confirmation("lights")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes.", "yes"),
        ("Yeah, go ahead.", "yes"),
        ("Um, sure, do it please", "yes"),
        ("OK", "yes"),
        ("No.", "no"),
        ("No, thank you.", "no"),
        ("Cancel that", "no"),
        ("Never mind.", "no"),
        ("Always.", "always"),
        ("Yes, always.", "always"),
        ("Yes and don't ask again", "always"),
        ("Yeah, you don't need to ask me again.", "always"),
        ("Yes, but the kitchen", None),
        ("No, the kitchen lights", None),
        ("What's the weather?", None),
        ("", None),
    ],
)
def test_spoken_answers(text, expected):
    assert confirmation_reply(text) == expected


def test_the_window_is_about_a_minute():
    assert 30 <= conversation.CONFIRM_WINDOW_S <= 120


def test_silence_after_a_confirmed_action_still_says_it_happened(service_for):
    service, _ = service_for(switch(), QUESTION, "")
    turn(service, "Turn off the office lights")
    said, _ = turn(service, "Yes")
    assert said == "Done."
    assert Lights.runs == [{"room": "office", "on": False}]


def test_the_question_drops_any_preamble(service_for):
    service, _ = service_for(switch(), "I need your OK first. " + QUESTION)
    said, _ = turn(service, "Turn off the office lights")
    assert said == QUESTION


def test_a_question_about_something_else_is_replaced_by_the_plain_one(service_for):
    # The held call is the garage; a question naming the office must not be agreed to.
    service, _ = service_for(switch("garage"), QUESTION, "Done.")
    said, _ = turn(service, "Turn off the garage lights")
    assert said == "Should I go ahead with lights, with room garage and on off?"
    turn(service, "Yes")
    assert Lights.runs == [{"room": "garage", "on": False}]


@pytest.mark.parametrize(
    "history",
    [
        [],  # The console's Clear, or a new voice session.
        [{"role": "user", "content": "Turn off the office lights"}],
        [
            {"role": "user", "content": "Turn off the office lights"},
            {"role": "assistant", "content": "Something else entirely."},
        ],
    ],
)
def test_a_yes_counts_only_right_after_the_question(service_for, history):
    service, model = service_for(switch(), QUESTION, "Okay.")
    turn(service, "Turn off the office lights")
    turn(service, "Yes", history)
    assert Lights.runs == []


def test_a_held_action_whose_tool_changed_is_not_run(service_for):
    service, _ = service_for(switch(), QUESTION)
    turn(service, "Turn off the office lights")
    service._pending.key = "mcp:other:switch"  # The name now belongs to another action.
    said, _ = turn(service, "Yes")
    assert said == conversation.GONE
    assert Lights.runs == []


def test_numbers_sent_as_text_stay_exact():
    from arietta_voice.tools.schema import coerce_arguments

    schema = {"type": "object", "properties": {"phone": {"type": "string"}}}
    assert coerce_arguments(schema, {"phone": 2065551234}) == {"phone": "2065551234"}
    assert coerce_arguments(schema, {"phone": 10.0}) == {"phone": "10"}
    assert coerce_arguments(schema, {"phone": 2.5}) == {"phone": "2.5"}
