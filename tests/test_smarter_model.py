"""The optional smarter model: off unless configured, and only where it helps."""

import io
import json
from types import SimpleNamespace

import pytest

from arietta_voice.config import ModelConfig, default_config, validate_config
from arietta_voice.conversation import SMARTER_FILLER, ConversationService
from arietta_voice.llm import ModelEvent, OllamaChatModel, _plain_messages, build_chat_model
from arietta_voice.tools import BaseTool, ToolResult
from arietta_voice.tools import registry as tool_registry

Q = '<|"|>'


class Smarter:
    def __init__(self, text="Jane Austen wrote it.", fail=False):
        self.text, self.fail, self.requests = text, fail, []

    def stream(self, messages, *, max_tokens=None, temperature=None):
        self.requests.append(messages)
        if self.fail:
            raise OSError("connection refused")
        yield ModelEvent("text", self.text)
        yield ModelEvent("done")


class Lookup(BaseTool):
    name = "lookup"
    description = "the user asks for a stored value"
    parameters = {
        "type": "object",
        "properties": {"what": {"type": "string"}},
        "required": ["what"],
    }

    def run(self, arguments):
        return ToolResult(content="it is 42")


@pytest.fixture
def service_for(tmp_path, monkeypatch):
    module = SimpleNamespace(create_tool=lambda context: Lookup(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))

    def build(*outputs, when="unsure", smarter=None, enabled=True):
        config = default_config(tmp_path)
        config.knowledge.enabled = False
        config.tools.enabled = ("lookup",)
        config.smarter_model.enabled = enabled
        config.smarter_model.model = "bigger:cloud"
        config.smarter_model.when = when
        answers = list(outputs)
        service = ConversationService(
            config, generate=lambda *a, **k: answers.pop(0), native_tools=True
        )
        fake = smarter or Smarter()
        if enabled:
            monkeypatch.setattr(service, "_smarter_model", lambda: fake)
        return service, fake

    return build


def turn(service, message="Who wrote Pride and Prejudice?"):
    events = list(service.stream(message=message))
    return [e["text"] for e in events if e["kind"] == "clause"], events[-1]["result"]


def test_a_fresh_install_never_uses_a_smarter_model(tmp_path):
    config = default_config(tmp_path)
    assert config.smarter_model.enabled is False and config.smarter_model.model == ""
    service = ConversationService(config, generate=lambda *a, **k: "Hi.")
    assert service._smarter_model() is None


def test_when_unsure_a_declined_answer_goes_to_the_smarter_model(service_for):
    service, smarter = service_for("I don't have that information right now.")
    clauses, result = turn(service)
    assert clauses == [SMARTER_FILLER, "Jane Austen wrote it."]
    assert result["response"] == "Jane Austen wrote it."
    assert result["smarter"] == "bigger:cloud"
    # It gets the conversation, persona included, but no tool declarations.
    assert smarter.requests[0][-1] == {"role": "user", "content": "Who wrote Pride and Prejudice?"}


def test_when_unsure_a_good_answer_stays_local(service_for):
    service, smarter = service_for("Jane Austen wrote it. She was English.")
    clauses, result = turn(service)
    assert clauses == ["Jane Austen wrote it.", "She was English."]
    assert "smarter" not in result and smarter.requests == []


def test_always_hands_over_every_answer_without_a_filler(service_for):
    service, smarter = service_for("Paris is the capital.", when="always")
    clauses, result = turn(service, "What's the capital of France?")
    assert clauses == ["Jane Austen wrote it."]  # The fake's answer, to show who spoke.
    assert result["smarter"] == "bigger:cloud"


def test_an_answer_from_a_tool_result_stays_local(service_for):
    call = f"<|tool_call>call:lookup{{what:{Q}x{Q}}}<tool_call|>"
    # "always" first asks the local model for a few tokens to see if it wants a tool.
    service, smarter = service_for(call, call, "It is 42.", when="always")
    clauses, result = turn(service, "What is the stored value?")
    assert clauses == ["It is 42."]
    assert result["model_tool"] == "lookup" and smarter.requests == []


def test_an_unreachable_smarter_model_leaves_the_local_answer(service_for):
    service, _ = service_for("I don't know that one.", smarter=Smarter(fail=True))
    clauses, result = turn(service)
    assert clauses[-1] == "I don't know that one."
    assert "smarter" not in result


def test_short_answers_are_judged_at_the_end(service_for):
    service, _ = service_for("No idea, sorry", when="unsure")  # No sentence end, not a decline.
    clauses, result = turn(service)
    assert clauses == ["No idea, sorry"] and "smarter" not in result


@pytest.mark.parametrize(
    "change,problem",
    [
        ({"enabled": True, "model": ""}, "must name a model"),
        ({"provider": "openai"}, "smarter_model.provider"),
        ({"when": "sometimes"}, "smarter_model.when"),
    ],
)
def test_settings_are_checked(tmp_path, change, problem):
    config = default_config(tmp_path)
    for key, value in change.items():
        setattr(config.smarter_model, key, value)
    with pytest.raises(ValueError, match=problem):
        validate_config(config)


# ----- Ollama -----


def _ollama_lines(*chunks, done=True, error=None):
    lines = [json.dumps({"message": {"content": c}, "done": False}) for c in chunks]
    if error:
        lines.append(json.dumps({"error": error}))
    if done:
        lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    return io.BytesIO(("\n".join(lines) + "\n").encode())


def test_ollama_streams_chat(monkeypatch):
    import urllib.request

    sent = {}

    def urlopen(request, timeout):
        sent["url"], sent["body"] = request.full_url, json.loads(request.data)
        return _ollama_lines("Jane", " Austen.")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    model = build_chat_model(
        ModelConfig(chat_provider="ollama", chat_model="gemma4:31b-cloud", ollama_url="http://h:1")
    )
    assert isinstance(model, OllamaChatModel) and model.native_tool_calls is False
    events = list(model.stream([{"role": "user", "content": "Who?"}], max_tokens=50))
    assert [e.text for e in events if e.kind == "text"] == ["Jane", " Austen."]
    assert events[-1].kind == "done"
    assert sent["url"] == "http://h:1/api/chat"
    assert sent["body"]["model"] == "gemma4:31b-cloud" and sent["body"]["stream"] is True
    assert sent["body"]["options"]["num_predict"] == 50


@pytest.mark.parametrize(
    "lines", [_ollama_lines("Hi", error="model not found"), _ollama_lines("Hi", done=False)]
)
def test_ollama_errors_are_raised(monkeypatch, lines):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: lines)
    with pytest.raises(RuntimeError):
        list(OllamaChatModel("m").stream([{"role": "user", "content": "Hi"}]))


def test_tool_messages_become_plain_notes():
    messages = [
        {"role": "system", "content": "Persona"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "it is 42"},
        {"role": "user", "content": "And?"},
    ]
    assert _plain_messages(messages) == [
        {"role": "system", "content": "Persona"},
        {"role": "system", "content": "Tool result:\nit is 42"},
        {"role": "user", "content": "And?"},
    ]


def test_always_checks_for_a_tool_with_one_short_local_call(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.tools.enabled = ("lookup",)
    module = SimpleNamespace(create_tool=lambda context: Lookup(context))
    monkeypatch.setattr(tool_registry, "load_module", lambda config, name: (module, "user", None))
    config.smarter_model.enabled, config.smarter_model.model = True, "bigger:cloud"
    config.smarter_model.when = "always"
    calls = []

    def local(messages, *, max_tokens=None, temperature=None, tools=None):
        calls.append(max_tokens)
        return "Paris is"

    service = ConversationService(config, generate=local, native_tools=True)
    monkeypatch.setattr(service, "_smarter_model", lambda: Smarter("Paris."))
    clauses, result = turn(service, "What's the capital of France?")
    assert clauses == ["Paris."] and result["smarter"] == "bigger:cloud"
    assert calls == [6]  # Only the short check: the local answer is never generated.
