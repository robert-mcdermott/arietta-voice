from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import pytest

from arietta_voice.config import default_config
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.model_tooling import parse_model_tool_request
from arietta_voice.tools import BaseTool, ToolContext, ToolError, ToolRegistry, ToolResult
from arietta_voice.tools.registry import (
    argument_example,
    describe_problems,
    enabled_tool_names,
    run_tool,
)
from arietta_voice.tools.schema import check_schema, validate_arguments

USER_TOOL = """
from arietta_voice.tools import BaseTool, ToolResult


class WeatherTool(BaseTool):
    name = "{name}"
    description = "the user asks about the weather"
    triggers = (r"\\bweather in (?P<city>[a-z ]+)",)
    parameters = {{
        "type": "object",
        "properties": {{"city": {{"type": "string", "description": "city name"}}}},
        "required": ["city"],
    }}

    def run(self, arguments):
        return ToolResult(content=f"Sunny in {{arguments['city']}}.", speak=f"It's sunny in {{arguments['city']}}.")


def create_tool(context):
    return WeatherTool(context)
"""


def write_user_tool(config, name="weather", source=None):
    directory = Path(config.tools.directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(source or USER_TOOL.format(name=name))


def knowledge_config(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "food.md").write_text(
        "---\ntitle: Food and Dining\ntags: food, eat, lunch, cafe\n---\n\n"
        "## On-Site Options\nVisitors can grab lunch at the cafeteria and coffee at the cafe.\n"
    )
    config = default_config(tmp_path)
    config.knowledge.directory = knowledge_dir
    return config, KnowledgeRetriever(config.knowledge)


# ----- Built-in tools -----


def test_identity_and_time_answer_instantly(tmp_path) -> None:
    registry = ToolRegistry(default_config(tmp_path))
    identity = registry.fast_path("What is your name?")
    assert identity.name == "assistant_identity"
    assert identity.result.speak == registry.config.assistant.identity
    time = registry.fast_path("what time is it")
    assert time.name == "local_time" and time.result.speak.startswith("It is ")
    assert registry.fast_path("How much RAM do you have?") is None


def test_local_time_gives_the_model_the_date_as_well() -> None:
    from arietta_voice.tools.builtin.local_time import LocalTimeTool

    result = LocalTimeTool(clock=lambda: datetime(2026, 11, 13, 9, 5)).run({})
    assert result.speak == "It is 9:05 AM."
    assert result.content == "The local time is 9:05 AM on Friday, November 13, 2026."


def test_knowledge_adds_context_without_answering_and_is_a_model_tool(tmp_path) -> None:
    config, retriever = knowledge_config(tmp_path)
    registry = ToolRegistry(config, knowledge_retriever=retriever)
    hit = registry.fast_path("Where can I get something to eat?")
    assert hit.name == "search_knowledge"
    assert hit.result.speak is None
    assert "Food and Dining" in hit.result.content
    # Nothing relevant adds nothing.
    assert registry.fast_path("Explain quantum chromodynamics") is None
    called = registry.run("search_knowledge", {"query": "lunch cafeteria"})
    assert "cafeteria" in called.content
    assert registry.run("search_knowledge", {"query": "zzz"}).empty


def test_knowledge_tool_is_unavailable_when_knowledge_is_off(tmp_path) -> None:
    registry = ToolRegistry(default_config(tmp_path))
    assert "search_knowledge" not in registry.tools
    assert registry.errors == {}


def test_system_info_is_only_called_by_the_model(tmp_path, monkeypatch) -> None:
    from arietta_voice.tools.builtin import system_info

    monkeypatch.setattr(system_info, "build_system_report", lambda root: "Synthetic Mac, 16 GB")
    registry = ToolRegistry(default_config(tmp_path))
    assert registry.fast_path("how much ram do you have") is None
    assert "Synthetic Mac, 16 GB" in registry.run("system_info", {}).content


# ----- Configuration and loading -----


def test_old_config_lists_and_names_are_merged_in_order(tmp_path) -> None:
    config = default_config(tmp_path)
    config.tools.enabled = ("assistant_identity", "local_time", "knowledge_retrieval")
    config.model_tools.enabled = ("search_knowledge", "system_info")
    assert enabled_tool_names(config) == [
        "assistant_identity",
        "local_time",
        "search_knowledge",
        "system_info",
    ]


def test_user_tools_load_from_the_tools_directory(tmp_path) -> None:
    config = default_config(tmp_path)
    config.tools.enabled = ("weather",)
    write_user_tool(config)
    registry = ToolRegistry(config)
    hit = registry.fast_path("What's the weather in paris today?")
    assert hit.name == "weather" and hit.arguments == {"city": "paris today"}
    assert hit.result.speak == "It's sunny in paris today."
    assert registry.run("weather", {"city": "Oslo"}).content == "Sunny in Oslo."


def test_a_user_tool_replaces_a_built_in_of_the_same_name(tmp_path) -> None:
    from arietta_voice.tool_admin import ToolAdmin

    config = default_config(tmp_path)
    config.tools.enabled = ("local_time",)
    config.tools.directory.mkdir(parents=True, exist_ok=True)
    (config.tools.directory / "local_time.py").write_text(
        "from arietta_voice.tools import BaseTool, ToolResult\n"
        "class Mine(BaseTool):\n"
        "    name = 'local_time'\n"
        "    description = 'my own clock'\n"
        "    triggers = (r'what time is it',)\n"
        "    def run(self, arguments):\n"
        "        return ToolResult(speak='Time for tea.')\n"
        "def create_tool(context):\n"
        "    return Mine(context)\n"
    )
    hit = ToolRegistry(config).fast_path("What time is it?")
    assert hit.result.speak == "Time for tea."
    record = next(r for r in ToolAdmin(config).inventory()["tools"] if r["name"] == "local_time")
    assert record["origin"] == "user" and record["description"] == "my own clock"


def test_broken_or_missing_tools_are_reported_without_disabling_the_rest(tmp_path) -> None:
    config = default_config(tmp_path)
    config.tools.enabled = ("local_time", "missing_tool", "broken", "misnamed")
    write_user_tool(config, "broken", "raise RuntimeError('import failed')\n")
    write_user_tool(config, "misnamed", USER_TOOL.format(name="something_else"))
    registry = ToolRegistry(config)
    assert list(registry.tools) == ["local_time"]
    assert set(registry.errors) == {"missing_tool", "broken", "misnamed"}
    assert "match its file" in registry.errors["misnamed"]


@pytest.mark.parametrize("name", ["../escape", "Upper", "has-dash", ""])
def test_tool_names_cannot_reach_outside_the_tools_directory(tmp_path, name) -> None:
    config = default_config(tmp_path)
    config.tools.enabled = (name,)
    registry = ToolRegistry(config)
    assert not registry.tools


def test_disabled_tools_never_run(tmp_path) -> None:
    config = default_config(tmp_path)
    config.tools.enabled = ("local_time",)
    registry = ToolRegistry(config)
    with pytest.raises(ToolError) as error:
        registry.run("system_info", {})
    assert error.value.kind == "unavailable"
    with pytest.raises(ToolError):
        registry.run("assistant_identity", {})  # Exists, but not model-callable.


# ----- Arguments, limits and results -----


class Echo(BaseTool):
    name = "echo"
    description = "testing"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": 10},
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
            "mode": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["text"],
    }
    triggers = (r"echo (?P<text>\w+) (?P<count>\d+)",)

    def run(self, arguments):
        return ToolResult(content=arguments["text"] * arguments.get("count", 1), data=arguments)


@pytest.mark.parametrize(
    "arguments, problem",
    [
        ({}, "missing 'text'"),
        ({"text": ["x"]}, "must be text"),
        ({"text": "x" * 11}, "too long"),
        ({"text": "x", "count": 4}, "at most 3"),
        ({"text": "x", "count": True}, "whole number"),
        ({"text": "x", "mode": "c"}, "one of a, b"),
        ({"text": "x", "extra": 1}, "unexpected argument 'extra'"),
    ],
)
def test_invalid_arguments_are_rejected_before_the_tool_runs(arguments, problem) -> None:
    with pytest.raises(ToolError) as error:
        run_tool(Echo(), arguments)
    assert error.value.kind == "invalid_arguments" and problem in str(error.value)


def test_numbers_are_accepted_as_text_and_aliases_as_names() -> None:
    class Aliased(Echo):
        aliases = {"message": "text"}

    assert run_tool(Echo(), {"text": 5}).content == "5"
    assert run_tool(Echo(), {"text": 2.5}).content == "2.5"
    assert run_tool(Aliased(), {"message": "hi"}).content == "hi"
    # The real name wins when both are sent.
    with pytest.raises(ToolError, match="unexpected argument 'message'"):
        run_tool(Aliased(), {"message": "hi", "text": "yo"})
    with pytest.raises(ToolError, match="arguments must be an object"):
        run_tool(Echo(), ["hi"])


def test_aliases_must_name_declared_arguments() -> None:
    class Broken(Echo):
        aliases = {"message": "nothing"}

    assert "aliases must map other names to declared arguments" in describe_problems(
        Broken(), "echo"
    )


def test_trigger_captures_are_converted_to_declared_types() -> None:
    tool = Echo()
    arguments = tool.match("please echo hi 2")
    assert arguments == {"text": "hi", "count": "2"}
    assert run_tool(tool, arguments).content == "hihi"


def test_slow_failing_or_oversized_tools_are_contained() -> None:
    release = threading.Event()

    class Slow(Echo):
        timeout_s = 0.05

        def run(self, arguments):
            release.wait(2)
            return ToolResult(content="late")

    class Failing(Echo):
        def run(self, arguments):
            raise KeyError("secret detail")

    class Huge(Echo):
        def run(self, arguments):
            return ToolResult(content="x" * 10_000, speak="y" * 5000)

    with pytest.raises(ToolError) as slow:
        run_tool(Slow(), {"text": "a"})
    release.set()
    assert slow.value.kind == "timeout"
    with pytest.raises(ToolError) as failing:
        run_tool(Failing(), {"text": "a"})
    assert failing.value.kind == "failed" and "secret" not in str(failing.value)
    huge = run_tool(Huge(), {"text": "a"})
    assert len(huge.content) == 4000 and len(huge.speak) == 1000


def test_a_failing_instant_tool_falls_through_to_the_next(tmp_path, monkeypatch) -> None:
    config = default_config(tmp_path)
    config.tools.enabled = ("local_time",)
    registry = ToolRegistry(config)

    def broken(arguments):
        raise RuntimeError("clock failure")

    monkeypatch.setattr(registry.tools["local_time"], "run", broken)
    assert registry.fast_path("what time is it") is None


def test_definitions_are_checked() -> None:
    class Bad(BaseTool):
        name = "bad"
        description = ""
        parameters = {"type": "object", "properties": {"x": {"type": "array"}}, "required": ["y"]}
        triggers = ("(",)
        timeout_s = 0

    problems = describe_problems(Bad(), "bad")
    assert any("description" in p for p in problems)
    assert any("property 'x' needs a type" in p for p in problems)
    assert any("required" in p for p in problems)
    assert any("trigger" in p for p in problems)
    assert any("timeout_s" in p for p in problems)
    assert check_schema({"type": "object", "properties": {}, "oneOf": []})
    assert validate_arguments({"type": "object", "properties": {}}, []) == [
        "arguments must be an object"
    ]


def test_model_instructions_describe_arguments_from_the_schema(tmp_path) -> None:
    config, retriever = knowledge_config(tmp_path)
    registry = ToolRegistry(config, knowledge_retriever=retriever)
    text = registry.instruction_text()
    assert '- name: search_knowledge\n- arguments: {"query": "<short search query>"}' in text
    assert "- name: assistant_identity" not in text  # Not model-callable.
    assert argument_example(Echo.parameters) == {
        "text": "<string>",
        "count": "<integer>",
        "mode": "a|b",
    }
    assert argument_example(
        {"type": "object", "properties": {"d": {"type": "string", "description": "x, such as 5"}}}
    ) == {"d": "5"}
    config.tools.enabled = ("assistant_identity",)
    assert ToolRegistry(config).instruction_text() is None


def test_tool_context_reaches_tools(tmp_path) -> None:
    config = default_config(tmp_path)
    tool = Echo(ToolContext(config))
    assert tool.context.config is config


def test_parse_model_tool_request_accepts_fenced_json() -> None:
    request = parse_model_tool_request(
        '```json\n{"tool_call":{"name":"search_knowledge","arguments":{"query":"it"}}}\n```'
    )
    assert request.name == "search_knowledge" and request.arguments == {"query": "it"}
