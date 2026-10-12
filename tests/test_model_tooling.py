from __future__ import annotations

from pathlib import Path

from arietta_voice.config import KnowledgeConfig, default_config
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.model_tools import ModelToolRegistry
from arietta_voice.model_tooling import maybe_generate_with_model_tools, parse_model_tool_request


def test_parse_model_tool_request_accepts_fenced_json() -> None:
    request = parse_model_tool_request(
        """```json
{"tool_call":{"name":"search_knowledge","arguments":{"query":"it leadership"}}}
```"""
    )

    assert request is not None
    assert request.name == "search_knowledge"
    assert request.arguments["query"] == "it leadership"


def test_model_tool_registry_only_loads_registered_tools(tmp_path) -> None:
    config = default_config(tmp_path)
    config.model_tools.enabled = ("system_info",)
    registry = ModelToolRegistry(config)

    instruction = registry.instruction_text()

    assert instruction is not None
    assert "get_system_info" in instruction
    assert "search_knowledge" not in instruction


def test_model_tool_registry_can_load_custom_registered_module(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "custom_model_tools"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "demo.py").write_text(
        """\
from arietta_voice.model_tools.base import ModelToolContext, ModelToolExecution


class DemoTool:
    name = "demo_tool"
    description = "you need a demo value from a custom registered module"
    arguments_example = {"topic": "<topic>"}

    def invoke(self, arguments: dict[str, object]) -> ModelToolExecution | None:
        topic = str(arguments.get("topic", "")).strip()
        if not topic:
            return None
        return ModelToolExecution(
            followup_context=f"Demo result for {topic}.",
            details={"topic": topic},
        )


def create_tool(context: ModelToolContext) -> DemoTool:
    del context
    return DemoTool()
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    config = default_config(tmp_path)
    config.model_tools.enabled = ("custom_model_tools.demo",)
    registry = ModelToolRegistry(config)

    instruction = registry.instruction_text()
    execution = registry.execute("demo_tool", {"topic": "testing"})

    assert instruction is not None
    assert "demo_tool" in instruction
    assert execution is not None
    assert execution.followup_context == "Demo result for testing."
    assert execution.details == {"topic": "testing"}


def test_maybe_generate_with_model_tools_returns_plain_answer_without_tool_call(tmp_path) -> None:
    calls: list[list[dict[str, str]]] = []
    registry = ModelToolRegistry(default_config(tmp_path))

    def fake_llm(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return "The local knowledge snippets already answer that."

    response, tool_use = maybe_generate_with_model_tools(
        [{"role": "user", "content": "Who runs IT?"}],
        model_tool_registry=registry,
        llm_generate=fake_llm,
    )

    assert response == "The local knowledge snippets already answer that."
    assert tool_use is None
    assert len(calls) == 1


def test_maybe_generate_with_model_tools_executes_followup_knowledge_search(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "leadership.md").write_text(
        """\
---
title: IT Leadership Team
tags: it, leadership, managers
---

## Leadership Roster
Jordan leads the IT team.
""",
        encoding="utf-8",
    )
    retriever = KnowledgeRetriever(KnowledgeConfig(directory=knowledge_dir, min_score=1.0))
    config = default_config(tmp_path)
    registry = ModelToolRegistry(config, knowledge_retriever=retriever)

    calls: list[list[dict[str, str]]] = []
    responses = iter(
        [
            '{"tool_call":{"name":"search_knowledge","arguments":{"query":"it leadership"}}}',
            "Jordan leads the IT team.",
        ]
    )

    def fake_llm(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return next(responses)

    response, tool_use = maybe_generate_with_model_tools(
        [{"role": "user", "content": "Who leads IT?"}],
        model_tool_registry=registry,
        llm_generate=fake_llm,
    )

    assert response == "Jordan leads the IT team."
    assert tool_use is not None
    assert tool_use.name == "search_knowledge"
    assert tool_use.details["query"] == "it leadership"
    assert tool_use.details["matches"] == 1
    assert len(calls) == 2
    assert any(
        "IT Leadership Team" in message["content"]
        for message in calls[1]
        if message["role"] == "system"
    )


def test_maybe_generate_with_model_tools_executes_system_info(tmp_path: Path, monkeypatch) -> None:
    from arietta_voice.model_tools import system_info

    monkeypatch.setattr(
        system_info,
        "build_system_report",
        lambda project_root: (
            "I'm running on Macmini9,1 with 16 GB RAM, 220 GB free disk space, "
            "and a 1-minute load average of 0.42."
        ),
    )
    config = default_config(tmp_path)
    registry = ModelToolRegistry(config)

    responses = iter(
        [
            '{"tool_call":{"name":"get_system_info","arguments":{}}}',
            "This machine is a Macmini9,1 with 16 GB of RAM and plenty of free disk space.",
        ]
    )
    calls: list[list[dict[str, str]]] = []

    def fake_llm(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return next(responses)

    response, tool_use = maybe_generate_with_model_tools(
        [{"role": "user", "content": "How much RAM do you have?"}],
        model_tool_registry=registry,
        llm_generate=fake_llm,
    )

    assert response == "This machine is a Macmini9,1 with 16 GB of RAM and plenty of free disk space."
    assert tool_use is not None
    assert tool_use.name == "get_system_info"
    assert tool_use.details == {}
    assert len(calls) == 2
    assert any(
        "Macmini9,1" in message["content"]
        for message in calls[1]
        if message["role"] == "system"
    )
