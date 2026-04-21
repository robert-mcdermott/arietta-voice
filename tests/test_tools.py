from __future__ import annotations

from pathlib import Path

from arietta_voice.config import default_config
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.tools import ToolRegistry


def test_tool_registry_handles_identity() -> None:
    registry = ToolRegistry(default_config())
    response = registry.maybe_handle("What is your name?")

    assert response is not None
    assert response.name == "get_assistant_identity"


def test_tool_registry_only_loads_registered_tools() -> None:
    config = default_config()
    config.tools.enabled = ("assistant_identity",)
    registry = ToolRegistry(config)

    assert registry.maybe_handle("What is your name?") is not None
    assert registry.maybe_handle("What time is it?") is None


def test_tool_registry_does_not_handle_system_info_deterministically() -> None:
    registry = ToolRegistry(default_config())

    assert registry.maybe_handle("How much RAM do you have?") is None


def test_tool_registry_handles_local_time() -> None:
    registry = ToolRegistry(default_config())
    response = registry.maybe_handle("What time is it?")

    assert response is not None
    assert response.name == "get_local_time"


def test_tool_registry_returns_knowledge_context(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "food.md").write_text(
        """\
---
title: Food and Dining
tags: food, eat, lunch, cafe
---

## On-Site Options
Visitors can grab lunch at the cafeteria and coffee at the cafe.
""",
        encoding="utf-8",
    )

    config = default_config()
    config.knowledge.directory = knowledge_dir
    retriever = KnowledgeRetriever(config.knowledge)
    registry = ToolRegistry(config, knowledge_retriever=retriever)

    response = registry.maybe_handle("Where can I get something to eat?")

    assert response is not None
    assert response.name == "retrieve_knowledge"
    assert response.bypass_model is False
    assert response.context is not None
    assert "Food and Dining" in response.context


def test_tool_registry_raises_for_unknown_registered_tool() -> None:
    config = default_config()
    config.tools.enabled = ("missing_tool",)

    try:
        ToolRegistry(config)
    except ValueError as exc:
        assert "missing_tool" in str(exc)
    else:
        raise AssertionError("Expected ToolRegistry to reject an unknown tool module")
