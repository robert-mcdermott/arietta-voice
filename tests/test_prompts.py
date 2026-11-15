from pathlib import Path

import pytest

from arietta_voice.config import default_config
from arietta_voice.knowledge import NOTES_RULE
from arietta_voice.models import ConversationTurn, ToolResponse
from arietta_voice.prompts import PromptManager


@pytest.mark.parametrize("limit", [0, 1, 8])
def test_history_limit_counts_complete_conversation_pairs(tmp_path: Path, limit: int) -> None:
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.assistant.max_history_turns = limit
    history = [
        ConversationTurn(role=role, content=f"{role} {turn}")
        for turn in range(10)
        for role in ("user", "assistant")
    ]
    result = PromptManager(config).build_messages(history, "Current question")
    expected = history[-limit * 2 :] if limit else []
    assert result == [
        *({"role": turn.role, "content": turn.content} for turn in expected),
        {"role": "user", "content": "Current question"},
    ]


def test_notes_follow_the_history_so_the_prompt_start_is_reused(tmp_path: Path) -> None:
    # Before, notes went right after the persona: every question with notes, and
    # the one after it, made the model read the whole conversation again.
    config = default_config(tmp_path)
    history = [ConversationTurn("user", "First"), ConversationTurn("assistant", "Answer")]
    notes = ToolResponse("search_knowledge", "", False, "Knowledge base notes: the bins")
    manager = PromptManager(config)
    with_notes = manager.build_messages(history, "When are the bins?", notes)
    without = manager.build_messages(history, "And the recycling?")
    assert with_notes[:-2] == without[:-1]
    assert with_notes[-2:] == [
        {"role": "system", "content": "Knowledge base notes: the bins"},
        {"role": "user", "content": "When are the bins?"},
    ]


def test_how_to_use_notes_is_in_the_system_prompt_only_with_knowledge(tmp_path: Path) -> None:
    config = default_config(tmp_path)
    assert NOTES_RULE in PromptManager(config).system_prompt()
    config.knowledge.enabled = False
    assert NOTES_RULE not in PromptManager(config).system_prompt()
