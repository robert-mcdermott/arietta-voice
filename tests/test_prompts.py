from pathlib import Path

import pytest

from arietta_voice.config import default_config
from arietta_voice.models import ConversationTurn
from arietta_voice.prompts import PromptManager


@pytest.mark.parametrize("limit", [0, 1, 8])
def test_history_limit_counts_complete_conversation_pairs(tmp_path: Path, limit: int) -> None:
    config = default_config(tmp_path)
    config.assistant.max_history_turns = limit
    history = [ConversationTurn(role=role, content=f"{role} {turn}")
               for turn in range(10) for role in ("user", "assistant")]
    result = PromptManager(config).build_messages(history, "Current question")
    expected = history[-limit * 2:] if limit else []
    assert result == [*({"role": turn.role, "content": turn.content} for turn in expected),
                      {"role": "user", "content": "Current question"}]
