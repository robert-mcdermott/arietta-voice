from __future__ import annotations

import re

from arietta_voice.config import AppConfig
from arietta_voice.knowledge import NOTES_RULE
from arietta_voice.models import ConversationTurn, ToolResponse

MARKDOWN_NOISE_RE = re.compile(r"[#*_`>-]+")
WHITESPACE_RE = re.compile(r"\s+")


class PromptManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def system_prompt(self) -> str:
        parts: list[str] = []
        if self.config.prompts.soul_path.exists():
            parts.append(self.config.prompts.soul_path.read_text(encoding="utf-8").strip())
        if self.config.memory.enabled and self.config.prompts.memory_path.exists():
            parts.append(self.config.prompts.memory_path.read_text(encoding="utf-8").strip())
        if self.config.knowledge.enabled:
            parts.append(NOTES_RULE)
        return "\n\n".join(part for part in parts if part)

    def build_messages(
        self,
        history: list[ConversationTurn],
        user_text: str,
        tool_response: ToolResponse | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        prompt = self.system_prompt()
        if prompt:
            messages.append({"role": "system", "content": prompt})
        history_limit = self.config.assistant.max_history_turns * 2
        recent_history = history[-history_limit:] if history_limit > 0 else []
        for turn in recent_history:
            messages.append({"role": turn.role, "content": turn.content})
        # Notes and tool results for this question go after the conversation so
        # far, next to the question. The persona and history then stay the same
        # from turn to turn and the model reuses them instead of reading the
        # whole conversation again.
        if tool_response and tool_response.context:
            messages.append({"role": "system", "content": tool_response.context})
        messages.append({"role": "user", "content": user_text})
        return messages


def sanitize_for_speech(text: str) -> str:
    cleaned = MARKDOWN_NOISE_RE.sub(" ", text)
    cleaned = cleaned.replace("\n", " ")
    return WHITESPACE_RE.sub(" ", cleaned).strip()
