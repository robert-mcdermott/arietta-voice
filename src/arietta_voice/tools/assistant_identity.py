from __future__ import annotations

import re

from arietta_voice.models import ToolResponse
from arietta_voice.tools.base import ToolContext


IDENTITY_PATTERNS = (
    re.compile(r"\bwho are you\b"),
    re.compile(r"\bwhat(?:'s| is) your name\b"),
    re.compile(r"\bwhat do you do\b"),
    re.compile(r"\bwhat(?:'s| is) your purpose\b"),
)


class AssistantIdentityTool:
    name = "get_assistant_identity"

    def __init__(self, identity: str) -> None:
        self.identity = identity

    def maybe_handle(self, user_text: str) -> ToolResponse | None:
        lowered = user_text.lower()
        if any(pattern.search(lowered) for pattern in IDENTITY_PATTERNS):
            return ToolResponse(name=self.name, content=self.identity)
        return None


def create_tool(context: ToolContext) -> AssistantIdentityTool:
    return AssistantIdentityTool(context.config.assistant.identity)
