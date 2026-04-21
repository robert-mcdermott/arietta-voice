from __future__ import annotations

import re

from arietta_voice.models import ToolResponse
from arietta_voice.tools.base import ToolContext


# Copy this file to a new module in this directory, then add the module name to
# [tools].enabled in config/arietta_voice.toml.
MATCH_PATTERNS = (
    re.compile(r"\bexample custom tool\b"),
)


class ExampleCustomTool:
    name = "example_custom_tool"

    def maybe_handle(self, user_text: str) -> ToolResponse | None:
        lowered = user_text.lower()
        if not any(pattern.search(lowered) for pattern in MATCH_PATTERNS):
            return None

        return ToolResponse(
            name=self.name,
            content="The example custom tool is wired up. Replace this with your real logic.",
        )


def create_tool(context: ToolContext) -> ExampleCustomTool:
    del context
    return ExampleCustomTool()
