from __future__ import annotations

from datetime import datetime
import re

from arietta_voice.models import ToolResponse
from arietta_voice.tools.base import ToolContext


TIME_PATTERNS = (
    re.compile(r"\bwhat(?:'s| is) the time\b"),
    re.compile(r"\bwhat time is it\b"),
    re.compile(r"\bcurrent time\b"),
)


def _format_local_time() -> str:
    now = datetime.now()
    hour = now.strftime("%I").lstrip("0") or "0"
    return f"It is {hour}:{now.strftime('%M %p')}."


class LocalTimeTool:
    name = "get_local_time"

    def maybe_handle(self, user_text: str) -> ToolResponse | None:
        lowered = user_text.lower()
        if any(pattern.search(lowered) for pattern in TIME_PATTERNS):
            return ToolResponse(name=self.name, content=_format_local_time())
        return None


def create_tool(context: ToolContext) -> LocalTimeTool:
    del context
    return LocalTimeTool()
