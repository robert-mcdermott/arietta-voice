from __future__ import annotations

from datetime import datetime

from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult


class LocalTimeTool(BaseTool):
    name = "local_time"
    description = "you need the current local time, date or day of the week"
    triggers = (r"\bwhat(?:'s| is) the time\b", r"\bwhat time is it\b", r"\bcurrent time\b")

    def __init__(self, context=None, clock=datetime.now) -> None:
        super().__init__(context)
        self.clock = clock

    def run(self, arguments):
        now = self.clock()
        hour = now.strftime("%I").lstrip("0") or "0"
        spoken = f"It is {hour}:{now.strftime('%M %p')}."
        date = f"{now.strftime('%A, %B')} {now.day}, {now.year}"
        return ToolResult(
            content=f"The local time is {hour}:{now.strftime('%M %p')} on {date}.",
            speak=spoken,
            data={"iso": now.isoformat(timespec="seconds")},
        )


def create_tool(context: ToolContext) -> LocalTimeTool:
    return LocalTimeTool(context)
