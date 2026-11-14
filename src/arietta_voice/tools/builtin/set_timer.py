from __future__ import annotations

from datetime import datetime

from arietta_voice.timephrases import (
    duration_adjective,
    format_clock,
    format_duration,
    parse_duration,
)
from arietta_voice.timers import TimerLimitError, timer_name
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult
from arietta_voice.tools.builtin._timers import clean_label, store_for, timer_request


class SetTimerTool(BaseTool):
    name = "set_timer"
    description = "the user wants a countdown timer"
    parameters = {
        "type": "object",
        "properties": {
            "duration": {"type": "string", "description": "how long, such as 10 minutes"},
            "label": {
                "type": "string",
                "description": "what it is for, such as pasta",
                "maxLength": 80,
            },
        },
        "required": ["duration"],
    }
    aliases = {"length": "duration", "time": "duration", "minutes": "duration", "name": "label"}
    timeout_s = 5.0
    speaks_for_itself = True

    def __init__(self, context=None, store=None) -> None:
        super().__init__(context)
        self.store = store if store is not None else store_for(context)

    def match(self, text):
        return timer_request(text)

    def run(self, arguments):
        seconds = parse_duration(arguments.get("duration", ""))
        if seconds is None:
            return ToolResult(
                content="No timer was set because no length was given. Ask how long it should be.",
                speak="How long should the timer be?",
            )
        try:
            alert = self.store.add_timer(seconds, clean_label(arguments.get("label", "")))
        except TimerLimitError as error:
            return ToolResult(content=f"No timer was set: {error}.", speak=f"Sorry, {error}.")
        length = format_duration(alert.duration_s)
        ends = format_clock(datetime.fromtimestamp(alert.due_at).astimezone())
        if alert.label:
            adjective = duration_adjective(alert.duration_s)
            number = adjective.split()[0]
            article = "an" if number.startswith("8") or number in ("11", "18") else "a"
            speak = f"Okay, {article} {adjective} {timer_name(alert)}, starting now."
        else:
            speak = f"Okay, {length}, starting now."
        called = f" called {alert.label}" if alert.label else ""
        return ToolResult(
            content=f"Started a timer for {length}{called}. It ends at {ends}.",
            speak=speak,
            data={"id": alert.id, "due_at": alert.due_at},
        )


def create_tool(context: ToolContext) -> SetTimerTool:
    return SetTimerTool(context)
