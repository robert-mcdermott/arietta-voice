from __future__ import annotations

from datetime import datetime, time, timedelta

from arietta_voice.timephrases import (
    describe_when,
    format_duration,
    local_datetime,
    parse_when,
)
from arietta_voice.timers import TimerLimitError, apology, reminder_parts
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult
from arietta_voice.tools.builtin._timers import reminder_request, store_for


def local_now() -> datetime:
    return datetime.now().astimezone()


class SetReminderTool(BaseTool):
    name = "set_reminder"
    description = "the user wants a reminder at a time or after a delay"
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "what about, starting with to or that, such as to call mom",
                "maxLength": 200,
            },
            "when": {"type": "string", "description": "such as at 5 pm"},
        },
        "required": ["text", "when"],
    }
    aliases = {
        "note": "text",
        "message": "text",
        "reminder": "text",
        "task": "text",
        "what": "text",
        "time": "when",
        "at": "when",
        "datetime": "when",
        "due": "when",
    }
    timeout_s = 5.0
    speaks_for_itself = True

    def __init__(self, context=None, store=None, clock=local_now) -> None:
        super().__init__(context)
        self.store = store if store is not None else store_for(context)
        self.clock = clock

    def match(self, text):
        return reminder_request(text, self.clock())

    def run(self, arguments):
        now = self.clock()
        form, body = reminder_parts(arguments.get("text", ""))
        when = parse_when(arguments.get("when", ""), now)
        if not body:
            return ToolResult(
                content="No reminder was set because it had nothing to say. Ask what it is for.",
                speak="What should I remind you about?",
            )
        if when is None:
            return ToolResult(
                content="No reminder was set because no time was given. Ask when.",
                speak="When should I remind you?",
            )
        due = when.due
        if when.relative and due - now >= timedelta(days=1) and not 7 <= due.hour < 22:
            # "In three weeks" at 2 in the morning means that day, at a waking hour.
            due = local_datetime(due.date(), time(9), now)
        try:
            alert = self.store.add_reminder(body, due, form)
        except TimerLimitError as error:
            return ToolResult(
                content=f"No reminder was set: {error}.", speak=f"Sorry, {apology(error)}."
            )
        seconds = due.timestamp() - now.timestamp()
        if when.relative and seconds < 6 * 3600:
            timing = f"in {format_duration(seconds)}"
        else:
            timing = describe_when(due, now)
        return ToolResult(
            speak=f"Okay, I'll remind you {form} {body} {timing}.",
            content=f"Reminder set {describe_when(due, now)}: {form} {body}.",
            data={"id": alert.id, "due_at": alert.due_at},
        )


def create_tool(context: ToolContext) -> SetReminderTool:
    return SetReminderTool(context)
