from __future__ import annotations

import re
from datetime import datetime, timedelta

from arietta_voice.timephrases import describe_when, parse_when
from arietta_voice.timers import TimerLimitError
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult
from arietta_voice.tools.builtin._timers import alarm_request, store_for

EXPLICIT_HALF = re.compile(
    r"\b(?:am|pm|a\.?\s?m|p\.?\s?m|noon|midnight|morning|afternoon|evening|tonight|night|in)\b",
    re.IGNORECASE,
)


LATER_DAY = re.compile(
    r"\b(?:tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)


def local_now() -> datetime:
    return datetime.now().astimezone()


class SetAlarmTool(BaseTool):
    """Instant answers for "wake me up at 7" and "set an alarm for 6:30".

    Not offered to the model: asking it for a reminder ("to wake up", at 7 AM)
    does the same, and one tool fewer keeps its instructions short.
    """

    name = "set_alarm"
    description = "the user wants an alarm at a time of day, such as to wake up"
    parameters = {
        "type": "object",
        "properties": {
            "when": {"type": "string", "description": "such as at 7 am or tomorrow at 6:30"},
            "label": {"type": "string", "maxLength": 80},
        },
        "required": ["when"],
    }
    model_callable = False
    timeout_s = 5.0
    speaks_for_itself = True

    def __init__(self, context=None, store=None, clock=local_now) -> None:
        super().__init__(context)
        self.store = store if store is not None else store_for(context)
        self.clock = clock

    def match(self, text):
        return alarm_request(text, self.clock())

    def run(self, arguments):
        now = self.clock()
        phrase = arguments.get("when", "")
        label = " ".join(str(arguments.get("label") or "").split())
        when = parse_when(phrase, now)
        if when is None:
            return ToolResult(
                content="No alarm was set because no time was given.",
                speak="What time should I set it for?",
            )
        due = when.due
        morning = label == "time to wake up" or LATER_DAY.search(phrase)
        if morning and not when.relative and not EXPLICIT_HALF.search(phrase):
            # "Wake me up at 7" and "an alarm for 5:45 tomorrow" mean the morning.
            if due.hour >= 12:
                due -= timedelta(hours=12)
                if due <= now:
                    due += timedelta(days=1)
        try:
            alert = self.store.add_alarm(due, label)
        except TimerLimitError as error:
            return ToolResult(content=f"No alarm was set: {error}.", speak=f"Sorry, {error}.")
        timing = describe_when(due, now)
        if label == "time to wake up":
            speak = f"Okay, I'll wake you {timing}."
        else:
            speak = f"Okay, your alarm is set {timing}."
        return ToolResult(
            content=f"Alarm set {timing}.",
            speak=speak,
            data={"id": alert.id, "due_at": alert.due_at},
        )


def create_tool(context: ToolContext) -> SetAlarmTool:
    return SetAlarmTool(context)
