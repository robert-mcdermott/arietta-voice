from __future__ import annotations

import time
from datetime import datetime

from arietta_voice.timephrases import describe_when
from arietta_voice.timers import RINGING, alarm_name, reminder_phrase, timer_name
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult
from arietta_voice.tools.builtin._timers import cancel_request, select_alerts, store_for

NOUNS = {
    "timer": "timers",
    "reminder": "reminders",
    "alarm": "alarms",
    "any": "timers, reminders or alarms",
}


class CancelTimerTool(BaseTool):
    name = "cancel_timer"
    description = "the user wants to cancel or stop a timer, reminder or alarm"
    parameters = {
        "type": "object",
        "properties": {
            "which": {
                "type": "string",
                "description": "which one, such as pasta or all",
                "maxLength": 200,
            },
            "kind": {"type": "string", "enum": ["timer", "reminder", "alarm", "any"]},
        },
    }
    timeout_s = 5.0
    speaks_for_itself = True

    def __init__(self, context=None, store=None) -> None:
        super().__init__(context)
        self.store = store if store is not None else store_for(context)

    def match(self, text):
        return cancel_request(text)

    def run(self, arguments):
        kind = arguments.get("kind") or "any"
        which = " ".join(str(arguments.get("which") or "").lower().split())
        candidates = [a for a in self.store.active() if kind == "any" or a.kind == kind]
        if not candidates:
            return _say(f"You don't have any {NOUNS[kind]}.")
        if which in ("all", "everything", "every one", "all of them"):
            chosen = candidates
        elif which:
            chosen = select_alerts(candidates, which)
            if not chosen:
                return _say(f"I couldn't find one for {which}.")
        else:
            ringing = [a for a in candidates if a.status == RINGING]
            chosen = ringing or (candidates if len(candidates) == 1 else [])
        if not chosen or (which and which != "all" and len(chosen) > 1):
            names = [_name(alert) for alert in (chosen or candidates)]
            listed = ", ".join(names[:-1]) + f" or {names[-1]}"
            return _say(f"Which one: {listed}?")
        cancelled = self.store.cancel([alert.id for alert in chosen])
        if not cancelled:
            return _say("That one has already finished.")
        if len(cancelled) == 1:
            return _say(f"Okay, I've cancelled {_name(cancelled[0])}.", cancelled)
        return _say(f"Okay, I've cancelled {len(cancelled)} {NOUNS[kind]}.", cancelled)


def _name(alert) -> str:
    if alert.kind == "timer":
        return f"your {timer_name(alert)}"
    if alert.kind == "alarm":
        due = datetime.fromtimestamp(alert.due_at).astimezone()
        when = describe_when(due, datetime.fromtimestamp(time.time()).astimezone())
        return f"your {alarm_name(alert)} {when}"
    return f"your reminder {reminder_phrase(alert)}"


def _say(answer: str, cancelled=()) -> ToolResult:
    return ToolResult(
        content=answer,
        speak=answer,
        data={"cancelled": [alert.id for alert in cancelled]},
    )


def create_tool(context: ToolContext) -> CancelTimerTool:
    return CancelTimerTool(context)
