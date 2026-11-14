from __future__ import annotations

import time

from arietta_voice.timephrases import format_remaining
from arietta_voice.timers import RINGING, describe, timer_name
from arietta_voice.tools.base import BaseTool, ToolContext, ToolResult
from arietta_voice.tools.builtin._timers import check_request, select_alerts, store_for

NOTHING = {
    "timer": "You don't have any timers running.",
    "reminder": "You don't have any reminders set.",
    "alarm": "You don't have any alarms set.",
    "any": "You don't have any timers, reminders or alarms.",
}


class CheckTimersTool(BaseTool):
    name = "check_timers"
    description = "the user asks about their timers, reminders or alarms"
    parameters = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["timer", "reminder", "alarm", "any"]},
            "which": {"type": "string", "description": "one timer, such as pasta", "maxLength": 80},
        },
    }
    timeout_s = 5.0

    def __init__(self, context=None, store=None, clock=time.time) -> None:
        super().__init__(context)
        self.store = store if store is not None else store_for(context)
        self.clock = clock

    def match(self, text):
        return check_request(text, self.store.active)

    def run(self, arguments):
        kind = arguments.get("kind") or "any"
        now = self.clock()
        alerts = [a for a in self.store.active() if kind == "any" or a.kind == kind]
        which = str(arguments.get("which") or "").strip()
        if which:
            alerts = select_alerts(alerts, which) or alerts
        alerts.sort(key=lambda a: (a.status != RINGING, a.kind != "timer", a.due_at))
        if not alerts:
            answer = NOTHING[kind]
        elif len(alerts) == 1 and alerts[0].kind == "timer":
            alert = alerts[0]
            if alert.status == RINGING:
                answer = f"Your {timer_name(alert)} is going off now."
            else:
                answer = (
                    f"Your {timer_name(alert)} has {format_remaining(alert.due_at - now)} left."
                )
        elif len(alerts) == 1:
            answer = f"You have {describe(alerts[0], now)}."
        else:
            items = [describe(alert, now) for alert in alerts]
            answer = f"You have {_count(alerts)}: " + ", ".join(items[:-1]) + f", and {items[-1]}."
        return ToolResult(
            content=answer,
            speak=answer,
            data={"alerts": [alert.to_dict() for alert in alerts]},
        )


def _count(alerts) -> str:
    parts = []
    for noun in ("timer", "reminder", "alarm"):
        count = sum(alert.kind == noun for alert in alerts)
        if count:
            parts.append(f"{count} {noun}" + ("s" if count != 1 else ""))
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def create_tool(context: ToolContext) -> CheckTimersTool:
    return CheckTimersTool(context)
