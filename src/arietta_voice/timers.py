"""Timers and reminders that survive restarts and are shared between processes.

Tools in the inference worker add them, the voice runtime announces them when
they are due, and the console lists and cancels them. All three use one small
JSON file. Every change takes an exclusive lock and replaces the file
atomically, so a worker stopped in the middle of a write cannot leave it half
written.

An alert is ``pending`` until it is due, then ``ringing``: it is announced again
every ``repeat_s`` seconds until someone answers it, up to ``max_repeats`` more
times. Answering it (or cancelling it) makes it ``done``. Finished alerts are
kept for a day so the console can show what happened.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

from arietta_voice.timephrases import (
    describe_when,
    duration_adjective,
    format_clock,
    format_duration,
    format_remaining,
    normalize,
    parse_duration,
    second_person,
)

PENDING, RINGING, DONE = "pending", "ringing", "done"
KINDS = ("timer", "reminder", "alarm")
MAX_ACTIVE = 50
MAX_TIMER_S = 24 * 3600
MAX_REMINDER_S = 366 * 24 * 3600
MAX_LABEL_CHARS = 200
KEEP_FINISHED_S = 24 * 3600
MAX_FINISHED = 100
# Announced later than this (the assistant was off, or the Mac was asleep), the
# announcement says so.
LATE_S = 120

logger = logging.getLogger("arietta_voice.timers")
T = TypeVar("T")


class TimerLimitError(ValueError):
    """Too many alerts, or one that is too long, too short or badly described."""


def apology(error: Exception) -> str:
    """An error message to follow "Sorry, " in speech."""
    text = str(error)
    return text[:1].lower() + text[1:]


@dataclass(slots=True)
class Alert:
    id: str
    kind: str
    due_at: float  # seconds since the epoch
    created_at: float
    label: str = ""  # what a timer is for, or what to be reminded about
    duration_s: float | None = None  # timers only
    form: str = ""  # reminders: "to", "that" or "about", as the user said it
    status: str = PENDING
    rings: int = 0
    rang_at: float | None = None
    finished_at: float | None = None
    outcome: str | None = None  # dismissed, snoozed, cancelled, acknowledged, unanswered

    @classmethod
    def from_dict(cls, data: dict) -> Alert | None:
        """An alert from the file, or None for an entry that is not well formed."""
        known = {item.name for item in fields(cls)}
        try:
            alert = cls(**{key: value for key, value in data.items() if key in known})
        except TypeError:
            return None

        def number(value, optional=False):
            if value is None:
                return optional
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        valid = (
            isinstance(alert.id, str)
            and alert.kind in KINDS
            and alert.status in (PENDING, RINGING, DONE)
            and number(alert.due_at)
            and number(alert.created_at)
            and isinstance(alert.label, str)
            and isinstance(alert.form, str)
            and isinstance(alert.rings, int)
            and not isinstance(alert.rings, bool)
            and all(
                number(value, optional=True)
                for value in (alert.duration_s, alert.rang_at, alert.finished_at)
            )
            and (alert.outcome is None or isinstance(alert.outcome, str))
        )
        return alert if valid else None

    @property
    def active(self) -> bool:
        return self.status != DONE

    def to_dict(self) -> dict:
        return asdict(self)


DETERMINERS = {"the", "a", "an", "my", "your", "his", "her", "our", "their", "this", "that"}


def reminder_parts(text: str) -> tuple[str, str]:
    """How to read a reminder back: "to call my mom" -> ("to", "call your mom").

    Without "to", "that" or "about", an action ("call mom") reads best with
    "to" and a thing ("the dentist") with "about".
    """
    text = " ".join(second_person(str(text or "")).split())
    found = re.match(r"(?P<form>to|that|about|of|for)\s+(?P<rest>.+)$", text, re.IGNORECASE)
    if found:
        form = found.group("form").lower()
        return ("about" if form in ("of", "for") else form), found.group("rest").strip()
    if not text:
        return "", ""
    return ("about" if text.split()[0].lower() in DETERMINERS else "to"), text


def timer_name(alert: Alert) -> str:
    """ "pasta timer" or "10 minute timer"."""
    label = re.sub(r"\s*\btimer\b\s*$", "", alert.label.strip(), flags=re.IGNORECASE)
    if re.match(r"(?:to|for)\s", label, re.IGNORECASE):
        return f"timer {label}"  # "timer to check the oven"
    if label:
        return f"{label} timer"
    return f"{duration_adjective(alert.duration_s or 0)} timer"


def reminder_phrase(alert: Alert) -> str:
    """ "to call your mom", "that the bins go out", "about the dentist"."""
    return f"{alert.form or 'about'} {alert.label}"


def alarm_name(alert: Alert) -> str:
    """ "alarm" or "gym alarm"."""
    label = re.sub(r"\s*\balarm\b\s*$", "", alert.label.strip(), flags=re.IGNORECASE)
    return f"{label} alarm" if label and not label.lower().startswith("time to") else "alarm"


def describe(alert: Alert, now: float) -> str:
    """One active alert, for "what timers do I have"."""
    if alert.kind == "timer":
        if alert.status == RINGING:
            return f"your {timer_name(alert)}, which is going off now"
        return f"your {timer_name(alert)}, with {format_remaining(alert.due_at - now)} left"
    due = datetime.fromtimestamp(alert.due_at).astimezone()
    when = describe_when(due, datetime.fromtimestamp(now).astimezone())
    if alert.kind == "alarm":
        if alert.status == RINGING:
            return f"your {alarm_name(alert)}, which is going off now"
        return (
            f"an {alarm_name(alert)} {when}"
            if alarm_name(alert) == "alarm"
            else (f"your {alarm_name(alert)} {when}")
        )
    return f"a reminder {reminder_phrase(alert)} {when}"


def announcement(alerts: list[Alert], now: float) -> str:
    """What to say when alerts go off."""
    sentences = []
    for alert in alerts:
        late = now - alert.due_at if alert.rings == 0 else 0
        if alert.kind == "timer":
            if late > LATE_S:
                sentences.append(f"Your {timer_name(alert)} finished {format_remaining(late)} ago.")
            else:
                sentences.append(f"Your {timer_name(alert)} is done.")
            continue
        if alert.kind == "alarm":
            clock = format_clock(datetime.fromtimestamp(alert.due_at).astimezone())
            if alert.label.lower().startswith("time to"):
                body = f"It's {clock}, {alert.label}."
            elif alert.label:
                body = f"It's {clock}. This is your {alarm_name(alert)}."
            else:
                body = f"It's {clock}. This is your alarm."
            if late > LATE_S:
                body = f"Your {clock} alarm went off {format_remaining(late)} ago."
            sentences.append(body)
            continue
        if alert.form == "to":
            body = f"It's time to {alert.label}."
        elif alert.form == "that":
            body = f"Just a reminder that {alert.label}."
        else:
            body = f"Here's your reminder about {alert.label}."
        if late > LATE_S:
            due = datetime.fromtimestamp(alert.due_at).astimezone()
            body = f"This reminder was due at {format_clock(due)}. {body}"
        sentences.append(body)
    return " ".join(sentences)


class TimerStore:
    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    def signature(self) -> tuple | None:
        """Changes whenever the file is replaced; cheap enough to check often."""
        try:
            stat = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size, stat.st_ino)

    def load(self) -> list[Alert]:
        """All alerts; an unreadable file reads as empty and is set aside on the next change."""
        try:
            return self._read()
        except (OSError, ValueError) as exc:
            logger.warning("Timers file could not be read (%s)", type(exc).__name__)
            return []

    def active(self) -> list[Alert]:
        return sorted((a for a in self.load() if a.active), key=lambda a: a.due_at)

    def finished(self) -> list[Alert]:
        done = [a for a in self.load() if not a.active]
        return sorted(done, key=lambda a: a.finished_at or 0, reverse=True)

    def mutate(self, change: Callable[[list[Alert], float], T]) -> T:
        """Apply ``change(alerts, now)`` under the lock and save the result."""
        with self._lock():
            try:
                alerts = self._read()
            except (OSError, ValueError):
                alerts = []
                if self.path.exists():
                    aside = self.path.with_name(f"{self.path.name}.unreadable-{int(time.time())}")
                    os.replace(self.path, aside)
                    logger.warning("Unreadable timers file moved to %s", aside.name)
            now = self.clock()
            result = change(alerts, now)
            self._write(_prune(alerts, now))
            return result

    def _read(self) -> list[Alert]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        payload = json.loads(text)
        if not isinstance(payload, dict) or not isinstance(payload.get("alerts"), list):
            raise ValueError("Expected an object with an alerts list")
        alerts = [Alert.from_dict(item) for item in payload["alerts"] if isinstance(item, dict)]
        return [alert for alert in alerts if alert is not None]

    @contextlib.contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path.with_name(f"{self.path.name}.lock"), os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _write(self, alerts: list[Alert]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = json.dumps(
            {"version": 1, "alerts": [alert.to_dict() for alert in alerts]}, indent=1
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise

    # Changes used by tools and the console --------------------------------

    def add_timer(self, seconds: float, label: str = "") -> Alert:
        if not 1 <= seconds <= MAX_TIMER_S:
            raise TimerLimitError(
                "Timers can run from 1 second to 24 hours; for longer, ask for a reminder"
            )
        return self._add("timer", seconds, label=label, duration_s=float(round(seconds)))

    def add_reminder(self, text: str, due: datetime, form: str = "") -> Alert:
        if not text.strip():
            raise TimerLimitError("A reminder needs something to remind you about")
        seconds = due.timestamp() - self.clock()
        if not 0 < seconds <= MAX_REMINDER_S:
            raise TimerLimitError("Reminders must be in the future and within a year")
        return self._add("reminder", seconds, label=text, form=form)

    def add_alarm(self, due: datetime, label: str = "") -> Alert:
        seconds = due.timestamp() - self.clock()
        if not 0 < seconds <= 7 * 24 * 3600:
            raise TimerLimitError("Alarms must be in the future and within a week")
        return self._add("alarm", seconds, label=label)

    def cancel(self, ids: list[str], outcome: str = "cancelled") -> list[Alert]:
        wanted = set(ids)

        def change(alerts, now):
            cancelled = []
            for alert in alerts:
                if alert.id in wanted and alert.active:
                    alert.status, alert.finished_at, alert.outcome = DONE, now, outcome
                    cancelled.append(alert)
            return cancelled

        return self.mutate(change)

    def _add(self, kind: str, seconds: float, **values) -> Alert:
        label = " ".join(str(values.pop("label", "")).split())[:MAX_LABEL_CHARS]

        def change(alerts, now):
            if sum(alert.active for alert in alerts) >= MAX_ACTIVE:
                raise TimerLimitError(f"There are already {MAX_ACTIVE} timers and reminders")
            alert = Alert(
                id=secrets.token_hex(4),
                kind=kind,
                due_at=now + seconds,
                created_at=now,
                label=label,
                **values,
            )
            alerts.append(alert)
            return alert

        return self.mutate(change)


def _prune(alerts: list[Alert], now: float) -> list[Alert]:
    active = [alert for alert in alerts if alert.active]
    finished = sorted(
        (a for a in alerts if not a.active and now - (a.finished_at or 0) < KEEP_FINISHED_S),
        key=lambda a: a.finished_at or 0,
        reverse=True,
    )[:MAX_FINISHED]
    alerts[:] = [*active, *finished]
    return alerts


class AlertScheduler:
    """The runtime's view of the store: what is due, and what is ringing."""

    def __init__(
        self,
        store: TimerStore,
        *,
        repeat_s: float = 30.0,
        max_repeats: int = 4,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store, self.repeat_s, self.max_repeats, self.clock = (
            store,
            repeat_s,
            max_repeats,
            clock,
        )
        self._signature: object = ()
        self._alerts: list[Alert] = []
        # Also remembered here, so a store that cannot be written does not
        # make an alert ring on every pass of the listening loop.
        self._last_rang: dict[str, float] = {}

    def _refresh(self) -> list[Alert]:
        signature = self.store.signature()
        if signature != self._signature:
            self._signature = signature
            self._alerts = self.store.load()
        return self._alerts

    def _quiet(self, alert: Alert, now: float) -> bool:
        last = self._last_rang.get(alert.id)
        return last is not None and now - last < self.repeat_s

    def due(self) -> list[Alert]:
        """Alerts to announce now, oldest first. Unanswered ones are retired here."""
        now = self.clock()
        due, expired = [], []
        for alert in self._refresh():
            if alert.status == PENDING and alert.due_at <= now and not self._quiet(alert, now):
                due.append(alert)
            elif alert.status == RINGING and now - (alert.rang_at or 0) >= self.repeat_s:
                if alert.rings > self.max_repeats:
                    expired.append(alert.id)
                elif not self._quiet(alert, now):
                    due.append(alert)
        if expired:
            self._finish(expired, "unanswered", only=(RINGING,))
        return sorted(due, key=lambda alert: alert.due_at)

    def ringing(self) -> list[Alert]:
        return [alert for alert in self._refresh() if alert.status == RINGING]

    def next_due_in(self) -> float | None:
        now = self.clock()
        pending = [a.due_at - now for a in self._refresh() if a.status == PENDING]
        return min(pending) if pending else None

    def rang(self, alerts: list[Alert]) -> None:
        ids = {alert.id for alert in alerts}
        now = self.clock()
        for alert_id in ids:
            self._last_rang[alert_id] = now

        def change(stored, when):
            for alert in stored:
                if alert.id in ids and alert.active:
                    alert.status, alert.rings, alert.rang_at = RINGING, alert.rings + 1, when

        self._change(change)

    def dismiss(self, outcome: str = "dismissed") -> list[Alert]:
        """Stop everything that is ringing."""
        ids = [alert.id for alert in self.ringing()]
        return self._finish(ids, outcome, only=(RINGING,)) if ids else []

    def snooze(self, seconds: float) -> list[Alert]:
        """Ring again after ``seconds`` instead of now."""
        ids = {alert.id for alert in self.ringing()}

        def change(stored, now):
            snoozed = []
            for alert in stored:
                if alert.id in ids and alert.status == RINGING:
                    alert.status, alert.due_at, alert.rings, alert.rang_at = (
                        PENDING,
                        now + seconds,
                        0,
                        None,
                    )
                    snoozed.append(alert)
            return snoozed

        return self._change(change) if ids else []

    def _finish(self, ids, outcome, *, only=(PENDING, RINGING)) -> list[Alert]:
        wanted = set(ids)

        def change(stored, now):
            finished = []
            for alert in stored:
                if alert.id in wanted and alert.status in only:
                    alert.status, alert.finished_at, alert.outcome = DONE, now, outcome
                    finished.append(alert)
            return finished

        return self._change(change) or []

    def _change(self, change):
        try:
            result = self.store.mutate(change)
        except OSError as exc:
            logger.warning("Timers file could not be updated (%s)", type(exc).__name__)
            return None
        self._signature = ()  # Read the saved state on the next check.
        return result


# What people say while an alert is going off --------------------------------

STOP = re.compile(
    r"(?:(?:ok(?:ay)?|alright|all right|yes|yeah|yep)\s*)*"
    r"(?:stop|stop it|stop that|dismiss|dismiss it|cancel|cancel it|quiet|be quiet|enough|"
    r"that's enough|shut up|silence|turn it off|turn that off|got it|i got it|i heard you|"
    r"heard you|thanks?|thank you|i'm up|i am up|ok(?:ay)?|alright|done|"
    r"(?:stop|dismiss|cancel|silence|turn off|end) (?:the |my |that |this )?"
    r"(?:\w+ )?(?:timer|timers|alarm|alert|reminder)s?(?: off)?)"
    r"(?:\s*(?:now|please|thanks|thank you))*"
)
SNOOZE = re.compile(
    r"(?:(?:ok(?:ay)?|alright|yes|yeah)\s*)*(?:please\s+)?"
    r"(?:snooze(?: it| that| the (?:timer|alarm|reminder))?(?: for)?(?P<a>.*)|"
    r"remind me again(?: in)?(?P<b>.*)|"
    r"(?:give me|i need)(?P<c>.+?) more(?: minutes?)?|"
    r"(?P<d>.+?) more minutes?)"
)


def _command_text(text: str) -> str:
    text = normalize(text)
    return re.sub(r"[^a-z0-9:' ]+", " ", text).strip()


def is_stop_command(text: str) -> bool:
    return bool(STOP.fullmatch(" ".join(_command_text(text).split())))


def snooze_request(text: str) -> float | None:
    """Seconds asked for ("snooze for ten minutes" is 600), 0.0 for a bare snooze, else None."""
    found = SNOOZE.fullmatch(" ".join(_command_text(text).split()))
    if found is None:
        return None
    group = next(name for name in "abcd" if found.group(name) is not None)
    rest = found.group(group).strip()
    if not rest:
        return 0.0 if group in "ab" else None
    if group in "cd" and not re.search(r"\b(?:seconds?|minutes?|hours?)$", rest):
        rest += " minutes"  # "give me five more" means minutes
    seconds = parse_duration(rest)
    return seconds if seconds and seconds <= MAX_TIMER_S else None


def snooze_reply(seconds: float) -> str:
    return f"Okay, I'll remind you again in {format_duration(seconds)}."
