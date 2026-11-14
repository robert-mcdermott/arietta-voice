"""Shared by the timer and reminder tools: the store, and reading requests.

Each tool's ``match`` recognizes the common ways of asking, so these answer
instantly without the model. Anything else reaches the model, which can call
the same tools with arguments.
"""

from __future__ import annotations

import re
from datetime import datetime

from arietta_voice.timephrases import find_duration, normalize, parse_duration, parse_when
from arietta_voice.timers import TimerStore

LEAD_IN = (
    r"^(?:(?:please|hey|ok(?:ay)?|so|and|also|now|can you|could you|would you|will you|"
    r"i need you to|i want you to|i'd like you to|go ahead and)\s+)*"
)
QUESTION = re.compile(
    r"^(?:how|what|what's|which|when|is|are|do|does|did|any|have|has)\b|"
    r"\b(?:left|remaining|cancel|stop|delete|remove|clear)\b"
)
SET_VERB = re.compile(r"\b(?:set|start|make|create|add|put|need|want|give|begin|run|launch)\b")
ARTICLES = r"(?:a|an|the|my|another|new|one more)"
LABEL_BEFORE = re.compile(rf"\b{ARTICLES}\s+(?P<label>[a-z][a-z' ]*?)\s+timer\b")
LABEL_AFTER = re.compile(
    rf"\btimer\b(?:\s+for)?\s+(?P<word>called|named|labell?ed|for|to)\s+"
    rf"(?:{ARTICLES}\s+)?(?P<label>\S.*)$"
)
NOT_A_LABEL = {"timer", "quick", "countdown", "kitchen"}


def store_for(context) -> TimerStore:
    return TimerStore(context.config.timers.path)


def clean_label(value: str) -> str:
    label = " ".join(str(value or "").split()).strip(" .,!?")
    label = re.sub(rf"^{ARTICLES}\s+", "", label, flags=re.IGNORECASE)
    return re.sub(r"\s*\btimer\b$", "", label, flags=re.IGNORECASE)


def timer_request(text: str) -> dict | None:
    """ "set a 10 minute pasta timer" -> {"duration": "10 minutes", "label": "pasta"}."""
    spoken = re.sub(r"[^a-z0-9:' ]+", " ", normalize(text))
    spoken = " ".join(re.sub(LEAD_IN, "", spoken).split())
    if not re.search(r"\btimer\b", spoken) or QUESTION.search(spoken):
        return None
    found = find_duration(spoken)
    if found is None:
        # "Set a timer" alone gets "how long?". With more said ("as long as an
        # egg takes"), the model may know the length, so it decides.
        bare = re.fullmatch(
            rf"(?:{SET_VERB.pattern} )?(?:me )?(?:{ARTICLES} )?timer(?: please)?", spoken
        )
        return {"duration": ""} if bare else None
    rest = f"{spoken[: found.start]} {spoken[found.end :]}"
    rest = " ".join(rest.split())
    label = ""
    after = LABEL_AFTER.search(rest)
    before = LABEL_BEFORE.search(rest)
    if after:
        label = clean_label(after.group("label"))
        if after.group("word") == "to" and label:
            label = f"to {label}"  # "your timer to check the oven"
    elif before and before.group("label") not in NOT_A_LABEL:
        label = clean_label(before.group("label"))
    arguments = {"duration": f"{round(found.seconds)} seconds"}
    if label:
        arguments["label"] = label
    return arguments


REMINDER_LEAD = re.compile(
    LEAD_IN + r"(?P<lead>remind me|ping me|nudge me|alert me|"
    r"give me (?:a|an) (?:reminder|nudge|ping|heads up|shout)|let me know|"
    r"don't let me forget|do not let me forget|make sure i remember|"
    r"(?:set|add|create|make|put in|set up) (?:a|an|up a) reminder)\b\s*",
    re.IGNORECASE,
)
# Only reminders when a time is given: "let me know when the oven is hot" is not.
LOOSE_LEADS = {"let me know", "don't let me forget", "do not let me forget", "make sure i remember"}
FORM = re.compile(r"\s*\b(?P<form>to|that|about|of|for|when to)\s+", re.IGNORECASE)
WHEN_START = re.compile(
    r"\s+(?=(?:in|at|on|by|tomorrow|tonight|today|this|next|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|noon|midnight|\d)\b)",
    re.IGNORECASE,
)


def reminder_request(text: str, now: datetime) -> dict | None:
    """ "remind me to call mom at 5" -> {"text": "to call mom", "when": "at 5"}."""
    spoken = text.replace("’", "'").strip().rstrip(".!?")
    lead = REMINDER_LEAD.match(spoken)
    if lead is None:
        return None
    # "Let me know" is only a reminder when a time is given.
    loose = lead.group("lead").lower() in LOOSE_LEADS
    rest = spoken[lead.end() :].strip(" ,")
    if parse_when(rest, now):
        return None if loose else {"text": "", "when": rest}  # The tool asks what for.
    # When first: "remind me in 20 minutes to check the oven".
    for form in FORM.finditer(rest):
        head = rest[: form.start()].strip(" ,")
        if head and parse_when(head, now):
            return {"text": _when_to(rest[form.start() :].strip(" ,")), "when": head}
    # What first: "remind me to put the kids in bed at 8"; the earliest split
    # that leaves a time phrase the whole way to the end wins.
    for boundary in WHEN_START.finditer(rest):
        tail = rest[boundary.end() :].strip(" ,")
        if parse_when(tail, now):
            return {"text": _when_to(rest[: boundary.start()].strip(" ,")), "when": tail}
    return None if loose else {"text": rest, "when": ""}


def _when_to(text: str) -> str:
    """ "when to flip the steak" reads back as "to flip the steak"."""
    return re.sub(r"^when\s+(?=to\s)", "", text, flags=re.IGNORECASE)


ALARM = re.compile(
    LEAD_IN + r"(?:"
    r"(?P<wake>wake me(?: up)?|get me up)|"
    r"(?:set|make|create|add|put on|give me) (?:an|the|my|a) (?:(?P<label>[a-z ]+?) )?alarm|"
    r"alarm"
    r")\s+(?P<when>.+)$",
    re.IGNORECASE,
)


def alarm_request(text: str, now: datetime) -> dict | None:
    """ "wake me up at 7" -> {"when": "at 7", "label": "time to wake up"}."""
    spoken = text.replace("’", "'").strip().rstrip(".!?").lower()
    found = ALARM.match(spoken)
    if found is None or not parse_when(found.group("when"), now):
        return None
    arguments = {"when": found.group("when")}
    if found.group("wake"):
        arguments["label"] = "time to wake up"
    elif found.group("label") and found.group("label") not in ("new", "another"):
        arguments["label"] = found.group("label")
    return arguments


CHECK = re.compile(
    LEAD_IN + r"(?:"
    r"how (?:much|long)(?: time)?(?: is| s)? (?:left|remaining)|"
    r"how much longer|how long until|how long till|"
    r"how (?:much time|long) (?:is |s )?(?:left|remaining)|"
    r"(?:what|which|any) (?:timers?|reminders?|alarms?)|"
    r"(?:what s|what is|how s|how is) (?:on |left on )?(?:the|my) (?:\w+ )?"
    r"(?:timers?|reminders?|alarms?)|"
    r"(?:list|show|check|read)(?: me)? (?:all )?(?:my|the)? ?(?:timers?|reminders?|alarms?)|"
    r"(?:do|have) i (?:have|got|set) any (?:timers?|reminders?|alarms?)|"
    r"(?:are|is) there any (?:timers?|reminders?|alarms?)|"
    r"what(?: s| is| are) my (?:timers?|reminders?|alarms?)|"
    r"when(?: s| is) my (?:\w+ )?alarm|is my (?:\w+ )?alarm set|"
    r"time (?:left|remaining) on"
    r")\b"
)
# Only answered instantly when such a timer exists; otherwise the model decides.
DONE_YET = re.compile(
    LEAD_IN + r"(?:is|are) (?:my|the) (?P<which>[a-z0-9 ]+?)(?: timer)? "
    r"(?:nearly |almost |about |close to |ready |done )?(?:done|ready|finished|up)\b"
)


def check_request(text: str, active=None) -> dict | None:
    """What to check; ``active()`` (read only when needed) settles "is my pasta done?"."""
    spoken = " ".join(re.sub(r"[^a-z0-9: ]+", " ", normalize(text)).split())
    done_yet = DONE_YET.match(spoken)
    if done_yet and active is not None:
        timers = [alert for alert in active() if alert.kind == "timer"]
        if select_alerts(timers, done_yet.group("which")):
            return {"kind": "timer", "which": done_yet.group("which")}
    if not CHECK.search(spoken):
        return None
    arguments = {"kind": _kind(spoken)}
    named = re.search(r"\b(?:on|for) (?:the |my )?(?P<which>[a-z0-9 ]+?) timer\b", spoken)
    if named and named.group("which") not in ("a", "this", "that"):
        arguments["which"] = named.group("which")
    return arguments


IGNORED_WORDS = {"the", "my", "a", "an", "to", "for", "and", "of", "on", "at", "in", "your"}


def select_alerts(candidates: list, which: str) -> list:
    """Alerts that best match a description: a length ("10 minute") or label words."""
    by_length = [alert for alert in candidates if same_duration(which, alert.duration_s)]
    if by_length:
        return by_length
    wanted = {w for w in re.findall(r"[a-z0-9']+", which.lower()) if w not in IGNORED_WORDS}
    scored = []
    for alert in candidates:
        score = len(wanted & set(re.findall(r"[a-z0-9']+", alert.label.lower())))
        if score:
            scored.append((score, alert))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [alert for score, alert in scored if score == best]


CANCEL = re.compile(
    LEAD_IN + r"(?:cancel|stop|delete|remove|clear|turn off|end|kill|drop|scratch|"
    r"forget(?: about)?|never ?mind(?: about)?|no need for)\b(?P<rest>.*)$"
)


def cancel_request(text: str) -> dict | None:
    spoken = " ".join(re.sub(r"[^a-z0-9:' ]+", " ", normalize(text)).split())
    found = CANCEL.match(spoken)
    if found is None or not re.search(r"\b(?:timers?|reminders?|alarms?)\b", spoken):
        return None
    rest = found.group("rest")
    kind = _kind(spoken)
    if re.search(r"\b(?:all|every|everything)\b", rest):
        return {"which": "all", "kind": kind}
    which = re.sub(
        r"\b(?:timers?|reminders?|alarms?|the|my|a|an|that|this|one|please|now)\b", " ", rest
    )
    which = re.sub(r"^\s*(?:to|for|about|that)\b", " ", " ".join(which.split()))
    return {"which": " ".join(which.split()), "kind": kind}


def _kind(spoken: str) -> str:
    kinds = [
        kind
        for kind, pattern in (
            ("timer", r"\b(?:timers?|left|remaining|longer)\b"),
            ("reminder", r"\breminders?\b"),
            ("alarm", r"\balarms?\b"),
        )
        if re.search(pattern, spoken)
    ]
    return kinds[0] if len(kinds) == 1 else "any"


def same_duration(which: str, seconds: float | None) -> bool | None:
    """None when ``which`` names no duration, else whether it matches."""
    found = find_duration(normalize(which))
    if found is None:
        parsed = parse_duration(which)
        if parsed is None:
            return None
        return seconds is not None and abs(parsed - seconds) < 1
    return seconds is not None and abs(found.seconds - seconds) < 1
