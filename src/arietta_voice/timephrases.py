"""Read and say durations and times of day the way people speak them.

Speech recognition writes numbers as digits or as words ("10 minutes", "ten
minutes", "an hour and a half", "5:30 p.m.", "five thirty pm"), so both are read.
Everything here is plain functions over text and ``datetime`` values, with the
current time passed in, so it is easy to test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta

ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
UNITS = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
    "day": 86400,
    "days": 86400,
    "week": 604800,
    "weeks": 604800,
}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# Default hour when only a part of the day is given ("tomorrow morning").
PERIOD_HOURS = {"morning": 9, "afternoon": 14, "evening": 18, "night": 20}
FILLERS = {"please", "about", "around", "approximately", "roughly", "exactly", "sharp", "by"}
TOKEN = re.compile(r"\d+:\d{2}|\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lower case with a.m./p.m. and o'clock folded into single words."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b([ap])\.?\s?m\b\.?", r" \1m ", text)
    text = re.sub(r"\bo'?\s?clock\b", " oclock ", text)
    text = re.sub(r"(\d)\s*([ap]m)\b", r"\1 \2", text)
    text = re.sub(r"[-–—]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[re.Match]:
    return list(TOKEN.finditer(text))


# Numbers -------------------------------------------------------------------


def _number(words: list[str], i: int) -> tuple[float, int] | None:
    """A number starting at ``words[i]``: digits, words to the hundreds, or "a"."""
    if i >= len(words):
        return None
    word = words[i]
    if NUMBER.fullmatch(word):
        value, j = float(word), i + 1
    elif word in ("a", "an"):
        if words[i + 1 : i + 2] == ["couple"]:
            j = i + 2 + (words[i + 2 : i + 3] == ["of"])
            return 2.0, j
        value, j = 1.0, i + 1
    elif word == "couple":
        return 2.0, i + 1 + (words[i + 1 : i + 2] == ["of"])
    elif word in TENS:
        value, j = float(TENS[word]), i + 1
        if j < len(words) and 0 < ONES.get(words[j], 0) < 10:
            value, j = value + ONES[words[j]], j + 1
    elif word in ONES:
        value, j = float(ONES[word]), i + 1
    else:
        return None
    if words[j : j + 1] == ["hundred"]:
        value, j = value * 100, j + 1
        if words[j : j + 1] == ["and"] and _small_number(words, j + 1):
            j += 1
        extra = _small_number(words, j)
        if extra:
            value, j = value + extra[0], extra[1]
    if words[j : j + 3] == ["and", "a", "half"]:
        value, j = value + 0.5, j + 3
    return value, j


def _small_number(words: list[str], i: int) -> tuple[int, int] | None:
    if i >= len(words):
        return None
    if words[i] in TENS:
        value, j = TENS[words[i]], i + 1
        if j < len(words) and 0 < ONES.get(words[j], 0) < 10:
            value, j = value + ONES[words[j]], j + 1
        return value, j
    if words[i] in ONES:
        return ONES[words[i]], i + 1
    return None


# Durations -----------------------------------------------------------------


def _duration_part(words: list[str], i: int) -> tuple[float, int] | None:
    """One "<number> <unit>" piece, including halves and quarters."""
    head = words[i : i + 4]
    # "half an hour", "half a minute"
    if head[:1] == ["half"] and head[1:2] in (["a"], ["an"]) and head[2:3] and head[2] in UNITS:
        return UNITS[head[2]] / 2, i + 3
    # "a half hour", "a quarter hour", "a quarter of an hour", "quarter of an hour"
    start = i + (head[:1] in (["a"], ["an"]))
    rest = words[start : start + 4]
    if rest[:1] in (["half"], ["quarter"]):
        fraction = 0.5 if rest[0] == "half" else 0.25
        if rest[1:2] and rest[1] in UNITS:
            return UNITS[rest[1]] * fraction, start + 2
        if rest[1:3] in (["of", "an"], ["of", "a"]) and rest[3:4] and rest[3] in UNITS:
            return UNITS[rest[3]] * fraction, start + 4
    number = _number(words, i)
    if number is None:
        return None
    value, j = number
    if j >= len(words) or words[j] not in UNITS:
        return None
    unit = UNITS[words[j]]
    seconds, j = value * unit, j + 1
    if words[j : j + 3] == ["and", "a", "half"]:  # "an hour and a half"
        seconds, j = seconds + unit / 2, j + 3
    return seconds, j


def _duration_at(words: list[str], i: int) -> tuple[float, int] | None:
    total, j, found = 0.0, i, False
    while j < len(words):
        part = _duration_part(words, j)
        if part is None:
            break
        total, j, found = total + part[0], part[1], True
        if words[j : j + 1] == ["and"] and _duration_part(words, j + 1):
            j += 1
    return (total, j) if found else None


@dataclass(slots=True, frozen=True)
class DurationMatch:
    seconds: float
    start: int  # character span in the normalized text
    end: int


def find_duration(text: str) -> DurationMatch | None:
    """The first spoken duration in already normalized text, with its span."""
    tokens = _tokens(text)
    words = [token.group() for token in tokens]
    for i in range(len(words)):
        found = _duration_at(words, i)
        if found and found[0] > 0:
            return DurationMatch(found[0], tokens[i].start(), tokens[found[1] - 1].end())
    return None


COMPACT = re.compile(
    r"(?:(?P<h>\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?)?\s*"
    r"(?:(?P<m>\d+(?:\.\d+)?)\s*m(?:in(?:ute)?s?)?)?\s*"
    r"(?:(?P<s>\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?)?"
)
ISO_DURATION = re.compile(
    r"p(?:t)?(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?"
)
CLOCK_DURATION = re.compile(r"(?:(?P<h>\d+):)?(?P<m>\d{1,2}):(?P<s>\d{2})")


def parse_duration(text: str) -> float | None:
    """Seconds for a whole phrase such as "10 minutes", "1h30m" or "00:10:00".

    A bare number is taken as minutes. Returns None unless the entire phrase
    is a duration (leading "for", "in" or "about" and a trailing "from now" are
    allowed).
    """
    text = normalize(str(text or ""))
    if not text:
        return None
    for pattern in (ISO_DURATION, COMPACT, CLOCK_DURATION):
        found = pattern.fullmatch(text.replace(" ", "") if pattern is not COMPACT else text)
        if found and any(found.groupdict().values()):
            parts = {key: float(value or 0) for key, value in found.groupdict().items()}
            seconds = parts["h"] * 3600 + parts["m"] * 60 + parts["s"]
            return seconds if seconds > 0 else None
    if NUMBER.fullmatch(text):
        return float(text) * 60 or None
    words = [token.group() for token in _tokens(text)]
    while words and words[0] in {"for", "in", "another", "just", *FILLERS}:
        words = words[1:]
    while words and words[-1] in {"long", "more", "later"}:
        words = words[:-1]
    if words[-2:] == ["from", "now"]:
        words = words[:-2]
    found = _duration_at(words, 0)
    if found and found[1] == len(words) and found[0] > 0:
        return found[0]
    return None


# Times of day --------------------------------------------------------------


def _hour_word(word: str) -> int | None:
    if NUMBER.fullmatch(word) and "." not in word and 0 <= int(word) <= 24:
        return int(word)
    value = ONES.get(word)
    return value if value is not None and 1 <= value <= 12 else None


def _minutes_after(words: list[str], i: int) -> tuple[int, int] | None:
    """Minutes spoken after an hour: "30", "thirty", "forty five", "oh five"."""
    if i >= len(words):
        return None
    word = words[i]
    if NUMBER.fullmatch(word) and len(word) == 2 and int(word) < 60:
        return int(word), i + 1
    if word == "oh" and i + 1 < len(words) and 0 < ONES.get(words[i + 1], 0) < 10:
        return ONES[words[i + 1]], i + 2
    small = _small_number(words, i)
    if small and 10 <= small[0] < 60:
        return small
    return None


def _meridiem(words: list[str], j: int) -> tuple[str | None, int]:
    if j < len(words) and words[j] in ("am", "pm"):
        return words[j], j + 1
    if j < len(words) and words[j] == "oclock":
        if j + 1 < len(words) and words[j + 1] in ("am", "pm"):
            return words[j + 1], j + 2
        return None, j + 1
    return None, j


def _time_at(words: list[str], i: int, *, after_at: bool) -> tuple[tuple, int] | None:
    """(hour, minute, meridiem) at ``words[i]``; a bare hour needs "at" before it."""
    if i >= len(words):
        return None
    word = words[i]
    if word in ("noon", "midday"):
        return (12, 0, "pm"), i + 1
    if word == "midnight":
        return (0, 0, "am"), i + 1
    # "half past five", "quarter to six", "ten past five"
    offset = None
    if word in ("half", "quarter"):
        offset, j = (30 if word == "half" else 15), i + 1
    else:
        small = _small_number(words, i) or (
            (int(word), i + 1) if NUMBER.fullmatch(word) and "." not in word else None
        )
        if small and small[1] < len(words) and words[small[1]] in ("past", "to", "after"):
            offset, j = small
    if offset is not None and j < len(words) and words[j] in ("past", "to", "after"):
        hour = _hour_word(words[j + 1]) if j + 1 < len(words) else None
        if hour is None or offset >= 60:
            return None
        meridiem, k = _meridiem(words, j + 2)
        if words[j] == "to":
            hour, minute = (12 if hour == 1 else hour - 1), 60 - offset
        else:
            minute = offset
        return (hour, minute, meridiem), k
    if re.fullmatch(r"\d{1,2}[:.]\d{2}", word):
        hour, minute = (int(part) for part in re.split(r"[:.]", word))
        if hour > 24 or minute >= 60:
            return None
        meridiem, j = _meridiem(words, i + 1)
        return (hour, minute, meridiem), j
    if NUMBER.fullmatch(word) and "." not in word and len(word) in (3, 4):
        hour, minute = int(word[:-2]), int(word[-2:])
        meridiem, j = _meridiem(words, i + 1)
        if hour > 24 or minute >= 60 or not (after_at or meridiem):
            return None
        return (hour, minute, meridiem), j
    hour = _hour_word(word)
    if hour is None:
        return None
    minute, j = 0, i + 1
    minutes = _minutes_after(words, j)
    if minutes:
        minute, j = minutes
    meridiem, k = _meridiem(words, j)
    spoken_clock = k > j or minutes is not None
    if not (after_at or spoken_clock):
        return None
    return (hour, minute, meridiem), k


@dataclass(slots=True, frozen=True)
class When:
    due: datetime
    relative: bool  # said as a delay ("in 20 minutes") rather than a clock time


def parse_when(text: str, now: datetime) -> When | None:
    """When something should happen: "in 20 minutes", "at 5 pm", "tomorrow at 9".

    ``now`` should be timezone-aware local time. Returns None unless the whole
    phrase is understood, so a caller can try several ways of splitting a
    sentence and keep the one that parses.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    try:  # Models sometimes pass ISO 8601.
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        pass
    else:
        parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=now.tzinfo)
        return When(parsed, relative=False) if parsed > now else None
    words = [token.group() for token in _tokens(normalize(raw))]
    duration = clock = weekday = period = None
    day_offset = None
    next_week = False
    i = 0
    while i < len(words):
        word = words[i]
        following = words[i + 1] if i + 1 < len(words) else None
        if word == "for" and clock is None and not _duration_at(words, i + 1):
            found = _time_at(words, i + 1, after_at=True)  # "an alarm for 9"
            if found is not None:
                clock, i = found
                continue
        if word in FILLERS or word in ("and", "then", "on", "the", "for"):
            if word in ("on", "the") and following not in (*WEEKDAYS, "day"):
                return None
            i += 1
            continue
        if word == "in" and following == "the" and i + 2 < len(words):
            part = words[i + 2]
            if part in PERIOD_HOURS:
                period, i = part, i + 3
                continue
            return None
        if word == "in" or (duration is None and _duration_at(words, i)):
            found = _duration_at(words, i + (word == "in"))
            if found is None or duration is not None:
                return None
            duration, i = found
            if words[i : i + 2] == ["from", "now"]:
                i += 2
            elif words[i : i + 1] == ["later"]:
                i += 1
            continue
        if word == "at":
            if following == "night":
                period, i = "night", i + 2
                continue
            found = _time_at(words, i + 1, after_at=True)
            if found is None or clock is not None:
                return None
            clock, i = found
            continue
        if word in ("today", "tomorrow", "tonight"):
            if day_offset is not None:
                return None
            day_offset = 1 if word == "tomorrow" else 0
            if word == "tonight":
                period = "night"
            i += 1
            continue
        if word == "day" and words[i : i + 3] == ["day", "after", "tomorrow"]:
            day_offset, i = 2, i + 3
            continue
        if word in ("this", "next") and following in PERIOD_HOURS:
            period, i = following, i + 2
            day_offset = 0 if day_offset is None else day_offset
            continue
        if word in ("this", "next") and following in WEEKDAYS:
            weekday, next_week, i = WEEKDAYS.index(following), word == "next", i + 2
            continue
        if word in WEEKDAYS:
            weekday, i = WEEKDAYS.index(word), i + 1
            continue
        if word in PERIOD_HOURS:
            period, i = word, i + 1
            continue
        found = _time_at(words, i, after_at=False)
        if found is not None and clock is None:
            clock, i = found
            continue
        return None

    if duration is not None:
        if clock or day_offset or weekday is not None or period:
            return None
        if duration <= 0:
            return None
        return When(now + timedelta(seconds=duration), relative=True)
    if clock is None and day_offset is None and weekday is None and period is None:
        return None

    explicit_day = day_offset is not None or weekday is not None
    date = now.date()
    if day_offset:
        date += timedelta(days=day_offset)
    elif weekday is not None:
        delta = (weekday - now.weekday()) % 7
        if delta == 0 and next_week:
            delta = 7
        date += timedelta(days=delta)

    if clock is None:
        hour, minute = PERIOD_HOURS.get(period or "morning"), 0
        candidates = [(hour, minute)]
    else:
        hour, minute, meridiem = clock
        if meridiem == "pm" and hour < 12:
            candidates = [(hour + 12, minute)]
        elif meridiem == "am":
            candidates = [(0 if hour == 12 else hour, minute)]
        elif hour > 12 or hour == 0:
            candidates = [(hour % 24, minute)]
        elif period in ("afternoon", "evening", "night"):
            candidates = [(hour + 12 if hour < 12 else hour, minute)]
        elif period == "morning":
            candidates = [(0 if hour == 12 else hour, minute)]
        elif explicit_day:
            # Without am or pm, 1 to 6 means the afternoon and 7 to 11 the morning.
            candidates = [(hour + 12 if 1 <= hour <= 6 else hour, minute)]
        else:
            # The next time the clock shows this hour, morning or evening.
            candidates = [(hour % 12, minute), (hour % 12 + 12, minute)]

    tz = now.tzinfo
    options = sorted(datetime.combine(date, time(h, m), tzinfo=tz) for h, m in candidates)
    for option in options:
        if option > now:
            return When(option, relative=False)
    if weekday is not None and not day_offset:
        return When(options[0] + timedelta(days=7), relative=False)
    if explicit_day:
        return None  # "today at 9 am" after nine.
    return When(options[0] + timedelta(days=1), relative=False)


# Saying them ---------------------------------------------------------------


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _split(seconds: float) -> tuple[int, int, int]:
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return hours, minutes, secs


def format_duration(seconds: float) -> str:
    """ "10 minutes", "1 hour and 30 minutes", "45 seconds"."""
    hours, minutes, secs = _split(seconds)
    parts = [
        _plural(value, unit)
        for value, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second"))
        if value
    ]
    return _join(parts) or "0 seconds"


def duration_adjective(seconds: float) -> str:
    """ "10 minute" or "1 hour 30 minute", as in "your 10 minute timer"."""
    hours, minutes, secs = _split(seconds)
    parts = [
        f"{value} {unit}"
        for value, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second"))
        if value
    ]
    return " ".join(parts) or "0 second"


def format_remaining(seconds: float) -> str:
    """Time left, rounded the way a person would say it."""
    seconds = max(0, seconds)
    if seconds >= 600:
        return format_duration(round(seconds / 60) * 60)
    if seconds >= 60:
        return format_duration(round(seconds / 5) * 5)
    return format_duration(max(1, round(seconds)))


def format_clock(moment: datetime) -> str:
    """ "5:30 PM", "5 PM", "noon" or "midnight"."""
    if moment.minute == 0 and moment.hour == 12:
        return "noon"
    if moment.minute == 0 and moment.hour == 0:
        return "midnight"
    hour = moment.hour % 12 or 12
    meridiem = "AM" if moment.hour < 12 else "PM"
    if moment.minute == 0:
        return f"{hour} {meridiem}"
    return f"{hour}:{moment.minute:02d} {meridiem}"


def describe_when(moment: datetime, now: datetime) -> str:
    """ "at 5 PM", "tomorrow at 9 AM", "on Friday at noon", "on March 3 at 8 AM"."""
    clock = format_clock(moment)
    days = (moment.date() - now.date()).days
    if days == 0:
        return f"at {clock}"
    if days == 1:
        return f"tomorrow at {clock}"
    if 1 < days < 7:
        return f"on {moment.strftime('%A')} at {clock}"
    return f"on {moment.strftime('%B')} {moment.day} at {clock}"


PRONOUNS = {
    "i am": "you are",
    "i'm": "you're",
    "i've": "you've",
    "i'll": "you'll",
    "i'd": "you'd",
    "i was": "you were",
    "i": "you",
    "me": "you",
    "my": "your",
    "mine": "yours",
    "myself": "yourself",
}
PRONOUN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, PRONOUNS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def second_person(text: str) -> str:
    """ "call my mom" -> "call your mom", so a reminder can be read back."""
    return PRONOUN.sub(lambda found: PRONOUNS[found.group().lower()], text)
