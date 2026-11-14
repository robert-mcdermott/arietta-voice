"""Questions about dates: days until, days since, days between, what day of the week.

"How many days until Christmas?", "What day of the week is July 4th?", "What's
the date in three weeks?", "How many days between March 1 and June 15?". Dates
can be holidays, month and day (in digits or words, with an optional year),
"12/25", "today", "tomorrow" or a weekday. Without a year, a date means its
next occurrence (or its last one, for "since" and "ago").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from arietta_voice.timephrases import ONES, TENS, WEEKDAYS, normalize, parse_duration

MONTHS = {
    name: number
    for number, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth (1-based, or -1 for last) given weekday of a month."""
    if n > 0:
        first = date(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    following = date(year + month // 12, month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return date(year, month, day)


HOLIDAYS = {
    "christmas eve": ("Christmas Eve", lambda y: date(y, 12, 24)),
    "christmas day": ("Christmas", lambda y: date(y, 12, 25)),
    "christmas": ("Christmas", lambda y: date(y, 12, 25)),
    "boxing day": ("Boxing Day", lambda y: date(y, 12, 26)),
    "new year's eve": ("New Year's Eve", lambda y: date(y, 12, 31)),
    "new years eve": ("New Year's Eve", lambda y: date(y, 12, 31)),
    "new year's day": ("New Year's Day", lambda y: date(y, 1, 1)),
    "new years day": ("New Year's Day", lambda y: date(y, 1, 1)),
    "new year's": ("New Year's Day", lambda y: date(y, 1, 1)),
    "new years": ("New Year's Day", lambda y: date(y, 1, 1)),
    "new year": ("New Year's Day", lambda y: date(y, 1, 1)),
    "halloween": ("Halloween", lambda y: date(y, 10, 31)),
    "valentine's day": ("Valentine's Day", lambda y: date(y, 2, 14)),
    "valentines day": ("Valentine's Day", lambda y: date(y, 2, 14)),
    "valentine's": ("Valentine's Day", lambda y: date(y, 2, 14)),
    "st patrick's day": ("St. Patrick's Day", lambda y: date(y, 3, 17)),
    "saint patrick's day": ("St. Patrick's Day", lambda y: date(y, 3, 17)),
    "st patricks day": ("St. Patrick's Day", lambda y: date(y, 3, 17)),
    "independence day": ("Independence Day", lambda y: date(y, 7, 4)),
    "the fourth of july": ("the Fourth of July", lambda y: date(y, 7, 4)),
    "fourth of july": ("the Fourth of July", lambda y: date(y, 7, 4)),
    "easter sunday": ("Easter", _easter),
    "easter": ("Easter", _easter),
    "thanksgiving": ("Thanksgiving", lambda y: _nth_weekday(y, 11, 3, 4)),
    "mother's day": ("Mother's Day", lambda y: _nth_weekday(y, 5, 6, 2)),
    "mothers day": ("Mother's Day", lambda y: _nth_weekday(y, 5, 6, 2)),
    "father's day": ("Father's Day", lambda y: _nth_weekday(y, 6, 6, 3)),
    "fathers day": ("Father's Day", lambda y: _nth_weekday(y, 6, 6, 3)),
    "labor day": ("Labor Day", lambda y: _nth_weekday(y, 9, 0, 1)),
    "labour day": ("Labor Day", lambda y: _nth_weekday(y, 9, 0, 1)),
    "memorial day": ("Memorial Day", lambda y: _nth_weekday(y, 5, 0, -1)),
}
HOLIDAY_PATTERN = "|".join(re.escape(name) for name in sorted(HOLIDAYS, key=len, reverse=True))


@dataclass(slots=True, frozen=True)
class NamedDate:
    day: date
    name: str | None  # "Christmas", or None for a plain date
    explicit_year: bool


def _day_number(text: str) -> int | None:
    """ "25", "25th", "twenty fifth", "first"."""
    text = text.strip()
    found = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", text)
    if found:
        return int(found.group(1))
    words = text.split()
    if len(words) == 1 and words[0] in ORDINAL_WORDS:
        return ORDINAL_WORDS[words[0]]
    if len(words) == 2 and words[0] in TENS and words[1] in ORDINAL_WORDS:
        return TENS[words[0]] + ORDINAL_WORDS[words[1]]
    if len(words) == 1 and words[0] in ONES:
        return ONES[words[0]]
    if len(words) == 2 and words[0] in TENS and words[1] in ONES:
        return TENS[words[0]] + ONES[words[1]]
    if len(words) == 1 and words[0] in TENS:
        return TENS[words[0]]
    return None


DAY = r"(?:\d{1,2}(?:st|nd|rd|th)?|(?:twenty |thirty )?[a-z]+)"
MONTH = r"(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
YEAR = r"(?:,?\s*(?P<year>\d{4}))?"


def _resolve(build, year: int | None, today: date, *, past: bool) -> date | None:
    """A date with this year if given, else the next (or last) occurrence."""
    try:
        if year is not None:
            return build(year)
        candidate = build(today.year)
        if past and candidate > today:
            return build(today.year - 1)
        if not past and candidate < today:
            return build(today.year + 1)
        return candidate
    except ValueError:  # February 30th, and the like
        return None


MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]  # fmt: skip


def _iso_as_words(text: str) -> str:
    """ "2026-12-25", as models write dates, becomes "december 25 2026"."""

    def words(found):
        month = int(found.group(2))
        if not 1 <= month <= 12:
            return found.group(0)
        return f"{MONTH_NAMES[month - 1]} {int(found.group(3))} {found.group(1)}"

    return re.sub(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", words, text)


def parse_date(text: str, today: date, *, past: bool = False) -> NamedDate | None:
    """A date phrase: "Christmas", "July 4th", "the 25th of December", "12/25"."""
    spoken = normalize(_iso_as_words(text)).strip(" ?.!,")
    spoken = re.sub(r"^(?:the date|the day|day)\s+", "", spoken)
    spoken = re.sub(
        r"\s+(?:this year|next year|last year)$", lambda m: f" {m.group(0).strip()}", spoken
    )
    shift = 0
    for suffix, years in ((" next year", 1), (" last year", -1), (" this year", 0)):
        if spoken.endswith(suffix):
            spoken, shift = spoken[: -len(suffix)].strip(), years
            break
    year_override = today.year + shift if shift or " this year" in text.lower() else None
    if spoken in ("today", "now"):
        return NamedDate(today, "today", False)
    if spoken == "tomorrow":
        return NamedDate(today + timedelta(days=1), "tomorrow", False)
    if spoken == "yesterday":
        return NamedDate(today - timedelta(days=1), "yesterday", False)
    found = re.fullmatch(
        rf"(?:on )?(?:(?P<which>next|last|this) )?(?P<weekday>{'|'.join(WEEKDAYS)})", spoken
    )
    if found:
        weekday = WEEKDAYS.index(found.group("weekday"))
        if past or found.group("which") == "last":
            delta = -((today.weekday() - weekday) % 7 or 7)
        else:
            delta = (weekday - today.weekday()) % 7 or 7
        return NamedDate(today + timedelta(days=delta), found.group("weekday").title(), False)
    found = re.fullmatch(rf"(?P<holiday>{HOLIDAY_PATTERN}){YEAR}", spoken)
    if found:
        name, build = HOLIDAYS[found.group("holiday")]
        year = int(found.group("year")) if found.group("year") else year_override
        day = _resolve(build, year, today, past=past)
        return NamedDate(day, name, year is not None) if day else None
    patterns = (
        rf"(?:on )?(?P<month>{MONTH}) (?:the )?(?P<day>{DAY}){YEAR}",
        rf"(?:on )?(?:the )?(?P<day>{DAY}) (?:of )?(?P<month>{MONTH}){YEAR}",
    )
    for pattern in patterns:
        found = re.fullmatch(pattern, spoken)
        if found:
            month = MONTHS[found.group("month").rstrip(".")]
            day_number = _day_number(found.group("day"))
            if day_number is None:
                return None
            year = int(found.group("year")) if found.group("year") else year_override
            day = _resolve(lambda y: date(y, month, day_number), year, today, past=past)
            return NamedDate(day, None, year is not None) if day else None
    found = re.fullmatch(r"(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/(?P<year>\d{2,4}))?", spoken)
    if found:
        month, day_number = int(found.group("month")), int(found.group("day"))
        year = found.group("year")
        year = (int(year) + 2000 if len(year) == 2 else int(year)) if year else year_override
        day = _resolve(lambda y: date(y, month, day_number), year, today, past=past)
        return NamedDate(day, None, year is not None) if day else None
    return None


def say_date(day: date, today: date) -> str:
    """ "Friday, December 25", with the year when it is not this year."""
    text = f"{day.strftime('%A, %B')} {day.day}"
    return text if day.year == today.year else f"{text}, {day.year}"


def say_days(days: int) -> str:
    return "1 day" if days == 1 else f"{days:,} days"


def _name(target: NamedDate, today: date) -> str:
    """ "Christmas", "today", "Friday", or "March 1" (with the year if not this one)."""
    if target.name:
        return target.name
    text = f"{target.day.strftime('%B')} {target.day.day}"
    return text if target.day.year == today.year else f"{text}, {target.day.year}"


LEAD = r"^(?:(?:hey|ok(?:ay)?|so|please|can you tell me|could you tell me|tell me)\s+)*"
UNTIL = re.compile(
    LEAD + r"(?:how many (?P<unit>days|weeks|sleeps) (?:are there |is it |are left |left |remain )?"
    r"(?:until|till|til|before|to)|how long (?:is it |is there )?(?:until|till|til|before|to)|"
    r"days (?:until|till|to)|countdown to) (?P<what>.+)$"
)
SINCE = re.compile(
    LEAD + r"(?:how many (?P<unit>days|weeks) (?:has it been |is it |have passed |since|ago was)?"
    r"\s*(?:since)? (?P<what>.+?)(?: ago)?|how long (?:ago (?:was|is) |since )(?P<what2>.+))$"
)
WEEKDAY = re.compile(
    LEAD + r"what day (?:of the week )?(?P<verb>is|was|will be|does|did|falls on|is it on) "
    r"(?P<what>.+?)"
    r"(?: (?:fall|land|fall on|land on|on))?$"
)
AFTER = re.compile(
    LEAD + r"what(?:'s| is| will| was)? (?:the )?(?:date|day)(?: will it be| be| is it| was it)?"
    r" (?P<what>(?:in .+|.+ from (?:now|today)|.+ ago))$"
)
IS_ON = re.compile(
    LEAD + r"(?P<verb>is|does|will|was|did) (?P<what>.+?) (?:be |fall |land )?on a "
    r"(?P<kind>weekend|weekday|week day|monday|tuesday|wednesday|thursday|friday|saturday|"
    r"sunday)(?P<rest> this year| next year)?$"
)
BETWEEN = re.compile(
    LEAD + r"how many (?P<unit>days|weeks) (?:are there )?(?:between|from) (?P<first>.+?) "
    r"(?:and|to|until|till) (?P<second>.+)$"
)


def answer(text: str, today: date) -> str | None:
    """A spoken answer to a date question, or None if it is not one."""
    spoken = normalize(_iso_as_words(text)).strip(" ?.!")
    found = BETWEEN.match(spoken)
    if found:
        first, second = parse_date(found["first"], today), parse_date(found["second"], today)
        if first is None or second is None:
            return None
        days = abs((second.day - first.day).days)
        return (
            f"There are {_count(days, found['unit'])} between {_name(first, today)} "
            f"and {_name(second, today)}."
        )
    found = IS_ON.match(spoken)
    if found:
        past = found["verb"] in ("was", "did")
        target = parse_date(found["what"] + (found["rest"] or ""), today, past=past)
        if target is None:
            return None
        weekday = target.day.strftime("%A")
        kind = found["kind"].replace("week day", "weekday")
        if kind == "weekend":
            yes = target.day.weekday() >= 5
        elif kind == "weekday":
            yes = target.day.weekday() < 5
        else:
            yes = weekday.lower() == kind
        verb = "was" if target.day < today else "is"
        return f"{'Yes' if yes else 'No'}, {_name(target, today)} {verb} on a {weekday}."
    found = UNTIL.match(spoken)
    if found:
        target = parse_date(found["what"], today)
        if target is None:
            return None
        days = (target.day - today).days
        name = _name(target, today)
        if days == 0:
            return f"{_capital(name)} is today."
        if days < 0:
            return f"{_capital(name)} was {_count(-days, found['unit'])} ago."
        return f"{_capital(name)} is in {_count(days, found['unit'])}."
    found = SINCE.match(spoken)
    if found:
        target = parse_date(found["what"] or found["what2"], today, past=True)
        if target is None:
            return None
        days = (today - target.day).days
        name = _name(target, today)
        if days == 0:
            return f"{_capital(name)} is today."
        if days < 0:
            return f"{_capital(name)} is in {_count(-days, found['unit'])}."
        return f"{_capital(name)} was {_count(days, found['unit'])} ago."
    found = WEEKDAY.match(spoken)
    if found:
        target = parse_date(found["what"], today, past=found["verb"] in ("was", "did"))
        if target is None:
            return None
        weekday = target.day.strftime("%A")
        verb = "was" if target.day < today else "is"
        if target.name in ("today", "tomorrow", "yesterday"):
            return f"{_capital(target.name)} {verb} {weekday}."
        if target.name == weekday:  # "what day is next Friday": give its date
            return f"{weekday} {verb} {target.day.strftime('%B')} {target.day.day}."
        if target.name:
            when = f"{target.day.strftime('%B')} {target.day.day}"
            if target.day.year != today.year:
                when += f", {target.day.year}"
            return f"{target.name} {verb} on a {weekday}, {when}."
        return f"{_name(target, today)} {verb} a {weekday}."
    found = AFTER.match(spoken)
    if found:
        what = found["what"]
        ago = what.endswith(" ago")
        phrase = re.sub(r"^in |\s+(?:from (?:now|today)|ago)$", "", what)
        seconds = parse_duration(phrase)
        if seconds is None or seconds < 86400 or seconds % 86400:
            return None
        days = int(seconds // 86400)
        day = today - timedelta(days=days) if ago else today + timedelta(days=days)
        if ago:
            return f"{phrase.capitalize()} ago it was {say_date(day, today)}."
        return f"In {phrase} it will be {say_date(day, today)}."
    return None


def _count(days: int, unit: str | None) -> str:
    if unit == "weeks":
        weeks, rest = divmod(days, 7)
        if not weeks:
            return say_days(rest)
        text = "1 week" if weeks == 1 else f"{weeks:,} weeks"
        return f"{text} and {say_days(rest)}" if rest else text
    if unit == "sleeps":
        return "1 sleep" if days == 1 else f"{days:,} sleeps"
    return say_days(days)


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]
