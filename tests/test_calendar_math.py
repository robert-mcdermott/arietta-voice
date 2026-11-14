from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from arietta_voice.calendar_math import _easter, answer, parse_date
from arietta_voice.config import default_config
from arietta_voice.tools import ToolRegistry
from arietta_voice.tools.builtin.date_math import DateMathTool
from arietta_voice.tools.registry import run_tool

TODAY = date(2026, 11, 14)  # A Saturday.


@pytest.mark.parametrize(
    "text,expected",
    [
        ("How many days until Christmas?", "Christmas is in 41 days."),
        ("how many weeks until christmas", "Christmas is in 5 weeks and 6 days."),
        ("how many sleeps until christmas", "Christmas is in 41 sleeps."),
        ("how long until thanksgiving", "Thanksgiving is in 12 days."),
        ("how many days until new year's", "New Year's Day is in 48 days."),
        ("days until halloween", "Halloween is in 351 days."),
        ("how many days until easter", "Easter is in 134 days."),
        ("how many days until 12/25", "December 25 is in 41 days."),
        ("how many days until december 25th", "December 25 is in 41 days."),
        ("how many days until the 1st of december", "December 1 is in 17 days."),
        ("how many days until friday", "Friday is in 6 days."),
        ("how many days until tomorrow", "Tomorrow is in 1 day."),
        ("how many days until 2026-11-14", "November 14 is today."),
        ("how many days until 2026-12-25", "December 25 is in 41 days."),
        ("how many days since halloween", "Halloween was 14 days ago."),
        ("how long ago was valentine's day", "Valentine's Day was 273 days ago."),
        ("how many days since january 1st", "January 1 was 317 days ago."),
        ("what day of the week is July 4th?", "July 4, 2027 is a Sunday."),
        ("what day is christmas this year", "Christmas is on a Friday, December 25."),
        ("what day is christmas next year", "Christmas is on a Saturday, December 25, 2027."),
        ("what day of the week was halloween", "Halloween was on a Saturday, October 31."),
        ("what day of the week is the 25th of december 2027", "December 25, 2027 is a Saturday."),
        ("what day of the week is september twenty first", "September 21, 2027 is a Tuesday."),
        ("what day is tomorrow", "Tomorrow is Sunday."),
        ("Is Christmas on a weekend this year?", "No, Christmas is on a Friday."),
        ("is christmas on a weekend next year", "Yes, Christmas is on a Saturday."),
        ("does new year's day fall on a friday", "Yes, New Year's Day is on a Friday."),
        ("was halloween on a weekday", "No, Halloween was on a Saturday."),
        ("what day is next friday", "Friday is November 20."),
        ("what's the date in three weeks", "In three weeks it will be Saturday, December 5."),
        ("what day will it be in 100 days", "In 100 days it will be Monday, February 22, 2027."),
        ("what was the date 10 days ago", "10 days ago it was Wednesday, November 4."),
        (
            "how many days between march 1 and june 15",
            "There are 106 days between March 1, 2027 and June 15, 2027.",
        ),
        ("how many days from today until easter", "There are 134 days between today and Easter."),
        (
            "how many weeks between today and new year's eve",
            "There are 6 weeks and 5 days between today and New Year's Eve.",
        ),
    ],
)
def test_date_questions(text, expected):
    assert answer(text, TODAY) == expected


@pytest.mark.parametrize(
    "text",
    [
        "how many days until february 30",
        "how many days until my birthday",
        "what day is it",
        "what's the date",
        "what's the date in 3 hours",
        "how many timers do I have",
        "what time is it",
    ],
)
def test_other_questions_are_left_alone(text):
    assert answer(text, TODAY) is None


def test_floating_holidays():
    assert _easter(2026) == date(2026, 4, 5)
    assert _easter(2027) == date(2027, 3, 28)
    assert parse_date("thanksgiving", TODAY).day == date(2026, 11, 26)
    assert parse_date("mother's day", TODAY).day == date(2027, 5, 9)
    assert parse_date("labor day 2027", TODAY).day == date(2027, 9, 6)
    assert parse_date("memorial day 2027", TODAY).day == date(2027, 5, 31)
    assert parse_date("christmas", TODAY, past=True).day == date(2025, 12, 25)


@pytest.fixture
def registry(tmp_path):
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    registry = ToolRegistry(config)
    registry.tools["date_math"].clock = lambda: TODAY
    return registry


def test_date_questions_are_answered_instantly(registry):
    hit = registry.fast_path("How many days until Christmas?")
    assert (hit.name, hit.result.speak) == ("date_math", "Christmas is in 41 days.")
    for text in ("What time is it?", "what's 2 plus 2", "remind me in 3 weeks to call Sam"):
        assert registry.fast_path(text).name != "date_math"


def test_the_model_can_ask_in_its_own_words():
    tool = DateMathTool(clock=lambda: TODAY)
    assert run_tool(tool, {"question": "days until 2026-12-25"}).speak == (
        "December 25 is in 41 days."
    )
    assert run_tool(tool, {"query": "what day of the week is christmas"}).speak == (
        "Christmas is on a Friday, December 25."
    )
    assert run_tool(tool, {"question": "something odd"}).speak == (
        "Sorry, I couldn't work out that date."
    )
    assert tool.speaks_for_itself


def test_long_relative_reminders_land_at_a_waking_hour(tmp_path):
    from arietta_voice.timers import TimerStore
    from arietta_voice.tools.builtin.set_reminder import SetReminderTool

    now = datetime(2026, 11, 14, 2, 41, tzinfo=ZoneInfo("America/Los_Angeles"))
    store = TimerStore(tmp_path / "timers.json", clock=lambda: now.timestamp())
    tool = SetReminderTool(store=store, clock=lambda: now)
    result = run_tool(tool, {"text": "to renew my passport", "when": "in 3 weeks"})
    assert result.speak == "Okay, I'll remind you to renew your passport on December 5 at 9 AM."
    soon = run_tool(tool, {"text": "to stretch", "when": "in 2 hours"})
    assert soon.speak == "Okay, I'll remind you to stretch in 2 hours."


def test_timer_limits_are_apologised_for_in_lower_case(tmp_path):
    from arietta_voice.timers import TimerStore
    from arietta_voice.tools.builtin.set_timer import SetTimerTool

    tool = SetTimerTool(store=TimerStore(tmp_path / "timers.json"))
    assert run_tool(tool, {"duration": "2 days"}).speak == (
        "Sorry, timers can run from 1 second to 24 hours; for longer, ask for a reminder."
    )


@pytest.mark.parametrize(
    "text", ["what's the date in 3000000 days", "how many days until december 25 99999"]
)
def test_dates_past_the_calendar_are_not_answered(text):
    assert answer(text, TODAY) is None


def test_how_long_until_a_date_is_a_date_question(registry):
    hit = registry.fast_path("How long until Christmas?")
    assert (hit.name, hit.result.speak) == ("date_math", "Christmas is in 41 days.")
    assert registry.fast_path("how long until my pasta timer goes off").name == "check_timers"
