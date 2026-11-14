from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from arietta_voice.timephrases import (
    describe_when,
    duration_adjective,
    find_duration,
    format_clock,
    format_duration,
    format_remaining,
    normalize,
    parse_duration,
    parse_when,
    second_person,
)

ZONE = ZoneInfo("America/Los_Angeles")
# A Saturday morning.
NOW = datetime(2026, 11, 14, 10, 0, tzinfo=ZONE)


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("10 minutes", 600),
        ("ten minutes", 600),
        ("10 minute", 600),
        ("forty-five mins", 2700),
        ("twenty five minutes", 1500),
        ("an hour and a half", 5400),
        ("one and a half hours", 5400),
        ("1.5 hours", 5400),
        ("half an hour", 1800),
        ("a quarter of an hour", 900),
        ("a half hour", 1800),
        ("90 seconds", 90),
        ("1 hour and 30 minutes", 5400),
        ("2 hours, 15 minutes", 8100),
        ("a couple of minutes", 120),
        ("a minute", 60),
        ("one hundred twenty seconds", 120),
        ("for 5 minutes", 300),
        ("5 minutes from now", 300),
        ("1h30m", 5400),
        ("90s", 90),
        ("PT10M", 600),
        ("00:10:00", 600),
        ("1:30", 90),
        ("10", 600),
    ],
)
def test_durations_are_read_as_people_say_them(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "soon", "a while", "10 apples", "0 minutes", "minutes"])
def test_anything_else_is_not_a_duration(text):
    assert parse_duration(text) is None


def test_a_duration_is_found_inside_a_sentence_with_its_span():
    text = normalize("Set a 10-minute pasta timer.")
    found = find_duration(text)
    assert found.seconds == 600
    assert text[found.start : found.end] == "10 minute"
    assert find_duration("set a pasta timer") is None


@pytest.mark.parametrize(
    "text,expected,relative",
    [
        ("in 20 minutes", NOW + timedelta(minutes=20), True),
        ("20 minutes from now", NOW + timedelta(minutes=20), True),
        ("in an hour and a half", NOW + timedelta(minutes=90), True),
        ("at 5 pm", NOW.replace(hour=17), False),
        ("at 5:30 p.m.", NOW.replace(hour=17, minute=30), False),
        ("five thirty pm", NOW.replace(hour=17, minute=30), False),
        ("at 5.30 pm", NOW.replace(hour=17, minute=30), False),
        ("at 17:30", NOW.replace(hour=17, minute=30), False),
        ("at half past five", NOW.replace(hour=17, minute=30), False),
        ("quarter to six pm", NOW.replace(hour=17, minute=45), False),
        ("at noon", NOW.replace(hour=12), False),
        # The next time the clock shows nine is this evening.
        ("at 9", NOW.replace(hour=21), False),
        ("at 11", NOW.replace(hour=11), False),
        ("tomorrow", NOW.replace(day=15, hour=9), False),
        ("tomorrow morning", NOW.replace(day=15, hour=9), False),
        ("at 9 tomorrow", NOW.replace(day=15, hour=9), False),
        ("tomorrow at 3", NOW.replace(day=15, hour=15), False),
        ("tonight at 8", NOW.replace(hour=20), False),
        ("this afternoon", NOW.replace(hour=14), False),
        ("on friday at 3pm", NOW.replace(day=20, hour=15), False),
        ("next monday at 10", NOW.replace(day=16, hour=10), False),
        ("the day after tomorrow at 7", NOW.replace(day=16, hour=7), False),
        # Saturday at nine has passed, so it is next week.
        ("saturday", NOW.replace(day=21, hour=9), False),
        ("2026-11-15T09:00:00", NOW.replace(day=15, hour=9), False),
    ],
)
def test_when_phrases(text, expected, relative):
    when = parse_when(text, NOW)
    assert when is not None, text
    assert when.due == expected
    assert when.relative is relative


@pytest.mark.parametrize(
    "text",
    ["", "bed at 8", "today at 9 am", "in a bit", "at banana", "2020-01-01T09:00:00", "on time"],
)
def test_unclear_or_past_times_are_rejected(text):
    assert parse_when(text, NOW) is None


def test_late_evening_bare_hour_means_tomorrow_morning():
    late = NOW.replace(hour=22)
    assert parse_when("at 9", late).due == late.replace(day=15, hour=9)


def test_saying_durations_and_times():
    assert format_duration(5400) == "1 hour and 30 minutes"
    assert format_duration(3725) == "1 hour, 2 minutes and 5 seconds"
    assert format_duration(1) == "1 second"
    assert duration_adjective(600) == "10 minute"
    assert duration_adjective(5400) == "1 hour 30 minute"
    assert format_remaining(452) == "7 minutes and 30 seconds"
    assert format_remaining(3725) == "1 hour and 2 minutes"
    assert format_remaining(0.2) == "1 second"
    assert format_clock(NOW.replace(hour=17, minute=5)) == "5:05 PM"
    assert format_clock(NOW.replace(hour=9)) == "9 AM"
    assert format_clock(NOW.replace(hour=12)) == "noon"
    assert format_clock(NOW.replace(hour=0)) == "midnight"
    assert describe_when(NOW.replace(hour=17), NOW) == "at 5 PM"
    assert describe_when(NOW.replace(day=15, hour=9), NOW) == "tomorrow at 9 AM"
    assert describe_when(NOW.replace(day=19, hour=12), NOW) == "on Thursday at noon"
    assert describe_when(NOW.replace(day=30, hour=8), NOW) == "on November 30 at 8 AM"


def test_reminders_are_read_back_to_the_user():
    assert second_person("call my mom and tell her I'm late") == (
        "call your mom and tell her you're late"
    )
    assert second_person("I am picking up Sam") == "you are picking up Sam"
    assert second_person("email me the summary") == "email you the summary"
