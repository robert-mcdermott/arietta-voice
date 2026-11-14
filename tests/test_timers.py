import json
import os
import stat
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from arietta_voice.config import default_config
from arietta_voice.timers import (
    DONE,
    MAX_ACTIVE,
    PENDING,
    RINGING,
    AlertScheduler,
    TimerLimitError,
    TimerStore,
    announcement,
    is_stop_command,
    reminder_parts,
    snooze_request,
    timer_name,
)
from arietta_voice.tools import ToolRegistry
from arietta_voice.tools.builtin.cancel_timer import CancelTimerTool
from arietta_voice.tools.builtin.check_timers import CheckTimersTool
from arietta_voice.tools.builtin.set_reminder import SetReminderTool
from arietta_voice.tools.builtin.set_timer import SetTimerTool

ZONE = ZoneInfo("America/Los_Angeles")
START = datetime(2026, 11, 14, 10, 0, tzinfo=ZONE).timestamp()


class Clock:
    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture(autouse=True)
def pacific_time(monkeypatch):
    """Spoken times use the machine's time zone; pin it to match ZONE."""
    import time

    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return TimerStore(tmp_path / "state" / "timers.json", clock=clock)


# Store -------------------------------------------------------------------


def test_alerts_are_saved_privately_and_survive_a_new_store(store, clock):
    alert = store.add_timer(600, "  pasta  ")
    assert alert.due_at == START + 600 and alert.label == "pasta"
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    again = TimerStore(store.path, clock=clock)
    assert [a.id for a in again.active()] == [alert.id]
    assert not list(store.path.parent.glob(".*.tmp"))


@pytest.mark.parametrize("seconds", [0, 0.5, 24 * 3600 + 1])
def test_timer_lengths_are_bounded(store, seconds):
    with pytest.raises(TimerLimitError):
        store.add_timer(seconds)


def test_reminders_must_be_in_the_future_and_say_something(store):
    future = datetime.fromtimestamp(START + 60, ZONE)
    with pytest.raises(TimerLimitError):
        store.add_reminder("", future)
    with pytest.raises(TimerLimitError):
        store.add_reminder("call", datetime.fromtimestamp(START - 60, ZONE))
    assert store.add_reminder("call", future, "to").form == "to"


def test_active_alerts_are_capped(store):
    for _ in range(MAX_ACTIVE):
        store.add_timer(60)
    with pytest.raises(TimerLimitError):
        store.add_timer(60)


def test_cancel_finishes_only_active_alerts(store):
    alert = store.add_timer(60)
    assert [a.id for a in store.cancel([alert.id])] == [alert.id]
    assert store.cancel([alert.id]) == []
    assert store.active() == []
    assert store.finished()[0].outcome == "cancelled"


def test_finished_alerts_are_kept_for_a_day(store, clock):
    alert = store.add_timer(60)
    store.cancel([alert.id])
    clock.now += 25 * 3600
    store.add_timer(60)
    assert store.finished() == []


def test_malformed_entries_are_dropped_and_unreadable_files_set_aside(store, tmp_path):
    store.path.parent.mkdir(parents=True)
    good = {"id": "abcd1234", "kind": "timer", "due_at": START, "created_at": START}
    bad = [
        {**good, "kind": "stopwatch"},
        {**good, "due_at": "soon"},
        {**good, "rings": True},
        {**good, "unknown": 1, "status": "lost"},
        "not an object",
    ]
    store.path.write_text(json.dumps({"version": 1, "alerts": [good, *bad]}))
    assert [a.id for a in store.load()] == ["abcd1234"]

    store.path.write_text("{ not json")
    assert store.load() == []
    store.add_timer(60)
    assert len(store.active()) == 1
    assert list(store.path.parent.glob("timers.json.unreadable-*"))


# Scheduler -----------------------------------------------------------------


def scheduler(store, clock, **kwargs):
    return AlertScheduler(store, clock=clock, **kwargs)


def test_due_alerts_ring_then_repeat_until_unanswered(store, clock):
    alerts = scheduler(store, clock, repeat_s=30, max_repeats=2)
    timer = store.add_timer(60, "eggs")
    assert alerts.due() == []
    assert alerts.next_due_in() == 60
    rings = 0
    clock.now += 60
    for _ in range(3):
        due = alerts.due()
        assert [a.id for a in due] == [timer.id]
        alerts.rang(due)
        rings += 1
        assert alerts.due() == []  # Quiet until the repeat interval passes.
        clock.now += 30
    assert alerts.due() == []
    assert store.finished()[0].outcome == "unanswered"
    assert rings == 3


def test_dismiss_and_snooze_change_only_what_is_ringing(store, clock):
    alerts = scheduler(store, clock)
    ringing = store.add_timer(10)
    waiting = store.add_timer(600)
    clock.now += 10
    alerts.rang(alerts.due())
    assert [a.id for a in alerts.ringing()] == [ringing.id]
    snoozed = alerts.snooze(300)
    assert [a.id for a in snoozed] == [ringing.id]
    assert alerts.ringing() == []
    clock.now += 300
    assert [a.id for a in alerts.due()] == [ringing.id]
    alerts.rang(alerts.due())
    assert [a.id for a in alerts.dismiss()] == [ringing.id]
    statuses = {a.id: a.status for a in store.load()}
    assert statuses == {ringing.id: DONE, waiting.id: PENDING}


def test_alerts_set_by_another_process_are_noticed(store, clock):
    alerts = scheduler(store, clock)
    assert alerts.due() == []
    TimerStore(store.path, clock=clock).add_timer(5)
    clock.now += 5
    assert len(alerts.due()) == 1


def test_an_alert_that_cannot_be_recorded_is_said_once(store, clock, monkeypatch):
    alerts = scheduler(store, clock, repeat_s=30)
    store.add_timer(5)
    clock.now += 5

    def fail(change):
        raise OSError("read-only")

    monkeypatch.setattr(store, "mutate", fail)
    alerts.rang(alerts.due())
    assert alerts.due() == []
    # Its ring was never saved, so it would look due forever; it is not repeated.
    clock.now += 3600
    assert alerts.due() == []


def test_entries_with_unsayable_times_are_dropped(store):
    store.path.parent.mkdir(parents=True)
    entry = {"id": "abcd1234", "kind": "timer", "created_at": START}
    store.path.write_text(
        json.dumps(
            {
                "version": 1,
                "alerts": [
                    {**entry, "due_at": 1e300},
                    {**entry, "due_at": float("nan")},
                    {**entry, "due_at": START},
                ],
            }
        )
    )
    assert [a.due_at for a in store.load()] == [START]


# What is said ---------------------------------------------------------------


def test_announcements(store, clock):
    pasta = store.add_timer(600, "pasta")
    plain = store.add_timer(90)
    oven = store.add_timer(60, "to check the oven")
    assert timer_name(pasta) == "pasta timer"
    assert timer_name(plain) == "1 minute 30 second timer"
    assert timer_name(oven) == "timer to check the oven"
    clock.now += 600
    assert announcement([pasta], clock.now) == "Your pasta timer is done."
    late = clock.now + 630
    assert announcement([plain], late) == ("Your 1 minute 30 second timer finished 19 minutes ago.")
    due = datetime.fromtimestamp(clock.now + 60, ZONE)
    call = store.add_reminder("call your mom", due, "to")
    bins = store.add_reminder("the bins go out", due, "that")
    dentist = store.add_reminder("the dentist", due, "about")
    assert announcement([call, bins], clock.now + 60) == (
        "It's time to call your mom. Just a reminder that the bins go out."
    )
    assert announcement([dentist], clock.now + 60) == "Here's your reminder about the dentist."
    assert announcement([call], clock.now + 3600).startswith("This reminder was due at ")


@pytest.mark.parametrize(
    "text",
    [
        "Stop.",
        "stop it",
        "Okay, thanks!",
        "OK",
        "thank you",
        "got it",
        "Stop the timer.",
        "turn off the alarm",
        "dismiss",
        "okay stop the pasta timer please",
        "I'm up",
    ],
)
def test_stop_commands(text):
    assert is_stop_command(text)


@pytest.mark.parametrize("text", ["stop the music", "what time is it", "yes", "okay set a timer"])
def test_other_requests_are_not_stop_commands(text):
    assert not is_stop_command(text)


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("snooze", 0.0),
        ("Snooze it.", 0.0),
        ("snooze for ten minutes", 600),
        ("remind me again in 15 minutes", 900),
        ("give me five more minutes", 300),
        ("five more minutes", 300),
        ("okay, snooze for an hour", 3600),
    ],
)
def test_snooze_requests(text, seconds):
    assert snooze_request(text) == seconds


@pytest.mark.parametrize("text", ["stop", "snooze for a while", "more coffee"])
def test_other_requests_are_not_snoozes(text):
    assert snooze_request(text) is None


@pytest.mark.parametrize(
    "text,parts",
    [
        ("to call my mom", ("to", "call your mom")),
        ("that I have a meeting", ("that", "you have a meeting")),
        ("of the dentist", ("about", "the dentist")),
        ("call mom", ("to", "call mom")),
        ("the dentist", ("about", "the dentist")),
        ("", ("", "")),
    ],
)
def test_reminder_parts(text, parts):
    assert reminder_parts(text) == parts


# Tools ----------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    return ToolRegistry(config)


def pin_clock(registry, now):
    """Fix "now" for the tools and for the store's own check that a time is ahead."""
    for name in ("set_reminder", "set_alarm"):
        registry.tools[name].clock = lambda: now
        registry.tools[name].store.clock = now.timestamp


def answer(registry, text):
    hit = registry.fast_path(text)
    return None if hit is None else (hit.name, hit.result.speak)


@pytest.mark.parametrize(
    "text,speak",
    [
        ("Set a 10-minute pasta timer.", "Okay, a 10 minute pasta timer, starting now."),
        ("Set a timer for ten minutes.", "Okay, 10 minutes, starting now."),
        ("timer for 5 minutes called eggs", "Okay, a 5 minute eggs timer, starting now."),
        ("Can you set an 8 minute egg timer", "Okay, an 8 minute egg timer, starting now."),
        (
            "Set a timer for 90 seconds to check the oven",
            "Okay, a 1 minute 30 second timer to check the oven, starting now.",
        ),
        ("Set a timer.", "How long should the timer be?"),
        # Speech recognition heard "pasta" as "pause to".
        ("Set a 10-minute pause to timer.", "Okay, a 10 minute pause to timer, starting now."),
    ],
)
def test_timers_are_set_instantly(registry, text, speak):
    assert answer(registry, text) == ("set_timer", speak)


def test_reminders_are_set_instantly(registry, monkeypatch):
    now = datetime(2026, 11, 14, 10, 0, tzinfo=ZONE)
    pin_clock(registry, now)
    cases = {
        "Remind me to call my mom at 5:30 p.m.": "Okay, I'll remind you to call your mom at 5:30 PM.",
        "Remind me in 20 minutes to take the bread out.": (
            "Okay, I'll remind you to take the bread out in 20 minutes."
        ),
        "remind me to put the kids in bed at 8": (
            "Okay, I'll remind you to put the kids in bed at 8 PM."
        ),
        "Remind me tomorrow morning that the bins go out": (
            "Okay, I'll remind you that the bins go out tomorrow at 9 AM."
        ),
        "Could you remind me about the dentist on Friday at 3?": (
            "Okay, I'll remind you about the dentist on Friday at 3 PM."
        ),
        "remind me to stretch": "When should I remind you?",
        "remind me at noon": "What should I remind you about?",
    }
    for text, speak in cases.items():
        assert answer(registry, text) == ("set_reminder", speak), text


def test_checking_and_cancelling(registry):
    assert answer(registry, "How much time is left?") == (
        "check_timers",
        "You don't have any timers running.",
    )
    registry.fast_path("set a 10 minute pasta timer")
    registry.fast_path("set a 5 minute egg timer")
    assert answer(registry, "How long is left on the egg timer?") == (
        "check_timers",
        "Your egg timer has 5 minutes left.",
    )
    name, speak = answer(registry, "What timers do I have?")
    assert speak == (
        "You have 2 timers: your egg timer, with 5 minutes left, "
        "and your pasta timer, with 10 minutes left."
    )
    assert answer(registry, "Cancel the timer") == (
        "cancel_timer",
        "Which one: your egg timer or your pasta timer?",
    )
    assert answer(registry, "Cancel the 10 minute timer.") == (
        "cancel_timer",
        "Okay, I've cancelled your pasta timer.",
    )
    assert answer(registry, "cancel the timer") == (
        "cancel_timer",
        "Okay, I've cancelled your egg timer.",
    )
    assert answer(registry, "cancel all my reminders") == (
        "cancel_timer",
        "You don't have any reminders.",
    )


@pytest.mark.parametrize(
    "text",
    ["What time is it?", "what's the weather", "how long is the drive to Portland", "stop"],
)
def test_unrelated_requests_are_left_alone(registry, text):
    hit = registry.fast_path(text)
    assert hit is None or hit.name not in {
        "set_timer",
        "set_reminder",
        "check_timers",
        "cancel_timer",
    }


def test_the_model_can_call_each_tool_with_arguments(store, clock):
    now = datetime.fromtimestamp(clock.now, ZONE)
    set_timer = SetTimerTool(store=store)
    set_reminder = SetReminderTool(store=store, clock=lambda: now)
    check = CheckTimersTool(store=store, clock=clock)
    cancel = CancelTimerTool(store=store)

    result = set_timer.run({"duration": "PT15M", "label": "the tea"})
    assert result.content == "Started a timer for 15 minutes called tea. It ends at 10:15 AM."
    result = set_reminder.run({"text": "call Sam", "when": "2026-11-14T18:00:00"})
    assert result.speak == "Okay, I'll remind you to call Sam at 6 PM."
    assert set_reminder.run({"text": "x", "when": "whenever"}).speak == "When should I remind you?"
    assert set_timer.run({"duration": "a while"}).speak == "How long should the timer be?"

    listed = check.run({"kind": "any"})
    assert listed.speak == (
        "You have 1 timer and 1 reminder: your tea timer, with 15 minutes left, "
        "and a reminder to call Sam at 6 PM."
    )
    assert len(listed.data["alerts"]) == 2
    assert cancel.run({"which": "sam", "kind": "reminder"}).speak == (
        "Okay, I've cancelled your reminder to call Sam."
    )
    assert cancel.run({"which": "coffee"}).speak == "I couldn't find one for coffee."
    assert cancel.run({"which": "all"}).speak == "Okay, I've cancelled your tea timer."


def test_cancel_prefers_what_is_ringing(store, clock):
    store.add_timer(600, "pasta")
    ringing = store.add_timer(5, "eggs")
    clock.now += 5
    alerts = AlertScheduler(store, clock=clock)
    alerts.rang(alerts.due())
    result = CancelTimerTool(store=store).run({"kind": "timer"})
    assert result.speak == "Okay, I've cancelled your eggs timer."
    assert result.data["cancelled"] == [ringing.id]
    assert [a.status for a in store.active()] == [PENDING]
    assert RINGING not in {a.status for a in store.load()}


def test_a_spoken_timer_request_never_reaches_the_model(tmp_path):
    from unittest.mock import Mock

    from arietta_voice.conversation import ConversationService

    config = default_config(tmp_path)
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "kitchen.md").write_text("# Kitchen\nThe timer is on the oven.\n")
    generate = Mock(side_effect=AssertionError("the model was asked"))
    service = ConversationService(config, generate=generate)
    result = service.chat(message="Set a timer for ten minutes")
    assert result.response == "Okay, 10 minutes, starting now."
    assert result.used_tool == "set_timer"
    assert len(TimerStore(config.timers.path).active()) == 1


# Alarms and other ways of asking ----------------------------------------------


@pytest.fixture
def afternoon_registry(registry):
    pin_clock(registry, datetime(2026, 11, 14, 14, 0, tzinfo=ZONE))
    return registry


@pytest.mark.parametrize(
    "text,speak",
    [
        # Without am or pm, waking up means the morning.
        ("Wake me up at 7.", "Okay, I'll wake you tomorrow at 7 AM."),
        ("wake me up at 7 pm", "Okay, I'll wake you at 7 PM."),
        ("get me up at half past six", "Okay, I'll wake you tomorrow at 6:30 AM."),
        ("Set an alarm for 6:30 am", "Okay, your alarm is set tomorrow at 6:30 AM."),
        ("set a gym alarm for 5:45 tomorrow", "Okay, your alarm is set tomorrow at 5:45 AM."),
        ("Set an alarm for 9", "Okay, your alarm is set at 9 PM."),
        ("wake me in 20 minutes", "Okay, I'll wake you at 2:20 PM."),
    ],
)
def test_alarms_are_set_instantly(afternoon_registry, text, speak):
    assert answer(afternoon_registry, text) == ("set_alarm", speak)


@pytest.mark.parametrize(
    "text,speak",
    [
        (
            "Ping me in a quarter of an hour about the laundry.",
            "Okay, I'll remind you about the laundry in 15 minutes.",
        ),
        (
            "Can you nudge me at 6 to start dinner?",
            "Okay, I'll remind you to start dinner at 6 PM.",
        ),
        (
            "Let me know in 10 minutes when to flip the steak.",
            "Okay, I'll remind you to flip the steak in 10 minutes.",
        ),
        (
            "Please set a reminder for my dentist appointment on Monday at 2pm",
            "Okay, I'll remind you about your dentist appointment on Monday at 2 PM.",
        ),
        ("set a reminder for 5 pm to call mom", "Okay, I'll remind you to call mom at 5 PM."),
        (
            "Don't let me forget to take my pills at nine tonight.",
            "Okay, I'll remind you to take your pills at 9 PM.",
        ),
    ],
)
def test_more_ways_of_asking_for_a_reminder(afternoon_registry, text, speak):
    assert answer(afternoon_registry, text) == ("set_reminder", speak)


@pytest.mark.parametrize(
    "text",
    [
        # The model may know how long an egg takes; "how long?" would be unhelpful.
        "Set a timer for as long as it takes to soft boil an egg.",
        # "Let me know" without a time is not a reminder.
        "let me know when the oven is hot",
        "don't let me forget my keys",
        # Only a timer that exists makes "is my ... done?" a timer question.
        "is my code done compiling",
    ],
)
def test_requests_the_model_should_decide(afternoon_registry, text):
    hit = afternoon_registry.fast_path(text)
    assert hit is None or hit.name not in {"set_timer", "set_reminder", "check_timers"}


def test_checking_and_cancelling_alarms(afternoon_registry):
    registry = afternoon_registry
    registry.fast_path("set a 5 minute pasta timer")
    assert answer(registry, "Is my pasta nearly done?") == (
        "check_timers",
        "Your pasta timer has 5 minutes left.",
    )
    registry.fast_path("wake me up at 7")
    registry.fast_path("set a gym alarm for 5:45 tomorrow")
    name, speak = answer(registry, "When is my alarm?")
    assert name == "check_timers" and speak.startswith("You have 2 alarms: ")
    assert answer(registry, "cancel the gym alarm")[1].startswith(
        "Okay, I've cancelled your gym alarm tomorrow at 5:45"
    )
    listed = CheckTimersTool(store=registry.tools["check_timers"].store).run({})
    assert listed.speak.startswith("You have 1 timer and 1 alarm: ")


def test_alarm_announcements(store, clock):
    due = datetime.fromtimestamp(START + 60, ZONE)
    wake = store.add_alarm(due, "time to wake up")
    gym = store.add_alarm(due, "gym")
    plain = store.add_alarm(due)
    clock.now += 60
    assert announcement([wake], clock.now) == "It's 10:01 AM, time to wake up."
    assert announcement([gym], clock.now) == "It's 10:01 AM. This is your gym alarm."
    assert announcement([plain], clock.now) == "It's 10:01 AM. This is your alarm."
    assert announcement([plain], clock.now + 600) == "Your 10:01 AM alarm went off 10 minutes ago."
    with pytest.raises(TimerLimitError):
        store.add_alarm(datetime.fromtimestamp(START + 8 * 86400, ZONE))


def test_a_model_s_near_miss_argument_names_are_accepted(store, clock):
    from arietta_voice.tools.registry import run_tool

    now = datetime.fromtimestamp(clock.now, ZONE)
    reminder = run_tool(
        SetReminderTool(store=store, clock=lambda: now),
        {"note": "call the vet", "time": "9 am tomorrow"},
    )
    assert reminder.speak == "Okay, I'll remind you to call the vet tomorrow at 9 AM."
    assert run_tool(SetTimerTool(store=store), {"minutes": 6}).speak == (
        "Okay, 6 minutes, starting now."
    )


def test_strict_dismissals_and_named_alerts():
    from arietta_voice.timers import stop_label

    assert is_stop_command("stop", strict=True)
    assert is_stop_command("okay, turn it off", strict=True)
    assert not is_stop_command("okay", strict=True)
    assert not is_stop_command("thanks", strict=True)
    assert is_stop_command("thanks")
    assert stop_label("cancel my dentist reminder") == "dentist"
    assert stop_label("stop the timer") is None
    assert stop_label("what time is it") is None
