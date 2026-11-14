import tomllib

import pytest

from arietta_voice.config import default_config, load_config
from arietta_voice.private_config import local_config_path
from arietta_voice.web import settings


@pytest.fixture
def setup(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    base = config_dir / "arietta_voice.toml"
    base.write_text('[assistant]\nname = "Bridget"\n', encoding="utf-8")
    return load_config(base), base


def field(described, section, key):
    return next(
        item
        for group in described["groups"]
        for item in group["fields"]
        if item["section"] == section and item["key"] == key
    )


def test_settings_are_described_with_values_and_origin(setup):
    config, base = setup
    described = settings.describe(config, base)
    assert field(described, "assistant", "name")["value"] == "Bridget"
    assert field(described, "wake_word", "phrases")["value"] == list(config.wake_word.phrases)
    assert field(described, "weather", "units")["choices"] == ["imperial", "metric"]
    assert field(described, "timers", "repeat_s")["private"] is False
    assert described["private_file"].endswith("arietta_voice.local.toml")


def test_changes_are_validated_saved_privately_and_applied(setup):
    config, base = setup
    changed = settings.apply(
        config,
        base,
        {
            "timers": {"repeat_s": 45, "alert_tone": False},
            "weather": {"location": "  Portland,   Maine ", "units": "metric"},
            "wake_word": {"phrases": "bridget, hey bridget"},
            "assistant": {"name": "Bridget"},  # Unchanged, so not written.
        },
    )
    assert changed == [
        "timers.repeat_s",
        "timers.alert_tone",
        "weather.location",
        "weather.units",
        "wake_word.phrases",
    ]
    private = tomllib.loads(local_config_path(base).read_text(encoding="utf-8"))
    assert private["timers"] == {"repeat_s": 45.0, "alert_tone": False}
    assert private["weather"] == {"location": "Portland, Maine", "units": "metric"}
    assert private["wake_word"]["phrases"] == ["bridget", "hey bridget"]
    assert "assistant" not in private
    assert config.timers.repeat_s == 45 and config.wake_word.phrases == ("bridget", "hey bridget")
    reloaded = load_config(base)
    assert reloaded.weather.location == "Portland, Maine"
    assert settings.describe(reloaded, base)["groups"][3]["fields"][0]["private"] is True


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"timers": {"repeat_s": 2}}, "at least 5"),
        ({"timers": {"max_repeats": 1.5}}, "whole number"),
        ({"timers": {"alert_tone": "yes"}}, "on or off"),
        ({"models": {"temperature": "hot"}}, "must be a number"),
        ({"weather": {"units": "kelvin"}}, "one of imperial, metric"),
        ({"assistant": {"name": "x" * 41}}, "at most 40"),
        ({"assistant": {"name": "   "}}, "must not be blank"),
        ({"wake_word": {"phrases": []}}, "must not be empty"),
        ({"models": {"chat_model": "other"}}, "cannot be changed here"),
        ({"admin": {"allow_source_editing": True}}, "cannot be changed here"),
        ({"timers": "fast"}, "must be an object"),
    ],
)
def test_bad_changes_are_refused_and_nothing_is_written(setup, changes, message):
    config, base = setup
    with pytest.raises(ValueError, match=message):
        settings.apply(config, base, changes)
    assert not local_config_path(base).exists()
    assert config.timers.repeat_s == default_config(base.parent.parent).timers.repeat_s
