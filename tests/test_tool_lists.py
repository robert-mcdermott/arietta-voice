"""A private tool list keeps its order and choices, and still gets new standard tools."""

from __future__ import annotations

import json

import pytest

from arietta_voice.config import (
    DEFAULT_ENABLED_TOOLS,
    load_config,
    merge_tool_lists,
    standard_tools,
)
from arietta_voice.private_config import initialize_private_config, local_config_path
from arietta_voice.tool_admin import ToolAdmin

RC1 = (
    "assistant_identity",
    "local_time",
    "set_timer",
    "set_reminder",
    "set_alarm",
    "check_timers",
    "cancel_timer",
    "search_knowledge",
    "system_info",
)


def write_configs(tmp_path, standard=DEFAULT_ENABLED_TOOLS, private=None):
    shipped = tmp_path / "config" / "arietta_voice.toml"
    shipped.parent.mkdir(exist_ok=True)
    shipped.write_text(f"[tools]\nenabled = {json.dumps(list(standard))}\n", encoding="utf-8")
    if private is not None:
        local_config_path(shipped).write_text(private, encoding="utf-8")
    return shipped


def test_without_a_private_list_the_standard_tools_run():
    assert merge_tool_lists(DEFAULT_ENABLED_TOOLS) == (DEFAULT_ENABLED_TOOLS, ())
    enabled, disabled = merge_tool_lists(DEFAULT_ENABLED_TOOLS, None, ["system_info"])
    assert "system_info" not in enabled and disabled == ("system_info",)


def test_a_saved_list_keeps_its_order_and_new_tools_join_in_their_standard_place():
    standard = ["local_time", "set_timer", "calculate", "search_knowledge", "system_info"]
    saved = ["system_info", "local_time", "my_lights", "search_knowledge"]
    enabled, disabled = merge_tool_lists(standard, saved, [])
    # New tools go just before the next standard tool in the list, so they come
    # after your own tools, whose triggers keep first say.
    assert enabled == (
        "system_info",
        "local_time",
        "my_lights",
        "set_timer",
        "calculate",
        "search_knowledge",
    )
    assert disabled == ()


def test_switched_off_tools_stay_off_even_if_also_listed():
    standard = ["local_time", "calculate", "search_knowledge"]
    enabled, disabled = merge_tool_lists(standard, ["local_time", "calculate"], ["calculate"])
    assert enabled == ("local_time", "search_knowledge")
    assert disabled == ("calculate",)


def test_a_list_from_before_tools_were_unified_gets_every_new_tool(tmp_path):
    shipped = write_configs(
        tmp_path,
        private='[tools]\nenabled = ["assistant_identity", "local_time", "knowledge_retrieval"]\n'
        '[model_tools]\nenabled = ["search_knowledge", "system_info"]\n',
    )
    config = load_config(shipped)
    assert config.tools.enabled == DEFAULT_ENABLED_TOOLS
    assert config.tools.disabled == ()
    assert config.model_tools.enabled == ()


def test_a_tool_an_old_list_left_out_stays_off(tmp_path):
    # Saved before 0.2, without system_info: it was switched off, so it stays off.
    shipped = write_configs(
        tmp_path, private='[tools]\nenabled = ["local_time", "knowledge_retrieval"]\n'
    )
    # A disabled list in the shipped file does not make an old private list look new.
    shipped.write_text(shipped.read_text() + "disabled = []\n", encoding="utf-8")
    config = load_config(shipped)
    assert "system_info" not in config.tools.enabled
    assert "assistant_identity" not in config.tools.enabled
    assert {"set_timer", "calculate", "date_math"} <= set(config.tools.enabled)
    assert config.tools.disabled == ("assistant_identity", "system_info")


def test_a_list_from_rc1_gets_rc2_tools_but_keeps_its_choices(tmp_path):
    saved = [name for name in RC1 if name != "set_alarm"]
    shipped = write_configs(tmp_path, private=f"[tools]\nenabled = {json.dumps(saved)}\n")
    config = load_config(shipped)
    assert "set_alarm" not in config.tools.enabled
    assert config.tools.enabled.index("date_math") < config.tools.enabled.index("search_knowledge")
    assert {"calculate", "date_math"} <= set(config.tools.enabled)


def test_an_empty_old_list_still_means_every_tool_off(tmp_path):
    shipped = write_configs(tmp_path, private="[tools]\nenabled = []\n")
    config = load_config(shipped)
    assert config.tools.enabled == ()
    assert config.tools.disabled == DEFAULT_ENABLED_TOOLS


def test_console_save_records_switched_off_tools_and_later_tools_still_arrive(tmp_path):
    shipped = write_configs(tmp_path, standard=RC1)
    admin = ToolAdmin(load_config(shipped), config_path=shipped)
    chosen = ["local_time", "set_timer", "check_timers", "weather_like_tool", "search_knowledge"]
    admin._load = lambda name: None  # Only the saved lists matter here.
    admin.validate = lambda name: {"validation": "valid", "origin": "builtin", "error": None}
    admin.set_enabled(chosen)

    saved = local_config_path(shipped).read_text(encoding="utf-8")
    assert '"set_alarm"' in saved.split("disabled = ")[1].splitlines()[0]
    assert load_config(shipped).tools.enabled == tuple(chosen)

    # A later release adds calculate to the standard list.
    write_configs(tmp_path, standard=DEFAULT_ENABLED_TOOLS)
    upgraded = load_config(shipped).tools.enabled
    assert "calculate" in upgraded and "date_math" in upgraded
    assert "set_alarm" not in upgraded and "system_info" not in upgraded
    assert upgraded.index("weather_like_tool") < upgraded.index("search_knowledge")


def test_a_private_config_snapshot_still_gets_later_tools(tmp_path):
    shipped = write_configs(tmp_path, standard=RC1)
    initialize_private_config(shipped)
    assert load_config(shipped).tools.enabled == RC1
    write_configs(tmp_path, standard=DEFAULT_ENABLED_TOOLS)
    assert load_config(shipped).tools.enabled == DEFAULT_ENABLED_TOOLS


def test_standard_tools_come_from_the_shipped_file_or_the_defaults(tmp_path):
    assert standard_tools(tmp_path / "missing.toml") == DEFAULT_ENABLED_TOOLS
    shipped = write_configs(tmp_path, standard=["local_time", "knowledge_retrieval"])
    assert standard_tools(shipped) == ("local_time", "search_knowledge")


def test_disabled_must_be_a_list_of_names(tmp_path):
    shipped = write_configs(tmp_path, private='[tools]\ndisabled = "system_info"\n')
    with pytest.raises(ValueError, match="tools.disabled"):
        load_config(shipped)
