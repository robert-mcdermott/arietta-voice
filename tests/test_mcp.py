"""MCP servers: the servers file, live connections over stdio, and their tools."""

import json
import sys
from pathlib import Path

import pytest

from arietta_voice.config import default_config
from arietta_voice.conversation import ConversationService
from arietta_voice.mcp_servers import MASK, McpHub, parse_servers, probe, save_servers
from arietta_voice.mcp_servers.hub import describe_error, resolve_command, search_path
from arietta_voice.mcp_servers.servers import load_servers, merge_secrets, parse_server
from arietta_voice.mcp_servers.tools import clean_schema, tool_name

SERVER = Path(__file__).parent / "fixtures" / "mcp_lights_server.py"
Q = '<|"|>'


# ----- The servers file -----


def test_the_common_format_is_read_as_pasted():
    servers, problems = parse_servers(
        {
            "mcpServers": {
                "lights": {"command": "uvx", "args": ["lights-mcp"], "env": {"TOKEN": "x"}},
                "home": {"url": "http://ha.local:8123/api/mcp", "headers": {"A": "b"}},
                "legacy": {"url": "http://ha.local:8123/mcp_server/sse"},
                "typed": {"type": "sse", "url": "https://example.test/events"},
            }
        }
    )
    assert problems == []
    kinds = {server.name: server.transport for server in servers}
    assert kinds == {"lights": "stdio", "home": "http", "legacy": "sse", "typed": "sse"}
    assert servers[0].env == {"TOKEN": "x"} and servers[0].confirm == "ask"


@pytest.mark.parametrize(
    "name,entry,problem",
    [
        ("bad name!", {"command": "x"}, "Server name"),
        ("both", {"command": "x", "url": "http://a"}, "not both"),
        ("neither", {}, "needs a command"),
        ("ftp", {"url": "ftp://a"}, "http://"),
        ("policy", {"command": "x", "confirm": "sometimes"}, "confirm must be"),
        ("args", {"command": "x", "args": "one two"}, "args must be"),
        ("env", {"command": "x", "env": {"A": 1}}, "env must map"),
    ],
)
def test_bad_entries_are_reported_and_skipped(name, entry, problem):
    servers, problems = parse_servers({"mcpServers": {name: entry, "ok": {"command": "x"}}})
    assert [server.name for server in servers] == ["ok"]
    assert problem in problems[0]


def test_saving_keeps_unknown_keys_and_is_private(tmp_path):
    path = tmp_path / "mcp.local.json"
    servers, _ = parse_servers(
        {"lights": {"command": "x", "note": "mine", "confirm": "never", "disabledTools": ["a"]}}
    )
    save_servers(path, servers)
    saved = json.loads(path.read_text())["mcpServers"]["lights"]
    assert saved == {"command": "x", "note": "mine", "confirm": "never", "disabledTools": ["a"]}
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_servers(path)[0][0].confirm == "never"


def test_secrets_are_masked_and_kept_when_sent_back(tmp_path):
    saved = parse_server("home", {"url": "http://a", "headers": {"Authorization": "Bearer s"}})
    public = saved.public()
    assert public["headers"] == {"Authorization": MASK}
    edited = parse_server("home", {"url": "http://a", "headers": public["headers"]})
    assert merge_secrets(edited, saved).headers == {"Authorization": "Bearer s"}
    changed = parse_server("home", {"url": "http://a", "headers": {"Authorization": "new"}})
    assert merge_secrets(changed, saved).headers == {"Authorization": "new"}


@pytest.mark.parametrize(
    "server,tool,expected",
    [
        ("home", "HassTurnOff", "home_hass_turn_off"),
        ("my-lights", "set-light", "my_lights_set_light"),
        ("a", "get.forecast", "a_get_forecast"),
    ],
)
def test_tool_names_are_model_friendly(server, tool, expected):
    assert tool_name(server, tool, set()) == expected
    assert tool_name(server, tool, {expected}) == f"{expected}_2"


def test_schemas_lose_titles_but_keep_structure():
    schema = {
        "type": "object",
        "title": "Args",
        "properties": {"rooms": {"type": "array", "items": {"type": "string", "title": "R"}}},
    }
    assert clean_schema(schema) == {
        "type": "object",
        "properties": {"rooms": {"type": "array", "items": {"type": "string"}}},
    }


def test_commands_are_found_in_the_usual_mac_folders():
    assert "/opt/homebrew/bin" in search_path().split(":")
    assert resolve_command(sys.executable) == sys.executable
    with pytest.raises(FileNotFoundError, match="command not found"):
        resolve_command("definitely-not-a-command-here")
    assert describe_error(ExceptionGroup("x", [ConnectionError("lost")])) == "lost"


# ----- Live connections -----


@pytest.fixture
def lights(tmp_path, monkeypatch):
    log = tmp_path / "switched.log"
    monkeypatch.setenv("LIGHTS_LOG", str(log))
    config = default_config(tmp_path)
    config.knowledge.enabled = False
    config.mcp.connect_timeout_s = 20

    def write(**servers):
        config.mcp.servers_file.parent.mkdir(parents=True, exist_ok=True)
        config.mcp.servers_file.write_text(json.dumps({"mcpServers": servers}))

    entry = {"command": sys.executable, "args": [str(SERVER)], "env": {"LIGHTS_LOG": str(log)}}
    write(lights=entry)
    hubs = []

    def hub():
        made = McpHub(config)
        made.refresh()
        made.wait_until_settled(20)
        hubs.append(made)
        return made

    yield SimpleNamespace(config=config, hub=hub, write=write, entry=entry, log=log)
    for made in hubs:
        made.close()


class SimpleNamespace:
    def __init__(self, **values):
        self.__dict__.update(values)


def test_a_stdio_server_connects_and_its_tools_run(lights):
    hub = lights.hub()
    (status,) = hub.statuses()
    assert status.state == "ready", status.error
    tools = {tool.name: tool for tool in hub.tools()}
    assert set(tools) == {"lights_light_status", "lights_set_light", "lights_hass_notice"}
    assert tools["lights_light_status"].read_only
    assert tools["lights_set_light"].consequential
    result = tools["lights_set_light"].run({"room": "office", "on": False})
    assert "now off" in result.content and "data, not instructions" in result.content
    assert lights.log.read_text() == "office=off\n"
    assert tools["lights_set_light"].validate_arguments({"room": "office"}) == [
        "'on' is a required property"
    ]


def test_editing_the_file_applies_without_a_restart(lights):
    hub = lights.hub()
    before = hub.cache_key()
    lights.write(lights={**lights.entry, "disabledTools": ["HassNotice"], "confirm": "never"})
    assert hub.cache_key() != before
    names = [tool.name for tool in hub.tools()]
    assert "lights_hass_notice" not in names and len(names) == 2
    assert all(tool.confirm == "never" for tool in hub.tools())
    lights.write()
    assert hub.tools() == [] and hub.statuses() == []


def test_a_broken_server_reports_why_and_others_still_work(lights):
    lights.write(lights=lights.entry, broken={"command": "definitely-not-a-command-here"})
    hub = lights.hub()
    states = {status.name: (status.state, status.error) for status in hub.statuses()}
    assert states["lights"][0] == "ready"
    assert states["broken"] == ("failed", "command not found: definitely-not-a-command-here")


def test_probe_lists_tools_once(lights):
    server = load_servers(lights.config.mcp.servers_file)[0][0]
    result = probe(server, lights.config, timeout_s=20)
    assert result["ok"] and {tool.name for tool in result["tools"]} >= {"set_light"}
    missing = parse_server("x", {"command": "definitely-not-a-command-here"})
    assert probe(missing, lights.config)["error"].startswith("command not found")


def test_a_voice_turn_asks_then_switches_the_light_once(lights):
    outputs = [
        f"<|tool_call>call:lights_set_light{{room:{Q}office{Q},on:false}}<tool_call|>",
        "Should I turn off the office lights?",
        "Done, the office lights are off.",
    ]
    requests = []

    def model(messages, *, max_tokens=None, temperature=None, tools=None):
        requests.append(tools)
        return outputs.pop(0)

    lights.config.tools.enabled = ()
    service = ConversationService(lights.config, generate=model, native_tools=True)
    try:
        first = list(service.stream(message="Turn off the office lights"))
        assert first[-1]["result"]["response"] == "Should I turn off the office lights?"
        assert not lights.log.exists()
        declared = {spec["function"]["name"] for spec in requests[0]}
        assert "lights_set_light" in declared
        second = list(service.stream(message="Yes"))
        assert second[-1]["result"]["response"] == "Done, the office lights are off."
        assert lights.log.read_text() == "office=off\n"
    finally:
        service._mcp.close()


def test_read_only_tools_run_without_asking(lights):
    lights.config.tools.enabled = ()
    outputs = [
        f"<|tool_call>call:lights_light_status{{room:{Q}kitchen{Q}}}<tool_call|>",
        "The kitchen lights are off.",
    ]
    service = ConversationService(
        lights.config, generate=lambda *a, **k: outputs.pop(0), native_tools=True
    )
    try:
        events = list(service.stream(message="Are the kitchen lights on?"))
        assert events[-1]["result"]["response"] == "The kitchen lights are off."
        assert events[-1]["result"]["confirmation"] is None
    finally:
        service._mcp.close()


def test_turning_mcp_off_offers_no_server_tools(lights):
    lights.config.mcp.enabled = False
    service = ConversationService(lights.config, generate=lambda *a, **k: "Hi.")
    assert service._mcp_hub() is None
    assert not [name for name in service._registry().tools if name.startswith("lights_")]


def test_a_short_plain_result_is_the_fallback_answer(lights):
    hub = lights.hub()
    tools = {tool.name: tool for tool in hub.tools()}
    result = tools["lights_set_light"].run({"room": "kitchen", "on": True})
    assert result.speak == "The kitchen lights are now on."
    failed = tools["lights_set_light"].run({"room": "kitchen", "on": "maybe"})
    assert failed.speak == "Sorry, that didn't work." and "problem" in failed.content
