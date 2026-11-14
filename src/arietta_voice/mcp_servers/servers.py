"""The MCP servers the assistant connects to, kept in a private JSON file.

The file uses the format other MCP clients use, so the snippet in a server's
README can be pasted in as it is::

    {"mcpServers": {
        "lights": {"command": "uvx", "args": ["lights-mcp"], "env": {"TOKEN": "..."}},
        "home": {"type": "http", "url": "http://homeassistant.local:8123/api/mcp",
                 "headers": {"Authorization": "Bearer ..."}}
    }}

Each server may also say whether it is ``enabled`` (default true), when to
``confirm`` before one of its tools acts (see CONFIRM_POLICIES), and which of its
tools to leave out (``disabledTools``). Other keys are kept as they are.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from arietta_voice.private_config import write_private_text

# When to ask before a tool that changes something runs (read-only tools never ask):
# "ask" every time, unless you answered "always" for that action; "first_use",
# only the first time each action is used; "never", for a server you trust.
CONFIRM_POLICIES = ("ask", "first_use", "never")
TRANSPORTS = ("stdio", "http", "sse")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
# Shown instead of a secret; sending it back unchanged keeps the saved value.
MASK = "••••••"
MAX_SERVERS = 20
_KNOWN = {"type", "command", "args", "env", "cwd", "url", "headers"}
_OURS = {"enabled", "confirm", "disabledTools"}


@dataclass(slots=True)
class ServerConfig:
    name: str
    transport: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    confirm: str = "ask"
    disabled_tools: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def runs_a_program(self) -> bool:
        return self.transport == "stdio"

    def connection_key(self) -> str:
        """What a connection depends on; a change to anything else needs no reconnect."""
        return json.dumps(
            [self.transport, self.command, self.args, self.env, self.cwd, self.url, self.headers],
            sort_keys=True,
        )

    def to_json(self) -> dict[str, Any]:
        entry: dict[str, Any] = dict(self.extra)
        if self.transport == "stdio":
            entry["command"] = self.command
            if self.args:
                entry["args"] = list(self.args)
            if self.env:
                entry["env"] = dict(self.env)
            if self.cwd:
                entry["cwd"] = self.cwd
        else:
            entry["type"] = self.transport
            entry["url"] = self.url
            if self.headers:
                entry["headers"] = dict(self.headers)
        if not self.enabled:
            entry["enabled"] = False
        if self.confirm != "ask":
            entry["confirm"] = self.confirm
        if self.disabled_tools:
            entry["disabledTools"] = list(self.disabled_tools)
        return entry

    def public(self) -> dict[str, Any]:
        """For the console: secrets in env and headers are masked."""
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args),
            "env": {key: MASK for key in self.env},
            "cwd": self.cwd,
            "url": self.url,
            "headers": {key: MASK for key in self.headers},
            "enabled": self.enabled,
            "confirm": self.confirm,
            "disabled_tools": list(self.disabled_tools),
        }


def parse_server(name: str, entry: Any) -> ServerConfig:
    """One server from its JSON entry; ValueError says what is wrong."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(
            f"Server name {name!r} must be 1-40 letters, digits, hyphens or underscores"
        )
    if not isinstance(entry, dict):
        raise ValueError(f"{name}: each server must be a JSON object")
    command, url = entry.get("command"), entry.get("url")
    kind = entry.get("type")
    if kind not in (None, *TRANSPORTS, "streamable-http", "streamableHttp"):
        raise ValueError(f"{name}: type must be stdio, http or sse")
    if command and url:
        raise ValueError(f"{name}: give a command or a url, not both")
    if command:
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"{name}: command must be text")
        transport = "stdio"
    elif url:
        if not isinstance(url, str) or not re.match(r"https?://", url.strip()):
            raise ValueError(f"{name}: url must start with http:// or https://")
        transport = "sse" if kind == "sse" or url.rstrip("/").endswith("/sse") else "http"
    else:
        raise ValueError(f"{name}: needs a command (a local program) or a url")
    confirm = entry.get("confirm", "ask")
    if confirm not in CONFIRM_POLICIES:
        raise ValueError(f"{name}: confirm must be one of {', '.join(CONFIRM_POLICIES)}")
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{name}: enabled must be true or false")
    return ServerConfig(
        name=name,
        transport=transport,
        command=command.strip() if command else "",
        args=_strings(entry.get("args", []), f"{name}: args"),
        env=_mapping(entry.get("env", {}), f"{name}: env"),
        cwd=_text(entry.get("cwd", ""), f"{name}: cwd"),
        url=url.strip() if url else "",
        headers=_mapping(entry.get("headers", {}), f"{name}: headers"),
        enabled=enabled,
        confirm=confirm,
        disabled_tools=_strings(entry.get("disabledTools", []), f"{name}: disabledTools"),
        extra={key: value for key, value in entry.items() if key not in _KNOWN | _OURS},
    )


def parse_servers(data: Any) -> tuple[list[ServerConfig], list[str]]:
    """Servers from a parsed file or snippet, and a problem for each bad entry."""
    if isinstance(data, dict) and "mcpServers" in data:
        data = data["mcpServers"]
    elif isinstance(data, dict) and "servers" in data and isinstance(data["servers"], dict):
        data = data["servers"]  # The format some editors use.
    if not isinstance(data, dict):
        return [], ['Expected {"mcpServers": {"name": {...}}}']
    servers, problems = [], []
    for name, entry in data.items():
        try:
            servers.append(parse_server(name, entry))
        except ValueError as error:
            problems.append(str(error))
    if len(servers) > MAX_SERVERS:
        problems.append(f"Only the first {MAX_SERVERS} servers are used")
        servers = servers[:MAX_SERVERS]
    return servers, problems


def load_servers(path: Path) -> tuple[list[ServerConfig], list[str]]:
    if not path.is_file():
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [], [f"{path.name} could not be read: {error}"]
    return parse_servers(data)


def save_servers(path: Path, servers: list[ServerConfig]) -> None:
    """Write the servers, keeping anything else at the top of an existing file."""
    if len(servers) > MAX_SERVERS:
        raise ValueError(f"At most {MAX_SERVERS} MCP servers can be connected")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if isinstance(data.get("servers"), dict):
        del data["servers"]  # Read as the servers list; written back under the usual key.
    data["mcpServers"] = {server.name: server.to_json() for server in servers}
    write_private_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def merge_secrets(updated: ServerConfig, saved: ServerConfig | None) -> ServerConfig:
    """Keep saved env and header values where the console sent back the mask.

    Only for the same destination: a saved token never follows a server to a
    new URL or program, so changing either needs the values entered again.
    """
    masked = MASK in updated.env.values() or MASK in updated.headers.values()
    moved = saved is None or (
        (updated.transport, updated.url, updated.command, updated.args)
        != (saved.transport, saved.url, saved.command, saved.args)
    )
    if masked and moved:
        raise ValueError(
            "The address or program changed, so enter the hidden values again "
            "(saved tokens are not sent anywhere new)"
        )
    previous_env = saved.env if saved else {}
    previous_headers = saved.headers if saved else {}
    return replace(
        updated,
        env={k: previous_env.get(k, "") if v == MASK else v for k, v in updated.env.items()},
        headers={
            k: previous_headers.get(k, "") if v == MASK else v for k, v in updated.headers.items()
        },
    )


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of text")
    return tuple(value)


def _mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{label} must map names to text")
    return dict(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return value.strip()
