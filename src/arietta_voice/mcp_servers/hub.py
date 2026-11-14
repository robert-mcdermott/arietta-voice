"""Connections to MCP servers, kept open on a background event loop.

The hub reads the servers file, connects to each enabled server, keeps the
connection alive with pings and reconnects with a growing delay when it drops.
It notices when the file changes, so servers added in the console reach a
running assistant without a restart. Tools run synchronously from the
conversation's threads through ``call``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import time
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arietta_voice.mcp_servers.servers import ServerConfig, load_servers

logger = logging.getLogger("arietta_voice.mcp")
PING_EVERY_S = 30.0
PING_TIMEOUT_S = 10.0
MAX_TOOLS_PER_SERVER = 100
RETRY_FIRST_S = 2.0
RETRY_MAX_S = 120.0
# Where command-line servers are usually installed on a Mac; added to PATH so
# "npx" and "uvx" are found even when the assistant starts outside a shell.
EXTRA_PATHS = ("/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local" / "bin"))


class McpUnavailable(RuntimeError):
    """The server is not connected (yet), so its tool cannot run."""


@dataclass(slots=True)
class ServerStatus:
    name: str
    state: str  # connecting, ready, failed
    error: str | None
    tools: list[Any]
    server_info: str | None = None


class McpHub:
    def __init__(self, config) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connections: dict[str, _Connection] = {}
        self._file_state: object = None
        self._loaded = False
        self.problems: list[str] = []

    # ----- Lifecycle -----

    def refresh(self) -> None:
        """Read the servers file again if it changed, and connect or disconnect to match."""
        path = Path(self.config.mcp.servers_file)
        try:
            stat = path.stat()
            state: object = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            state = None
        with self._lock:
            if self._loaded and state == self._file_state:
                return
            self._loaded, self._file_state = True, state
            servers, self.problems = load_servers(path) if state else ([], [])
            for problem in self.problems:
                logger.warning("MCP servers file: %s", problem)
            wanted = {
                server.name: server
                for server in servers
                if server.enabled and self.config.mcp.enabled
            }
            for name, connection in list(self._connections.items()):
                server = wanted.get(name)
                if server is None or server.connection_key() != connection.server.connection_key():
                    connection.stop()
                    del self._connections[name]
                else:
                    connection.server = server  # Confirm policy or tool list only.
            for name, server in wanted.items():
                if name not in self._connections:
                    connection = _Connection(server, self.config)
                    self._connections[name] = connection
                    connection.start(self._ensure_loop())

    def close(self) -> None:
        with self._lock:
            for connection in self._connections.values():
                connection.stop()
            self._connections.clear()
            loop, self._loop = self._loop, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="arietta-mcp", daemon=True)
            thread.start()
            self._loop, self._thread = loop, thread
        return self._loop

    # ----- What the conversation uses -----

    def statuses(self) -> list[ServerStatus]:
        self.refresh()
        with self._lock:
            connections = list(self._connections.values())
        return [connection.status() for connection in connections]

    def cache_key(self) -> tuple:
        """Changes whenever the tools on offer or their confirmation rules change."""
        self.refresh()
        with self._lock:
            return tuple(
                (name, c.generation, c.server.confirm, c.server.disabled_tools)
                for name, c in sorted(self._connections.items())
            )

    def tools(self) -> list:
        """A tool for each enabled tool of each connected server."""
        from arietta_voice.mcp_servers.tools import McpTool, tool_name

        self.refresh()
        with self._lock:
            connections = sorted(self._connections.items())
        tools, taken = [], set()
        for _, connection in connections:
            if connection.state != "ready":
                continue
            server = connection.server
            for spec in connection.tools:
                if spec.name in server.disabled_tools:
                    continue
                name = tool_name(server.name, spec.name, taken)
                taken.add(name)
                tools.append(McpTool(self, server, spec, name, self.config.mcp.call_timeout_s))
        return tools

    def wait_until_settled(self, timeout_s: float) -> None:
        """Wait for servers still making their first connection, for up to ``timeout_s``."""
        deadline = time.monotonic() + timeout_s
        with self._lock:
            connections = list(self._connections.values())
        for connection in connections:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            connection.settled.wait(remaining)

    def call(self, server: str, tool: str, arguments: dict[str, Any], timeout_s: float):
        with self._lock:
            connection = self._connections.get(server)
            loop = self._loop
        if connection is None or loop is None:
            raise McpUnavailable(f"{server} is not connected")
        future = asyncio.run_coroutine_threadsafe(connection.call(tool, arguments, timeout_s), loop)
        try:
            return future.result(timeout_s + 5)
        except TimeoutError:
            future.cancel()
            raise


class _Connection:
    def __init__(self, server: ServerConfig, config) -> None:
        self.server = server
        self.config = config
        self.state = "connecting"
        self.error: str | None = None
        self.tools: list[Any] = []
        self.server_info: str | None = None
        self.generation = 0
        self.settled = threading.Event()  # The first attempt has finished.
        self._client = None
        self._broken: asyncio.Event | None = None
        self._future = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._future = asyncio.run_coroutine_threadsafe(self._run(), loop)

    def stop(self) -> None:
        if self._future is not None:
            self._future.cancel()

    def status(self) -> ServerStatus:
        return ServerStatus(self.server.name, self.state, self.error, self.tools, self.server_info)

    def _set(self, state: str, error: str | None = None) -> None:
        self.state, self.error = state, error
        self.generation += 1

    async def _run(self) -> None:
        delay = RETRY_FIRST_S
        while True:
            try:
                await self._attempt()
                delay = RETRY_FIRST_S  # The server ended the session; reconnect soon.
            except asyncio.CancelledError:
                raise
            except Exception as error:  # Any failure: report it and try again later.
                message = describe_error(error)
                logger.warning("MCP server %s: %s", self.server.name, message)
                self._set("failed", message)
            finally:
                self._client = None
                self.settled.set()
            await asyncio.sleep(delay)
            delay = min(delay * 2, RETRY_MAX_S)

    async def _attempt(self) -> None:
        """Connect, list tools and stay connected; the whole session runs in one task."""
        self._set("connecting", self.error)
        ready = asyncio.Event()
        session = asyncio.create_task(self._session(ready))
        waiter = asyncio.create_task(ready.wait())
        try:
            done, _ = await asyncio.wait(
                {session, waiter},
                timeout=self.config.mcp.connect_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(f"no answer within {self.config.mcp.connect_timeout_s:g} s")
            await session
        finally:
            # Stopped, timed out or failed: the session (and any program it
            # started) must end with this attempt, never run on unseen.
            waiter.cancel()
            if not session.done():
                session.cancel()
                with suppress(BaseException):
                    await session

    async def _session(self, ready: asyncio.Event) -> None:
        import anyio
        from mcp import Client, Implementation

        from arietta_voice import __version__

        async with AsyncExitStack() as stack:
            transport = await _open_transport(self.server, self.config, stack)
            client = await stack.enter_async_context(
                Client(
                    transport,
                    read_timeout_seconds=self.config.mcp.call_timeout_s,
                    client_info=Implementation(name="arietta-voice", version=__version__),
                )
            )
            self.tools = await _list_tools(client)
            info = client.server_info
            self.server_info = f"{info.name} {info.version}".strip() if info else None
            self._broken = asyncio.Event()
            self._client = client
            self._set("ready")
            self.settled.set()
            ready.set()
            while True:
                with anyio.move_on_after(PING_EVERY_S):
                    await self._broken.wait()
                if self._broken.is_set():
                    raise ConnectionError("the connection was lost")
                with anyio.fail_after(PING_TIMEOUT_S):
                    await client.send_ping()

    async def call(self, tool: str, arguments: dict[str, Any], timeout_s: float):
        import anyio

        client = self._client
        if client is None or self.state != "ready":
            raise McpUnavailable(f"{self.server.name} is not connected")
        try:
            with anyio.fail_after(timeout_s):
                return await client.call_tool(tool, arguments, read_timeout_seconds=timeout_s)
        except TimeoutError:
            raise  # A slow tool; the connection itself is fine.
        except (ConnectionError, OSError, anyio.ClosedResourceError, anyio.BrokenResourceError):
            if self._broken is not None:
                self._broken.set()  # Reconnect; this call has failed.
            raise


async def _list_tools(client) -> list[Any]:
    tools, cursor = [], None
    while True:
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor or len(tools) >= MAX_TOOLS_PER_SERVER:
            return tools[:MAX_TOOLS_PER_SERVER]


async def _open_transport(server: ServerConfig, config, stack: AsyncExitStack):
    if server.transport == "stdio":
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        errlog = stack.enter_context(_server_log(server, config))
        params = StdioServerParameters(
            command=resolve_command(server.command),
            args=list(server.args),
            env={"PATH": search_path(), **server.env},
            cwd=server.cwd or None,
        )
        return stdio_client(params, errlog=errlog)
    from mcp.shared._httpx_utils import create_mcp_http_client

    if server.transport == "sse":
        from mcp.client.sse import sse_client

        return sse_client(server.url, headers=server.headers or None)
    from mcp.client.streamable_http import streamable_http_client

    http_client = await stack.enter_async_context(
        create_mcp_http_client(headers=server.headers or None)
    )
    return streamable_http_client(server.url, http_client=http_client)


def search_path() -> str:
    parts = os.environ.get("PATH", "").split(os.pathsep)
    return os.pathsep.join(dict.fromkeys([*filter(None, parts), *EXTRA_PATHS]))


def resolve_command(command: str) -> str:
    """The program to run, found on PATH or in the usual Mac install folders."""
    if os.sep in command:
        return command
    found = shutil.which(command, path=search_path())
    if found is None:
        raise FileNotFoundError(f"command not found: {command}")
    return found


def _server_log(server: ServerConfig, config):
    """A command-line server's own messages go to logs/mcp-<name>.log, not the terminal."""
    directory = Path(config.logging.directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"mcp-{server.name}.log"
    if path.exists() and path.stat().st_size > 1_000_000:
        path.unlink()
    return open(path, "a", encoding="utf-8")  # noqa: SIM115 - closed by the exit stack


def describe_error(error: BaseException) -> str:
    """A short reason for a failed connection, from inside any exception groups."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    if isinstance(error, FileNotFoundError):
        return str(error) if "command not found" in str(error) else f"not found: {error}"
    text = str(error).strip() or type(error).__name__
    if isinstance(error, (TimeoutError, ConnectionError)):
        return text
    return f"{type(error).__name__}: {text}"[:300]


def probe(server: ServerConfig, config, timeout_s: float | None = None) -> dict[str, Any]:
    """Connect once, list the server's tools and disconnect (the console's Test)."""
    timeout_s = timeout_s or config.mcp.connect_timeout_s
    started = time.monotonic()

    async def run():
        from mcp import Client

        async with AsyncExitStack() as stack:
            transport = await _open_transport(server, config, stack)
            client = await stack.enter_async_context(Client(transport))
            tools = await _list_tools(client)
            info = client.server_info
            return tools, (f"{info.name} {info.version}".strip() if info else None)

    async def bounded():
        task = asyncio.create_task(run())
        try:
            return await asyncio.wait_for(task, timeout_s)
        except TimeoutError as error:
            raise TimeoutError(f"no answer within {timeout_s:g} s") from error

    try:
        tools, info = asyncio.run(bounded())
    except Exception as error:
        return {"ok": False, "error": describe_error(error), "tools": [], "server_info": None}
    finally:
        elapsed = int((time.monotonic() - started) * 1000)
    return {"ok": True, "error": None, "tools": tools, "server_info": info, "elapsed_ms": elapsed}
