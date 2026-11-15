"""The console in a real browser: every page loads and works without an error.

The static checks in test_console_static.py catch broken imports and missing
element ids. These catch what only shows when the page runs: an exception while
a view renders, an API call that fails, an error banner. The console runs
against a scratch setup and is driven through Playwright in the Chrome that is
already installed (as on the macOS CI runners); without Chrome they are skipped,
except where ARIETTA_REQUIRE_BROWSER is set, as in CI.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

PASSWORD = "browser-test-only-password"
VIEWS = {
    "home": "Home",
    "chat": "Chat",
    "knowledge": "Knowledge",
    "persona": "Persona",
    "tools": "Tools",
    "connections": "Connections",
    "activity": "Activity",
    "system": "System",
    "settings": "Settings",
}


def _write_config(root: Path) -> Path:
    for name in ("config", "knowledge", "logs"):
        (root / name).mkdir()
    (root / "config" / "SOUL.md").write_text("You are a concise test assistant.")
    (root / "knowledge" / "food.md").write_text("# Food\nLunch is at the cafeteria.")
    config_path = root / "config" / "arietta_voice.toml"
    config_path.write_text(
        """
[assistant]
name = "Browser Test"

[tools]
enabled = ["assistant_identity", "local_time"]

[prompts]
soul_path = "SOUL.md"
memory_path = "MEMORY.md"

[logging]
directory = "../logs"
events_file = "../logs/events.jsonl"

[knowledge]
backend = "keyword"
directory = "../knowledge"

[mcp]
enabled = false
""".strip()
    )
    return config_path


@pytest.fixture(scope="module")
def console(tmp_path_factory):
    import uvicorn

    from arietta_voice.web.app import create_app
    from arietta_voice.web.service import AriettaApiService

    patch = pytest.MonkeyPatch()
    patch.setenv("ARIETTA_ADMIN_USERNAME", "admin")
    patch.setenv("ARIETTA_ADMIN_PASSWORD", PASSWORD)
    try:
        config_path = _write_config(tmp_path_factory.mktemp("console"))
        app = create_app(service=AriettaApiService(config_path))
    finally:
        patch.undo()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=1
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            pytest.fail("The console server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(10)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
        except Exception as error:  # Chrome is not installed here.
            reason = f"Chrome is not available: {str(error).splitlines()[0]}"
            if os.environ.get("ARIETTA_REQUIRE_BROWSER"):
                pytest.fail(reason)  # CI must run these, not skip them.
            pytest.skip(reason)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(http_credentials={"username": "admin", "password": PASSWORD})
    page = context.new_page()
    page.problems = []
    page.on("pageerror", lambda error: page.problems.append(f"script error: {error}"))
    page.on(
        "console",
        lambda message: (
            page.problems.append(f"console error: {message.text}")
            if message.type == "error"
            else None
        ),
    )
    page.on(
        "response",
        lambda response: (
            page.problems.append(f"HTTP {response.status}: {response.url}")
            if response.status >= 400
            else None
        ),
    )
    yield page
    context.close()


def _settle(page, view: str, wait_ms: int = 1200) -> None:
    page.wait_for_function(
        "view => document.getElementById(view).classList.contains('active')", arg=view
    )
    page.wait_for_timeout(wait_ms)  # The view's own requests.


@pytest.mark.parametrize("view", VIEWS)
def test_every_page_loads_without_an_error(console, page, view):
    page.goto(f"{console}/#/{view}")
    _settle(page, view)
    assert page.locator("#view-title").inner_text() == VIEWS[view]
    assert page.locator("#error-notice").is_hidden()
    assert page.problems == []


def test_the_navigation_and_theme_work(console, page):
    page.goto(console)
    _settle(page, "home")
    for view, title in VIEWS.items():
        page.click(f'.nav-button[data-view="{view}"]')
        _settle(page, view, wait_ms=200)  # Each page's requests are checked above.
        assert page.url.endswith(f"#/{view}")
        assert page.locator("#view-title").inner_text() == title
        assert page.title() == f"{title} · Arietta Voice"
    before = page.evaluate("document.documentElement.dataset.theme || ''")
    page.click("#theme-toggle")
    assert page.evaluate("document.documentElement.dataset.theme") != before
    assert page.locator("#error-notice").is_hidden()
    assert page.problems == []


def test_the_home_page_shows_the_assistant_is_stopped(console, page):
    page.goto(console)
    _settle(page, "home")
    page.wait_for_timeout(1000)
    text = page.locator("#home").inner_text().lower()
    assert "stopped" in text or "not running" in text
    assert page.problems == []


def test_a_script_error_or_failed_request_is_caught(console, page):
    # Without this, a listener that never fired would pass every test above.
    page.goto(console)
    _settle(page, "home")
    page.evaluate("setTimeout(() => { throw new Error('synthetic failure'); })")
    page.evaluate("fetch('/api/no-such-endpoint')")
    page.wait_for_timeout(500)
    assert any("synthetic failure" in problem for problem in page.problems)
    assert any("HTTP 404" in problem for problem in page.problems)
