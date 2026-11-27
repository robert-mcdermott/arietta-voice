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
import re
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
enabled = ["assistant_identity", "local_time", "search_knowledge"]

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
    yield from _serve(tmp_path_factory.mktemp("console"))


@pytest.fixture(scope="module")
def memory_console(tmp_path_factory):
    yield from _serve(tmp_path_factory.mktemp("memory-console"), memory=True)


def _serve(root, *, memory=False):
    import uvicorn

    from arietta_voice.web.app import create_app
    from arietta_voice.web.service import AriettaApiService

    patch = pytest.MonkeyPatch()
    patch.setenv("ARIETTA_ADMIN_USERNAME", "admin")
    patch.setenv("ARIETTA_ADMIN_PASSWORD", PASSWORD)
    try:
        config_path = _write_config(root)
        if memory:
            with config_path.open("a") as handle:
                handle.write("\n[memory]\nenabled = true\n")
        service = AriettaApiService(config_path)
        service.worker.handler_factory = _fixture_engine
        app = create_app(service=service)
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


def _fixture_engine(config):
    """Exercise the real worker and retrieval without downloading a model in CI."""
    from arietta_voice.inference import InferenceEngine
    from arietta_voice.llm import ModelEvent

    class MemoryModel:
        def stream(self, messages, **kwargs):
            user = (
                messages[0]["content"].partition("User said: ")[2].partition("\nAssistant said:")[0]
            )
            yield ModelEvent(
                "text", "- Prefers jasmine tea." if user == "I prefer jasmine tea." else "NONE"
            )
            yield ModelEvent("done")

    engine = InferenceEngine(config)
    engine.service()._llm = MemoryModel()

    def answer(messages, **kwargs):
        if messages[-1]["content"] == "I prefer jasmine tea.":
            return "Understood."
        if messages[-1]["content"] == "Which tea do I prefer?":
            system = " ".join(m["content"] for m in messages if m["role"] == "system").lower()
            return next(
                (
                    f"You prefer {tea} tea."
                    for tea in ("jasmine", "oolong")
                    if f"{tea} tea" in system
                ),
                "I don't know your tea preference.",
            )
        for message in messages:
            hit = re.search(
                r"\[(S\d+)\] [^\n]+: The spare key is in the ([^.]+)\.", message["content"]
            )
            if hit:
                return f"The spare key is in the {hit[2]} [{hit[1]}]."
        return "I couldn't find the spare key in your notes."

    engine.service()._generate_override = answer
    return engine


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


def test_phone_navigation_and_first_chat(console, page):
    page.set_viewport_size({"width": 375, "height": 800})
    page.goto(console)
    page.get_by_role("button", name="Open text chat").click()
    page.get_by_role("button", name="What time is it?").click()
    composer = page.get_by_role("textbox", name="Message", exact=True)
    assert composer.input_value() == "What time is it?"
    arithmetic = page.get_by_role("button", name="What is 17 times 23?")
    arithmetic.click()
    assert composer.input_value() == "What is 17 times 23?"
    composer.fill("My unfinished question")
    sync_api.expect(arithmetic).to_be_disabled()
    composer.fill("")
    page.get_by_role("button", name="What time is it?").click()
    picker = page.get_by_role("combobox", name="Console section")
    picker.select_option("settings")
    page.go_back()
    sync_api.expect(picker).to_have_value("chat")
    page.go_forward()
    sync_api.expect(picker).to_have_value("settings")
    page.go_back()
    sync_api.expect(picker).to_have_value("chat")
    assert composer.input_value() == "What time is it?"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    # Exercise the real instant tool through the streaming endpoint.
    page.get_by_role("button", name="Send", exact=True).click()
    page.locator("#chat-log .assistant .msg-meta").wait_for()
    assert "time" in page.locator("#chat-log .assistant .msg-body").inner_text().lower() or (
        "local_time" in page.locator("#chat-log .assistant .msg-meta").inner_text()
    )
    assert page.problems == []


def test_saving_one_editor_preserves_the_other_draft(console, page):
    page.goto(f"{console}/#/persona")
    sync_api.expect(page.locator("#persona-save")).to_be_enabled()
    page.fill("#persona-content", "A scratch persona.")
    page.get_by_text("Edit the full memory file", exact=True).click()
    page.fill("#memory-content", "An unfinished memory.")
    page.click("#persona-save")
    sync_api.expect(page.locator("#persona-save")).to_have_text("Save")
    assert page.input_value("#memory-content") == "An unfinished memory."
    assert page.locator("#memory-save").inner_text() == "Save changes"
    page.fill("#persona-content", "Another unfinished persona.")
    page.click("#memory-save")
    sync_api.expect(page.locator("#memory-save")).to_have_text("Save")
    assert page.input_value("#persona-content") == "Another unfinished persona."
    assert page.problems == []


def test_chat_cites_current_note_passage_after_edit_and_deletion(console, page):
    path = "household/entry & storage.md"
    url = f"{console}/api/knowledge/files/household/entry%20%26%20storage.md"
    question = "According to my notes, where is my spare key?"
    page.goto(f"{console}/#/chat")

    def ask():
        page.click("#chat-clear")
        page.fill("#chat-message", question)
        page.click("#chat-send")
        page.locator("#chat-log .assistant .msg-meta, #error-notice:visible").first.wait_for(
            timeout=120000
        )
        assert page.locator("#error-notice").is_hidden(), page.locator("#error-notice").inner_text()
        return page.locator("#chat-log .assistant .msg-body")

    old_link = None
    for location in ("blue drawer", "hall closet"):
        note = (
            "# House notes\n\n## Storage\nCamping chairs are in the shed.\n\n"
            f"## Entry\nThe spare key is in the {location}.\n"
        )
        assert page.request.put(url, data={"content": note}).ok
        reply = ask()
        assert location in reply.inner_text().lower()
        link = reply.get_by_role("link", name=re.compile("House notes.*Entry"))
        with page.expect_popup() as opened:
            link.click()
        reader = opened.value
        sync_api.expect(reader.get_by_role("heading", name="Entry")).to_be_visible()
        assert path in reader.locator("main").inner_text()
        assert reader.locator("blockquote").inner_text() == f"The spare key is in the {location}."
        if old_link:
            assert page.request.get(old_link).status == 404
        old_link = reader.url
        reader.close()

    assert page.request.delete(url).ok
    reply = ask()
    assert not reply.get_by_role("link").count()
    text = reply.inner_text().lower()
    assert "blue drawer" not in text and "hall closet" not in text
    assert re.search(r"not|couldn't|couldn’t|can't|can’t|no |don't|don’t|unable", text), text
    assert "notes" in text and "access" not in text, text
    assert page.request.get(old_link).status == 404
    assert page.problems == []


def test_learn_correct_forget_in_fresh_chats(memory_console, page):
    def fresh_chat(message):
        page.goto(f"{memory_console}/#/chat")
        page.reload()  # Discard all in-browser conversation history.
        assert page.locator("#chat-log .msg").count() == 0
        page.fill("#chat-message", message)
        page.click("#chat-send")
        page.locator("#chat-log .assistant .msg-meta").wait_for(timeout=90000)
        return page.locator("#chat-log .assistant .msg-body").inner_text().lower()

    fresh_chat("I prefer jasmine tea.")
    page.goto(f"{memory_console}/#/persona")
    fact = page.locator("#memory-entries .result-card").filter(has_text=re.compile("jasmine", re.I))
    sync_api.expect(fact).to_be_visible(timeout=60000)
    assert "Learned in Chat" in fact.inner_text()
    assert "jasmine" in fresh_chat("Which tea do I prefer?")

    page.goto(f"{memory_console}/#/persona")
    fact.get_by_role("button", name="Edit", exact=True).click()
    page.get_by_label("Correct saved fact").fill("Prefers oolong tea.")
    page.get_by_role("button", name="Save correction").click()
    updated = page.locator("#memory-entries .result-card").filter(has_text="oolong")
    sync_api.expect(updated).to_contain_text("corrected")
    answer = fresh_chat("Which tea do I prefer?")
    assert "oolong" in answer and "jasmine" not in answer

    page.goto(f"{memory_console}/#/persona")
    page.once("dialog", lambda dialog: dialog.accept())
    updated.get_by_role("button", name="Forget").click()
    sync_api.expect(page.locator("#memory-entries")).to_contain_text("No saved facts")
    answer = fresh_chat("Which tea do I prefer?")
    assert "oolong" not in answer and "jasmine" not in answer
    assert re.search(r"not|don.t|haven.t|no |unknown|which|tell me", answer), answer
    assert not page.request.get(f"{memory_console}/api/persona").json()["memory"]["entries"]
    assert page.problems == []
