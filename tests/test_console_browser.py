"""The console in a real browser: every page loads and works without an error.

The static checks in test_console_static.py catch broken imports and missing
element ids. These catch what only shows when the page runs: an exception while
a view renders, an API call that fails, an error banner. The console runs
against a scratch setup and is driven through Playwright in the Chrome that is
already installed (as on the macOS CI runners); without Chrome they are skipped,
except where ARIETTA_REQUIRE_BROWSER is set, as in CI.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

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


def _write_config(root: Path, *, indexed=False) -> Path:
    for name in ("config", "knowledge", "logs"):
        (root / name).mkdir()
    (root / "config" / "SOUL.md").write_text("You are a concise test assistant.")
    (root / "knowledge" / "food.md").write_text("# Food\nLunch is at the cafeteria.")
    config_path = root / "config" / "arietta_voice.toml"
    backend = "hybrid" if indexed else "keyword"
    config_path.write_text(
        f"""
[assistant]
name = "Browser Test"

[tools]
enabled = ["assistant_identity", "local_time", "set_reminder", "search_knowledge"]

[prompts]
soul_path = "SOUL.md"
memory_path = "MEMORY.md"

[logging]
directory = "../logs"
events_file = "../logs/events.jsonl"

[knowledge]
backend = "{backend}"
embedding_model = "synthetic-key"
index_directory = "../index"
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


@pytest.fixture(scope="module")
def reminder_console(tmp_path_factory):
    root = tmp_path_factory.mktemp("reminder-console")
    for url in _serve(root):
        yield url, root / "config" / "arietta_voice.toml"


def _serve(root, *, memory=False, indexed=False):
    import uvicorn

    from arietta_voice.web.app import create_app
    from arietta_voice.web.service import AriettaApiService

    patch = pytest.MonkeyPatch()
    patch.setenv("ARIETTA_ADMIN_USERNAME", "admin")
    patch.setenv("ARIETTA_ADMIN_PASSWORD", PASSWORD)
    try:
        config_path = _write_config(root, indexed=indexed)
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
    if config.knowledge.backend == "hybrid" and config.knowledge.embedding_model == "synthetic-key":
        engine.service().knowledge.backend.vector_backend._embedder = _KeyEmbedder()
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


@pytest.mark.parametrize("width", [375, 1440], ids=["phone", "desktop"])
def test_navigation_and_first_chat(console, page, width):
    page.set_viewport_size({"width": width, "height": 800})
    page.goto(console)
    page.get_by_role("button", name="Open text chat").click()
    send = page.get_by_role("button", name="Send", exact=True)
    sync_api.expect(send).to_be_disabled()
    page.get_by_role("button", name="What time is it?").click()
    composer = page.get_by_role("textbox", name="Message", exact=True)
    assert composer.input_value() == "What time is it?"
    arithmetic = page.get_by_role("button", name="What is 17 times 23?")
    arithmetic.click()
    assert composer.input_value() == "What is 17 times 23?"
    composer.fill("My unfinished question")
    sync_api.expect(arithmetic).to_be_disabled()
    composer.fill("   ")
    sync_api.expect(send).to_be_disabled()
    composer.fill("")
    page.get_by_role("button", name="What time is it?").click()
    if width == 375:
        page.get_by_role("combobox", name="Console section").select_option("settings")
    else:
        page.get_by_role("button", name="Settings", exact=True).click()
    page.go_back()
    sync_api.expect(page).to_have_url(f"{console}/#/chat")
    page.go_forward()
    sync_api.expect(page).to_have_url(f"{console}/#/settings")
    page.go_back()
    sync_api.expect(page).to_have_url(f"{console}/#/chat")
    assert composer.input_value() == "What time is it?"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    # Exercise the real instant tool through the streaming endpoint.
    send.click()
    page.locator("#chat-log .assistant .msg-meta").wait_for()
    sync_api.expect(send).to_be_disabled()
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


def _ask_spare_key(page):
    page.click("#chat-clear")
    page.fill("#chat-message", "According to my notes, where is my spare key?")
    page.click("#chat-send")
    page.locator("#chat-log .assistant .msg-meta, #error-notice:visible").first.wait_for(
        timeout=120000
    )
    assert page.locator("#error-notice").is_hidden(), page.locator("#error-notice").inner_text()
    return page.locator("#chat-log .assistant .msg-body")


def test_chat_cites_current_note_passage_after_edit_and_deletion(console, page):
    path = "household/entry & storage.md"
    url = f"{console}/api/knowledge/files/household/entry%20%26%20storage.md"
    page.goto(f"{console}/#/chat")

    old_link = None
    for location in ("blue drawer", "hall closet"):
        note = (
            "# House notes\n\n## Storage\nCamping chairs are in the shed.\n\n"
            f"## Entry\nThe spare key is in the {location}.\n"
        )
        assert page.request.put(url, data={"content": note}).ok
        reply = _ask_spare_key(page)
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
    reply = _ask_spare_key(page)
    assert not reply.get_by_role("link").count()
    text = reply.inner_text().lower()
    assert "blue drawer" not in text and "hall closet" not in text
    assert re.search(r"not|couldn't|couldn’t|can't|can’t|no |don't|don’t|unable", text), text
    assert "notes" in text and "access" not in text, text
    assert page.request.get(old_link).status == 404
    assert page.problems == []


class _KeyEmbedder:
    model_name = "synthetic-key"

    def embed_query(self, text):
        import numpy as np

        return np.array([1.0, 0.0] if "key" in text.lower() else [0.0, 1.0], dtype=np.float32)

    def embed_documents(self, texts):
        import numpy as np

        return np.stack([self.embed_query(text) for text in texts])


@pytest.fixture
def indexed_console(tmp_path):
    yield from _serve(tmp_path, indexed=True)


def test_chat_remains_available_during_rebuild_and_rejects_deleted_passage(
    indexed_console, page, monkeypatch
):
    url = indexed_console
    index_url = f"{url}/api/knowledge/index"
    note_url = f"{url}/api/knowledge/files/key.md"
    started, resume = threading.Event(), threading.Event()

    class PausedEmbedder(_KeyEmbedder):
        def embed_documents(self, texts):
            started.set()
            assert resume.wait(90), "test did not release the index builder"
            return super().embed_documents(texts)

    def save(location):
        content = f"# Home\n\n## Entry\nThe spare key is in the {location}."
        assert page.request.put(note_url, data={"content": content}).ok

    def check_answer(location):
        body = _ask_spare_key(page)
        assert location in body.inner_text()
        link = body.get_by_role("link", name=re.compile("Home.*Entry"))
        passage = page.request.get(link.evaluate("a => a.href"))
        assert passage.ok and f"The spare key is in the {location}." in passage.text()
        return passage.url

    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda config: _KeyEmbedder()
    )
    page.goto(f"{url}/#/chat")
    save("blue drawer")
    assert page.request.post(index_url).json()["updated"]
    original = check_answer("blue drawer")
    save("hall closet")
    monkeypatch.setattr(
        "arietta_voice.knowledge._create_text_embedder", lambda config: PausedEmbedder()
    )
    # A second tab starts a build while the first keeps chatting.
    builder = page.context.new_page()
    builder.goto(f"{url}/#/knowledge")
    rebuild = builder.get_by_role("button", name="Rebuild index")

    @contextmanager
    def building():
        started.clear()
        resume.clear()
        with builder.expect_response(index_url, timeout=120000) as response:
            try:
                rebuild.click()
                assert started.wait(15)
                yield response
            finally:
                resume.set()

    with building() as build:
        assert page.request.get(original).status == 404
        edited = check_answer("hall closet")
    assert build.value.json()["updated"]
    sync_api.expect(builder.locator("#toasts")).to_contain_text("Index rebuilt · 2 sections")
    assert check_answer("hall closet") == edited

    # Another build has read the key note when the owner deletes it.
    assert page.request.put(
        f"{url}/api/knowledge/files/weather.md", data={"content": "# Weather\nRain today."}
    ).ok
    with building() as build:
        assert page.request.delete(note_url).ok
        assert page.request.get(edited).status == 404
    assert build.value.status == 409
    sync_api.expect(builder.locator("#error-message")).to_contain_text("Notes changed")
    builder.close()
    body = _ask_spare_key(page)
    assert "couldn't find" in body.inner_text()
    assert not body.get_by_role("link").count()
    assert page.request.post(index_url).json()["updated"]
    assert page.request.get(edited).status == 404
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


def _silent_reminder_runtime(config_path, connection):
    """Run the real quiet listening loop up to its next microphone read on each tick."""
    from arietta_voice.config import load_config
    from arietta_voice.runtime import AriettaVoiceRuntime

    config = load_config(config_path)
    config.audio.tts_enabled = config.audio.chime_enabled = False
    runtime = AriettaVoiceRuntime(config)
    microphone = Mock(dropped_chunks=0, input_overflows=0)
    microphone.get.side_effect = StopIteration
    spoken = []
    runtime._print_response = spoken.append
    try:
        while (now := connection.recv()) is not None:
            runtime.alerts.clock = runtime.alerts.store.clock = lambda: now
            spoken.clear()
            with pytest.raises(StopIteration):
                runtime._listen(microphone, Mock())
            connection.send(spoken)
    finally:
        runtime.memory_jobs.close()
        runtime.worker.close()
        connection.close()


@contextmanager
def _reminder_runtime(config_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_silent_reminder_runtime, args=(config_path, child))
    process.start()
    child.close()

    def tick(now):
        parent.send(now)
        assert parent.poll(20), "The reminder runtime did not finish its clock tick"
        return parent.recv()

    try:
        yield tick
    finally:
        try:
            if process.is_alive():
                parent.send(None)
        finally:
            parent.close()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            assert process.exitcode == 0
            process.close()


def test_chat_reminder_fires_once_and_stays_dismissed_across_restarts(reminder_console, page):
    url, config_path = reminder_console
    label = "water the test fern"

    def state():
        response = page.request.get(f"{url}/api/timers")
        assert response.ok
        return response.json()

    with _reminder_runtime(config_path) as tick:
        assert tick(time.time()) == []
        page.goto(f"{url}/#/chat")
        page.fill("#chat-message", f"Remind me in ten minutes to {label}.")
        page.click("#chat-send")
        sync_api.expect(page.locator("#chat-log .assistant .msg-body")).to_contain_text(label)
        page.get_by_role("button", name="Home", exact=True).click()
        row = page.locator("#timers-list li").filter(has_text=label)
        sync_api.expect(row.get_by_role("button", name="Cancel")).to_be_visible()
        (pending,) = state()["active"]
        assert (pending["kind"], pending["label"], pending["status"], pending["rings"]) == (
            "reminder",
            label,
            "pending",
            0,
        )
        due = pending["due_at"]
        assert tick(due - 1) == []

    with _reminder_runtime(config_path) as tick:
        assert tick(due - 1) == []
        assert tick(due) == [f"It's time to {label}."]
        assert tick(due + 1) == []
        (ringing,) = state()["active"]
        assert (ringing["id"], ringing["status"], ringing["rings"]) == (pending["id"], "ringing", 1)
        page.reload()
        row.get_by_role("button", name="Stop", exact=True).click()
        sync_api.expect(row).to_have_count(0)
        assert tick(due + 3600) == []

    with _reminder_runtime(config_path) as tick:
        assert tick(due + 7200) == []
        page.reload()
        page.locator("#timers-finished summary").click()
        earlier = page.locator("#timers-finished-list li").filter(has_text=label)
        sync_api.expect(earlier).to_contain_text("Stopped")
        final = state()
        assert final["active"] == []
        (finished,) = final["finished"]
        assert (finished["id"], finished["status"], finished["outcome"], finished["rings"]) == (
            pending["id"],
            "done",
            "dismissed",
            1,
        )
    assert page.problems == []
