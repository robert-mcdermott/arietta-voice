from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arietta_voice import __version__
from arietta_voice.api import AriettaApiService, create_app
from arietta_voice.config import load_config
from arietta_voice.runtime_status import runtime_status_path


@pytest.fixture(autouse=True)
def explicit_admin_credentials(monkeypatch):
    monkeypatch.setenv("ARIETTA_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ARIETTA_ADMIN_PASSWORD", "test-only-long-password")
    monkeypatch.delenv("ARIETTA_ADMIN_ALLOWED_HOSTS", raising=False)


def _write_api_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    knowledge_dir = tmp_path / "knowledge"
    logs_dir = tmp_path / "logs"
    config_dir.mkdir()
    knowledge_dir.mkdir()
    logs_dir.mkdir()

    (config_dir / "SOUL.md").write_text("You are a concise test assistant.", encoding="utf-8")
    (knowledge_dir / "food.md").write_text(
        """\
---
title: Food and Dining
tags: food, lunch, cafe
---

## On-Site Options
Visitors can get lunch at the cafeteria and coffee at the cafe.
""",
        encoding="utf-8",
    )

    config_path = config_dir / "arietta_voice.toml"
    config_path.write_text(
        """
[assistant]
name = "Test Assistant"
identity = "I am Test Assistant, an Arietta Voice API test assistant."

[tools]
enabled = ["assistant_identity"]

[model_tools]
enabled = []

[prompts]
soul_path = "SOUL.md"
memory_path = "MEMORY.md"

[logging]
directory = "../logs"
events_file = "../logs/events.jsonl"

[knowledge]
backend = "keyword"
directory = "../knowledge"
min_score = 1.0
""".strip(),
        encoding="utf-8",
    )
    return config_path


def _client(config_path: Path) -> TestClient:
    service = AriettaApiService(config_path)
    return TestClient(create_app(service=service), base_url="http://localhost")


class FakeRuntimeSupervisor:
    def __init__(self) -> None:
        self.started = False

    def status(self):
        return {
            "managed_by_api": self.started,
            "managed_process_id": 111 if self.started else None,
            "managed_process_running": self.started,
            "managed_started_at": "2026-01-01T00:00:00Z" if self.started else None,
            "managed_exit_code": None,
            "managed_command": ["fake"],
            "managed_stdout_log": "/tmp/fake.out",
            "managed_stderr_log": "/tmp/fake.err",
            "managed_last_error": None,
        }

    def start(self):
        self.started = True
        return {"changed": True, "message": "started", **self.status()}

    def stop(self):
        self.started = False
        return {"changed": True, "message": "stopped", **self.status()}

    def restart(self):
        self.started = True
        return {"changed": True, "message": "restarted", **self.status()}


def _client_with_service(config_path: Path) -> tuple[TestClient, AriettaApiService]:
    service = AriettaApiService(config_path)
    from dataclasses import asdict
    from unittest.mock import Mock

    from arietta_voice.api import ChatRequest

    def in_process(operation, payload, **kwargs):
        assert operation == "chat"
        payload.pop("enabled_tools")
        payload.pop("enabled_model_tools", None)
        return asdict(service._chat_in_process(ChatRequest(**payload), record=False))

    service.worker.close()
    service.worker = Mock(call=in_process, pid=None, restarts=0, operation=None)
    client = TestClient(create_app(service=service), base_url="http://localhost")
    return client, service


def test_api_refuses_to_start_without_admin_setup(tmp_path, monkeypatch):
    monkeypatch.delenv("ARIETTA_ADMIN_PASSWORD")
    with pytest.raises(ValueError, match="admin-setup"):
        create_app(_write_api_config(tmp_path))


def test_legacy_default_password_is_not_accepted(tmp_path):
    client = _client(_write_api_config(tmp_path))
    response = client.get("/api/config", auth=("admin", "password"))
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"


def test_api_loads_private_credentials_and_source_permission(tmp_path, monkeypatch):
    from arietta_voice.admin_auth import save_admin_credentials
    from arietta_voice.private_config import update_private_section

    config_path = _write_api_config(tmp_path)
    monkeypatch.delenv("ARIETTA_ADMIN_PASSWORD")
    save_admin_credentials(config_path, "owner", "private-login-password")
    update_private_section(config_path, "admin", {"allow_source_editing": True})
    client = _client(config_path)
    auth = ("owner", "private-login-password")
    assert client.get("/api/tools", auth=auth).json()["source_editing_enabled"] is True
    payload = client.get("/api/config", auth=auth)
    assert payload.status_code == 200
    assert "private-login-password" not in payload.text
    assert "password_hash" not in payload.text


@pytest.mark.parametrize(
    "method, path, payload",
    [
        ("post", "/api/tools/create", {"name": "blocked"}),
        (
            "put",
            "/api/tools/local_time/source",
            {"content": "raise RuntimeError('should not execute')"},
        ),
    ],
)
def test_source_mutations_are_disabled_by_default(tmp_path, monkeypatch, method, path, payload):
    client, service = _client_with_service(_write_api_config(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("Source mutation reached the service while disabled")

    for name in ("create_tool", "write_tool_source"):
        monkeypatch.setattr(service, name, forbidden)
    assert client.request(method, path, json=payload).status_code == 401
    result = client.request(method, path, json=payload, auth=("admin", "test-only-long-password"))
    assert result.status_code == 403
    assert "Source editing is disabled" in result.json()["detail"]


def test_browser_boundaries_and_no_store(tmp_path):
    client = _client(_write_api_config(tmp_path))
    auth = ("admin", "test-only-long-password")
    assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 400
    for headers in (
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
    ):
        response = client.post(
            "/api/tools/enabled", json={"enabled": []}, auth=auth, headers=headers
        )
        assert response.status_code == 403
    response = client.post(
        "/api/tools/enabled",
        json={"enabled": []},
        auth=auth,
        headers={"Origin": "http://localhost"},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert client.get("/api/tools/instructions", auth=auth).status_code == 405


def test_static_inventory_and_source_reads_do_not_execute_plugins(tmp_path):
    client, service = _client_with_service(_write_api_config(tmp_path))
    marker = tmp_path / "executed"
    tools_dir = tmp_path / "plugins"
    tools_dir.mkdir()
    (tools_dir / "side_effect.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "class Tool:\n    name = 'side_effect'\n    description = 'never'\n"
        "def create_tool(context):\n    raise RuntimeError('factory executed')\n"
    )
    service.config.tools.directory = tools_dir
    service.config.tools.enabled = ("assistant_identity", "side_effect")
    auth = ("admin", "test-only-long-password")
    for path in ("/api/state", "/api/tools", "/api/tools/side_effect/source"):
        assert client.get(path, auth=auth).status_code == 200
        assert not marker.exists()
    inventory = client.get("/api/tools", auth=auth).json()
    record = next(tool for tool in inventory["tools"] if tool["name"] == "side_effect")
    assert record["origin"] == "user" and record["description"] == "never"
    assert all(tool["validation"] == "not_run" for tool in inventory["tools"])
    # Validation is an explicit authenticated action that is allowed to import.
    validated = client.post("/api/tools/side_effect/validate", auth=auth).json()
    assert validated["validation"] == "failed" and "factory executed" in validated["error"]
    assert marker.read_text() == "imported"


def test_health_is_public_and_contains_only_liveness(tmp_path: Path, monkeypatch) -> None:
    client, service = _client_with_service(_write_api_config(tmp_path))
    runtime_status_path(service.config).write_text(
        '{"session_active":true,"last_user_transcript":"PRIVATE TRANSCRIPT"}',
        encoding="utf-8",
    )
    service.last_error = "PRIVATE ERROR"

    def unexpected_diagnostics():
        raise AssertionError("Public liveness must not inspect private diagnostics")

    monkeypatch.setattr(service, "health", unexpected_diagnostics)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "api_version": __version__}
    assert client.app.version == __version__


def test_detailed_health_requires_authentication(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))
    assert client.get("/api/health").status_code == 401
    response = client.get("/api/health", auth=("admin", "test-only-long-password"))
    assert response.status_code == 200
    assert response.json()["assistant"] == "Test Assistant"
    assert "diagnostics" in response.json()


def test_chat_without_tools_does_not_load_or_execute_plugins(tmp_path: Path, monkeypatch) -> None:

    client, service = _client_with_service(_write_api_config(tmp_path))
    service.config.model_tools.enabled = ("missing_private_plugin",)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled tools must not load or execute")

    def generate(messages, **kwargs):
        calls.append(messages)
        return '{"tool_call":{"name":"forbidden","arguments":{}}}'

    monkeypatch.setattr(service, "probe_tools", forbidden)
    from arietta_voice import tools

    monkeypatch.setattr(tools, "ToolRegistry", forbidden)
    monkeypatch.setattr(service, "_llm_generate", generate)
    response = client.post(
        "/api/chat",
        auth=("admin", "test-only-long-password"),
        json={"message": "What is your name?", "use_tools": False},
    )
    assert response.status_code == 200
    assert response.json()["used_tool"] is None
    assert response.json()["model_tool"] is None
    assert len(calls) == 1
    assert calls[0][-1] == {"role": "user", "content": "What is your name?"}


def test_admin_console_requires_basic_auth(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    unauthenticated = client.get("/")
    authenticated = client.get("/", auth=("admin", "test-only-long-password"))

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert "Arietta Voice" in authenticated.text


def test_console_assets_are_authenticated_and_contained(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))
    auth = ("admin", "test-only-long-password")

    assert client.get("/static/js/main.js").status_code == 401

    page = client.get("/admin", auth=auth)
    assert page.headers["content-type"].startswith("text/html")
    policy = page.headers["content-security-policy"]
    assert "script-src 'self'" in policy and "unsafe-inline" not in policy
    # The policy only holds if the page carries no inline script or style.
    assert "<style" not in page.text and " style=" not in page.text
    assert all("src=" in tag for tag in page.text.split("<script")[1:])

    script = client.get("/static/js/main.js", auth=auth)
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert (
        client.get("/static/css/console.css", auth=auth)
        .headers["content-type"]
        .startswith("text/css")
    )

    for escape in (
        "/static/../app.py",
        "/static/%2e%2e/app.py",
        "/static/..%2fapp.py",
        "/static/js/../../service.py",
        "/static/js",
        "/static/missing.js",
    ):
        assert client.get(escape, auth=auth).status_code == 404, escape


def test_console_page_references_only_packaged_assets(tmp_path: Path) -> None:
    import re

    from arietta_voice.web.app import STATIC_DIR, STATIC_MEDIA_TYPES

    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    references = re.findall(r'(?:src|href)="(/static/[^"]+)"', page)
    assert references
    for reference in references:
        assert (STATIC_DIR / reference.removeprefix("/static/")).is_file(), reference
    for script in (STATIC_DIR / "js").glob("*.js"):
        for imported in re.findall(r'from "\./([^"]+)"', script.read_text(encoding="utf-8")):
            assert (STATIC_DIR / "js" / imported).is_file(), f"{script.name} imports {imported}"
    assert {path.suffix for path in STATIC_DIR.rglob("*") if path.is_file()} <= set(
        STATIC_MEDIA_TYPES
    )


def test_state_and_config_are_authenticated(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    unauthorized = client.get("/api/state")
    authorized = client.get("/api/state", auth=("admin", "test-only-long-password"))
    config_response = client.get("/api/config", auth=("admin", "test-only-long-password"))

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["api"]["version"] == __version__
    assert authorized.json()["runtime"]["mode"] == "web_api"
    assert authorized.json()["runtime"]["state"] == "not_running"
    assert config_response.json()["assistant"]["name"] == "Test Assistant"


def test_runtime_endpoint_reports_unmanaged_runtime_status(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.get("/api/runtime", auth=("admin", "test-only-long-password"))

    assert response.status_code == 200
    assert response.json()["managed_by_api"] is False
    assert response.json()["state"] == "not_running"


def test_runtime_control_endpoints_delegate_to_supervisor(tmp_path: Path) -> None:
    client, service = _client_with_service(_write_api_config(tmp_path))
    service.runtime_supervisor = FakeRuntimeSupervisor()

    start = client.post("/api/runtime/start", auth=("admin", "test-only-long-password"))
    state = client.get("/api/state", auth=("admin", "test-only-long-password"))
    stop = client.post("/api/runtime/stop", auth=("admin", "test-only-long-password"))

    assert start.status_code == 200
    assert start.json()["managed_process_running"] is True
    assert state.json()["runtime"]["managed_process_running"] is True
    assert stop.status_code == 200
    assert stop.json()["managed_process_running"] is False


def test_knowledge_search_endpoint(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.post(
        "/api/knowledge/search",
        auth=("admin", "test-only-long-password"),
        json={"query": "where can visitors get lunch"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["matches"]
    assert payload["matches"][0]["title"] == "Food and Dining"


def test_knowledge_file_crud_endpoints(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    created = client.put(
        "/api/knowledge/files/new-note.md",
        auth=("admin", "test-only-long-password"),
        json={"content": "# New Note\n\nThe dashboard can edit knowledge."},
    )
    listed = client.get("/api/knowledge/files", auth=("admin", "test-only-long-password"))
    read = client.get("/api/knowledge/files/new-note.md", auth=("admin", "test-only-long-password"))
    deleted = client.delete(
        "/api/knowledge/files/new-note.md", auth=("admin", "test-only-long-password")
    )

    assert created.status_code == 200
    assert created.json()["created"] is True
    assert any(file["path"] == "new-note.md" for file in listed.json()["files"])
    assert "dashboard can edit" in read.json()["content"]
    assert deleted.json()["deleted"] is True


def test_knowledge_file_endpoint_rejects_path_traversal(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.put(
        "/api/knowledge/files/%2E%2E/secret.md",
        auth=("admin", "test-only-long-password"),
        json={"content": "nope"},
    )

    assert response.status_code == 400


def test_logs_endpoint_reads_latest_events(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    events_path = tmp_path / "logs" / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                '{"ts":"2026-01-01T00:00:00Z","event":"older"}',
                '{"ts":"2026-01-01T00:00:01Z","event":"newer"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = _client(config_path)

    response = client.get("/api/logs?limit=1", auth=("admin", "test-only-long-password"))

    assert response.status_code == 200
    assert response.json()["items"][0]["event"] == "newer"
    assert response.json()["total"] == 2
    assert response.json()["has_next"] is True


def test_logs_endpoint_paginates_and_filters(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    events_path = tmp_path / "logs" / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                '{"ts":"2026-01-01T00:00:00Z","event":"alpha","text":"first apple"}',
                '{"ts":"2026-01-01T00:00:01Z","event":"beta","text":"second banana"}',
                '{"ts":"2026-01-01T00:00:02Z","event":"alpha","text":"third apricot"}',
                '{"ts":"2026-01-01T00:00:03Z","event":"gamma","text":"fourth grape"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = _client(config_path)

    first_page = client.get("/api/logs?limit=2", auth=("admin", "test-only-long-password"))
    second_page = client.get(
        "/api/logs?limit=2&offset=2", auth=("admin", "test-only-long-password")
    )
    event_filtered = client.get(
        "/api/logs?limit=10&event=alpha", auth=("admin", "test-only-long-password")
    )
    search_filtered = client.get(
        "/api/logs?limit=10&q=banana", auth=("admin", "test-only-long-password")
    )

    assert [item["event"] for item in first_page.json()["items"]] == ["gamma", "alpha"]
    assert first_page.json()["has_next"] is True
    assert first_page.json()["has_previous"] is False
    assert [item["event"] for item in second_page.json()["items"]] == ["beta", "alpha"]
    assert second_page.json()["has_previous"] is True
    assert event_filtered.json()["total"] == 2
    assert [item["text"] for item in event_filtered.json()["items"]] == [
        "third apricot",
        "first apple",
    ]
    assert search_filtered.json()["total"] == 1
    assert search_filtered.json()["items"][0]["event"] == "beta"


def test_chat_endpoint_can_answer_with_deterministic_tool(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.post(
        "/api/chat",
        auth=("admin", "test-only-long-password"),
        json={"message": "What is your name?"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["source"] == "tool"
    assert payload["used_tool"] == "assistant_identity"
    assert "Test Assistant" in payload["response"]


AUTH = ("admin", "test-only-long-password")


def _editable_client(tmp_path):
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    service.config.admin.allow_source_editing = True
    service.config.tools.directory = tmp_path / "my_tools"
    return config_path, client, service


def test_tools_endpoint_lists_one_inventory(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))
    payload = client.get("/api/tools", auth=AUTH).json()
    records = {tool["name"]: tool for tool in payload["tools"]}
    assert payload["enabled"] == ["assistant_identity"]
    assert payload["errors"] == []
    assert records["assistant_identity"]["enabled"] is True
    assert records["assistant_identity"]["origin"] == "builtin"
    assert records["assistant_identity"]["model_callable"] is False
    assert records["local_time"]["enabled"] is False
    assert records["local_time"]["triggers"]
    assert records["search_knowledge"]["parameters"]["required"] == ["query"]
    validated = client.post("/api/tools/assistant_identity/validate", auth=AUTH).json()
    assert validated["validation"] == "valid" and validated["enabled"] is True


def test_tools_endpoint_reports_broken_user_tools(tmp_path: Path) -> None:
    client, service = _client_with_service(_write_api_config(tmp_path))
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "broken_tool.py").write_text("BROKEN = True\n", encoding="utf-8")
    service.config.tools.directory = tools_dir
    service.config.tools.enabled = ("assistant_identity", "broken_tool")
    payload = client.get("/api/tools", auth=AUTH).json()
    record = next(tool for tool in payload["tools"] if tool["name"] == "broken_tool")
    assert record["origin"] == "user" and "Could not read" in record["error"]
    assert payload["errors"][0]["name"] == "broken_tool"


def test_tools_enabled_endpoint_persists_one_ordered_list(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    original = config_path.read_bytes()
    client = _client(config_path)
    response = client.post(
        "/api/tools/enabled", auth=AUTH, json={"enabled": ["local_time", "assistant_identity"]}
    )
    assert response.status_code == 200
    assert response.json()["enabled"] == ["local_time", "assistant_identity"]
    saved = load_config(config_path)
    assert saved.tools.enabled == ("local_time", "assistant_identity")
    assert saved.model_tools.enabled == ()
    assert config_path.read_bytes() == original
    missing = client.post("/api/tools/enabled", auth=AUTH, json={"enabled": ["nope"]})
    assert missing.status_code == 400


def test_a_disabled_tool_can_be_tested_by_request_or_arguments(tmp_path: Path, monkeypatch) -> None:
    from arietta_voice.tools.builtin import system_info

    monkeypatch.setattr(system_info, "build_system_report", lambda root: "Synthetic Mac")
    client = _client(_write_api_config(tmp_path))
    by_text = client.post(
        "/api/tools/local_time/test", auth=AUTH, json={"text": "What time is it?"}
    )
    assert by_text.status_code == 200
    assert by_text.json()["matched"] is True
    assert by_text.json()["result"]["speak"].startswith("It is ")
    no_match = client.post("/api/tools/local_time/test", auth=AUTH, json={"text": "Hello"})
    assert no_match.json()["matched"] is False
    by_arguments = client.post("/api/tools/system_info/test", auth=AUTH, json={"arguments": {}})
    assert "Synthetic Mac" in by_arguments.json()["result"]["content"]
    invalid = client.post(
        "/api/tools/system_info/test", auth=AUTH, json={"arguments": {"surprise": 1}}
    )
    assert invalid.json()["error"]["kind"] == "invalid_arguments"
    both = client.post("/api/tools/local_time/test", auth=AUTH, json={"text": "x", "arguments": {}})
    assert both.status_code == 400


def test_probe_reports_which_enabled_tool_answers(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))
    hit = client.post("/api/tools/probe", auth=AUTH, json={"text": "Who are you?"}).json()
    assert hit["matched"] and hit["name"] == "assistant_identity" and hit["answers_directly"]
    miss = client.post("/api/tools/probe", auth=AUTH, json={"text": "What time is it?"}).json()
    assert miss["matched"] is False  # local_time is not enabled in this config.


def test_create_writes_a_user_tool_outside_the_package(tmp_path: Path) -> None:
    config_path, client, service = _editable_client(tmp_path)
    created = client.post(
        "/api/tools/create",
        auth=AUTH,
        json={
            "name": "Demo Tool",
            "description": "the user asks for a banana checkout",
            "trigger": r"\bbanana checkout\b",
            "response": "Banana checkout handled.",
        },
    )
    assert created.status_code == 200
    assert created.json()["name"] == "demo_tool"
    assert (tmp_path / "my_tools" / "demo_tool.py").exists()
    assert created.json()["tool"]["validation"] == "valid"
    assert "demo_tool" in load_config(config_path).tools.enabled
    probe = client.post("/api/tools/probe", auth=AUTH, json={"text": "please banana checkout"})
    assert probe.json()["name"] == "demo_tool"
    assert probe.json()["result"]["speak"] == "Banana checkout handled."
    duplicate = client.post("/api/tools/create", auth=AUTH, json={"name": "demo_tool"})
    assert duplicate.status_code == 409
    builtin = client.post("/api/tools/create", auth=AUTH, json={"name": "local_time"})
    assert builtin.status_code == 409


def test_source_endpoints_edit_user_tools_and_keep_builtins_read_only(tmp_path: Path) -> None:
    _, client, _ = _editable_client(tmp_path)
    client.post(
        "/api/tools/create",
        auth=AUTH,
        json={"name": "source_demo", "trigger": "source demo", "response": "Original."},
    )
    read = client.get("/api/tools/source_demo/source", auth=AUTH).json()
    assert read["editable"] is True and read["file"].endswith("source_demo.py")
    saved = client.put(
        "/api/tools/source_demo/source",
        auth=AUTH,
        json={"content": read["content"].replace("Original.", "Edited.")},
    )
    assert saved.status_code == 200 and saved.json()["tool"]["validation"] == "valid"
    tested = client.post("/api/tools/source_demo/test", auth=AUTH, json={"text": "source demo"})
    assert tested.json()["result"]["speak"] == "Edited."
    builtin = client.get("/api/tools/local_time/source", auth=AUTH).json()
    assert builtin["editable"] is False and "class LocalTimeTool" in builtin["content"]
    blocked = client.put(
        "/api/tools/local_time/source", auth=AUTH, json={"content": "raise SystemExit"}
    )
    assert blocked.status_code == 400
    assert client.get("/api/tools/..%2Fescape/source", auth=AUTH).status_code in (400, 404)


def test_instructions_endpoint_shows_what_the_model_is_told(tmp_path: Path) -> None:
    client, service = _client_with_service(_write_api_config(tmp_path))
    service.config.tools.enabled = ("assistant_identity", "system_info")
    payload = client.post("/api/tools/instructions", auth=AUTH).json()
    assert payload["enabled"] == ["assistant_identity", "system_info"]
    assert "- name: system_info" in payload["instruction_text"]
    assert "assistant_identity" not in payload["instruction_text"]


@pytest.mark.parametrize(
    "failure, code", [("WorkerTimeout", 504), ("WorkerBusy", 503), ("WorkerFailure", 502)]
)
def test_chat_failure_has_safe_actionable_error_and_can_recover(tmp_path, failure, code):
    from unittest.mock import Mock

    from arietta_voice import worker

    client, service = _client_with_service(_write_api_config(tmp_path))
    service.worker.call = Mock(
        side_effect=[
            getattr(worker, failure)("PRIVATE failure details"),
            {"response": "Recovered", "source": "model", "used_tool": None, "model_tool": None},
        ]
    )
    auth = ("admin", "test-only-long-password")
    response = client.post("/api/chat", auth=auth, json={"message": "hello"})
    assert response.status_code == code
    assert "PRIVATE" not in response.text
    assert "PRIVATE" not in service.config.logging.events_file.read_text()
    assert service.chat_turns == 0
    response = client.post("/api/chat", auth=auth, json={"message": "new request"})
    assert response.status_code == 200
    assert service.chat_turns == 1
    assert service.last_error is None


def test_cancel_route_requires_auth_and_remembers_early_stop(tmp_path):
    client, service = _client_with_service(_write_api_config(tmp_path))
    auth = ("admin", "test-only-long-password")
    assert client.post("/api/chat/turn-one/cancel").status_code == 401
    assert client.post("/api/chat/turn-one/cancel", auth=auth).json() == {"cancelled": True}
    result = client.post(
        "/api/chat", auth=auth, json={"message": "Synthetic", "request_id": "turn-one"}
    )
    assert result.status_code == 409
    assert service.chat_turns == 0
    assert client.post("/api/chat/turn-one/cancel", auth=auth).json() == {"cancelled": False}
    assert (
        client.post(
            "/api/chat", auth=auth, json={"message": "What is your name?", "request_id": "turn-two"}
        ).status_code
        == 200
    )


def test_cancel_active_chat_suppresses_late_result_and_preserves_new_turn(tmp_path):
    import threading
    from unittest.mock import Mock

    from arietta_voice.api import ChatRequest
    from arietta_voice.worker import WorkerCancelled

    _, service = _client_with_service(_write_api_config(tmp_path))
    entered, release, errors = threading.Event(), threading.Event(), []

    def late_result(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        # Deliberately ignore the token here; service must still reject stale output.
        return dict(response="Late answer", source="model", used_tool=None, model_tool=None)

    service.worker = Mock(call=late_result, pid=None)

    def run():
        try:
            service.chat(ChatRequest(message="Synthetic", request_id="old"))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(2)
    assert service.cancel_chat("old")
    release.set()
    thread.join(2)
    assert not thread.is_alive() and isinstance(errors[0], WorkerCancelled)
    assert service.chat_turns == 0
    service.worker.call = lambda *a, **kw: dict(
        response="New", source="model", used_tool=None, model_tool=None
    )
    assert service.chat(ChatRequest(message="New", request_id="new")).response == "New"
    assert not service.cancel_chat("old")
    assert service.chat_turns == 1


def test_timer_endpoints_list_add_stop_and_cancel(tmp_path: Path) -> None:
    from arietta_voice.timers import AlertScheduler, TimerStore

    client, service = _client_with_service(_write_api_config(tmp_path))
    assert client.get("/api/timers").status_code == 401
    empty = client.get("/api/timers", auth=AUTH).json()
    assert empty["active"] == [] and empty["finished"] == []
    assert empty["runtime_running"] is False
    assert empty["tools_enabled"] == []  # This config predates the timer tools.

    added = client.post(
        "/api/timers", auth=AUTH, json={"kind": "timer", "duration": "ten minutes", "label": "tea"}
    ).json()
    assert added["added"]["name"] == "tea timer"
    assert added["added"]["duration_s"] == 600
    reminder = client.post(
        "/api/timers",
        auth=AUTH,
        json={"kind": "reminder", "when": "in 2 hours", "label": "to call my sister"},
    ).json()
    assert reminder["added"]["name"] == "to call your sister"
    assert [alert["kind"] for alert in reminder["active"]] == ["timer", "reminder"]
    assert (tmp_path / "state" / "timers.json").exists()

    for body, message in (
        ({"kind": "timer", "duration": "a while"}, "Say how long"),
        ({"kind": "reminder", "when": "someday", "label": "x"}, "Say when"),
        ({"kind": "timer", "duration": "48 hours"}, "24 hours"),
    ):
        response = client.post("/api/timers", auth=AUTH, json=body)
        assert response.status_code == 400 and message in response.json()["detail"]

    store = TimerStore(service.config.timers.path)
    store.mutate(lambda alerts, now: [setattr(alerts[0], "due_at", now - 1)])
    alerts = AlertScheduler(store)
    alerts.rang(alerts.due())
    assert client.get("/api/timers", auth=AUTH).json()["active"][0]["status"] == "ringing"
    stopped = client.post("/api/timers/stop", auth=AUTH).json()
    assert stopped["stopped"] == [added["added"]["id"]]
    assert stopped["finished"][0]["outcome"] == "dismissed"

    reminder_id = reminder["added"]["id"]
    cancelled = client.post(f"/api/timers/{reminder_id}/cancel", auth=AUTH).json()
    assert cancelled["active"] == []
    assert client.post(f"/api/timers/{reminder_id}/cancel", auth=AUTH).status_code == 404
    assert client.post("/api/timers/..%2Fx/cancel", auth=AUTH).status_code in (404, 422)


def _streaming_client(tmp_path, *, fail_at=None):
    """A client whose worker streams the in-process conversation, like the real one."""
    from unittest.mock import Mock

    client, service = _client_with_service(_write_api_config(tmp_path))
    state = {}

    def call(operation, payload=None, **kwargs):
        token = kwargs.get("cancellation")
        if token is not None:
            token.check()
        if operation == "chat_start":
            payload = dict(payload)
            payload.pop("enabled_tools")
            service._conversation._generate_override = lambda *a, **kw: "Synthetic answer. Two."
            state["stream"] = service._conversation.stream(**payload)
            state["steps"] = 0
            return None
        if operation == "chat_next":
            state["steps"] += 1
            if fail_at == state["steps"]:
                raise RuntimeError("synthetic failure")
            return next(state["stream"])
        if operation == "chat_close":
            state["closed"] = True
            return None
        raise AssertionError(operation)

    service.worker = Mock(call=call, pid=123, restarts=0, operation=None)
    return client, service, state


def _events(response):
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]


def test_chat_streams_clauses_then_the_result(tmp_path):
    client, service, state = _streaming_client(tmp_path)
    response = client.post("/api/chat/stream", auth=AUTH, json={"message": "Hello there"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)
    assert [e["kind"] for e in events] == ["clause", "clause", "done"]
    assert [e["text"] for e in events[:2]] == ["Synthetic answer.", "Two."]
    assert events[-1]["result"]["response"] == "Synthetic answer. Two."
    assert state["closed"] and service._active_chat is None
    assert service.chat_turns == 1


def test_chat_stream_requires_auth_and_reports_failures(tmp_path):
    client, service, _ = _streaming_client(tmp_path, fail_at=2)
    assert client.post("/api/chat/stream", json={"message": "Hi"}).status_code == 401
    events = _events(client.post("/api/chat/stream", auth=AUTH, json={"message": "Hi"}))
    assert events[-1] == {
        "kind": "error",
        "status": 502,
        "detail": "The model or tool failed. The next request will use a fresh worker.",
    }
    assert service._active_chat is None  # The next turn can start.
    assert _events(client.post("/api/chat/stream", auth=AUTH, json={"message": "Hi"}))


def test_chat_stream_refuses_a_second_turn_while_one_runs(tmp_path):
    from arietta_voice.turns import CancellationToken

    client, service, _ = _streaming_client(tmp_path)
    service._active_chat = ("other", CancellationToken())
    response = client.post("/api/chat/stream", auth=AUTH, json={"message": "Hi"})
    assert response.status_code == 503
