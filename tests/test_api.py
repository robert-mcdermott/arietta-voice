from __future__ import annotations

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
        payload.pop("enabled_model_tools")
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


@pytest.mark.parametrize("method, path, payload", [
    ("post", "/api/tools/create", {"module_name": "blocked"}),
    ("post", "/api/model-tools/create", {"module_name": "blocked"}),
    ("put", "/api/tools/local_time/source", {"content": "raise RuntimeError('should not execute')"}),
    ("put", "/api/model-tools/system_info/source", {"content": "raise RuntimeError('should not execute')"}),
])
def test_source_mutations_are_disabled_by_default(tmp_path, monkeypatch, method, path, payload):
    client, service = _client_with_service(_write_api_config(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("Source mutation reached the service while disabled")

    for name in ("create_tool", "create_model_tool", "write_tool_source", "write_model_tool_source"):
        monkeypatch.setattr(service, name, forbidden)
    assert client.request(method, path, json=payload).status_code == 401
    result = client.request(method, path, json=payload, auth=("admin", "test-only-long-password"))
    assert result.status_code == 403
    assert "Source editing is disabled" in result.json()["detail"]


def test_browser_boundaries_and_no_store(tmp_path):
    client = _client(_write_api_config(tmp_path))
    auth = ("admin", "test-only-long-password")
    assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 400
    for headers in ({"Origin": "https://attacker.example"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
        response = client.post("/api/tools/enabled", json={"enabled": []}, auth=auth, headers=headers)
        assert response.status_code == 403
    response = client.post("/api/tools/enabled", json={"enabled": []}, auth=auth, headers={"Origin": "http://localhost"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert client.get("/api/model-tools/instructions", auth=auth).status_code == 405


def test_static_inventory_and_source_reads_do_not_execute_plugins(tmp_path, monkeypatch):
    client, service = _client_with_service(_write_api_config(tmp_path))
    marker = tmp_path / "executed"
    tools_dir = tmp_path / "plugins"
    tools_dir.mkdir()
    (tools_dir / "side_effect.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "def create_tool(context):\n    raise RuntimeError('factory executed')\n"
    )
    external = tmp_path / "external_side_effect"
    external.mkdir()
    (external / "__init__.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('external import')")
    monkeypatch.syspath_prepend(str(tmp_path))
    service.tool_admin.tools_directory = tools_dir
    service.model_tool_admin.tools_directory = tools_dir
    service.config.tools.enabled = ("side_effect", "external_side_effect.plugin")
    service.config.model_tools.enabled = ("side_effect", "external_side_effect.plugin")
    auth = ("admin", "test-only-long-password")
    for path in ("/api/state", "/api/tools", "/api/tools/side_effect/source", "/api/model-tools/side_effect/source"):
        assert client.get(path, auth=auth).status_code == 200
        assert not marker.exists()
    inventory = client.get("/api/tools", auth=auth).json()
    assert inventory["assistant_tools"]["loaded"] == []
    assert inventory["model_tools"]["instruction_text"] is None
    assert all(tool["valid"] is None for tool in inventory["assistant_tools"]["tools"])
    # Validation is an explicit authenticated action that is allowed to import.
    assert client.post("/api/tools/side_effect/validate", auth=auth).status_code == 200
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
    from arietta_voice import api

    client, service = _client_with_service(_write_api_config(tmp_path))
    service.config.model_tools.enabled = ("missing_private_plugin",)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled tools must not load or execute")

    def generate(messages, **kwargs):
        calls.append(messages)
        return '{"tool_call":{"name":"forbidden","arguments":{}}}'

    monkeypatch.setattr(service, "probe_tools", forbidden)
    monkeypatch.setattr(api, "_AdHocModelToolRegistry", forbidden)
    monkeypatch.setattr(service, "_llm_generate", generate)
    response = client.post("/api/chat", auth=("admin", "test-only-long-password"),
                           json={"message": "What is your name?", "use_tools": False})
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
    deleted = client.delete("/api/knowledge/files/new-note.md", auth=("admin", "test-only-long-password"))

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
    second_page = client.get("/api/logs?limit=2&offset=2", auth=("admin", "test-only-long-password"))
    event_filtered = client.get("/api/logs?limit=10&event=alpha", auth=("admin", "test-only-long-password"))
    search_filtered = client.get("/api/logs?limit=10&q=banana", auth=("admin", "test-only-long-password"))

    assert [item["event"] for item in first_page.json()["items"]] == ["gamma", "alpha"]
    assert first_page.json()["has_next"] is True
    assert first_page.json()["has_previous"] is False
    assert [item["event"] for item in second_page.json()["items"]] == ["beta", "alpha"]
    assert second_page.json()["has_previous"] is True
    assert event_filtered.json()["total"] == 2
    assert [item["text"] for item in event_filtered.json()["items"]] == ["third apricot", "first apple"]
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
    assert payload["used_tool"] == "get_assistant_identity"
    assert "Test Assistant" in payload["response"]


def test_tools_endpoint_lists_registry_inventory(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.get("/api/tools", auth=("admin", "test-only-long-password"))

    payload = response.json()
    assistant_tools = payload["assistant_tools"]["tools"]
    assert response.status_code == 200
    assert any(
        tool["configured_name"] == "assistant_identity" and tool["enabled"] and tool["syntax_valid"]
        for tool in assistant_tools
    )
    assert any(
        tool["configured_name"] == "local_time" and not tool["enabled"] and tool["syntax_valid"]
        for tool in assistant_tools
    )
    assert payload["assistant_tools"]["load_errors"] == []

    validated = client.post("/api/tools/assistant_identity/validate", auth=("admin", "test-only-long-password"))
    assert validated.status_code == 200
    assert validated.json()["enabled"] is True


def test_tools_endpoint_reports_load_errors(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "broken_tool.py").write_text("BROKEN = True\n", encoding="utf-8")
    service.tool_admin.tools_directory = tools_dir

    response = client.get("/api/tools", auth=("admin", "test-only-long-password"))

    load_errors = response.json()["assistant_tools"]["load_errors"]
    assert response.status_code == 200
    assert any(error["configured_name"] == "broken_tool" for error in load_errors)


def test_tools_enabled_endpoint_persists_config(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    original = config_path.read_bytes()
    client = _client(config_path)

    response = client.post(
        "/api/tools/enabled",
        auth=("admin", "test-only-long-password"),
        json={"enabled": ["local_time"]},
    )

    assert response.status_code == 200
    assert response.json()["enabled"] == ["local_time"]
    assert load_config(config_path).tools.enabled == ("local_time",)
    assert config_path.read_bytes() == original


def test_specific_tool_probe_can_run_disabled_tool(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.post(
        "/api/tools/local_time/probe",
        auth=("admin", "test-only-long-password"),
        json={"text": "What time is it?"},
    )

    assert response.status_code == 200
    assert response.json()["matched"] is True
    assert response.json()["enabled"] is False
    assert response.json()["tool"] == "get_local_time"


def test_tool_create_endpoint_writes_valid_tool(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    service.config.admin.allow_source_editing = True
    service.tool_admin.tools_directory = tmp_path / "generated_tools"

    created = client.post(
        "/api/tools/create",
        auth=("admin", "test-only-long-password"),
        json={
            "module_name": "Demo Tool",
            "trigger_phrase": "banana checkout",
            "response": "Banana checkout handled.",
            "enable": True,
        },
    )
    probe = client.post(
        "/api/tools/demo_tool/probe",
        auth=("admin", "test-only-long-password"),
        json={"text": "please run banana checkout"},
    )
    validated = client.post("/api/tools/demo_tool/validate", auth=("admin", "test-only-long-password"))

    assert created.status_code == 200
    assert created.json()["module_name"] == "demo_tool"
    assert (tmp_path / "generated_tools" / "demo_tool.py").exists()
    assert created.json()["tool"]["valid"] is True
    assert "demo_tool" in load_config(config_path).tools.enabled
    assert probe.status_code == 200
    assert probe.json()["matched"] is True
    assert probe.json()["enabled"] is True
    assert probe.json()["content"] == "Banana checkout handled."
    assert validated.status_code == 200
    assert validated.json()["valid"] is True
    assert validated.json()["enabled"] is True


def test_tool_source_endpoints_read_write_and_validate(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    service.config.admin.allow_source_editing = True
    service.tool_admin.tools_directory = tmp_path / "generated_tools"

    created = client.post(
        "/api/tools/create",
        auth=("admin", "test-only-long-password"),
        json={
            "module_name": "Source Demo",
            "trigger_phrase": "source demo",
            "response": "Original source response.",
            "enable": True,
        },
    )
    read = client.get("/api/tools/source_demo/source", auth=("admin", "test-only-long-password"))
    updated_content = read.json()["content"].replace(
        "Original source response.",
        "Edited source response.",
    )
    saved = client.put(
        "/api/tools/source_demo/source",
        auth=("admin", "test-only-long-password"),
        json={"content": updated_content},
    )
    probe = client.post(
        "/api/tools/source_demo/probe",
        auth=("admin", "test-only-long-password"),
        json={"text": "please run source demo"},
    )

    assert created.status_code == 200
    assert read.status_code == 200
    assert read.json()["file"].endswith("source_demo.py")
    assert saved.status_code == 200
    assert saved.json()["saved"] is True
    assert saved.json()["tool"]["valid"] is True
    assert probe.json()["content"] == "Edited source response."


def test_model_tools_endpoint_lists_registry_inventory(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.get("/api/tools", auth=("admin", "test-only-long-password"))

    payload = response.json()
    model_tools = payload["model_tools"]["tools"]
    assert response.status_code == 200
    assert any(
        tool["configured_name"] == "system_info"
        and not tool["enabled"]
        and tool["syntax_valid"]
        and tool["active"] is None
        for tool in model_tools
    )
    assert payload["model_tools"]["load_errors"] == []


def test_model_tools_enabled_endpoint_persists_config_and_instructions(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client = _client(config_path)

    enabled = client.post(
        "/api/model-tools/enabled",
        auth=("admin", "test-only-long-password"),
        json={"enabled": ["system_info"]},
    )
    instructions = client.post("/api/model-tools/instructions", auth=("admin", "test-only-long-password"))

    assert enabled.status_code == 200
    assert enabled.json()["enabled"] == ["system_info"]
    assert load_config(config_path).model_tools.enabled == ("system_info",)
    assert instructions.status_code == 200
    assert "get_system_info" in instructions.json()["instruction_text"]


def test_model_tool_invoke_can_run_disabled_tool(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.post(
        "/api/model-tools/system_info/invoke",
        auth=("admin", "test-only-long-password"),
        json={"arguments": {}},
    )

    assert response.status_code == 200
    assert response.json()["invoked"] is True
    assert response.json()["enabled"] is False
    assert response.json()["tool"] == "get_system_info"
    assert response.json()["arguments"] == {}
    assert response.json()["arguments_example"] == {}
    assert "System information requested" in response.json()["followup_context"]


def test_model_tool_create_endpoint_writes_valid_tool(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    service.config.admin.allow_source_editing = True
    service.model_tool_admin.tools_directory = tmp_path / "generated_model_tools"

    created = client.post(
        "/api/model-tools/create",
        auth=("admin", "test-only-long-password"),
        json={
            "module_name": "Demo Model Tool",
            "description": "you need a demo model tool result",
            "arguments_example": {"topic": "<topic>"},
            "followup_context": "Demo model tool follow-up.",
            "enable": True,
        },
    )
    invoked = client.post(
        "/api/model-tools/demo_model_tool/invoke",
        auth=("admin", "test-only-long-password"),
        json={"arguments": {"topic": "widgets"}},
    )
    validated = client.post(
        "/api/model-tools/demo_model_tool/validate",
        auth=("admin", "test-only-long-password"),
    )

    assert created.status_code == 200
    assert created.json()["module_name"] == "demo_model_tool"
    assert (tmp_path / "generated_model_tools" / "demo_model_tool.py").exists()
    assert created.json()["tool"]["valid"] is True
    assert "demo_model_tool" in load_config(config_path).model_tools.enabled
    assert invoked.status_code == 200
    assert invoked.json()["invoked"] is True
    assert invoked.json()["enabled"] is True
    assert invoked.json()["arguments"] == {"topic": "widgets"}
    assert invoked.json()["arguments_example"] == {"topic": "<topic>"}
    assert "Demo model tool follow-up." in invoked.json()["followup_context"]
    assert invoked.json()["details"]["arguments"] == {"topic": "widgets"}
    assert validated.status_code == 200
    assert validated.json()["valid"] is True
    assert validated.json()["enabled"] is True


def test_model_tool_source_endpoints_read_write_and_invoke(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    service.config.admin.allow_source_editing = True
    service.model_tool_admin.tools_directory = tmp_path / "generated_model_tools"

    created = client.post(
        "/api/model-tools/create",
        auth=("admin", "test-only-long-password"),
        json={
            "module_name": "Source Model Tool",
            "description": "you need a source model tool result",
            "arguments_example": {"topic": "<topic>"},
            "followup_context": "Original model follow-up.",
            "enable": True,
        },
    )
    read = client.get("/api/model-tools/source_model_tool/source", auth=("admin", "test-only-long-password"))
    updated_content = read.json()["content"].replace(
        "Original model follow-up.",
        "Edited model follow-up.",
    )
    saved = client.put(
        "/api/model-tools/source_model_tool/source",
        auth=("admin", "test-only-long-password"),
        json={"content": updated_content},
    )
    invoked = client.post(
        "/api/model-tools/source_model_tool/invoke",
        auth=("admin", "test-only-long-password"),
        json={"arguments": {"topic": "source"}},
    )

    assert created.status_code == 200
    assert read.status_code == 200
    assert read.json()["file"].endswith("source_model_tool.py")
    assert saved.status_code == 200
    assert saved.json()["saved"] is True
    assert saved.json()["tool"]["valid"] is True
    assert invoked.status_code == 200
    assert "Edited model follow-up." in invoked.json()["followup_context"]


@pytest.mark.parametrize("failure, code", [("WorkerTimeout", 504), ("WorkerBusy", 503), ("WorkerFailure", 502)])
def test_chat_failure_has_safe_actionable_error_and_can_recover(tmp_path, failure, code):
    from unittest.mock import Mock
    from arietta_voice import worker
    client, service = _client_with_service(_write_api_config(tmp_path))
    service.worker.call = Mock(side_effect=[getattr(worker, failure)("PRIVATE failure details"),
        {"response": "Recovered", "source": "model", "used_tool": None, "model_tool": None}])
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
