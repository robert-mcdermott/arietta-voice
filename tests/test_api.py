from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from arietta_voice.api import AriettaApiService, create_app
from arietta_voice.config import load_config


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
    return TestClient(create_app(service=service))


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
    client = TestClient(create_app(service=service))
    return client, service


def test_health_is_public(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["assistant"] == "Test Assistant"


def test_admin_console_requires_basic_auth(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    unauthenticated = client.get("/")
    authenticated = client.get("/", auth=("admin", "password"))

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert "Arietta Voice" in authenticated.text


def test_state_and_config_are_authenticated(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    unauthorized = client.get("/api/state")
    authorized = client.get("/api/state", auth=("admin", "password"))
    config_response = client.get("/api/config", auth=("admin", "password"))

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["runtime"]["mode"] == "web_api"
    assert authorized.json()["runtime"]["state"] == "not_running"
    assert config_response.json()["assistant"]["name"] == "Test Assistant"


def test_runtime_endpoint_reports_unmanaged_runtime_status(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.get("/api/runtime", auth=("admin", "password"))

    assert response.status_code == 200
    assert response.json()["managed_by_api"] is False
    assert response.json()["state"] == "not_running"


def test_runtime_control_endpoints_delegate_to_supervisor(tmp_path: Path) -> None:
    client, service = _client_with_service(_write_api_config(tmp_path))
    service.runtime_supervisor = FakeRuntimeSupervisor()

    start = client.post("/api/runtime/start", auth=("admin", "password"))
    state = client.get("/api/state", auth=("admin", "password"))
    stop = client.post("/api/runtime/stop", auth=("admin", "password"))

    assert start.status_code == 200
    assert start.json()["managed_process_running"] is True
    assert state.json()["runtime"]["managed_process_running"] is True
    assert stop.status_code == 200
    assert stop.json()["managed_process_running"] is False


def test_knowledge_search_endpoint(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.post(
        "/api/knowledge/search",
        auth=("admin", "password"),
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
        auth=("admin", "password"),
        json={"content": "# New Note\n\nThe dashboard can edit knowledge."},
    )
    listed = client.get("/api/knowledge/files", auth=("admin", "password"))
    read = client.get("/api/knowledge/files/new-note.md", auth=("admin", "password"))
    deleted = client.delete("/api/knowledge/files/new-note.md", auth=("admin", "password"))

    assert created.status_code == 200
    assert created.json()["created"] is True
    assert any(file["path"] == "new-note.md" for file in listed.json()["files"])
    assert "dashboard can edit" in read.json()["content"]
    assert deleted.json()["deleted"] is True


def test_knowledge_file_endpoint_rejects_path_traversal(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.put(
        "/api/knowledge/files/%2E%2E/secret.md",
        auth=("admin", "password"),
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

    response = client.get("/api/logs?limit=1", auth=("admin", "password"))

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

    first_page = client.get("/api/logs?limit=2", auth=("admin", "password"))
    second_page = client.get("/api/logs?limit=2&offset=2", auth=("admin", "password"))
    event_filtered = client.get("/api/logs?limit=10&event=alpha", auth=("admin", "password"))
    search_filtered = client.get("/api/logs?limit=10&q=banana", auth=("admin", "password"))

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
        auth=("admin", "password"),
        json={"message": "What is your name?"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["source"] == "tool"
    assert payload["used_tool"] == "get_assistant_identity"
    assert "Test Assistant" in payload["response"]


def test_tools_endpoint_lists_registry_inventory(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.get("/api/tools", auth=("admin", "password"))

    payload = response.json()
    assistant_tools = payload["assistant_tools"]["tools"]
    assert response.status_code == 200
    assert any(
        tool["configured_name"] == "assistant_identity" and tool["enabled"] and tool["valid"]
        for tool in assistant_tools
    )
    assert any(
        tool["configured_name"] == "local_time" and not tool["enabled"] and tool["valid"]
        for tool in assistant_tools
    )
    assert payload["assistant_tools"]["load_errors"] == []

    validated = client.post("/api/tools/assistant_identity/validate", auth=("admin", "password"))
    assert validated.status_code == 200
    assert validated.json()["enabled"] is True


def test_tools_endpoint_reports_load_errors(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "broken_tool.py").write_text("BROKEN = True\n", encoding="utf-8")
    service.tool_admin.tools_directory = tools_dir

    response = client.get("/api/tools", auth=("admin", "password"))

    load_errors = response.json()["assistant_tools"]["load_errors"]
    assert response.status_code == 200
    assert any(error["configured_name"] == "broken_tool" for error in load_errors)


def test_tools_enabled_endpoint_persists_config(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client = _client(config_path)

    response = client.post(
        "/api/tools/enabled",
        auth=("admin", "password"),
        json={"enabled": ["local_time"]},
    )

    assert response.status_code == 200
    assert response.json()["enabled"] == ["local_time"]
    assert load_config(config_path).tools.enabled == ("local_time",)


def test_specific_tool_probe_can_run_disabled_tool(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.post(
        "/api/tools/local_time/probe",
        auth=("admin", "password"),
        json={"text": "What time is it?"},
    )

    assert response.status_code == 200
    assert response.json()["matched"] is True
    assert response.json()["enabled"] is False
    assert response.json()["tool"] == "get_local_time"


def test_tool_create_endpoint_writes_valid_tool(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client, service = _client_with_service(config_path)
    service.tool_admin.tools_directory = tmp_path / "generated_tools"

    created = client.post(
        "/api/tools/create",
        auth=("admin", "password"),
        json={
            "module_name": "Demo Tool",
            "trigger_phrase": "banana checkout",
            "response": "Banana checkout handled.",
            "enable": True,
        },
    )
    probe = client.post(
        "/api/tools/demo_tool/probe",
        auth=("admin", "password"),
        json={"text": "please run banana checkout"},
    )
    validated = client.post("/api/tools/demo_tool/validate", auth=("admin", "password"))

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
    service.tool_admin.tools_directory = tmp_path / "generated_tools"

    created = client.post(
        "/api/tools/create",
        auth=("admin", "password"),
        json={
            "module_name": "Source Demo",
            "trigger_phrase": "source demo",
            "response": "Original source response.",
            "enable": True,
        },
    )
    read = client.get("/api/tools/source_demo/source", auth=("admin", "password"))
    updated_content = read.json()["content"].replace(
        "Original source response.",
        "Edited source response.",
    )
    saved = client.put(
        "/api/tools/source_demo/source",
        auth=("admin", "password"),
        json={"content": updated_content},
    )
    probe = client.post(
        "/api/tools/source_demo/probe",
        auth=("admin", "password"),
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

    response = client.get("/api/tools", auth=("admin", "password"))

    payload = response.json()
    model_tools = payload["model_tools"]["tools"]
    assert response.status_code == 200
    assert any(
        tool["configured_name"] == "system_info"
        and not tool["enabled"]
        and tool["valid"]
        and tool["active"]
        for tool in model_tools
    )
    assert payload["model_tools"]["load_errors"] == []


def test_model_tools_enabled_endpoint_persists_config_and_instructions(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path)
    client = _client(config_path)

    enabled = client.post(
        "/api/model-tools/enabled",
        auth=("admin", "password"),
        json={"enabled": ["system_info"]},
    )
    instructions = client.get("/api/model-tools/instructions", auth=("admin", "password"))

    assert enabled.status_code == 200
    assert enabled.json()["enabled"] == ["system_info"]
    assert load_config(config_path).model_tools.enabled == ("system_info",)
    assert instructions.status_code == 200
    assert "get_system_info" in instructions.json()["instruction_text"]


def test_model_tool_invoke_can_run_disabled_tool(tmp_path: Path) -> None:
    client = _client(_write_api_config(tmp_path))

    response = client.post(
        "/api/model-tools/system_info/invoke",
        auth=("admin", "password"),
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
    service.model_tool_admin.tools_directory = tmp_path / "generated_model_tools"

    created = client.post(
        "/api/model-tools/create",
        auth=("admin", "password"),
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
        auth=("admin", "password"),
        json={"arguments": {"topic": "widgets"}},
    )
    validated = client.post(
        "/api/model-tools/demo_model_tool/validate",
        auth=("admin", "password"),
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
    service.model_tool_admin.tools_directory = tmp_path / "generated_model_tools"

    created = client.post(
        "/api/model-tools/create",
        auth=("admin", "password"),
        json={
            "module_name": "Source Model Tool",
            "description": "you need a source model tool result",
            "arguments_example": {"topic": "<topic>"},
            "followup_context": "Original model follow-up.",
            "enable": True,
        },
    )
    read = client.get("/api/model-tools/source_model_tool/source", auth=("admin", "password"))
    updated_content = read.json()["content"].replace(
        "Original model follow-up.",
        "Edited model follow-up.",
    )
    saved = client.put(
        "/api/model-tools/source_model_tool/source",
        auth=("admin", "password"),
        json={"content": updated_content},
    )
    invoked = client.post(
        "/api/model-tools/source_model_tool/invoke",
        auth=("admin", "password"),
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
