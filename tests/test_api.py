from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from arietta_voice.api import AriettaApiService, create_app


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
