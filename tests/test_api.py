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
    assert config_response.json()["assistant"]["name"] == "Test Assistant"


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
