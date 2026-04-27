from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from arietta_voice.config import default_config
from arietta_voice.runtime_status import read_runtime_status, runtime_status_path, write_runtime_status


def _config(tmp_path: Path):
    config = default_config(tmp_path)
    config.logging.directory = tmp_path / "logs"
    config.logging.events_file = tmp_path / "logs" / "events.jsonl"
    return config


def _iso(seconds_ago: float = 0.0) -> str:
    value = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def test_missing_runtime_status_reports_not_running(tmp_path: Path) -> None:
    config = _config(tmp_path)

    status = read_runtime_status(config)

    assert status["state"] == "not_running"
    assert status["is_running"] is False
    assert status["status_file"] == str(runtime_status_path(config))


def test_fresh_runtime_status_reports_voice_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    write_runtime_status(
        config,
        {
            "heartbeat_at": _iso(),
            "lifecycle": "running",
            "process_id": 12345,
            "started_at": _iso(2),
            "state": "waiting_for_wake",
            "session_active": False,
            "last_assistant_text": "Listening for the wake word.",
        },
    )

    status = read_runtime_status(config)

    assert status["state"] == "waiting_for_wake"
    assert status["is_running"] is True
    assert status["is_stale"] is False
    assert status["process_id"] == 12345


def test_stale_runtime_status_reports_stale(tmp_path: Path) -> None:
    config = _config(tmp_path)
    write_runtime_status(
        config,
        {
            "heartbeat_at": _iso(60),
            "lifecycle": "running",
            "process_id": 12345,
            "started_at": _iso(90),
            "state": "listening",
        },
    )

    status = read_runtime_status(config, stale_after_seconds=5)

    assert status["state"] == "stale"
    assert status["is_running"] is False
    assert status["is_stale"] is True


def test_stopped_runtime_status_reports_clean_exit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    write_runtime_status(
        config,
        {
            "heartbeat_at": _iso(60),
            "lifecycle": "stopped",
            "process_id": 12345,
            "started_at": _iso(90),
            "state": "stopping",
        },
    )

    status = read_runtime_status(config)

    assert status["state"] == "stopped"
    assert status["is_running"] is False
    assert status["is_stale"] is False
