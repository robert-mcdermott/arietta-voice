from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pytest

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


def test_waiting_for_audio_is_alive_but_not_ready(tmp_path):
    config = _config(tmp_path)
    write_runtime_status(config, {'heartbeat_at': _iso(), 'lifecycle': 'running',
        'state': 'waiting_for_audio', 'session_active': False,
        'reliability': {'devices': {'connection': 'waiting', 'reconnect_attempts': 2}}})
    status = read_runtime_status(config)
    assert status['is_running'] and not status['session_active']
    assert 'Waiting for the configured audio devices' in status['note']
    assert status['reliability']['devices']['reconnect_attempts'] == 2


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


@pytest.mark.parametrize("content_enabled, active, visible", [
    (False, True, False), (False, False, False), (True, False, False), (True, True, True),
])
def test_status_content_privacy_on_disk_and_on_read(tmp_path, content_enabled, active, visible) -> None:
    config = _config(tmp_path)
    config.logging.content_enabled = content_enabled
    payload = {"heartbeat_at": _iso(), "lifecycle": "running", "session_active": active,
               "last_user_transcript": "PRIVATE USER", "last_assistant_text": "PRIVATE ANSWER"}
    write_runtime_status(config, payload)
    assert ("PRIVATE" in runtime_status_path(config).read_text()) is visible
    assert ("PRIVATE" in json.dumps(read_runtime_status(config))) is visible

    # An older runtime's status file must not bypass the current read policy.
    runtime_status_path(config).write_text(json.dumps(payload))
    assert ("PRIVATE" in json.dumps(read_runtime_status(config))) is visible


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


def test_heartbeat_continues_during_blocking_work(tmp_path):
    import time
    from types import SimpleNamespace
    from arietta_voice.runtime_status import RuntimeStatusPublisher
    publisher = RuntimeStatusPublisher(_config(tmp_path), interval_seconds=0.02)
    runtime = SimpleNamespace(state="thinking", reliability_snapshot=lambda: {"turn_stage": "chat"})
    writes = []
    runtime._publish_status = lambda **kwargs: (writes.append(time.monotonic()), publisher.publish(runtime, lifecycle=kwargs.get("lifecycle", "running")))
    publisher.start(runtime)
    try:
        time.sleep(0.09)
    finally:
        publisher.stop(runtime)
    assert len(writes) >= 3
    status = read_runtime_status(publisher.config)
    assert status["lifecycle"] == "stopped"
    assert status["reliability"]["turn_stage"] == "chat"


@pytest.mark.parametrize("heartbeat", [None, "garbage", "2026-01-01T00:00:00"])
def test_missing_or_invalid_heartbeat_is_never_reported_running(tmp_path, heartbeat):
    config = _config(tmp_path)
    write_runtime_status(config, {"heartbeat_at": heartbeat, "lifecycle": "running"})
    assert not read_runtime_status(config)["is_running"]
