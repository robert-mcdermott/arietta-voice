from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from arietta_voice.config import AppConfig


RUNTIME_STATUS_FILENAME = "runtime_status.json"
DEFAULT_STALE_AFTER_SECONDS = 60.0


def runtime_status_path(config: AppConfig) -> Path:
    return config.logging.directory / RUNTIME_STATUS_FILENAME


def read_runtime_status(
    config: AppConfig,
    *,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> dict[str, Any]:
    path = runtime_status_path(config)
    base = {
        "mode": "web_api",
        "managed_by_api": False,
        "status_file": str(path),
        "state": "not_running",
        "lifecycle": "missing",
        "is_running": False,
        "is_stale": False,
        "last_heartbeat_at": None,
        "heartbeat_age_seconds": None,
        "process_id": None,
        "started_at": None,
        "session_active": False,
        "note": "No voice runtime heartbeat has been published yet.",
    }
    if not path.exists():
        return base

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "state": "unknown",
            "lifecycle": "unreadable",
            "note": f"Runtime status file could not be read: {type(exc).__name__}",
        }

    heartbeat_at = str(payload.get("heartbeat_at") or "").strip() or None
    heartbeat_age = _heartbeat_age_seconds(heartbeat_at)
    lifecycle = str(payload.get("lifecycle") or "running")
    is_stale = (
        lifecycle == "running"
        and heartbeat_age is not None
        and heartbeat_age > stale_after_seconds
    )
    is_running = lifecycle == "running" and not is_stale

    if lifecycle == "stopped":
        state = "stopped"
        note = "The voice runtime exited cleanly."
    elif is_stale:
        state = "stale"
        note = "The voice runtime heartbeat is stale; the process may have exited unexpectedly."
    elif is_running:
        state = str(payload.get("state") or "running")
        note = "The separate voice runtime is publishing a fresh heartbeat."
    else:
        state = str(payload.get("state") or "unknown")
        note = "The voice runtime status file exists, but its lifecycle is not active."

    return {
        **base,
        "state": state,
        "lifecycle": lifecycle,
        "is_running": is_running,
        "is_stale": is_stale,
        "last_heartbeat_at": heartbeat_at,
        "heartbeat_age_seconds": None if heartbeat_age is None else round(heartbeat_age, 1),
        "process_id": payload.get("process_id"),
        "started_at": payload.get("started_at"),
        "session_active": bool(payload.get("session_active", False)),
        "session_turn_count": int(payload.get("session_turn_count", 0) or 0),
        "last_wake_phrase": payload.get("last_wake_phrase"),
        "last_user_transcript": payload.get("last_user_transcript", ""),
        "last_assistant_text": payload.get("last_assistant_text", ""),
        "display_speech_level": float(payload.get("display_speech_level", 0.0) or 0.0),
        "display_speaking_active": bool(payload.get("display_speaking_active", False)),
        "last_error": payload.get("last_error"),
        "note": note,
    }


def write_runtime_status(config: AppConfig, payload: dict[str, Any]) -> None:
    path = runtime_status_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


class RuntimeStatusPublisher:
    def __init__(self, config: AppConfig, *, interval_seconds: float = 1.0) -> None:
        self.config = config
        self.interval_seconds = interval_seconds
        self.process_id = os.getpid()
        self.started_at = _now_iso()
        self._last_publish_monotonic = 0.0

    def publish(self, runtime: Any, *, lifecycle: str = "running") -> None:
        write_runtime_status(self.config, self._payload(runtime, lifecycle=lifecycle))
        self._last_publish_monotonic = time.monotonic()

    def publish_if_due(self, runtime: Any, *, lifecycle: str = "running") -> None:
        now = time.monotonic()
        if (now - self._last_publish_monotonic) < self.interval_seconds:
            return
        self.publish(runtime, lifecycle=lifecycle)

    def mark_stopped(self, runtime: Any) -> None:
        self.publish(runtime, lifecycle="stopped")

    def _payload(self, runtime: Any, *, lifecycle: str) -> dict[str, Any]:
        session = getattr(runtime, "session", None)
        state = getattr(runtime, "state", None)
        return {
            "heartbeat_at": _now_iso(),
            "lifecycle": lifecycle,
            "process_id": self.process_id,
            "started_at": self.started_at,
            "state": getattr(state, "value", str(state or "unknown")),
            "session_active": bool(getattr(session, "active", False)),
            "session_turn_count": int(getattr(session, "turn_count", 0) or 0),
            "last_wake_phrase": getattr(session, "last_wake_phrase", None),
            "last_user_transcript": getattr(runtime, "last_user_transcript", ""),
            "last_assistant_text": getattr(runtime, "last_assistant_text", ""),
            "display_speech_level": float(getattr(runtime, "display_speech_level", 0.0) or 0.0),
            "display_speaking_active": bool(getattr(runtime, "display_speaking_active", False)),
            "last_error": getattr(runtime, "last_error", None),
        }


def _heartbeat_age_seconds(heartbeat_at: str | None) -> float | None:
    if not heartbeat_at:
        return None
    try:
        parsed = datetime.fromisoformat(heartbeat_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
