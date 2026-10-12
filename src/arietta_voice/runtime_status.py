from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import threading
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
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
    except (OSError, ValueError) as exc:
        return {
            **base,
            "state": "unknown",
            "lifecycle": "unreadable",
            "note": f"Runtime status file could not be read: {type(exc).__name__}",
        }

    payload = _redact_conversation(config, payload)

    heartbeat_at = str(payload.get("heartbeat_at") or "").strip() or None
    heartbeat_age = _heartbeat_age_seconds(heartbeat_at)
    lifecycle = str(payload.get("lifecycle") or "running")
    is_stale = (
        lifecycle == "running"
        and (heartbeat_age is None or heartbeat_age > stale_after_seconds)
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
        "reliability": payload.get("reliability", {}),
        "note": note,
    }


def write_runtime_status(config: AppConfig, payload: dict[str, Any]) -> None:
    path = runtime_status_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with os.fdopen(os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(_redact_conversation(config, payload), handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _redact_conversation(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any]:
    # Also filter legacy status files on read. Passive speech is never exposed,
    # even when the user opts into active-conversation content logging.
    if not config.logging.content_enabled or not payload.get("session_active", False):
        return {**payload, "last_user_transcript": "", "last_assistant_text": ""}
    return payload


class RuntimeStatusPublisher:
    def __init__(self, config: AppConfig, *, interval_seconds: float = 1.0) -> None:
        self.config = config
        self.interval_seconds = interval_seconds
        self.process_id = os.getpid()
        self.started_at = _now_iso()
        self._last_publish_monotonic = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, runtime: Any) -> None:
        self._stop_event.clear()
        def heartbeat():
            while not self._stop_event.wait(self.interval_seconds):
                runtime._publish_status()
        self._thread = threading.Thread(target=heartbeat, name="arietta-heartbeat", daemon=True)
        self._thread.start()
        runtime._publish_status(force=True)

    def stop(self, runtime: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        runtime._publish_status(lifecycle="stopped", force=True)

    def publish(self, runtime: Any, *, lifecycle: str = "running") -> None:
        with self._lock:
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
            "reliability": runtime.reliability_snapshot() if hasattr(runtime, "reliability_snapshot") else {},
        }


def _heartbeat_age_seconds(heartbeat_at: str | None) -> float | None:
    if not heartbeat_at:
        return None
    try:
        parsed = datetime.fromisoformat(heartbeat_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
