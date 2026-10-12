from __future__ import annotations

import json
import logging
import fcntl
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from arietta_voice.config import LoggingConfig


CONTENT_FIELDS = {"text", "query", "arguments", "context", "content", "response", "message", "goodbye", "error"}


def _without_content(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_content(item) for key, item in value.items() if key not in CONTENT_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_without_content(item) for item in value]
    return value


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("arietta_voice")


class EventLogger:
    def __init__(
        self, path: Path, *, content_enabled: bool = False,
        max_bytes: int = 5_000_000, backup_count: int = 3,
    ) -> None:
        if max_bytes < 1024 or backup_count < 0:
            raise ValueError("Event logs require max_bytes >= 1024 and backup_count >= 0")
        self.path = path
        self.content_enabled = content_enabled
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    @classmethod
    def from_config(cls, config: LoggingConfig) -> EventLogger:
        return cls(config.events_file, content_enabled=config.content_enabled,
                   max_bytes=config.max_bytes, backup_count=config.backup_count)

    def log(self, event: str, *, content: dict[str, Any] | None = None, **fields: Any) -> None:
        fields = {**fields, **(content or {})} if self.content_enabled else _without_content(fields)
        payload = {
            **fields,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "event": event,
        }
        encoded = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
        if len(encoded) > self.max_bytes:
            # A large model/plugin result must not defeat the storage limit.
            encoded = (json.dumps({"ts": payload["ts"], "event": "oversized_event",
                                   "fields_omitted": True}) + "\n").encode("utf-8")

        # Runtime and API may share this path. Lock a stable sidecar inode so
        # rotation and appends are serialized across processes and threads.
        lock_path = self.path.with_name(self.path.name + ".lock")
        with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists() and self.path.stat().st_size + len(encoded) > self.max_bytes:
                self._rotate()
            with os.fdopen(os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600), "ab") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(encoded)

    def _rotate(self) -> None:
        # Honor a reduced backup count as well as the initial setting. Only
        # numeric archives belonging to this log are eligible for removal.
        prefix = self.path.name + "."
        for path in self.path.parent.iterdir():
            suffix = path.name.removeprefix(prefix) if path.name.startswith(prefix) else ""
            if suffix.isdigit() and int(suffix) > self.backup_count and path.is_file():
                path.unlink()
        for index in range(self.backup_count, 0, -1):
            source = self.path if index == 1 else self.path.with_name(f"{self.path.name}.{index - 1}")
            destination = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.chmod(0o600)
                source.replace(destination)
        if self.backup_count == 0:
            self.path.unlink(missing_ok=True)
