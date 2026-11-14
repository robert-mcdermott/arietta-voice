"""Actions the owner has approved for good, so the assistant stops asking.

Kept in a small private JSON file shared by the voice runtime and the console.
Keys name the action: ``mcp:<server>:<tool>`` for an MCP server's tool and
``tool:<name>`` for a local one.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from arietta_voice.private_config import write_private_text

MAX_APPROVALS = 500


class Approvals:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def all(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        approved = data.get("approved") if isinstance(data, dict) else None
        if not isinstance(approved, list):
            return []
        return [key for key in approved if isinstance(key, str)]

    def has(self, key: str) -> bool:
        return key in self.all()

    def add(self, key: str) -> None:
        self._change(lambda keys: keys if key in keys else [*keys, key][-MAX_APPROVALS:])

    def remove(self, key: str | None = None, *, prefix: str | None = None) -> None:
        def keep(keys):
            return [
                item
                for item in keys
                if item != key and not (prefix is not None and item.startswith(prefix))
            ]

        self._change(keep)

    def _change(self, update) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            keys = update(self.all())
            write_private_text(self.path, json.dumps({"approved": keys}, indent=2) + "\n")
