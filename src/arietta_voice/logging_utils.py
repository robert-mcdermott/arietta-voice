from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger("arietta_voice")


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
