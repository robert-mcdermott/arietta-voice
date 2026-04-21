from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


def cache_root() -> Path:
    return Path.home() / "Library" / "Caches" / "arietta-voice"


def ensure_download(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")

    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    tmp_path.replace(destination)
    return destination
