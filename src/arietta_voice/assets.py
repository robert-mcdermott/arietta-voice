from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path


def cache_root() -> Path:
    return Path.home() / "Library" / "Caches" / "arietta-voice"


def ensure_download(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Voice and console workers may need the same model on first use.
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".tmp", delete_on_close=False
    ) as handle:
        with urllib.request.urlopen(url, timeout=60) as response:
            # A 206 response can have a matching Content-Length for just one range.
            if response.status == 206 or response.headers.get("Content-Range"):
                raise OSError(f"Incomplete download of {destination.name}. Try again.")
            shutil.copyfileobj(response, handle)
            size = handle.tell()
            expected = response.headers.get("Content-Length")
            if not size or (expected is not None and size != int(expected)):
                raise OSError(f"Incomplete download of {destination.name}. Try again.")
        handle.close()
        Path(handle.name).replace(destination)
    return destination
