from __future__ import annotations

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any
import uuid


def local_config_path(base_path: Path) -> Path:
    if base_path.name.endswith(".local.toml"):
        return base_path
    return base_path.with_name(f"{base_path.stem}.local.toml")


def write_private_text(path: Path, content: str) -> None:
    """Atomically replace a private file without following a destination symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {path.name}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def render_config_toml(data: dict[str, Any]) -> str:
    def literal(value: Any) -> str:
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, tuple):
            value = list(value)
        return json.dumps(value, ensure_ascii=False, allow_nan=False)

    lines = ["# Private local settings. Do not commit this file."]
    for section, values in data.items():
        lines.extend(["", f"[{section}]"])
        for key, value in values.items():
            if value is not None:
                lines.append(f"{key} = {literal(value)}")
    return "\n".join(lines) + "\n"


def update_private_section(base_path: Path, section: str, values: dict[str, Any]) -> None:
    path = local_config_path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw.setdefault(section, {}).update(values)
        write_private_text(path, render_config_toml(raw))


def initialize_private_config(base_path: Path) -> dict[str, str]:
    """Snapshot effective settings/persona; never rewrite or delete the originals."""
    from arietta_voice.config import load_config

    base_path = base_path.resolve()
    local_path = local_config_path(base_path)
    soul_path = base_path.parent / "SOUL.local.md"
    if local_path.exists() or soul_path.exists():
        raise ValueError("Private config or SOUL.local.md already exists; existing files were preserved")
    config = load_config(base_path)
    soul_text = config.prompts.soul_path.read_text(encoding="utf-8") if config.prompts.soul_path.exists() else ""
    backup_root = base_path.parent / ".private-backups"
    backup_root.mkdir(exist_ok=True, mode=0o700)
    backup_dir = backup_root / uuid.uuid4().hex
    backup_dir.mkdir(mode=0o700)
    write_private_text(backup_dir / base_path.name, base_path.read_text(encoding="utf-8"))
    write_private_text(backup_dir / "SOUL.md", soul_text)
    data = asdict(config)
    data.pop("project_root")
    data["prompts"]["soul_path"] = soul_path.name
    write_private_text(soul_path, soul_text)
    write_private_text(local_path, render_config_toml(data))
    return {"config": str(local_path), "persona": str(soul_path), "backup": str(backup_dir)}
