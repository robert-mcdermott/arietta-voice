from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets

from arietta_voice.private_config import write_private_text


ITERATIONS = 600_000


def credential_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem.removesuffix(".local") + ".admin.json")


def _validate_credentials(username: str, password: str) -> None:
    if not isinstance(username, str) or not username.strip() or ":" in username or not username.isascii() or not username.isprintable():
        raise ValueError("Admin username must be nonblank ASCII without a colon")
    if not isinstance(password, str) or len(password) < 12 or not password.isascii() or not password.strip() or not password.isprintable():
        raise ValueError("Admin password must contain at least 12 ASCII characters")


@dataclass(frozen=True)
class AdminCredentials:
    username: str
    salt: bytes
    password_hash: bytes

    def verify(self, username: str, password: str) -> bool:
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), self.salt, ITERATIONS)
        username_ok = secrets.compare_digest(username.encode(), self.username.encode())
        password_ok = secrets.compare_digest(candidate, self.password_hash)
        return username_ok and password_ok

    @classmethod
    def from_password(cls, username: str, password: str) -> AdminCredentials:
        _validate_credentials(username, password)
        salt = secrets.token_bytes(16)
        return cls(username, salt, hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS))


def save_admin_credentials(config_path: Path, username: str, password: str, *, replace: bool = False) -> Path:
    path = credential_path(config_path)
    if path.exists() and not replace:
        raise ValueError("Admin credentials already exist; use admin-setup --replace to rotate them")
    credentials = AdminCredentials.from_password(username, password)
    write_private_text(path, json.dumps({
        "version": 1,
        "algorithm": "pbkdf2-sha256",
        "iterations": ITERATIONS,
        "username": credentials.username,
        "salt": credentials.salt.hex(),
        "password_hash": credentials.password_hash.hex(),
    }, indent=2) + "\n")
    return path


def load_admin_credentials(config_path: Path) -> AdminCredentials:
    # An explicitly set but invalid environment password fails closed instead
    # of silently falling back to a different credential source.
    if "ARIETTA_ADMIN_PASSWORD" in os.environ:
        return AdminCredentials.from_password(
            os.environ.get("ARIETTA_ADMIN_USERNAME", "admin"), os.environ["ARIETTA_ADMIN_PASSWORD"],
        )
    path = credential_path(config_path)
    if not path.exists():
        raise ValueError("Admin credentials are not configured. Run 'arietta-voice admin-setup' with the same --config, or set ARIETTA_ADMIN_PASSWORD.")
    try:
        if path.stat().st_mode & 0o077:
            raise ValueError("Admin credential file must be owner-only (chmod 600)")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != 1 or data["algorithm"] != "pbkdf2-sha256" or data["iterations"] != ITERATIONS:
            raise ValueError("Unsupported admin credential format")
        username = data["username"]
        _validate_credentials(username, "format-check-only")
        salt, password_hash = bytes.fromhex(data["salt"]), bytes.fromhex(data["password_hash"])
        if len(salt) != 16 or len(password_hash) != 32:
            raise ValueError("Invalid admin credential hash")
        return AdminCredentials(username, salt, password_hash)
    except (KeyError, TypeError, ValueError) as exc:
        # Never include file contents or a parsed secret in this error.
        raise ValueError("Invalid admin credential file; check permissions or run admin-setup --replace") from exc
