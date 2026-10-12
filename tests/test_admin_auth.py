import json

import pytest

from arietta_voice.admin_auth import credential_path, load_admin_credentials, save_admin_credentials


@pytest.fixture(autouse=True)
def clear_admin_environment(monkeypatch):
    monkeypatch.delenv("ARIETTA_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ARIETTA_ADMIN_USERNAME", raising=False)


def test_missing_credentials_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="admin-setup"):
        load_admin_credentials(tmp_path / "app.toml")


@pytest.mark.parametrize("password", ["", "password", "short", " " * 12])
def test_blank_or_weak_environment_password_is_rejected(tmp_path, monkeypatch, password):
    monkeypatch.setenv("ARIETTA_ADMIN_PASSWORD", password)
    with pytest.raises(ValueError, match="at least 12"):
        load_admin_credentials(tmp_path / "app.toml")


def test_private_hashed_credentials_and_rotation(tmp_path):
    base = tmp_path / "app.toml"
    password = "synthetic-long-password"
    path = save_admin_credentials(base, "owner", password)
    assert password not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600
    credentials = load_admin_credentials(base)
    assert credentials.verify("owner", password)
    assert not credentials.verify("admin", "password")
    assert not credentials.verify("owner", "wrong-password")
    assert not credentials.verify("other", password)
    with pytest.raises(ValueError, match="already exist"):
        save_admin_credentials(base, "owner", "new-long-password")
    save_admin_credentials(base, "owner", "new-long-password", replace=True)
    assert not load_admin_credentials(base).verify("owner", password)
    assert credential_path(tmp_path / "app.local.toml") == path


def test_explicit_environment_credentials_override_file(tmp_path, monkeypatch):
    base = tmp_path / "app.toml"
    save_admin_credentials(base, "owner", "file-long-password")
    monkeypatch.setenv("ARIETTA_ADMIN_PASSWORD", "env-long-password")
    assert load_admin_credentials(base).verify("admin", "env-long-password")
    monkeypatch.setenv("ARIETTA_ADMIN_PASSWORD", "")
    with pytest.raises(ValueError):
        load_admin_credentials(base)


def test_invalid_or_world_readable_credentials_are_rejected(tmp_path):
    base = tmp_path / "app.toml"
    path = save_admin_credentials(base, "owner", "file-long-password")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="Invalid admin credential"):
        load_admin_credentials(base)
    path.chmod(0o600)
    data = json.loads(path.read_text())
    data["password_hash"] = "invalid-private-content"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid admin credential") as error:
        load_admin_credentials(base)
    assert "invalid-private-content" not in str(error.value)
