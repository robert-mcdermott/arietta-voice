import sys
from unittest.mock import Mock

import pytest

from arietta_voice import cli


@pytest.mark.parametrize("arguments, field", [
    (["--session-timeout", "nan"], "wake_word.session_timeout_s"),
    (["--session-timeout", "-1"], "wake_word.session_timeout_s"),
    (["--wake-word", "!!!"], "wake_word.phrases"),
])
def test_cli_rejects_invalid_overrides_before_importing_runtime(tmp_path, monkeypatch, capsys, arguments, field):
    path = tmp_path / "config.toml"
    path.write_text("")
    monkeypatch.setattr(sys, "argv", ["arietta-voice", "--config", str(path), *arguments])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    error = capsys.readouterr().err
    assert field in error
    assert "Traceback" not in error


@pytest.mark.parametrize("content_enabled", [False, True])
def test_runtime_failures_do_not_leak_prompt_text_by_default(tmp_path, monkeypatch, caplog, content_enabled):
    from arietta_voice import runtime

    path = tmp_path / "config.toml"
    path.write_text(f"[logging]\ncontent_enabled = {str(content_enabled).lower()}\n")
    monkeypatch.setattr(sys, "argv", ["arietta-voice", "--config", str(path)])
    failed_runtime = Mock()
    failed_runtime.run.side_effect = RuntimeError("PRIVATE PROMPT")
    monkeypatch.setattr(runtime, "AriettaVoiceRuntime", lambda *args: failed_runtime)
    assert cli.main() == 1
    assert "RuntimeError" in caplog.text
    assert ("PRIVATE PROMPT" in caplog.text) is content_enabled


def test_admin_setup_uses_hidden_prompts_and_never_prints_password(tmp_path, monkeypatch, capsys):
    from arietta_voice.admin_auth import load_admin_credentials

    path = tmp_path / "app.toml"
    path.write_text("")
    monkeypatch.delenv("ARIETTA_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(sys, "argv", ["arietta-voice", "admin-setup", "--config", str(path)])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "synthetic-long-password")
    assert cli.main() == 0
    assert "synthetic-long-password" not in capsys.readouterr().out
    assert load_admin_credentials(path).verify("admin", "synthetic-long-password")


def test_admin_setup_refuses_echoing_password_fallback(tmp_path, monkeypatch, capsys):
    import warnings

    path = tmp_path / "app.toml"
    path.write_text("")
    monkeypatch.setattr(sys, "argv", ["arietta-voice", "admin-setup", "--config", str(path)])

    def unsafe_prompt(prompt):
        warnings.warn("Cannot hide input", cli.getpass.GetPassWarning)
        raise AssertionError("Password input must not proceed without echo suppression")

    monkeypatch.setattr(cli.getpass, "getpass", unsafe_prompt)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "interactive terminal" in capsys.readouterr().err
    assert not (tmp_path / "app.admin.json").exists()
