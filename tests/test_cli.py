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
