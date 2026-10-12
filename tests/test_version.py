from importlib.metadata import version
from pathlib import Path
import sys
import tomllib

import pytest

from arietta_voice import __version__, cli


def test_installed_version_matches_project_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == version("arietta-voice") == project["project"]["version"]


@pytest.mark.parametrize("arguments", [["--version"], ["run", "--version"]])
def test_version_exits_before_loading_config(arguments, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["arietta-voice", *arguments])

    def unexpected_config_load(*args, **kwargs):
        pytest.fail("Version reporting must not load config or start the assistant")

    monkeypatch.setattr(cli, "load_config", unexpected_config_load)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == f"arietta-voice {__version__}"
