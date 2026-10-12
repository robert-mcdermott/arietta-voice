from dataclasses import asdict
from pathlib import Path
import subprocess

import pytest

from arietta_voice.config import load_config
from arietta_voice.private_config import initialize_private_config, local_config_path, update_private_section


def test_local_overrides_merge_by_setting_and_resolve_relative_paths(tmp_path):
    base = tmp_path / "custom.toml"
    base.write_text('[assistant]\nname="Original"\nmax_history_turns=2\n[tools]\nenabled=["local_time"]')
    local_config_path(base).write_text('[assistant]\nname="Private"\n[tools]\nenabled=[]\n[prompts]\nsoul_path="SOUL.local.md"')
    config = load_config(base)
    assert config.assistant.name == "Private"
    assert config.assistant.max_history_turns == 2
    assert config.tools.enabled == ()
    assert config.prompts.soul_path == tmp_path / "SOUL.local.md"
    assert load_config(local_config_path(base)).assistant.name == "Private"


def test_explicit_config_does_not_load_unrelated_private_settings(tmp_path):
    (tmp_path / "arietta_voice.local.toml").write_text('[assistant]\nname="Private"')
    custom = tmp_path / "other.toml"
    custom.write_text('[assistant]\nname="Other"')
    assert load_config(custom).assistant.name == "Other"


def test_private_config_is_validated_and_unknown_fields_rejected(tmp_path):
    base = tmp_path / "custom.toml"
    base.write_text("")
    local_config_path(base).write_text('[admin]\nallow_source_editing="false"')
    with pytest.raises(ValueError, match="admin.allow_source_editing"):
        load_config(base)


def test_initialization_preserves_effective_settings_and_original_files(tmp_path):
    base = tmp_path / "arietta_voice.toml"
    base.write_text('[assistant]\nname="Private persona"\n[models]\naws_secret_access_key="synthetic-secret"\n[prompts]\nsoul_path="custom-soul.md"')
    soul = tmp_path / "custom-soul.md"
    soul.write_text("Private custom persona.\n")
    original_config, original_soul = base.read_bytes(), soul.read_bytes()
    before = asdict(load_config(base))
    result = initialize_private_config(base)
    after = asdict(load_config(base))
    after["prompts"]["soul_path"] = before["prompts"]["soul_path"]
    assert before == after
    assert base.read_bytes() == original_config
    assert soul.read_bytes() == original_soul
    assert Path(result["persona"]).read_bytes() == original_soul
    assert Path(result["backup"], base.name).read_bytes() == original_config
    assert Path(result["config"]).stat().st_mode & 0o777 == 0o600
    assert Path(result["persona"]).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="already exists"):
        initialize_private_config(base)


def test_admin_updates_only_private_config_and_preserves_unrelated_values(tmp_path):
    base = tmp_path / "arietta_voice.toml"
    base.write_text('[assistant]\nname="Original"\n[tools]\nenabled=[]')
    original = base.read_bytes()
    local = local_config_path(base)
    local.write_text('[models]\naws_secret_access_key="synthetic-secret"\n[assistant]\nname="Private"')
    update_private_section(base, "tools", {"enabled": ["local_time"]})
    update_private_section(base, "model_tools", {"enabled": []})
    config = load_config(base)
    assert base.read_bytes() == original
    assert config.models.aws_secret_access_key == "synthetic-secret"
    assert config.assistant.name == "Private"
    assert config.tools.enabled == ("local_time",)
    assert config.model_tools.enabled == ()
    assert local.stat().st_mode & 0o777 == 0o600


def test_repository_ignores_private_settings_and_backups():
    root = Path(__file__).resolve().parents[1]
    private_paths = ["config/arietta_voice.local.toml", "config/SOUL.local.md",
                     "config/arietta_voice.admin.json", "config/arietta_voice.local.toml.lock",
                     "config/.private-backups/example/original.toml", "config/arietta_voice.toml.working",
                     "config/.arietta_voice.local.toml.partial", "config/.SOUL.local.md.partial",
                     "config/.arietta_voice.admin.json.partial"]
    result = subprocess.run(["git", "check-ignore", "--no-index", "--", *private_paths], cwd=root, capture_output=True, text=True, check=True)
    assert set(result.stdout.splitlines()) == set(private_paths)
