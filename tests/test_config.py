from __future__ import annotations

from pathlib import Path

import pytest

from arietta_voice.config import config_to_dict, load_config


def test_load_config_resolves_relative_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    file_path = config_dir / "arietta_voice.toml"
    file_path.write_text(
        """
[assistant]
name = "Test Assistant"

[wake_word]
phrases = ["test assistant"]
session_timeout_s = 42.0

[tools]
enabled = ["assistant_identity"]

[model_tools]
enabled = ["system_info"]

[prompts]
soul_path = "SOUL.md"
memory_path = "MEMORY.md"

[models]
chat_provider = "bedrock"
chat_model = "us.anthropic.claude-sonnet-4-6"
bedrock_region = "us-west-2"

[knowledge]
backend = "hybrid"
directory = "../knowledge"
index_directory = "../.knowledge_index"
embedding_model = "BAAI/bge-small-en-v1.5"
hybrid_alpha = 0.6
min_vector_score = 0.25

[memory]
enabled = true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(file_path)

    assert config.assistant.name == "Test Assistant"
    assert config.prompts.soul_path == (config_dir / "SOUL.md").resolve()
    assert config.prompts.memory_path == (config_dir / "MEMORY.md").resolve()
    assert config.knowledge.directory == (tmp_path / "knowledge").resolve()
    assert config.knowledge.index_directory == (tmp_path / ".knowledge_index").resolve()
    assert config.knowledge.backend == "hybrid"
    assert config.knowledge.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.knowledge.hybrid_alpha == 0.6
    assert config.knowledge.min_vector_score == 0.25
    assert config.tools.enabled == ("assistant_identity",)
    assert config.model_tools.enabled == ("system_info",)
    assert config.wake_word.phrases == ("test assistant",)
    assert config.wake_word.session_timeout_s == 42.0
    assert config.models.chat_provider == "bedrock"
    assert config.models.chat_model == "us.anthropic.claude-sonnet-4-6"
    assert config.models.bedrock_region == "us-west-2"
    assert config.memory.enabled is True


def test_config_to_dict_redacts_aws_secrets() -> None:
    config = load_config()
    config.models.aws_access_key_id = "AKIATESTVALUE"
    config.models.aws_secret_access_key = "secret-value"
    config.models.aws_session_token = "session-value"

    payload = config_to_dict(config)

    assert payload["models"]["aws_access_key_id"] == "***REDACTED***"
    assert payload["models"]["aws_secret_access_key"] == "***REDACTED***"
    assert payload["models"]["aws_session_token"] == "***REDACTED***"


def test_load_config_allows_disabling_all_tools(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    file_path = config_dir / "arietta_voice.toml"
    file_path.write_text(
        """
[tools]
enabled = []
""".strip(),
        encoding="utf-8",
    )

    config = load_config(file_path)

    assert config.tools.enabled == ()


def test_load_config_allows_disabling_all_model_tools(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    file_path = config_dir / "arietta_voice.toml"
    file_path.write_text(
        """
[model_tools]
enabled = []
""".strip(),
        encoding="utf-8",
    )

    config = load_config(file_path)

    assert config.model_tools.enabled == ()


@pytest.mark.parametrize("section, setting", [
    ("assistant", 'max_history_turns = -1'),
    ("assistant", 'max_history_turns = 1.5'),
    ("assistant", 'max_history_turns = true'),
    ("wake_word", 'enabled = "false"'),
    ("wake_word", 'phrases = "Bridget"'),
    ("wake_word", 'phrases = []'),
    ("wake_word", 'phrases = ["?!"]'),
    ("wake_word", 'backend = ""'),
    ("wake_word", 'session_timeout_s = nan'),
    ("wake_word", 'session_timeout_s = 0'),
    ("audio", 'sample_rate = 48000'),
    ("audio", 'chunk_samples = 160'),
    ("audio", 'silence_ms = 0'),
    ("audio", 'barge_in_threshold_chunks = -1'),
    ("audio", 'tts_speed = inf'),
    ("models", 'chat_provider = "unsupported"'),
    ("models", 'chat_provider = ""'),
    ("models", 'max_tokens = 0'),
    ("models", 'temperature = -0.1'),
    ("memory", 'consolidate_every_turns = 0'),
    ("knowledge", 'backend = "unsupported"'),
    ("knowledge", 'backend = ""'),
    ("knowledge", 'embedding_model = ""'),
    ("knowledge", 'hybrid_alpha = 1.5'),
    ("knowledge", 'min_vector_score = nan'),
    ("knowledge", 'max_matches = 0'),
    ("knowledge", 'snippet_char_limit = -1'),
    ("logging", 'content_enabled = "false"'),
    ("logging", 'max_bytes = 100'),
    ("logging", 'backup_count = -1'),
])
def test_invalid_config_identifies_setting(tmp_path: Path, section: str, setting: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(f"[{section}]\n{setting}\n")
    key = setting.split("=", 1)[0].strip()
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        load_config(path)


@pytest.mark.parametrize("text, message", [
    ('[logging]\ncontent_enable = true', 'logging.content_enable'),
    ('[unknown]\nenabled = true', 'unknown'),
    ('logging = false', 'logging must be a TOML table'),
])
def test_unknown_or_malformed_sections_are_rejected(tmp_path, text, message) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_zero_history_and_disabled_empty_wake_are_supported(tmp_path) -> None:
    path = tmp_path / "valid.toml"
    path.write_text('[assistant]\nmax_history_turns = 0\n[wake_word]\nenabled = false\nphrases = []')
    config = load_config(path)
    assert config.assistant.max_history_turns == 0
    assert config.wake_word.phrases == ()
    assert not config.logging.content_enabled


def test_logging_configuration(tmp_path) -> None:
    path = tmp_path / "valid.toml"
    path.write_text('[logging]\ncontent_enabled = true\nmax_bytes = 2048\nbackup_count = 1')
    config = load_config(path)
    assert config.logging.content_enabled
    assert config.logging.max_bytes == 2048
    assert config.logging.backup_count == 1
