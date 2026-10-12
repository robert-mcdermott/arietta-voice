from __future__ import annotations

from arietta_voice.config import default_config
from arietta_voice.diagnostics import run_startup_diagnostics


def test_startup_diagnostics_without_hardware(tmp_path) -> None:
    config = default_config(tmp_path)
    config.prompts.soul_path = tmp_path / "SOUL.md"
    config.prompts.memory_path = tmp_path / "MEMORY.md"
    config.logging.directory = tmp_path / "logs"
    config.logging.events_file = config.logging.directory / "events.jsonl"
    config.knowledge.directory = tmp_path / "knowledge"
    config.prompts.soul_path.write_text("hello", encoding="utf-8")

    report = run_startup_diagnostics(config, include_hardware=False)

    assert report["overall_status"] == "ok"
    assert any(check["name"] == "prompt_soul" and check["ok"] for check in report["checks"])
    assert any(check["name"] == "wake_word_phrases" and check["ok"] for check in report["checks"])
    assert any(check["name"] == "logging_directory" and check["ok"] for check in report["checks"])
    assert any(check["name"] == "knowledge_directory" and check["ok"] for check in report["checks"])
