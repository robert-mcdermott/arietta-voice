from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from arietta_voice.logging_utils import EventLogger


def test_content_is_omitted_unless_enabled(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path)
    logger.log("user_transcript", content={"text": "PRIVATE SPEECH"}, latency_ms=42)
    logger.log("tool_call", content={"details": {"custom_key": "PRIVATE RESULT"}}, tool="lookup")
    logger.log("legacy_event", text="PRIVATE LEGACY", details={"query": "PRIVATE QUERY"}, error="PRIVATE ERROR")
    text = path.read_text()
    assert "PRIVATE" not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert records[0]["latency_ms"] == 42
    assert records[1]["tool"] == "lookup"

    EventLogger(path, content_enabled=True).log("user_transcript", content={"text": "OPTED IN"})
    assert json.loads(path.read_text().splitlines()[-1])["text"] == "OPTED IN"


def test_rotation_bounds_storage_and_keeps_newest_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, max_bytes=1024, backup_count=2)
    for index in range(100):
        logger.log("test", index=index, padding="x" * 100)
    files = [path, path.with_name("events.jsonl.1"), path.with_name("events.jsonl.2")]
    assert all(file.exists() and file.stat().st_size <= 1024 for file in files)
    assert not path.with_name("events.jsonl.3").exists()
    assert json.loads(path.read_text().splitlines()[-1])["index"] == 99
    for file in files:
        assert all(isinstance(json.loads(line), dict) for line in file.read_text().splitlines())


def test_zero_backups_and_oversized_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path, content_enabled=True, max_bytes=1024, backup_count=0)
    logger.log("large", content={"text": "x" * 2000})
    assert json.loads(path.read_text())["fields_omitted"]
    for index in range(30):
        logger.log("test", index=index)
    assert path.stat().st_size <= 1024
    assert not path.with_name("events.jsonl.1").exists()


def test_rotation_applies_reduced_retention_and_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    for suffix in ("", ".1", ".2", ".3"):
        log = path.with_name(path.name + suffix)
        log.write_text("x" * 1024)
        log.chmod(0o644)
    unrelated = tmp_path / "other.jsonl.3"
    unrelated.write_text("unrelated")
    EventLogger(path, max_bytes=1024, backup_count=1).log("new")
    assert not path.with_name("events.jsonl.2").exists()
    assert not path.with_name("events.jsonl.3").exists()
    assert unrelated.exists()
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.with_name("events.jsonl.1").stat().st_mode & 0o777 == 0o600


def _write_events(path: str, writer: int) -> None:
    logger = EventLogger(Path(path))
    for index in range(30):
        logger.log("concurrent", writer=writer, index=index)


def test_separate_processes_can_share_event_log(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = [pool.submit(_write_events, str(path), writer) for writer in range(2)]
        for result in results:
            result.result(timeout=30)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {(item["writer"], item["index"]) for item in records} == {
        (writer, index) for writer in range(2) for index in range(30)
    }
    assert len(records) == 60
