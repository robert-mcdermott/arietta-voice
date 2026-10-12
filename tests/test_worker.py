"""Exercise real spawned workers without loading models or audio hardware."""
import os
from pathlib import Path
import threading
import time

import pytest

from arietta_voice.worker import InferenceWorker, WorkerBusy, WorkerFailure, WorkerTimeout


class Handler:
    def __init__(self, config):
        self.config = config

    def handle(self, operation, payload):
        if operation == "config":
            return self.config.assistant.name
        if operation == "fail":
            raise RuntimeError("PRIVATE prompt")
        if operation == "crash":
            os._exit(3)
        if operation == "late_write":
            time.sleep(1)
            Path(payload).write_text("should never happen")
        if operation == "wait":
            time.sleep(2)
        return os.getpid()


@pytest.fixture
def worker():
    instance = InferenceWorker(None, handler_factory=Handler)
    yield instance
    instance.close()


def test_worker_serializes_reuses_and_reloads(worker):
    first = worker.call("ok", timeout_s=5)
    assert worker.call("ok", timeout_s=5) == first
    worker.invalidate()
    assert worker.call("ok", timeout_s=5) != first


def test_spawned_worker_receives_config_snapshot_and_reloads_only_when_invalidated(tmp_path):
    from arietta_voice.config import default_config
    config = default_config(tmp_path)
    config.assistant.name = "Synthetic first"
    instance = InferenceWorker(config, handler_factory=Handler)
    try:
        assert instance.call("config", timeout_s=5) == "Synthetic first"
        config.assistant.name = "Synthetic updated"
        assert instance.call("config", timeout_s=5) == "Synthetic first"
        instance.invalidate()
        assert instance.call("config", timeout_s=5) == "Synthetic updated"
    finally:
        instance.close()


def test_timeout_kills_work_before_next_request(worker, tmp_path):
    first = worker.call("ok", timeout_s=5)
    marker = tmp_path / "late_action"
    with pytest.raises(WorkerTimeout):
        worker.call("late_write", str(marker), timeout_s=0.1)
    assert worker.pid is None
    assert worker.call("ok", timeout_s=5) != first
    time.sleep(1.1)
    assert not marker.exists()
    assert worker.restarts == 1


@pytest.mark.parametrize("operation", ["fail", "crash"])
def test_crashes_and_exceptions_allow_next_request(worker, operation):
    with pytest.raises(WorkerFailure) as error:
        worker.call(operation, timeout_s=5)
    assert "PRIVATE" not in str(error.value)
    assert worker.pid is None
    assert worker.call("ok", timeout_s=5) == worker.pid


def test_busy_request_has_no_queue_and_shutdown_cancels_active_work(worker):
    worker.call("ok", timeout_s=5)
    waiting = threading.Event()
    errors = []
    def run():
        try:
            worker.call("wait", timeout_s=5, on_wait=waiting.set)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    assert waiting.wait(2)
    with pytest.raises(WorkerBusy):
        worker.call("ok", timeout_s=5)
    worker.close()
    thread.join(2)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert worker.pid is None
    with pytest.raises(WorkerFailure, match="closed"):
        worker.call("ok", timeout_s=5)
