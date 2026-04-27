from __future__ import annotations

from pathlib import Path
import signal

from arietta_voice.config import default_config
from arietta_voice.runtime_supervisor import RuntimeSupervisor


class FakeProcess:
    pid = 4321

    def __init__(self) -> None:
        self.returncode = None
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, value: int) -> None:
        self.signals.append(value)
        self.returncode = 0

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None):
        return self.returncode


def _config(tmp_path: Path):
    config = default_config(tmp_path)
    config.logging.directory = tmp_path / "logs"
    config.logging.events_file = tmp_path / "logs" / "events.jsonl"
    return config


def test_runtime_supervisor_builds_module_command(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "arietta_voice.toml"
    supervisor = RuntimeSupervisor(_config(tmp_path), config_path=config_path)

    command = supervisor.command()

    assert command[1:] == ["-m", "arietta_voice.cli", "run", "--config", str(config_path)]


def test_runtime_supervisor_starts_and_stops_fake_process(tmp_path: Path) -> None:
    created: list[FakeProcess] = []

    def popen_factory(*args, **kwargs):
        process = FakeProcess()
        created.append(process)
        return process

    supervisor = RuntimeSupervisor(_config(tmp_path), popen_factory=popen_factory)

    start = supervisor.start()
    stop = supervisor.stop()

    assert start["changed"] is True
    assert start["managed_process_id"] == 4321
    assert stop["changed"] is True
    assert created[0].signals == [signal.SIGINT]
    assert stop["managed_exit_code"] == 0
