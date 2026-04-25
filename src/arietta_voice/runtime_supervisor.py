from __future__ import annotations

from datetime import datetime, timezone
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from arietta_voice.config import AppConfig
from arietta_voice.runtime_status import read_runtime_status


PopenFactory = Callable[..., subprocess.Popen]


class RuntimeSupervisor:
    def __init__(
        self,
        config: AppConfig,
        *,
        config_path: Path | None = None,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.popen_factory = popen_factory
        self.process: subprocess.Popen | None = None
        self.started_at: str | None = None
        self.last_exit_code: int | None = None
        self.last_error: str | None = None
        self.stdout_path = config.logging.directory / "managed_runtime.stdout.log"
        self.stderr_path = config.logging.directory / "managed_runtime.stderr.log"

    def command(self) -> list[str]:
        command = [sys.executable, "-m", "arietta_voice.cli", "run"]
        if self.config_path is not None:
            command.extend(["--config", str(self.config_path)])
        return command

    def status(self) -> dict[str, Any]:
        process = self.process
        running = process is not None and process.poll() is None
        if process is not None and not running:
            self.last_exit_code = process.returncode
        return {
            "managed_by_api": running,
            "managed_process_id": process.pid if process is not None else None,
            "managed_process_running": running,
            "managed_started_at": self.started_at,
            "managed_exit_code": None if running else self.last_exit_code,
            "managed_command": self.command(),
            "managed_stdout_log": str(self.stdout_path),
            "managed_stderr_log": str(self.stderr_path),
            "managed_last_error": self.last_error,
        }

    def start(self) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            return {"changed": False, "message": "Voice runtime is already managed by the API.", **self.status()}

        observed = read_runtime_status(self.config)
        if observed.get("is_running"):
            return {
                "changed": False,
                "message": "A separate voice runtime is already publishing a fresh heartbeat.",
                **self.status(),
                "observed_runtime": observed,
            }

        self.config.logging.directory.mkdir(parents=True, exist_ok=True)
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = self.stdout_path.open("ab")
        stderr = self.stderr_path.open("ab")
        try:
            self.process = self.popen_factory(
                self.command(),
                cwd=str(self.config.project_root),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stdout.close()
            stderr.close()

        self.started_at = _now_iso()
        self.last_exit_code = None
        self.last_error = None
        return {"changed": True, "message": "Voice runtime started.", **self.status()}

    def stop(self, *, timeout_seconds: float = 8.0) -> dict[str, Any]:
        process = self.process
        if process is None or process.poll() is not None:
            if process is not None:
                self.last_exit_code = process.returncode
            return {
                "changed": False,
                "message": "No API-managed voice runtime is running.",
                **self.status(),
            }

        process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                self.last_exit_code = exit_code
                return {"changed": True, "message": "Voice runtime stopped.", **self.status()}
            time.sleep(0.1)

        process.terminate()
        try:
            exit_code = process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait(timeout=3)
        self.last_exit_code = exit_code
        return {
            "changed": True,
            "message": "Voice runtime was forced to stop after the graceful timeout.",
            **self.status(),
        }

    def restart(self) -> dict[str, Any]:
        stop_result = self.stop()
        start_result = self.start()
        return {
            "changed": bool(stop_result.get("changed")) or bool(start_result.get("changed")),
            "message": "Voice runtime restart requested.",
            "stop": stop_result,
            "start": start_result,
            **self.status(),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
