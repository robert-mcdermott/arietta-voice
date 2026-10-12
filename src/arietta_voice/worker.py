"""One killable, serialized inference owner. No queues of abandoned jobs."""
from __future__ import annotations

import atexit
import multiprocessing
import signal
import threading
import time
from typing import Any, Callable


class WorkerTimeout(TimeoutError):
    pass


class WorkerBusy(RuntimeError):
    pass


class WorkerFailure(RuntimeError):
    pass


def _worker_main(connection, config, handler_factory) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    handler = None
    try:
        connection.send(("ready", None))
        while True:
            operation, payload = connection.recv()
            try:
                if handler is None:
                    handler = handler_factory(config)
                result = handler.handle(operation, payload)
                connection.send(("ok", result))
            except Exception as exc:
                # Do not transport arbitrary exception text containing prompts.
                connection.send(("error", type(exc).__name__))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def _default_handler(config):
    from arietta_voice.inference import InferenceEngine
    return InferenceEngine(config)


class InferenceWorker:
    def __init__(self, config, *, handler_factory: Callable = _default_handler) -> None:
        self.config = config
        self.handler_factory = handler_factory
        self._process = None
        self._connection = None
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._closed = False
        self._invalidated = False
        self.operation: str | None = None
        self.restarts = 0
        atexit.register(self.close)

    @property
    def pid(self) -> int | None:
        with self._lifecycle_lock:
            return self._process.pid if self._process is not None and self._process.is_alive() else None

    def invalidate(self) -> None:
        """Reload configuration/plugins before the next job, without replaying this one."""
        with self._lifecycle_lock:
            self._invalidated = True

    def _receive(self, deadline: float, on_wait: Callable | None):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout(f"{self.operation} exceeded its time limit")
            connection = self._connection
            try:
                if connection is None:
                    raise WorkerFailure("Inference worker stopped")
                if connection.poll(min(remaining, 0.1)):
                    return connection.recv()
            except (EOFError, OSError, ValueError) as exc:
                raise WorkerFailure("Inference worker connection closed") from exc
            if on_wait is not None:
                on_wait()
            if self.pid is None:
                raise WorkerFailure("Inference worker exited")

    def call(self, operation: str, payload: Any = None, *, timeout_s: float, on_wait: Callable | None = None):
        if not self._lock.acquire(blocking=False):
            raise WorkerBusy("Inference is busy; wait for the current operation")
        self.operation = operation
        deadline = time.monotonic() + timeout_s
        try:
            with self._lifecycle_lock:
                if self._closed:
                    raise WorkerFailure("Inference worker is closed")
                needs_start = self.pid is None or self._invalidated
                if needs_start:
                    self._stop()
                    context = multiprocessing.get_context("spawn")
                    self._connection, child = context.Pipe()
                    self._process = context.Process(target=_worker_main, args=(child, self.config, self.handler_factory), daemon=True)
                    try:
                        self._process.start()
                    finally:
                        child.close()
                    self._invalidated = False
            if needs_start:
                if self._receive(deadline, on_wait)[0] != "ready":
                    raise WorkerFailure("Inference worker did not start")
            self._connection.send((operation, payload))
            kind, result = self._receive(deadline, on_wait)
            if kind != "ok":
                raise WorkerFailure(f"{operation} failed ({result})")
            return result
        except BaseException:
            self.restarts += 1
            self._stop()
            raise
        finally:
            self.operation = None
            self._lock.release()

    def _stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        process = self._process
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.5)
            if process.is_alive():
                # Never create a replacement while the previous owner lives.
                raise WorkerFailure("Inference worker could not be stopped")
            if process.pid is not None:
                process.join(timeout=0)
            process.close()
        self._process = None
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def close(self) -> None:
        # Shutdown is bounded, including when another request is waiting.
        with self._lifecycle_lock:
            self._closed = True
            self._stop()
        atexit.unregister(self.close)
