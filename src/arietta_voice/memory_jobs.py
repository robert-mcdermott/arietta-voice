"""Idle-only memory work on the existing serialized inference owner.

Only delivered turns are submitted. Pending text stays in bounded RAM; interrupted
jobs are discarded, never persisted or retried. Foreground callers must suspend
this scheduler before using the worker, and allow idle work only while listening.
"""
from collections import deque
import threading
import time

from arietta_voice.turns import CancellationToken
from arietta_voice.worker import WorkerCancelled, WorkerFailure


class MemoryJobs:
    MAX_PENDING = 2
    MAX_TEXT_CHARS = 12_000
    IDLE_DELAY_S = 1.0
    PREEMPT_GRACE_S = .1

    def __init__(self, worker, *, timeout_s, log):
        self.worker = worker
        self.timeout_s = min(timeout_s, 30.0)
        self._log = log
        self._condition = threading.Condition()
        self._pending = deque()
        self._thread = None
        self._closed = False
        self._allowed = False
        self._ready_at = 0.0
        self._preempt_at = None
        self._active = False
        self._token = None
        self._counts = dict(completed=0, cancelled=0, failed=0, dropped=0, conflicts=0)
        self._last_error = None

    def log(self, event, **fields):
        try:
            self._log(event, **fields)
        except OSError:
            # Losing diagnostics must not lose the scheduler or delivered turn.
            pass

    def submit(self, user, assistant, turn_count):
        with self._condition:
            if self._closed:
                return False
            if max(len(user), len(assistant)) > self.MAX_TEXT_CHARS:
                self._counts['dropped'] += 1
                self.log('memory_job_dropped', reason='text_limit')
                return False
            if len(self._pending) == self.MAX_PENDING:
                self._pending.popleft()
                self._counts['dropped'] += 1
                self.log('memory_job_dropped', reason='queue_full')
            self._pending.append(dict(user=user, assistant=assistant, turn_count=turn_count))
            self._condition.notify_all()
            return True

    def allow_idle(self):
        with self._condition:
            if self._closed:
                return
            if not self._allowed:
                self._allowed = True
                self._ready_at = time.monotonic() + self.IDLE_DELAY_S
            if self._pending and self._thread is None:
                self._thread = threading.Thread(target=self._run, name='arietta-memory', daemon=True)
                self._thread.start()
            self._condition.notify_all()

    def suspend(self, *, wait=True):
        with self._condition:
            self._allowed = False
            if self._preempt_at is None:
                self._preempt_at = time.monotonic()
            self._condition.notify_all()
            if not wait:
                return
            end = time.monotonic() + 2.5
            while self._active:
                if time.monotonic() >= end:
                    raise WorkerFailure('Memory worker did not release inference ownership')
                self._condition.wait(.05)

    def _on_wait(self):
        with self._condition:
            if (not self._allowed and self._token is not None and self._preempt_at is not None
                    and time.monotonic() - self._preempt_at >= self.PREEMPT_GRACE_S):
                self._token.cancel()

    def _check_idle(self):
        with self._condition:
            if not self._allowed or self._closed:
                raise WorkerCancelled('Foreground work has priority')

    def _run(self):
        while True:
            with self._condition:
                while not self._closed:
                    remaining = self._ready_at - time.monotonic()
                    if self._allowed and self._pending and remaining <= 0:
                        break
                    self._condition.wait(max(.01, remaining) if self._allowed and self._pending else None)
                if self._closed:
                    return
                payload = self._pending.popleft()
                self._active = True
                self._preempt_at = None
                token = self._token = CancellationToken()
            deadline = time.monotonic() + self.timeout_s
            def call(operation, data=None):
                self._check_idle()
                token.check(deadline)
                return self.worker.call(operation, data, timeout_s=max(.001, deadline-time.monotonic()),
                                        cancellation=token, on_wait=self._on_wait)
            try:
                call('memory_start', payload)
                while True:
                    result = call('memory_next')
                    if result['kind'] == 'prepared':
                        break
                    if result['kind'] != 'progress':
                        raise ValueError('Invalid memory event')
                result = call('memory_commit')
                with self._condition:
                    self._counts['conflicts' if result.get('conflict') else 'completed'] += 1
                    self._last_error = None
                self.log('memory_update_conflict' if result.get('conflict') else 'memory_updated', **result)
            except WorkerCancelled:
                with self._condition:
                    self._counts['cancelled'] += 1
                self.log('memory_update_cancelled')
            except Exception as exc:
                with self._condition:
                    self._counts['failed'] += 1
                    self._last_error = type(exc).__name__
                self.log('memory_update_failed', error_type=type(exc).__name__)
            finally:
                # Close the paused model generator before foreground can reuse
                # its processor/detokenizer. A stuck close is also killable.
                if self.worker.pid is not None:
                    try:
                        self.worker.call('memory_close', timeout_s=.5)
                    except Exception:
                        pass
                payload = result = None  # Release private text even while idle.
                with self._condition:
                    self._active = False
                    self._token = None
                    self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return dict(queued=len(self._pending), active=self._active, **self._counts,
                        last_error=self._last_error)

    def close(self):
        with self._condition:
            self._closed = True
            self._allowed = False
            self._preempt_at = time.monotonic() - self.PREEMPT_GRACE_S
            self._counts['dropped'] += len(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        self.suspend()
        if self._thread is not None:
            self._thread.join(timeout=1)
            if self._thread.is_alive():
                raise WorkerFailure('Memory scheduler did not stop')
