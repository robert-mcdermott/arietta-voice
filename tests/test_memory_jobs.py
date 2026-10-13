import threading
import time
from unittest.mock import Mock

import pytest

from arietta_voice.memory_jobs import MemoryJobs
from arietta_voice.worker import InferenceWorker


def until(predicate, timeout=3):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, 'Background job did not reach expected state'
        time.sleep(.005)


class CooperativeWorker:
    pid = 1
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.operations = []
    def call(self, op, payload=None, **kw):
        self.operations.append(op)
        if op == 'memory_next':
            self.entered.set()
            assert self.release.wait(2)
            return {'kind':'prepared'}
        if op == 'memory_commit':
            return {'facts_added':1, 'consolidated':False, 'conflict':False}


def make_jobs(worker, log=None):
    jobs = MemoryJobs(worker, timeout_s=2, log=log or Mock())
    jobs.IDLE_DELAY_S = 0
    return jobs


def test_queue_is_bounded_private_ram_and_does_not_start_until_idle():
    worker = CooperativeWorker()
    jobs = make_jobs(worker)
    try:
        for i in range(6):
            assert jobs.submit('Synthetic user '+str(i),'Answer',i+1)
        assert jobs.snapshot()['queued'] == 2
        assert jobs.snapshot()['dropped'] == 4
        assert not jobs.submit('x'*12001,'Answer',7)
        assert not worker.operations
        assert 'Synthetic' not in str(jobs.snapshot())
        jobs.close()
        assert jobs.snapshot()['queued'] == 0
        assert not jobs.submit('Later','Answer',8)
    finally:
        jobs.close()


def test_cooperative_preemption_reuses_worker_without_commit_or_retry():
    worker = CooperativeWorker()
    jobs = make_jobs(worker)
    try:
        jobs.submit('Synthetic','Answer',1)
        jobs.allow_idle()
        assert worker.entered.wait(1)
        jobs.suspend(wait=False)
        worker.release.set()
        jobs.suspend()
        assert jobs.snapshot()['cancelled'] == 1
        assert 'memory_commit' not in worker.operations
        assert worker.operations[-1] == 'memory_close'
        jobs.allow_idle()
        assert jobs.snapshot()['queued'] == 0
    finally:
        worker.release.set()
        jobs.close()


def test_memory_finishes_in_background_and_failures_have_separate_status():
    worker = CooperativeWorker()
    worker.release.set()
    jobs = make_jobs(worker)
    try:
        jobs.submit('Synthetic','Answer',1)
        jobs.allow_idle()
        until(lambda: jobs.snapshot()['completed'] == 1 and not jobs.snapshot()['active'])
        assert worker.operations[-2:] == ['memory_commit','memory_close']
        worker.call = Mock(side_effect=RuntimeError('PRIVATE failure'))
        jobs.submit('Synthetic','Answer',2)
        until(lambda: jobs.snapshot()['failed'] == 1 and not jobs.snapshot()['active'])
        assert jobs.snapshot()['last_error'] == 'RuntimeError'
        assert 'PRIVATE' not in str(jobs.snapshot())
        assert jobs.snapshot()['queued'] == 0
    finally:
        jobs.close()


class StuckMemoryHandler:
    def __init__(self, config): pass
    def handle(self, operation, payload):
        if operation == 'memory_next':
            time.sleep(10)
        return {'kind':'progress'}


def test_stuck_native_memory_is_killed_before_foreground_and_shutdown():
    worker = InferenceWorker(None, handler_factory=StuckMemoryHandler)
    worker.call('warmup', timeout_s=5)
    jobs = make_jobs(worker)
    try:
        jobs.submit('Synthetic','Answer',1)
        jobs.allow_idle()
        until(lambda: worker.operation == 'memory_next')
        started = time.monotonic()
        jobs.suspend()
        assert time.monotonic() - started < 2.5
        assert jobs.snapshot()['cancelled'] == 1
        assert worker.pid is None
        assert worker.call('foreground',timeout_s=5) == {'kind':'progress'}
        jobs.submit('Synthetic','Answer',2)
        jobs.allow_idle()
        until(lambda: worker.operation == 'memory_next')
        jobs.close()
        assert worker.pid is None
        assert not jobs._thread.is_alive()
    finally:
        jobs.close()
        worker.close()


def test_logging_failure_cannot_stop_later_memory_jobs():
    worker = CooperativeWorker()
    worker.release.set()
    jobs = make_jobs(worker, log=Mock(side_effect=OSError('Synthetic log failure')))
    try:
        jobs.submit('Synthetic','Answer',1)
        jobs.allow_idle()
        until(lambda: jobs.snapshot()['completed'] == 1 and not jobs.snapshot()['active'])
        jobs.submit('Next','Answer',2)
        until(lambda: jobs.snapshot()['completed'] == 2 and not jobs.snapshot()['active'])
        assert jobs.snapshot()['failed'] == 0
    finally:
        jobs.close()


def test_background_deadline_terminates_native_work_without_retry():
    worker = InferenceWorker(None, handler_factory=StuckMemoryHandler)
    worker.call('warmup',timeout_s=5)
    jobs = MemoryJobs(worker,timeout_s=.1,log=Mock())
    jobs.IDLE_DELAY_S = 0
    try:
        jobs.submit('Synthetic','Answer',1)
        jobs.allow_idle()
        until(lambda: jobs.snapshot()['failed'] == 1 and not jobs.snapshot()['active'])
        assert jobs.snapshot()['last_error'] == 'TurnTimeout'
        assert jobs.snapshot()['queued'] == 0
        assert worker.pid is None
        assert worker.call('foreground',timeout_s=5) == {'kind':'progress'}
    finally:
        jobs.close()
        worker.close()
