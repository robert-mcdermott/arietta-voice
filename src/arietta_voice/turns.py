"""Turn-scoped cancellation and bounded producer/consumer channels.

Limits apply to item size as well as item count. Cancellation clears queued data,
wakes both sides and never allows a cancelled channel to be reused for a new turn.
"""
from collections import deque
import threading
import time


class TurnCancelled(Exception):
    pass


class TurnTimeout(TimeoutError):
    pass


class ChannelClosed(Exception):
    pass


class ChannelEmpty(Exception):
    pass


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks = set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def check(self, deadline=None):
        if self.cancelled:
            raise TurnCancelled('Turn cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise TurnTimeout('Turn exceeded its time limit')

    def cancel(self):
        with self._lock:
            self._event.set()
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            callback()

    def subscribe(self, callback):
        with self._lock:
            self._callbacks.add(callback)
            cancelled = self.cancelled
        if cancelled:
            callback()

    def unsubscribe(self, callback):
        with self._lock:
            self._callbacks.discard(callback)


class BoundedChannel:
    def __init__(self, token, *, max_items, max_item_size, size=len, deadline=None):
        if max_items < 1 or max_item_size < 1:
            raise ValueError('Channel limits must be positive')
        self.token, self.deadline = token, deadline
        self.max_items, self.max_item_size, self.size = max_items, max_item_size, size
        self._items = deque()
        self._condition = threading.Condition()
        self._finished = False
        self._error = None
        token.subscribe(self._discard)

    def _discard(self):
        with self._condition:
            self._items.clear()
            self._condition.notify_all()

    @property
    def queued(self):
        with self._condition:
            return len(self._items)

    def put(self, item):
        if self.size(item) > self.max_item_size:
            raise ValueError('Stream item exceeds its size limit')
        with self._condition:
            while True:
                self.token.check(self.deadline)
                if self._finished:
                    raise ChannelClosed('Channel closed')
                if len(self._items) < self.max_items:
                    self._items.append(item)
                    self._condition.notify_all()
                    return
                self._condition.wait(.05)

    def get(self, on_wait=None, timeout=None):
        until = time.monotonic() + timeout if timeout is not None else None
        with self._condition:
            while True:
                self.token.check(self.deadline)
                if self._error is not None:
                    raise self._error
                if self._items:
                    item = self._items.popleft()
                    self._condition.notify_all()
                    return item
                if self._finished:
                    raise ChannelClosed('Channel drained')
                if until is not None and time.monotonic() >= until:
                    raise ChannelEmpty()
                wait = .05 if until is None else min(.05, max(0, until - time.monotonic()))
                self._condition.wait(wait)
                if on_wait is not None:
                    on_wait()

    def finish(self, error=None):
        with self._condition:
            self._finished = True
            self._error = error
            if error is not None:
                self._items.clear()
            self._condition.notify_all()

    def close(self):
        self.finish()
        self._discard()
        self.token.unsubscribe(self._discard)


TEXT_QUEUE_ITEMS = 2
TEXT_ITEM_CHARS = 12_000
AUDIO_QUEUE_ITEMS = 8
AUDIO_ITEM_SAMPLES = 4096
