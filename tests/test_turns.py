import threading
import time

import pytest

from arietta_voice.turns import (
    BoundedChannel, CancellationToken, ChannelClosed, TurnCancelled, TurnTimeout,
)


def test_full_queue_blocks_then_cancel_discards_and_wakes_producer():
    token = CancellationToken()
    channel = BoundedChannel(token, max_items=1, max_item_size=4)
    channel.put('old')
    errors, started = [], threading.Event()
    def produce():
        started.set()
        try:
            channel.put('next')
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=produce)
    thread.start()
    assert started.wait(1)
    assert channel.queued == 1
    token.cancel()
    thread.join(1)
    assert not thread.is_alive()
    assert isinstance(errors[0], TurnCancelled)
    assert channel.queued == 0
    with pytest.raises(TurnCancelled):
        channel.get()
    channel.close()


def test_finished_channel_drains_but_errors_discard_pending_data():
    token = CancellationToken()
    channel = BoundedChannel(token, max_items=2, max_item_size=3)
    with pytest.raises(ValueError):
        channel.put('oversized')
    channel.put('one')
    channel.finish()
    assert channel.get() == 'one'
    with pytest.raises(ChannelClosed):
        channel.get()
    with pytest.raises(ChannelClosed):
        channel.put('two')
    channel.close()
    failed = BoundedChannel(token, max_items=2, max_item_size=3)
    failed.put('old')
    failed.finish(RuntimeError('synthetic'))
    with pytest.raises(RuntimeError):
        failed.get()
    assert failed.queued == 0
    failed.close()


def test_empty_queue_wait_obeys_deadline_and_cancellation():
    token = CancellationToken()
    channel = BoundedChannel(token, max_items=1, max_item_size=3, deadline=time.monotonic() + .02)
    with pytest.raises(TurnTimeout):
        channel.get()
    channel.close()
    token.cancel()
    cancelled = BoundedChannel(token, max_items=1, max_item_size=3)
    with pytest.raises(TurnCancelled):
        cancelled.put('new')
    cancelled.close()


def test_short_consumer_timeout_does_not_sleep_for_full_poll_interval(monkeypatch):
    from arietta_voice import turns
    now, waits = [10.], []
    monkeypatch.setattr(turns.time, 'monotonic', lambda:now[0])
    channel = BoundedChannel(CancellationToken(), max_items=1, max_item_size=3)
    def wait(duration):
        waits.append(duration)
        now[0] += duration
    monkeypatch.setattr(channel._condition, 'wait', wait)
    with pytest.raises(turns.ChannelEmpty):
        channel.get(timeout=.01)
    assert sum(waits) == pytest.approx(.01)
    channel.close()
