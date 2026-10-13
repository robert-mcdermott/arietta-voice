import asyncio
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice.streaming import speech_stream
from arietta_voice.turns import CancellationToken


def test_slow_playback_has_bounded_prefetch_and_closing_stops_producer():
    worker = Mock(pid=42)
    token = CancellationToken()
    count, filled = [], threading.Event()
    def infer(operation, payload=None, *, cancellation):
        cancellation.check()
        if operation == 'tts_next':
            count.append(1)
            if len(count) == 10:
                filled.set()
            return np.zeros(4096, dtype=np.float32), 24000
    async def consume():
        stream = speech_stream('Synthetic', infer=infer, worker=worker, cancellation=token)
        await anext(stream)
        assert filled.wait(1)
        assert len(count) == 10  # One consumed, eight queued, one producer-held.
        await stream.aclose()
    asyncio.run(consume())
    assert not token.cancelled  # Normal stream disposal does not cancel memory work.
    worker.call.assert_called_once_with('tts_close', timeout_s=1)
    assert not any(t.name == 'arietta-speech-producer' for t in threading.enumerate())


@pytest.mark.parametrize('failure', ['text', 'audio', 'model'])
def test_stream_errors_do_not_leave_background_work_or_stale_audio(failure):
    worker, token = Mock(pid=42), CancellationToken()
    def infer(operation, payload=None, **kwargs):
        if operation == 'tts_next':
            if failure == 'model':
                raise RuntimeError('Synthetic failure')
            return np.zeros(4097, dtype=np.float32), 24000
    async def consume():
        stream = speech_stream('x' * 12001 if failure == 'text' else 'Synthetic',
                               infer=infer, worker=worker, cancellation=token)
        with pytest.raises((ValueError, RuntimeError)):
            await anext(stream)
        await stream.aclose()
    asyncio.run(consume())
    assert not any(t.name == 'arietta-speech-producer' for t in threading.enumerate())
