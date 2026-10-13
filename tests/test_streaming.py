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


def test_early_audio_precedes_later_tokens_and_cancellation_closes_both_streams():
    token, worker = CancellationToken(), Mock(pid=42)
    first_played = threading.Event()
    blocked = threading.Event()
    chat_calls, tts_calls = [], []
    def infer(operation, payload=None, *, cancellation):
        cancellation.check()
        if operation == 'chat_next':
            chat_calls.append(True)
            if len(chat_calls) == 1:
                return {'kind':'clause', 'text':'First sentence.'}
            blocked.set()
            assert first_played.wait(1), 'Generation outran early playback'
            while not cancellation.cancelled:
                cancellation._event.wait(.01)
            cancellation.check()
        if operation == 'tts_next':
            tts_calls.append(True)
            return (np.zeros(4096, dtype=np.float32), 24000) if len(tts_calls) == 1 else None
    async def consume():
        stream = speech_stream(infer=infer, worker=worker, cancellation=token, request={})
        frame, rate = await anext(stream)
        assert rate == 24000 and len(frame) == 4096
        first_played.set()
        assert blocked.wait(1)
        # Closing during pending generation must cancel and join its producer.
        await stream.aclose()
    asyncio.run(consume())
    worker.call.assert_called_once_with('speech_close', timeout_s=1)
    assert not any(t.name == 'arietta-speech-producer' for t in threading.enumerate())


def test_stream_result_only_after_all_clauses_and_error_discards_partial_result():
    for fail in (False, True):
        result, spoken = {}, []
        events = iter([{'kind':'clause','text':'First.'}, {'kind':'clause','text':'Second.'},
                       {'kind':'done','result':{'response':'First. Second.'}}])
        frames = []
        def infer(operation, payload=None, **kwargs):
            if operation == 'chat_next':
                event = next(events)
                if fail and event['kind'] == 'done':
                    raise RuntimeError('Synthetic provider failure')
                return event
            if operation == 'tts_start':
                spoken.append(payload)
                frames[:] = [(np.zeros(4096,dtype=np.float32),24000), None]
            if operation == 'tts_next':
                return frames.pop(0)
        async def consume():
            async for _ in speech_stream(infer=infer, worker=Mock(pid=1), cancellation=CancellationToken(),
                                         request={}, on_result=result.update):
                pass
        if fail:
            with pytest.raises(RuntimeError):
                asyncio.run(consume())
            assert result == {}
        else:
            asyncio.run(consume())
            assert result['response'] == 'First. Second.'
        assert spoken == ['First.', 'Second.']
