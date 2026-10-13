"""Bounded text-to-audio handoff; provider token streaming is a separate adapter."""
import threading

from arietta_voice.turns import (
    CancellationToken, BoundedChannel, ChannelClosed, TurnCancelled,
    TEXT_QUEUE_ITEMS, TEXT_ITEM_CHARS, AUDIO_QUEUE_ITEMS, AUDIO_ITEM_SAMPLES,
)
from arietta_voice.worker import WorkerFailure


async def speech_stream(text, *, infer, worker, cancellation, deadline=None, on_wait=None):
    """Prefetch at most eight PCM frames with one serialized inference producer.

    The current adapter supplies one completed reply. Text/channel contracts are
    also suitable for future speakable clauses; no provider token streaming is
    implied. A local token stops prefetch without cancelling a successful turn.
    """
    stop = CancellationToken()
    cancellation.subscribe(stop.cancel)
    texts = BoundedChannel(stop, max_items=TEXT_QUEUE_ITEMS, max_item_size=TEXT_ITEM_CHARS, deadline=deadline)
    audio = BoundedChannel(stop, max_items=AUDIO_QUEUE_ITEMS, max_item_size=AUDIO_ITEM_SAMPLES * 4,
                           size=lambda frame: frame[0].nbytes, deadline=deadline)
    producer = None
    try:
        texts.put(text)
        texts.finish()

        def produce():
            try:
                while True:
                    try:
                        sentence = texts.get()
                    except ChannelClosed:
                        break
                    infer('tts_start', sentence, cancellation=stop)
                    while True:
                        frame = infer('tts_next', cancellation=stop)
                        if frame is None:
                            break
                        samples, rate = frame
                        if samples.ndim != 1 or samples.dtype.name != 'float32' or not 8000 <= rate <= 96000:
                            raise ValueError('Invalid audio frame')
                        audio.put((samples.copy(), rate))
                audio.finish()
            except Exception as exc:
                audio.finish(exc)
            finally:
                if worker.pid is not None:
                    try:
                        worker.call('tts_close', timeout_s=1)
                    except Exception:
                        pass  # Worker failure already discards the generator.

        producer = threading.Thread(target=produce, name='arietta-speech-producer', daemon=True)
        producer.start()
        while True:
            try:
                frame = audio.get(on_wait=on_wait)
            except ChannelClosed:
                return
            stop.check(deadline)
            yield frame
    finally:
        stop.cancel()
        if producer is not None:
            producer.join(timeout=2)
        texts.close()
        audio.close()
        cancellation.unsubscribe(stop.cancel)
        if producer is not None and producer.is_alive():
            raise WorkerFailure('Speech producer did not stop')
