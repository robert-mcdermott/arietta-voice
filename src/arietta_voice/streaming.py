"""Bounded speech handoff with serialized token-generation and TTS ownership."""

import asyncio
import threading

from arietta_voice.turns import (
    AUDIO_ITEM_SAMPLES,
    AUDIO_QUEUE_ITEMS,
    TEXT_ITEM_CHARS,
    TEXT_QUEUE_ITEMS,
    BoundedChannel,
    CancellationToken,
    ChannelClosed,
    ChannelEmpty,
)
from arietta_voice.worker import WorkerFailure

# Longest wait for one in-flight inference call when a turn is discarded.
DISCARD_WAIT_S = 5.0


async def speech_stream(
    text=None,
    *,
    infer,
    worker,
    cancellation,
    deadline=None,
    on_wait=None,
    request=None,
    on_result=None,
    playback_events=False,
    discard=None,
):
    """Prefetch at most eight PCM frames with one serialized inference producer.

    Pull a clause, synthesize it, then resume tokens while queued audio plays.
    Model and TTS operations stay on one inference owner. A local token stops
    prefetch without cancelling a successful turn's later memory update.
    With playback_events, (None, None) polls and (exception, None) failures let
    the player stop the device before calling aclose() and joining the producer.
    A set ``discard`` event abandons the turn gently: the producer finishes its
    current inference call and stops, because cancelling a call mid-flight
    terminates the worker and its loaded models.
    """
    stop = CancellationToken()
    cancellation.subscribe(stop.cancel)
    texts = BoundedChannel(
        stop, max_items=TEXT_QUEUE_ITEMS, max_item_size=TEXT_ITEM_CHARS, deadline=deadline
    )
    audio = BoundedChannel(
        stop,
        max_items=AUDIO_QUEUE_ITEMS,
        max_item_size=AUDIO_ITEM_SAMPLES * 4,
        size=lambda frame: frame[0].nbytes,
        deadline=deadline,
    )
    producer = None
    try:
        if request is None:
            texts.put(text)
            texts.finish()

        def run(operation, payload=None):
            if discard is not None and discard.is_set():
                raise ChannelClosed("Turn discarded")
            return infer(operation, payload, cancellation=stop)

        def produce():
            try:
                if request is not None:
                    run("chat_start", request)
                while True:
                    if request is not None:
                        event = run("chat_next")
                        if event["kind"] == "done":
                            if on_result is not None:
                                on_result(event["result"])
                            break
                        if event["kind"] != "clause":
                            raise ValueError("Invalid conversation event")
                        texts.put(event["text"])
                    try:
                        sentence = texts.get()
                    except ChannelClosed:
                        break
                    run("tts_start", sentence)
                    while True:
                        frame = run("tts_next")
                        if frame is None:
                            break
                        samples, rate = frame
                        if (
                            samples.ndim != 1
                            or samples.dtype.name != "float32"
                            or not 8000 <= rate <= 96000
                        ):
                            raise ValueError("Invalid audio frame")
                        audio.put((samples.copy(), rate))
                audio.finish()
            except Exception as exc:
                audio.finish(exc)
            finally:
                if worker.pid is not None:
                    try:
                        worker.call(
                            "speech_close" if request is not None else "tts_close", timeout_s=1
                        )
                    except Exception:
                        pass  # Worker failure already discards the generator.

        producer = threading.Thread(target=produce, name="arietta-speech-producer", daemon=True)
        producer.start()
        while True:
            try:
                # Keep an active blocking output stream fed through clause gaps.
                frame = audio.get(on_wait=on_wait, timeout=0.01 if playback_events else 0.05)
            except ChannelClosed:
                if playback_events:
                    # Native generator closure can still be pending. Keep the
                    # player in control of the device during that wait.
                    while producer.is_alive():
                        yield None, None
                        await asyncio.sleep(0.01)
                return
            except ChannelEmpty:
                # Let playback poll speech interruption even between clauses.
                yield None, None
                continue
            stop.check(deadline)
            yield frame
    except Exception as exc:
        if playback_events:
            # Deliver failure before joining the producer. The player raises it,
            # aborts its device, then closes this generator and waits for cleanup.
            yield exc, None
        raise
    finally:
        if discard is not None and discard.is_set() and producer is not None:
            # Unblock a producer waiting on a full queue, then let its current
            # inference call return before anything is cancelled.
            texts.finish()
            audio.finish()
            producer.join(timeout=DISCARD_WAIT_S)
        stop.cancel()
        if producer is not None:
            producer.join(timeout=2)
        texts.close()
        audio.close()
        cancellation.unsubscribe(stop.cancel)
        if producer is not None and producer.is_alive():
            raise WorkerFailure("Speech producer did not stop")
