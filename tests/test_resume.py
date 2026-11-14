import queue
from unittest.mock import Mock

import numpy as np

from arietta_voice.resume import ResumeWatch


class QueuedMicrophone:
    """Capture already waiting in the queue; values above zero count as speech."""

    dropped_chunks = input_overflows = 0

    def __init__(self, values=()):
        self.pending = list(values)
        self.drain = Mock()

    def push(self, *values):
        self.pending.extend(values)

    def get_nowait(self):
        if not self.pending:
            raise queue.Empty
        return np.full(512, self.pending.pop(0), dtype=np.float32)


def speech_vad():
    return Mock(probability=Mock(side_effect=lambda chunk: float(chunk[0] > 0)))


def test_two_consecutive_speech_chunks_resume_and_keep_all_captured_audio():
    microphone = QueuedMicrophone([0, 1, 0, 1, 1, 0.5])
    watch = ResumeWatch(speech_vad(), window_end=100.0, clock=lambda: 0.0)
    assert watch.poll(microphone) is True
    assert watch.resumed and not watch.active and watch.discard.is_set()
    # A single speech chunk did not resume; everything heard so far is kept in order.
    assert [chunk[0] for chunk in watch.chunks] == [0, 1, 0, 1, 1]
    assert [len(chunk) for chunk in watch.chunks] == [512] * 5
    # The chunk after the decision stays in the queue for the listening loop.
    assert watch.poll(microphone) is True
    assert microphone.pending == [0.5]


def test_quiet_window_expires_once_and_releases_the_deferred_chime():
    now = [0.0]
    microphone = QueuedMicrophone([0, 0])
    chime = Mock()
    watch = ResumeWatch(speech_vad(), window_end=0.7, clock=lambda: now[0])
    watch.on_expire = chime
    assert watch.poll(microphone) is False
    assert watch.active and len(watch.chunks) == 2
    chime.assert_not_called()

    now[0] = 0.7
    microphone.push(0)
    assert watch.poll(microphone) is False
    assert not watch.active and not watch.resumed and not watch.discard.is_set()
    assert watch.chunks == []
    chime.assert_called_once()

    # Once the decision stands the watch leaves the microphone alone.
    microphone.push(1, 1)
    assert watch.poll(microphone) is False
    assert microphone.pending == [1, 1]
    watch.finish()
    chime.assert_called_once()


def test_finishing_for_playback_does_not_start_the_chime():
    chime = Mock()
    watch = ResumeWatch(speech_vad(), window_end=100.0, clock=lambda: 0.0)
    watch.on_expire = chime
    watch.finish(release=False)
    assert not watch.active
    chime.assert_not_called()
