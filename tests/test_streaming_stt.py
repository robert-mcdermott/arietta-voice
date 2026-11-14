"""Transcribing while the user speaks (experimental, off by default)."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice import runtime as runtime_module
from arietta_voice.audio import UtteranceBuffer
from arietta_voice.config import default_config
from arietta_voice.inference import InferenceEngine
from arietta_voice.stt import StreamingTranscription
from arietta_voice.stt_feed import FEED_SAMPLES, StreamFeed

CHUNK = 512


class Calls:
    def __init__(self, results=None):
        self.calls, self.results = [], dict(results or {})

    def __call__(self, operation, payload):
        self.calls.append((operation, None if payload is None else len(payload)))
        result = self.results.get(operation, True)
        if isinstance(result, Exception):
            raise result
        return result


def speak(buffer, feed, seconds):
    for _ in range(int(seconds * 16000 / CHUNK)):
        buffer.append(np.ones(CHUNK, dtype=np.float32))
        feed.feed(buffer)


def test_audio_is_fed_half_a_second_at_a_time_and_the_rest_at_the_end():
    calls, buffer = Calls(), UtteranceBuffer(16000 * 30)
    feed = StreamFeed(calls)
    speak(buffer, feed, 1.2)
    assert calls.calls[0] == ("stt_stream_start", None)
    adds = [size for op, size in calls.calls if op == "stt_stream_add"]
    assert adds and all(size >= FEED_SAMPLES for size in adds)
    feed.prepare(buffer)
    audio = buffer.audio()
    rest = feed.take_rest(audio)
    assert len(rest) == len(audio) - sum(adds)
    # Used up: a resumed attempt on the same utterance is transcribed the usual way.
    feed.prepare(buffer)
    assert feed.take_rest(buffer.audio()) is None


def test_a_new_utterance_cancels_the_old_stream():
    calls, buffer = Calls(), UtteranceBuffer(16000 * 30)
    feed = StreamFeed(calls)
    speak(buffer, feed, 0.2)
    buffer.clear()
    speak(buffer, feed, 0.2)
    assert [op for op, _ in calls.calls] == [
        "stt_stream_start",
        "stt_stream_cancel",
        "stt_stream_start",
    ]


@pytest.mark.parametrize(
    "results", [{"stt_stream_start": False}, {"stt_stream_add": RuntimeError("busy")}]
)
def test_any_failure_leaves_the_usual_transcription(results):
    calls, buffer = Calls(results), UtteranceBuffer(16000 * 30)
    feed = StreamFeed(calls)
    speak(buffer, feed, 1.0)
    feed.prepare(buffer)
    assert feed.state == "failed" and feed.take_rest(buffer.audio()) is None


def test_only_the_prepared_utterance_is_finished():
    calls, buffer = Calls(), UtteranceBuffer(16000 * 30)
    feed = StreamFeed(calls)
    speak(buffer, feed, 1.0)
    assert feed.take_rest(buffer.audio()) is None  # Not prepared by the listening loop.


class FakeStream:
    def __init__(self, fail=False):
        self.audio, self.fail, self.closed = [], fail, False

    def start(self):
        pass

    def add_audio(self, samples, rate):
        if self.fail:
            raise RuntimeError("native failure")
        self.audio.extend(samples)

    def stop(self):
        return SimpleNamespace(lines=[SimpleNamespace(text=f"heard {len(self.audio)}")])

    def close(self):
        self.closed = True


def fake_stt(stream):
    transcriber = SimpleNamespace(create_stream=lambda update_interval: stream)
    return SimpleNamespace(ensure_loaded=lambda: None, _transcriber=transcriber)


def test_streaming_transcription_works_in_the_background():
    stream = FakeStream()
    streaming = StreamingTranscription(fake_stt(stream), 16000)
    streaming.add(np.ones(100, dtype=np.float32))
    assert streaming.finish(np.ones(20, dtype=np.float32)) == "heard 120"
    assert stream.closed


def test_a_failed_pass_is_reported_at_the_end():
    streaming = StreamingTranscription(fake_stt(FakeStream(fail=True)), 16000)
    streaming.add(np.ones(10, dtype=np.float32))
    with pytest.raises(RuntimeError):
        streaming.finish(np.zeros(0, dtype=np.float32))


def test_worker_operations_report_failure_instead_of_raising(tmp_path):
    # A raised error would restart the worker and reload every model.
    engine = InferenceEngine(default_config(tmp_path))
    stream = FakeStream()
    engine._stt = fake_stt(stream)
    assert engine.handle("stt_stream_finish", np.zeros(1)) is None
    assert engine.handle("stt_stream_add", np.zeros(1)) is False
    assert engine.handle("stt_stream_start", None) is True
    assert engine.handle("stt_stream_add", np.ones(5, dtype=np.float32)) is True
    assert engine.handle("stt_stream_finish", np.ones(3, dtype=np.float32)) == "heard 8"
    engine._stt = fake_stt(FakeStream(fail=True))
    engine.handle("stt_stream_start", None)
    engine.handle("stt_stream_add", np.ones(5, dtype=np.float32))
    assert engine.handle("stt_stream_finish", np.zeros(0)) is None


def test_streaming_is_off_by_default(tmp_path):
    assert default_config(tmp_path).audio.streaming_stt is False


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    config.audio.streaming_stt = True
    config.audio.tts_enabled = False
    config.knowledge.enabled = False
    monkeypatch.setattr(runtime_module, "SileroVoiceActivityDetector", lambda *a: Mock())
    monkeypatch.setattr(runtime_module, "ChimePlayer", lambda *a: Mock())
    worker = SimpleNamespace(pid=None, restarts=0, call=None, close=lambda: None)
    return runtime_module.AriettaVoiceRuntime(config, worker=worker)


def test_the_runtime_uses_the_stream_and_falls_back(runtime):
    calls = []

    def call(operation, payload=None, **kwargs):
        calls.append(operation)
        if operation == "stt_stream_finish":
            return finish_result
        if operation == "transcribe":
            return "batch text"
        return True

    runtime.worker.call = call
    buffer = UtteranceBuffer(16000 * 30)
    for _ in range(40):
        buffer.append(np.ones(CHUNK, dtype=np.float32))
        runtime._stt_feed.feed(buffer)
    runtime._stt_feed.prepare(buffer)
    finish_result = " streamed text "
    assert runtime._transcribe(buffer.audio()) == "streamed text"
    assert calls.count("transcribe") == 0
    # The stream could not finish: the whole utterance is transcribed as usual.
    buffer.clear()
    for _ in range(40):
        buffer.append(np.ones(CHUNK, dtype=np.float32))
        runtime._stt_feed.feed(buffer)
    runtime._stt_feed.prepare(buffer)
    finish_result = None
    assert runtime._transcribe(buffer.audio()) == "batch text"
