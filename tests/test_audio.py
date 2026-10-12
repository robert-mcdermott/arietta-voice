import wave
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice import audio
from arietta_voice.worker import WorkerTimeout


def test_microphone_overflow_is_bounded_and_keeps_recent_audio():
    microphone = audio.MicrophoneStream(16000, 512, max_queue_chunks=2)
    for value in range(10):
        microphone._callback(np.full((512, 1), value, dtype=np.float32), 512, None, False)
    assert microphone.statistics()["queued_chunks"] == 2
    assert microphone.dropped_chunks == 8
    assert microphone.get()[0] == 8
    assert microphone.get()[0] == 9
    microphone.drain()
    assert microphone.empty()


def test_recording_queue_and_file_are_bounded(tmp_path):
    path = tmp_path / "test.wav"
    writer = audio.RecordingWriter(path, 16000, max_chunks=2, max_seconds=0.05)
    # Fill without starting the consumer to deterministically exercise pressure.
    writer.append(np.zeros(300, dtype=np.float32))
    writer.append(np.zeros(300, dtype=np.float32))
    writer.append(np.zeros(300, dtype=np.float32))
    assert writer.dropped_chunks == 1
    assert writer.queue.qsize() == 2
    writer.start()
    writer.close()
    with wave.open(str(path)) as recording:
        assert recording.getnframes() == 600
    assert path.stat().st_mode & 0o777 == 0o600
    assert writer.error is None


def test_recording_duration_limit_trims_last_chunk(tmp_path):
    writer = audio.RecordingWriter(tmp_path / "limit.wav", 100, 8, 1)
    for _ in range(4):
        writer.append(np.zeros(60, dtype=np.float32))
    assert writer.accepted_samples == 100
    assert writer.limit_reached
    assert writer.queue.qsize() == 2
    writer.start()
    writer.close()
    with wave.open(str(writer.path)) as recording:
        assert recording.getnframes() == 100


def test_recording_failure_does_not_escape_callback(tmp_path):
    writer = audio.RecordingWriter(tmp_path, 16000, 2, 1)
    writer.start()
    writer.close()
    assert writer.error == "IsADirectoryError"
    writer.append(np.zeros(512))
    assert writer.queue.empty()


def test_utterance_and_reference_buffers_release_old_samples():
    buffer = audio.UtteranceBuffer(10)
    assert buffer.append(np.zeros(6))
    assert not buffer.append(np.zeros(6))
    assert buffer.samples == 6
    buffer.clear()
    assert not buffer.chunks
    reference = audio.AudioReferenceBuffer(4)
    for value in range(100):
        reference.append(np.array([value], dtype=np.float32))
    assert reference.samples.base is None
    np.testing.assert_equal(reference.samples, [96, 97, 98, 99])
    np.testing.assert_equal(reference.read(94, 8), [0, 0, 96, 97, 98, 99, 0, 0])


@pytest.mark.parametrize("failure", ["generation", "write", "start", "stop", "timeout", "interrupt"])
def test_playback_always_releases_audio_and_stream(monkeypatch, failure):
    stream = Mock()
    if failure in {"write", "start", "stop"}:
        getattr(stream, failure).side_effect = RuntimeError("device failure")
    monkeypatch.setattr(audio.sd, "OutputStream", Mock(return_value=stream))
    chime, mic, vad, key, level = Mock(), Mock(), Mock(), Mock(), Mock()
    mic.dropped_chunks = mic.input_overflows = 0
    key.pressed.return_value = failure == "interrupt"
    closed = []
    async def chunks():
        try:
            yield np.zeros(4096, dtype=np.float32), 24000
            if failure == "generation":
                raise RuntimeError("generation failed")
        finally:
            closed.append(True)
    options = dict(tts_stream=chunks(), microphone=mic, vad=vad, keypress_monitor=key,
                   chime_player=chime, sample_rate=16000, use_aec=False,
                   barge_in_threshold_chunks=5, playback_level_callback=level,
                   deadline=0 if failure == "timeout" else None)
    if failure == "interrupt":
        assert audio.play_tts_stream(**options).interrupted
    else:
        with pytest.raises((RuntimeError, WorkerTimeout)):
            audio.play_tts_stream(**options)
    if failure != "timeout":
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
    assert closed == [True]
    chime.stop.assert_called_once()
    mic.drain.assert_called()
    vad.reset.assert_called()
    level.assert_called_with(0.0, False)


def test_input_start_failure_closes_recorder_and_device(monkeypatch, tmp_path):
    device = Mock()
    device.start.side_effect = RuntimeError("unavailable")
    monkeypatch.setattr(audio.sd, "InputStream", Mock(return_value=device))
    microphone = audio.MicrophoneStream(16000, 512, record_path=tmp_path / "partial.wav")
    with pytest.raises(RuntimeError):
        with microphone:
            pass
    device.close.assert_called_once()
    assert not microphone._recorder._thread.is_alive()


def test_voice_interruption_preserves_bounded_followup_audio(monkeypatch):
    import itertools
    ticks = itertools.count()
    monkeypatch.setattr(audio.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(audio, 'make_aec_processor', lambda rate: lambda chunk, reference: chunk)
    microphone = audio.MicrophoneStream(16000, 512, max_queue_chunks=8)
    output = Mock()
    def supply_followup(chunk):
        for _ in range(4):
            microphone._callback(np.ones((512, 1), dtype=np.float32), 512, None, False)
    output.write.side_effect = supply_followup
    monkeypatch.setattr(audio.sd, 'OutputStream', Mock(return_value=output))
    async def chunks():
        yield np.zeros(4096, dtype=np.float32), 24000
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=microphone,
        vad=Mock(probability=Mock(return_value=1.0)), keypress_monitor=Mock(pressed=Mock(return_value=False)),
        chime_player=Mock(), sample_rate=16000, use_aec=True, barge_in_threshold_chunks=3)
    assert result.interrupted
    assert len(result.buffered_chunks) == 3
    assert microphone.statistics()['queued_chunks'] == 1
    output.close.assert_called_once()
