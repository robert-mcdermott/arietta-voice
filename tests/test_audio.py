import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice import audio
from arietta_voice.worker import WorkerTimeout


def test_external_recording_survives_microphone_reopen(monkeypatch, tmp_path):
    writer = audio.RecordingWriter(tmp_path / 'reconnect.wav', 16000, 8, 10)
    writer.start()
    stream_factory = Mock(side_effect=lambda **kw: Mock(active=True))
    monkeypatch.setattr(audio.sd, 'InputStream', stream_factory)
    for index in (3, 7):
        with audio.MicrophoneStream(16000, 512, device=index, recorder=writer) as mic:
            mic._callback(np.ones((512, 1), dtype=np.float32), 512, None, False)
        assert writer._thread.is_alive()
    writer.close()
    with wave.open(str(writer.path)) as recording:
        assert recording.getnframes() == 1024
    assert [call.kwargs['device'] for call in stream_factory.call_args_list] == [3, 7]


def test_inactive_capture_rejects_already_queued_audio(monkeypatch):
    from arietta_voice.devices import AudioDeviceError
    monkeypatch.setattr(audio.sd, 'InputStream', lambda **kw: Mock(active=False))
    with audio.MicrophoneStream(16000, 512) as mic:
        mic._callback(np.ones((512, 1)), 512, None, False)
        with pytest.raises(AudioDeviceError, match='Inactive'):
            mic.get()


def test_health_failure_aborts_selected_output_before_generator_cleanup(monkeypatch):
    from arietta_voice.devices import AudioDeviceError
    order = []
    device = Mock()
    device.abort.side_effect = lambda: order.append('abort')
    device.close.side_effect = lambda: order.append('close')
    output = Mock(return_value=device)
    monkeypatch.setattr(audio.sd, 'OutputStream', output)
    async def chunks():
        try:
            yield np.zeros(512, dtype=np.float32), 24000
            yield None, None
        finally:
            order.append('generator_close')
    def health():
        if device.write.called:
            raise AudioDeviceError('CaptureStalled')
    with pytest.raises(AudioDeviceError):
        audio.play_tts_stream_passive(tts_stream=chunks(), chime_player=Mock(),
                                     output_device=8, health_check=health)
    assert output.call_args.kwargs['device'] == 8
    assert order == ['abort', 'close', 'generator_close']


def test_chime_routes_and_closes_selected_device(monkeypatch):
    play, stop, stream = Mock(), Mock(), Mock()
    monkeypatch.setattr(audio.sd, 'play', play)
    monkeypatch.setattr(audio.sd, 'stop', stop)
    monkeypatch.setattr(audio.sd, 'get_stream', lambda: stream)
    chime = audio.ChimePlayer(True, device=9)
    chime.start()
    chime.stop()
    assert play.call_args.kwargs['device'] == 9
    stop.assert_called_once()
    stream.close.assert_called_once()


def test_failed_chime_start_closes_partially_opened_stream(monkeypatch):
    from arietta_voice.devices import AudioDeviceError
    stream = Mock()
    monkeypatch.setattr(audio.sd, 'play', Mock(side_effect=AudioDeviceError('OutputUnavailable')))
    monkeypatch.setattr(audio.sd, 'stop', Mock())
    monkeypatch.setattr(audio.sd, 'get_stream', lambda: stream)
    chime = audio.ChimePlayer(True, device=9)
    with pytest.raises(AudioDeviceError, match='OutputUnavailable'):
        chime.start()
    assert chime.started_at == 0
    stream.close.assert_called_once()


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
        if failure in {"interrupt", "generation", "write", "start"}:
            stream.abort.assert_called_once()
            stream.stop.assert_not_called()
        else:
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
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(audio, 'make_aec_processor', lambda rate: lambda chunk, reference: chunk)
    microphone = audio.MicrophoneStream(16000, 512, max_queue_chunks=8)
    output = Mock()
    def supply_followup(chunk):
        for _ in range(4):
            now[0] += .032
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


def test_cancel_aborts_device_before_waiting_for_generator_cleanup(monkeypatch):
    from arietta_voice.turns import CancellationToken
    order = []
    device = Mock()
    device.abort.side_effect = lambda: order.append('abort')
    device.close.side_effect = lambda: order.append('device_closed')
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    token = CancellationToken()
    async def chunks():
        try:
            yield np.ones(4096, dtype=np.float32), 24000
        finally:
            order.append('generator_closed')
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=None, vad=None,
        keypress_monitor=Mock(pressed=Mock(return_value=True)), chime_player=Mock(),
        sample_rate=16000, use_aec=False, barge_in_threshold_chunks=3, cancellation=token)
    assert result.interrupted and token.cancelled
    assert order == ['abort', 'device_closed', 'generator_closed']
    device.write.assert_not_called()


def test_voice_interrupt_during_gap_between_clauses(monkeypatch):
    from arietta_voice.turns import CancellationToken
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(audio, 'make_aec_processor', lambda rate: lambda chunk, reference: chunk)
    microphone = audio.MicrophoneStream(16000,512,max_queue_chunks=8)
    device = Mock()
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    token = CancellationToken()
    async def chunks():
        yield np.zeros(512,dtype=np.float32),24000
        for _ in range(3):
            now[0] += .032
            microphone._callback(np.ones((512,1),dtype=np.float32),512,None,False)
        yield None, None
        raise AssertionError('Cancelled stream was resumed')
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=microphone,
        vad=Mock(probability=lambda _: 1.), keypress_monitor=Mock(pressed=lambda: False),
        chime_player=Mock(), sample_rate=16000,use_aec=True,barge_in_threshold_chunks=3,cancellation=token)
    assert result.interrupted and token.cancelled
    assert len(result.buffered_chunks) == 3
    device.abort.assert_called_once()
    assert device.write.call_count == 1


def test_capture_timestamp_survives_queue_delay_and_overflow(monkeypatch):
    now = [100.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    mic = audio.MicrophoneStream(16000, 512, max_queue_chunks=1)
    timing = SimpleNamespace(currentTime=20., inputBufferAdcTime=19.96)
    mic._callback(np.ones((512, 1)), 512, timing, False)
    now[0] = 101.
    assert mic.get_capture_nowait().captured_at == pytest.approx(99.96)
    for n in range(2):
        mic._callback(np.full((512, 1), n), 512, timing, False)
    assert mic.get_capture_nowait().samples[0] == 1
    assert mic.dropped_chunks == 1


def test_timed_reference_keeps_gaps_resamples_and_bounds_memory():
    ref = audio.TimedAudioReferenceBuffer(.1)
    ref.append(np.ones(480), 24000, 10.)
    ref.append(np.full(480, 2.), 24000, 10.04)
    output = ref.read(10., 960, 16000)
    np.testing.assert_allclose(output[:320], 1)
    np.testing.assert_allclose(output[321:640], 0)
    np.testing.assert_allclose(output[641:], 2)
    ref.append(np.ones(24000), 24000, 11.)
    assert sum(len(block[2]) for block in ref.blocks) <= 2400
    assert not ref.read(10., 512, 16000).any()


def test_aec_continuous_ten_ms_frames_preserve_every_sample(monkeypatch):
    from livekit.rtc.apm import AudioProcessingModule
    reverse, forward = [], []
    apm = Mock(spec=AudioProcessingModule)
    apm.process_reverse_stream.side_effect = lambda frame: reverse.append(bytes(frame.data))
    apm.process_stream.side_effect = lambda frame: forward.append(bytes(frame.data))
    monkeypatch.setattr('livekit.rtc.apm.AudioProcessingModule', lambda **kw: apm)
    process = audio.make_aec_processor(16000)
    signal = np.linspace(-.5, .5, 2560, dtype=np.float32)
    cleaned = [process(piece, piece) for piece in np.split(signal, 5)]
    assert len(forward) == len(reverse) == 16  # 160ms, not five padded 40ms chunks
    assert all(len(frame) == 320 for frame in forward)
    joined = np.concatenate(cleaned + [process.flush()])
    np.testing.assert_allclose(joined[:160], 0)
    np.testing.assert_allclose(joined[160:], signal, atol=1/32767)
    assert apm.set_stream_delay_ms.call_count == 16


def test_aec_flush_keeps_partial_native_frame_without_duplicate_samples(monkeypatch):
    apm = Mock()
    monkeypatch.setattr('livekit.rtc.apm.AudioProcessingModule', lambda **kw: apm)
    process = audio.make_aec_processor(16000)
    signal = np.full(512, .25, dtype=np.float32)
    output = np.concatenate([process(signal, signal), process.flush()])
    assert len(output) == 672
    np.testing.assert_allclose(output[160:], signal, atol=1/32767)
    assert len(process.flush()) == 0


@pytest.mark.parametrize('phase', ['initial_gap', 'clause_gap', 'drain'])
def test_keyboard_cancellation_covers_generation_gaps_and_device_tail(monkeypatch, phase):
    from arietta_voice.turns import CancellationToken
    now, pressed, order = [10.], [False], []
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda delay: now.__setitem__(0, now[0]+delay))
    device = Mock(latency=.1)
    device.abort.side_effect = lambda: order.append('abort')
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    token = CancellationToken()
    async def chunks():
        try:
            if phase != 'initial_gap':
                yield np.ones(512, dtype=np.float32), 24000
            pressed[0] = True
            if phase != 'drain':
                yield None, None
                raise AssertionError('Cancelled producer resumed')
        finally:
            order.append('producer_close')
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=None, vad=None,
        keypress_monitor=Mock(pressed=lambda: pressed[0]), chime_player=Mock(),
        sample_rate=16000, use_aec=False, barge_in_threshold_chunks=3, cancellation=token)
    assert result.interrupted and token.cancelled
    if phase != 'initial_gap':
        device.abort.assert_called_once()
        device.stop.assert_not_called()
    if phase == 'clause_gap':
        assert order == ['abort', 'producer_close']


def test_barge_in_handoff_is_cleaned_and_includes_pre_roll(monkeypatch):
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda delay: now.__setitem__(0, now[0]+delay))
    monkeypatch.setattr(audio, 'make_aec_processor', lambda rate: lambda mic, ref: mic/2)
    mic = audio.MicrophoneStream(16000, 512)
    device = Mock(latency=0.)
    def feed(chunk):
        for value in [.2, .4, 1., 1., 1.]:
            now[0] += .032
            mic._callback(np.full((512, 1), value, dtype=np.float32), 512, None, False)
    device.write.side_effect = feed
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    async def chunks():
        yield np.zeros(2048, dtype=np.float32), 24000
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=mic,
        vad=Mock(probability=lambda chunk: float(chunk[0] > .3)), keypress_monitor=None,
        chime_player=Mock(), sample_rate=16000, use_aec=True, barge_in_threshold_chunks=3,
        pre_roll_ms=64)
    assert result.interrupted
    assert len(result.buffered_chunks) == 5
    np.testing.assert_allclose([c[0] for c in result.buffered_chunks], [.1, .2, .5, .5, .5])
    device.abort.assert_called_once()


def test_capture_discontinuity_discards_old_barge_in_prefix(monkeypatch):
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda delay: now.__setitem__(0, now[0]+delay))
    factory = Mock(side_effect=lambda rate: lambda mic, ref: mic)
    monkeypatch.setattr(audio, 'make_aec_processor', factory)
    mic = audio.MicrophoneStream(16000, 512)
    device = Mock(latency=0.)
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    async def chunks():
        yield np.zeros(512, dtype=np.float32), 24000
        for t, value in [(10., 1.), (10.032, 1.), (10.5, .6), (10.532, .6), (10.564, .6)]:
            now[0] = t + .032
            mic._callback(np.full((512, 1), value, dtype=np.float32), 512, None, False)
        yield None, None
        raise AssertionError('Cancelled stream resumed')
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=mic,
        vad=Mock(probability=lambda chunk: 1.), keypress_monitor=None,
        chime_player=Mock(), sample_rate=16000, use_aec=True,
        barge_in_threshold_chunks=3, pre_roll_ms=64)
    assert result.interrupted and result.alignment_resets == 1
    assert result.alignment_reset_reasons == {'timestamp_gap': 1}
    assert factory.call_count == 2
    np.testing.assert_allclose(np.concatenate(result.buffered_chunks), .6)


def test_output_underflow_invalidates_reference_and_resets_aec(monkeypatch):
    device = Mock(latency=0., write=Mock(return_value=True))
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    factory = Mock(side_effect=lambda rate: lambda mic, ref: mic)
    monkeypatch.setattr(audio, 'make_aec_processor', factory)
    mic = audio.MicrophoneStream(16000, 512)
    async def chunks():
        yield np.zeros(512, dtype=np.float32), 24000
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=mic,
        vad=Mock(), keypress_monitor=None, chime_player=Mock(), sample_rate=16000,
        use_aec=True, barge_in_threshold_chunks=3)
    assert result.output_underflows == result.alignment_resets == 1
    assert result.alignment_reset_reasons == {'output_underflow': 1}
    assert factory.call_count == 2


def test_drop_during_producer_cleanup_cannot_join_discontinuous_followup(monkeypatch):
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio, 'make_aec_processor', lambda rate: lambda mic, ref: mic)
    mic = audio.MicrophoneStream(16000, 512, max_queue_chunks=3)
    device = Mock(latency=0.)
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    def feed():
        now[0] += .032
        mic._callback(np.ones((512, 1), dtype=np.float32), 512, None, False)
    async def chunks():
        try:
            yield np.zeros(512, dtype=np.float32), 24000
            for _ in range(3):
                feed()
            yield None, None
        finally:
            for _ in range(4):
                feed()
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=mic,
        vad=Mock(probability=lambda _: 1.), keypress_monitor=None, chime_player=Mock(),
        sample_rate=16000, use_aec=True, barge_in_threshold_chunks=3)
    assert result.interrupted
    assert result.buffered_chunks == []
    assert mic.empty()


def test_expected_startup_packet_does_not_reset_fresh_aec(monkeypatch):
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    factory = Mock(side_effect=lambda rate: lambda mic, ref: mic)
    monkeypatch.setattr(audio, 'make_aec_processor', factory)
    mic = audio.MicrophoneStream(16000, 512)
    device = Mock(latency=0.)
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: device)
    async def chunks():
        yield np.zeros(512, dtype=np.float32), 24000
        # Arrives after output starts, but includes samples from before it.
        now[0] = 10.016
        mic._callback(np.ones((512, 1)), 512, None, False)
        for _ in range(3):
            now[0] += .032
            mic._callback(np.full((512, 1), .5), 512, None, False)
        yield None, None
        raise AssertionError('Cancelled stream resumed')
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=mic,
        vad=Mock(probability=lambda _: 1.), keypress_monitor=None, chime_player=Mock(),
        sample_rate=16000, use_aec=True, barge_in_threshold_chunks=3)
    assert result.interrupted and result.alignment_resets == 0
    assert result.capture_discard_reasons == {'before_playback': 1}
    assert factory.call_count == 1
    np.testing.assert_allclose(np.concatenate(result.buffered_chunks), .5)
    assert mic.capture_phase == 'listening'


def test_initial_generation_wait_retires_unused_capture_without_overflow(monkeypatch):
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda delay: now.__setitem__(0, now[0]+delay))
    monkeypatch.setattr(audio, 'make_aec_processor', lambda rate: lambda mic, ref: mic)
    mic = audio.MicrophoneStream(16000, 512, max_queue_chunks=4)
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: Mock(latency=0.))
    async def chunks():
        for _ in range(100):  # Longer than the production queue's 2.048 seconds.
            now[0] += .032
            mic._callback(np.ones((512, 1)), 512, None, False)
            yield None, None
        yield np.zeros(512, dtype=np.float32), 24000
        for _ in range(3):
            now[0] += .032
            mic._callback(np.full((512, 1), .5), 512, None, False)
        yield None, None
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=mic,
        vad=Mock(probability=lambda _: 1.), keypress_monitor=None, chime_player=Mock(),
        sample_rate=16000, use_aec=True, barge_in_threshold_chunks=3)
    assert result.interrupted and result.alignment_resets == 0
    assert mic.dropped_chunks == 0
    assert mic.statistics()['discarded_by_reason'] == {'before_output': 100}
    np.testing.assert_allclose(np.concatenate(result.buffered_chunks), .5)


def test_active_capture_overflow_still_invalidates_interruption_prefix(monkeypatch):
    now = [10.]
    monkeypatch.setattr(audio.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(audio.time, 'sleep', lambda delay: now.__setitem__(0, now[0]+delay))
    monkeypatch.setattr(audio, 'make_aec_processor', lambda rate: lambda mic, ref: mic)
    mic = audio.MicrophoneStream(16000, 512, max_queue_chunks=3)
    monkeypatch.setattr(audio.sd, 'OutputStream', lambda **kw: Mock(latency=0.))
    def feed(value):
        now[0] += .032
        mic._callback(np.full((512, 1), value), 512, None, False)
    async def chunks():
        yield np.zeros(512, dtype=np.float32), 24000
        feed(1.)
        yield None, None
        for _ in range(4):
            feed(1.)
        yield None, None
        for _ in range(3):
            feed(.5)
        yield None, None
    result = audio.play_tts_stream(tts_stream=chunks(), microphone=mic,
        vad=Mock(probability=lambda _: 1.), keypress_monitor=None, chime_player=Mock(),
        sample_rate=16000, use_aec=True, barge_in_threshold_chunks=3)
    assert result.interrupted
    assert result.alignment_reset_reasons == {'capture_overflow': 1}
    assert mic.statistics()['drops_by_phase'] == {'playback': 1}
    assert mic.statistics()['discarded_by_reason'] == {'capture_gap': 3}
    np.testing.assert_allclose(np.concatenate(result.buffered_chunks), .5)


def test_intentional_discard_is_bounded_and_does_not_drop_recording(monkeypatch):
    recorder = Mock()
    mic = audio.MicrophoneStream(16000, 512, max_queue_chunks=2, recorder=recorder)
    def feed():
        mic._callback(np.zeros((512, 1)), 512, None, False)
    feed()
    original = mic._audio_q.get_nowait
    def consume_and_refill():
        chunk = original()
        feed()
        return chunk
    monkeypatch.setattr(mic._audio_q, 'get_nowait', consume_and_refill)
    mic.discard_pending('before_output')
    assert mic.statistics()['queued_chunks'] == 1
    assert mic.statistics()['discarded_chunks'] == 1
    assert mic.dropped_chunks == 0
    assert recorder.append.call_count == 2
