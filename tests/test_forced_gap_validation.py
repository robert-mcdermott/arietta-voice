import asyncio
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import forced_gap_validation as gap


@pytest.mark.parametrize('seconds,mode,aec,tts', [
    (-1,'live',True,True), (11,'live',True,True), (float('nan'),'live',True,True),
    (float('inf'),'live',True,True), (4,'components',True,True),
    (4,'live',False,True), (4,'live',True,False),
])
def test_invalid_injection_settings_are_rejected(seconds,mode,aec,tts):
    with pytest.raises(ValueError):
        gap.validate_gap_options(mode,seconds,aec=aec,tts=tts)


def test_only_exact_test_phrase_is_intercepted():
    assert gap.is_gap_request('Start the gap test!')
    assert not gap.is_gap_request('What does start the gap test mean?')
    assert not gap.is_gap_request('What is three plus three?')


@pytest.mark.parametrize('close_phase', ['lead_in','gap','after_gap','complete'])
def test_injection_yields_polls_and_never_starts_cancelled_continuation(monkeypatch,close_phase):
    now = [10.]
    monkeypatch.setattr(gap.time,'monotonic',lambda:now[0])
    async def sleep(seconds):
        now[0] += max(seconds, 1e-9)
    monkeypatch.setattr(gap.asyncio,'sleep',sleep)
    opened, closed, yielded = [], [], []
    async def speech(text):
        opened.append(text)
        try:
            yield np.ones(512,dtype=np.float32),24000
        finally:
            closed.append(text)
    state = dict(case_id=1,phase='lead_in',gap_started=False,gap_completed=False,
                 continuation_started=False,gap_polls=0)
    log = Mock()
    async def consume():
        stream = gap.forced_gap_stream(speech,.6,state,log)
        try:
            async for samples,rate in stream:
                yielded.append(samples)
                if state['phase'] == close_phase:
                    break
        finally:
            await stream.aclose()
    asyncio.run(consume())
    assert opened == closed
    assert len(opened) == (1 if close_phase in ('lead_in','gap') else 2)
    if close_phase == 'gap':
        assert not state['gap_completed'] and not state['continuation_started']
        assert state['gap_polls'] == 1
        assert state['gap_elapsed_s'] < .6
    elif close_phase in ('after_gap','complete'):
        assert state['gap_completed'] and state['continuation_started']
        assert state['gap_elapsed_s'] == pytest.approx(.6)
        assert sum(item is None for item in yielded) >= 59


def test_gap_gates_require_each_operator_scenario_and_actual_padding():
    cases = [dict(phase='after_gap',gap_started=True,gap_completed=True,continuation_started=True,
                  interrupted=False,cancelled=False,gap_polls=400,gap_silence_writes=200),
             dict(phase='gap',gap_started=True,gap_completed=False,continuation_started=False,
                  interrupted=True,cancelled=True,gap_polls=100,gap_silence_writes=50),
             dict(phase='after_gap',gap_started=True,gap_completed=True,continuation_started=True,
                  interrupted=True,cancelled=True,gap_polls=400,gap_silence_writes=200)]
    report = dict(events=dict(forced_gap_cases=cases,output_underflows=0,alignment_resets=0,
                             session_end_reasons={'user_goodbye':1}),
                  reliability={'audio':{'dropped_chunks':0}})
    assert all(gap.forced_gap_gates(report).values())
    cases[1]['gap_silence_writes'] = 0
    assert not gap.forced_gap_gates(report)['padding_exercised_each_case']
    cases.pop()
    assert not gap.forced_gap_gates(report)['interruption_after_gap']
    assert not all(gap.forced_gap_gates({}).values())


def test_recovery_uses_normal_conversation_path(monkeypatch):
    runtime = gap.ForcedGapRuntime.__new__(gap.ForcedGapRuntime)
    respond = Mock(return_value=('Six.',[]))
    monkeypatch.setattr(gap.AriettaVoiceRuntime,'_respond_with_speech',respond)
    mic,key = object(),object()
    assert runtime._respond_with_speech('What is three plus three?',mic,key) == ('Six.',[])
    respond.assert_called_once_with('What is three plus three?',mic,key)


@pytest.mark.parametrize('interrupt_phase', ['gap','after_gap'])
def test_real_player_hands_off_capture_during_and_after_injected_gap(monkeypatch,interrupt_phase):
    from types import SimpleNamespace
    from arietta_voice import audio
    from arietta_voice.turns import CancellationToken
    now, next_capture, refs = [10.], [10.], []
    state = dict(case_id=1,phase='lead_in',gap_started=False,gap_completed=False,
                 continuation_started=False,gap_polls=0)
    microphone = audio.MicrophoneStream(16000,512,max_queue_chunks=8)
    monkeypatch.setattr(audio.time,'monotonic',lambda:now[0])
    def advance(seconds):
        now[0] += max(seconds,1e-9)
        while next_capture[0]+.032 <= now[0]:
            speaking = state['phase'] == interrupt_phase and state['gap_polls'] >= 8
            timing = SimpleNamespace(currentTime=now[0],inputBufferAdcTime=next_capture[0])
            microphone._callback(np.full((512,1),float(speaking),dtype=np.float32),512,timing,False)
            next_capture[0] += .032
    monkeypatch.setattr(audio.time,'sleep',advance)
    async def sleep(seconds):
        advance(seconds)
    monkeypatch.setattr(gap.asyncio,'sleep',sleep)
    def clean(mic,reference):
        refs.append(reference.copy())
        return mic
    monkeypatch.setattr(audio,'make_aec_processor',lambda rate:clean)
    device = Mock(latency=0.,write_available=1024)
    def write(samples):
        advance(len(samples)/24000)
        return False
    device.write.side_effect = write
    monkeypatch.setattr(audio.sd,'OutputStream',lambda **kw:device)
    async def speech(text):
        yield np.full(14400,.2,dtype=np.float32),24000
    token = CancellationToken()
    result = audio.play_tts_stream(tts_stream=gap.forced_gap_stream(speech,.6,state,Mock()),
        microphone=microphone,vad=Mock(probability=lambda mic:float(mic.max()>.5)),
        keypress_monitor=None,chime_player=Mock(),sample_rate=16000,use_aec=True,
        barge_in_threshold_chunks=3,pre_roll_ms=64,cancellation=token)
    assert result.interrupted and token.cancelled
    assert result.buffered_chunks and len(result.buffered_chunks) <= 5
    assert result.output_timing['gap_silence_writes'] > 0
    assert result.alignment_resets == result.output_underflows == microphone.dropped_chunks == 0
    assert state['continuation_started'] == (interrupt_phase == 'after_gap')
    assert any(not ref.any() for ref in refs)
    device.abort.assert_called_once()
