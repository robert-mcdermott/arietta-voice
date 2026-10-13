import logging
from unittest.mock import Mock

import numpy as np
import pytest

from arietta_voice import runtime as runtime_module
from arietta_voice.config import default_config
from arietta_voice.models import AssistantState
from arietta_voice.prompts import PromptManager
from arietta_voice.runtime_status import runtime_status_path


@pytest.mark.parametrize('wake_enabled', [False, True])
def test_device_failure_closes_before_refresh_clears_session_and_never_replays(runtime, monkeypatch, wake_enabled):
    from arietta_voice.devices import AudioDeviceError
    from arietta_voice.turns import CancellationToken
    runtime.config.wake_word.enabled = wake_enabled
    order, waits = [], []
    opened = []
    token = CancellationToken()
    monkeypatch.setattr(runtime, 'cancel_turn', token.cancel)
    runtime.memory_jobs = Mock(snapshot=Mock(return_value={}))
    def select(_):
        return {'selector': 'CoreAudio::Mic', 'index': len(opened)+1}, {'selector': 'CoreAudio::Speaker', 'index': 9}
    monkeypatch.setattr(runtime_module, 'select_devices', select)
    class Mic:
        def __init__(self, *a, **kw):
            self.device = kw['device']
        def __enter__(self):
            opened.append(self.device)
            order.append('open')
            return self
        def __exit__(self, *a):
            order.append('close')
        def get(self, **kw):
            return np.zeros(512)
        def statistics(self):
            return {}
    monkeypatch.setattr(runtime_module, 'MicrophoneStream', Mic)
    monkeypatch.setattr(runtime_module, 'refresh_audio_backend', lambda: order.append('refresh'))
    def backoff(delay):
        assert runtime.state == AssistantState.WAITING_FOR_AUDIO
        assert not runtime.session.active and not runtime.session.history
        assert runtime.last_user_transcript == ''
        assert token.cancelled
        waits.append(delay)
    monkeypatch.setattr(runtime_module.time, 'sleep', backoff)
    def listen(mic, keys):
        assert runtime.output_device == runtime.chime_player.device == 9
        assert runtime.session.active is (not wake_enabled)
        assert not runtime.session.history
        if len(opened) == 1:
            runtime._store_turn('old question', 'old answer')
            runtime.last_user_transcript = 'old'
            raise AudioDeviceError('InputStreamInactive')
    monkeypatch.setattr(runtime, '_listen', listen)
    runtime._run_audio(Mock())
    assert order == ['open', 'close', 'refresh', 'open', 'close']
    assert opened == [1, 2] and waits == [1.0]
    assert runtime.audio_reconnect_attempts == runtime.audio_recoveries == 1
    assert runtime.device_connection == 'connected'
    assert runtime.microphone is None
    runtime.memory_jobs.suspend.assert_called_once()


def test_missing_device_backs_off_without_falling_back(runtime, monkeypatch):
    from arietta_voice.devices import AudioDeviceError
    select = Mock(side_effect=AudioDeviceError('InputUnavailable'))
    mic = Mock()
    monkeypatch.setattr(runtime_module, 'select_devices', select)
    monkeypatch.setattr(runtime_module, 'MicrophoneStream', mic)
    monkeypatch.setattr(runtime_module, 'refresh_audio_backend', lambda: None)
    delays = []
    def sleep(delay):
        delays.append(delay)
        if len(delays) == 8:
            raise KeyboardInterrupt
    monkeypatch.setattr(runtime_module.time, 'sleep', sleep)
    with pytest.raises(KeyboardInterrupt):
        runtime._run_audio(Mock())
    assert delays == [1, 2, 4, 8, 16, 30, 30, 30]
    mic.assert_not_called()
    assert runtime.audio_recoveries == 0


def test_open_stream_without_callback_is_not_reported_as_recovered(runtime, monkeypatch):
    from arietta_voice.devices import AudioDeviceError
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    stream.get.side_effect = [None, AudioDeviceError('CaptureStalled')]
    stream.statistics.return_value = {}
    monkeypatch.setattr(runtime_module, 'select_devices', lambda _: ({'selector': 'Mic', 'index': 1}, None))
    monkeypatch.setattr(runtime_module, 'MicrophoneStream', lambda *a, **kw: stream)
    monkeypatch.setattr(runtime_module.time, 'sleep', Mock(side_effect=KeyboardInterrupt))
    listen = Mock()
    monkeypatch.setattr(runtime, '_listen', listen)
    with pytest.raises(KeyboardInterrupt):
        runtime._run_audio(Mock())
    listen.assert_not_called()
    stream.__exit__.assert_called_once()
    assert runtime.device_connection == 'waiting'
    assert runtime.audio_recoveries == 0


def test_device_error_escapes_turn_recovery(runtime, monkeypatch):
    from arietta_voice.devices import AudioDeviceError
    monkeypatch.setattr(runtime, '_infer', Mock(side_effect=AudioDeviceError('CaptureStalled')))
    recover = Mock()
    monkeypatch.setattr(runtime, '_recover', recover)
    with pytest.raises(AudioDeviceError):
        runtime.process_utterance(np.zeros(512), Mock(), Mock())
    recover.assert_not_called()
    assert not runtime.session.history


class FakeWorker:
    pid = None
    restarts = 0
    transcript = ""
    response = "Answer"

    def call(self, operation, payload=None, **kwargs):
        if operation == "transcribe":
            return self.transcript
        if operation == "chat":
            return {"response": self.response, "source": "model"}
        if operation == "memory":
            return {"facts_added": 0, "consolidated": False}
        if operation == "endpoint":
            return 1.0

    def close(self):
        pass


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    config.audio.tts_enabled = False
    config.audio.smart_turn_enabled = False
    config.tools.enabled = ()
    config.model_tools.enabled = ()
    config.knowledge.enabled = False
    monkeypatch.setattr(runtime_module, "SileroVoiceActivityDetector", lambda *args: Mock())
    monkeypatch.setattr(runtime_module, "ChimePlayer", lambda *args: Mock())
    return runtime_module.AriettaVoiceRuntime(config, worker=FakeWorker())


@pytest.mark.parametrize("content_enabled", [False, True])
def test_passive_speech_never_reaches_logs_or_status(runtime, content_enabled, caplog, capsys) -> None:
    runtime.config.logging.content_enabled = content_enabled
    runtime.event_logger.content_enabled = content_enabled
    runtime.worker.transcript = "PRIVATE ambient conversation"
    with caplog.at_level(logging.INFO):
        runtime.process_utterance(np.zeros(512), Mock(), Mock())
    runtime._publish_status(force=True)
    assert not runtime.session.active
    assert runtime.last_user_transcript == ""
    assert "PRIVATE" not in caplog.text
    assert "PRIVATE" not in capsys.readouterr().out
    assert "PRIVATE" not in runtime.config.logging.events_file.read_text()
    assert "PRIVATE" not in runtime_status_path(runtime.config).read_text()


@pytest.mark.parametrize('stage,foreground,discarded', [
    ('transcribe', True, True), ('endpoint', True, False),
    ('chat', True, False), ('transcribe', False, False),
])
def test_only_foreground_transcription_wait_retires_capture(runtime, stage, foreground, discarded):
    import threading
    from arietta_voice.audio import MicrophoneStream
    mic = MicrophoneStream(16000, 512)
    mic._callback(np.ones((512, 1)), 512, None, False)
    runtime._turn_microphone = mic
    runtime._turn_thread_ident = threading.get_ident() if foreground else -1
    runtime.turn_stage = stage
    runtime._inference_wait()
    assert mic.empty() == discarded
    assert mic.dropped_chunks == 0
    assert mic.statistics()['discarded_by_reason'] == ({'transcribing': 1} if discarded else {})


@pytest.mark.parametrize("content_enabled", [False, True])
def test_active_content_respects_opt_in_everywhere(runtime, content_enabled, caplog, capsys, monkeypatch) -> None:
    runtime.config.logging.content_enabled = content_enabled
    runtime.event_logger.content_enabled = content_enabled
    runtime.worker.transcript = "Bridget, PRIVATE_CMD -5.2*7!"
    respond = Mock(return_value="PRIVATE ANSWER")
    monkeypatch.setattr(runtime, "_respond", respond)
    with caplog.at_level(logging.INFO):
        runtime.process_utterance(np.zeros(512), Mock(), Mock())
    runtime._publish_status(force=True)
    respond.assert_called_once_with("PRIVATE_CMD -5.2*7!")
    for output in (caplog.text, capsys.readouterr().out,
                   runtime.config.logging.events_file.read_text(),
                   runtime_status_path(runtime.config).read_text()):
        assert ("PRIVATE" in output) is content_enabled


def test_text_only_response_is_visible_in_interactive_terminal(runtime, monkeypatch, capsys) -> None:
    monkeypatch.setattr(runtime_module.sys.stdout, "isatty", lambda: True)
    runtime._deliver_response("Visible answer", microphone=Mock(), keypress_monitor=Mock(),
                              next_state=AssistantState.LISTENING)
    assert "Visible answer" in capsys.readouterr().out
    assert "Visible answer" not in runtime.config.logging.events_file.read_text()


@pytest.mark.parametrize("limit", [0, 1, 8])
def test_runtime_and_prompt_history_limits_agree(runtime, limit) -> None:
    runtime.config.assistant.max_history_turns = limit
    for index in range(10):
        runtime._store_turn(f"Question {index}", f"Answer {index}")
    assert len(runtime.session.history) == limit * 2
    messages = PromptManager(runtime.config).build_messages(runtime.session.history, "Next")
    assert len(messages) == limit * 2 + 1
    assert runtime.session.turn_count == 10


@pytest.mark.parametrize("text, should_end", [
    ("Goodbye.", True), ("Thanks, goodbye!", True), ("I'm done.", True),
    ("That’s all for now.", True), ("Okay, see you later!", True),
    ("Bye bye", True), ("Talk to you later, thank you.", True),
    ("What does goodbye mean?", False), ("Don't say goodbye yet", False),
    ("I'm done with my homework; what next?", False),
    ("Write a poem called Goodbye", False), ("That is all the information I have", False),
])
def test_only_direct_farewells_end_session(runtime, text, should_end, monkeypatch) -> None:
    respond = Mock(return_value="Answer")
    monkeypatch.setattr(runtime, "_respond", respond)
    result = runtime._handle_active_turn(text, Mock(), Mock())
    assert result.end_session_requested is should_end
    assert respond.called is not should_end


def test_ending_session_clears_private_display_state(runtime) -> None:
    runtime.last_user_transcript = "PRIVATE"
    runtime.last_assistant_text = "PRIVATE"
    runtime.reset_session()
    assert runtime.last_user_transcript == runtime.last_assistant_text == ""


@pytest.mark.parametrize("stage", ["transcribe", "chat", "tts_next"])
@pytest.mark.parametrize("timeout", [False, True])
def test_failed_turn_is_not_stored_and_next_turn_recovers(runtime, monkeypatch, stage, timeout):
    from arietta_voice.worker import WorkerFailure, WorkerTimeout
    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = "Test command"
    original = runtime.worker.call
    budgets = []
    def fail(operation, payload=None, **kwargs):
        budgets.append(kwargs["timeout_s"])
        if operation == stage:
            raise (WorkerTimeout if timeout else WorkerFailure)("PRIVATE failure")
        return original(operation, payload, **kwargs)
    runtime.worker.call = fail
    if stage == "tts_next":
        def deliver(*args, **kwargs):
            runtime._infer("tts_next")
            return []
        monkeypatch.setattr(runtime, "_deliver_response", deliver)
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.turn_failures == 1
    assert runtime.turn_timeouts == int(timeout)
    assert not runtime.session.history
    assert runtime.state == AssistantState.LISTENING
    assert "PRIVATE" not in runtime.config.logging.events_file.read_text()
    assert budgets == sorted(budgets, reverse=True)
    runtime.worker.call = original
    monkeypatch.setattr(runtime, "_deliver_response", lambda *args, **kwargs: [])
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.session.turn_count == 1
    assert runtime.last_error is None
    assert runtime.turn_stage == "idle"


def test_delivered_turn_queues_memory_without_calling_model(runtime):
    runtime.config.wake_word.enabled = False
    runtime.config.memory.enabled = True
    runtime.worker.transcript = "Remember a synthetic fact"
    original = runtime.worker.call
    calls = []
    def record(operation, payload=None, **kwargs):
        calls.append(operation)
        assert not operation.startswith('memory')
        return original(operation, payload, **kwargs)
    runtime.worker.call = record
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.session.turn_count == 1
    assert runtime.memory_jobs.snapshot()['queued'] == 1
    assert runtime.last_error is None
    assert runtime.turn_failures == 0
    assert runtime.turn_stage == 'idle'
    runtime.memory_jobs.close()


def test_disabled_memory_never_queues_private_text(runtime):
    runtime.config.memory.enabled = False
    runtime._remember_turn('Synthetic fact', 'Answer')
    assert runtime.memory_jobs.snapshot()['queued'] == 0


class ScriptedMicrophone:
    dropped_chunks = 0
    input_overflows = 0
    def __init__(self, samples):
        self.samples = iter(samples)
    def get(self, **kwargs):
        try:
            value = next(self.samples)
        except StopIteration:
            raise KeyboardInterrupt()
        if value == 2:
            self.dropped_chunks += 1
        return np.full(512, value, dtype=np.float32)
    def drain(self):
        pass


@pytest.mark.parametrize('pre_roll_ms,expected', [(0, [1, 0]), (64, [.2, .3, 1, 0])])
def test_listening_keeps_only_bounded_pre_roll(runtime, pre_roll_ms, expected):
    from arietta_voice.models import UtteranceResult
    runtime.config.audio.pre_roll_ms = pre_roll_ms
    runtime.config.audio.silence_ms = 32
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] >= 1)
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([.1, .2, .3, 1, 0]), Mock())
    recorded = runtime.process_utterance.call_args.args[0]
    np.testing.assert_allclose(recorded[::512], expected)


@pytest.mark.parametrize("prefix", [[1, 1, 1, 1, 0], [1, 2, 1, 0]])
def test_discard_long_or_gapped_speech_until_quiet_then_accept_new_turn(runtime, prefix):
    from arietta_voice.models import UtteranceResult
    runtime.config.audio.max_utterance_s = 1536 / 16000
    runtime.config.audio.silence_ms = 32
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone(prefix + [1, 0]), Mock())
    assert runtime.process_utterance.call_count == 1
    assert len(runtime.process_utterance.call_args.args[0]) == 1024


def test_negative_endpoint_cannot_accumulate_unbounded_audio(runtime):
    runtime.config.audio.smart_turn_enabled = True
    runtime.config.audio.silence_ms = 32
    runtime.config.audio.max_utterance_s = 1536 / 16000
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.worker.call = Mock(return_value=0)
    runtime.process_utterance = Mock()
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1] + [0] * 20), Mock())
    runtime.process_utterance.assert_not_called()
    assert runtime.utterances_discarded == 1
    assert all(len(call.args[1]) <= 1536 for call in runtime.worker.call.call_args_list)


def test_endpoint_timeout_recovers_and_accepts_next_turn(runtime):
    from arietta_voice.worker import WorkerTimeout
    from arietta_voice.models import UtteranceResult
    runtime.config.audio.smart_turn_enabled = True
    runtime.config.audio.silence_ms = 32
    runtime.vad.probability.side_effect = lambda chunk: float(chunk[0] > 0)
    runtime.worker.call = Mock(side_effect=[WorkerTimeout(), 1.0])
    runtime.process_utterance = Mock(return_value=UtteranceResult(buffered_chunks=[]))
    with pytest.raises(KeyboardInterrupt):
        runtime._listen(ScriptedMicrophone([1, 0, 0, 1, 0]), Mock())
    assert runtime.turn_timeouts == 1
    runtime.process_utterance.assert_called_once()


def test_startup_failure_stops_worker_and_publishes_final_status(runtime):
    import json
    runtime.warmup = Mock(side_effect=RuntimeError("failed warmup"))
    runtime.worker.close = Mock()
    runtime.chime_player.stop.side_effect = RuntimeError("failed chime cleanup")
    with pytest.raises(RuntimeError):
        runtime.run()
    runtime.worker.close.assert_called_once()
    payload = json.loads(runtime_status_path(runtime.config).read_text())
    assert payload["lifecycle"] == "stopped"
    assert payload["reliability"]["turn_stage"] == "stopped"
    assert runtime.status_publisher._thread is None


def test_cancelled_generation_is_not_a_failure_or_history_entry_and_next_turn_works(runtime):
    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = 'Synthetic question'
    original = runtime.worker.call
    def cancel(operation, payload=None, **kwargs):
        if operation == 'chat':
            runtime.cancel_turn()
            kwargs['cancellation'].check()
        return original(operation, payload, **kwargs)
    runtime.worker.call = cancel
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.turn_cancellations == 1
    assert runtime.turn_failures == runtime.turn_timeouts == 0
    assert not runtime.session.history
    assert runtime.state == AssistantState.LISTENING
    runtime.worker.call = original
    runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert runtime.session.turn_count == 1


def test_interrupted_reply_is_not_stored_or_memorized(runtime, monkeypatch):
    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = 'Synthetic question'
    followup = [np.ones(512)]
    def deliver(*args, **kwargs):
        runtime.cancel_turn()
        return followup
    monkeypatch.setattr(runtime, '_deliver_response', deliver)
    remember = Mock()
    monkeypatch.setattr(runtime, '_remember_turn', remember)
    result = runtime.process_utterance(np.zeros(512), Mock(), Mock())
    assert result.buffered_chunks is followup
    assert runtime.turn_cancellations == 1
    assert not runtime.session.history
    remember.assert_not_called()


@pytest.mark.parametrize('outcome', ['complete', 'cancel', 'failure'])
def test_streamed_voice_only_commits_fully_delivered_answers(runtime, monkeypatch, outcome):
    import asyncio
    runtime.config.audio.tts_enabled = True
    runtime.config.wake_word.enabled = False
    runtime.worker.transcript = 'Synthetic question'
    original = runtime.worker.call
    events = iter([{'kind':'clause','text':'First sentence.'},
                   {'kind':'done','result':{'response':'First sentence.','source':'model','used_tool':None,'model_tool':None}}])
    frames = []
    def infer(operation, payload=None, **kwargs):
        kwargs['cancellation'].check()
        if operation == 'chat_next':
            event = next(events)
            if outcome == 'failure' and event['kind'] == 'done':
                raise RuntimeError('Synthetic streaming failure')
            return event
        if operation == 'tts_start':
            frames[:] = [(np.zeros(4096,dtype=np.float32),24000), None]
        if operation == 'tts_next':
            return frames.pop(0)
        return original(operation,payload,**kwargs)
    runtime.worker.call = infer
    def play(*, tts_stream, **kwargs):
        async def consume():
            try:
                async for samples, rate in tts_stream:
                    if samples is not None and outcome == 'cancel':
                        runtime.cancel_turn()
                        break
            finally:
                await tts_stream.aclose()
        asyncio.run(consume())
        return PlaybackResult(interrupted=outcome=='cancel', buffered_chunks=[])
    from arietta_voice.audio import PlaybackResult
    monkeypatch.setattr(runtime_module, 'play_tts_stream', play)
    runtime.process_utterance(np.zeros(512), Mock(), Mock(pressed=lambda: False))
    assert runtime.session.turn_count == (1 if outcome == 'complete' else 0)
    assert runtime.turn_failures == (1 if outcome == 'failure' else 0)
    assert runtime.turn_cancellations == (1 if outcome == 'cancel' else 0)
    if outcome == 'complete':
        assert runtime.session.history[-1].content == 'First sentence.'
