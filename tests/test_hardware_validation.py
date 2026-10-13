"""Safety/reporting contracts only; CI must never open devices or load models."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arietta_voice.config import default_config
from scripts import hardware_validation as validation


def test_test_profile_preserves_source_settings_and_excludes_private_context(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = default_config(source_root)
    source.models.aws_secret_access_key = "synthetic-private-secret"
    source.models.bedrock_profile = "synthetic-private-profile"
    source.assistant.identity = "synthetic private persona"
    source.wake_word.phrases = ("private wake phrase",)
    source.memory.enabled = True
    source.logging.content_enabled = True
    source.admin.allow_source_editing = True
    source.tools.enabled = ("custom.private_tool",)
    source.models.max_tokens = 123
    before = asdict(source)
    target = tmp_path / "run"
    target.mkdir()
    config = validation.isolated_config(source, target)
    assert asdict(source) == before
    assert config.models.chat_model == source.models.chat_model
    assert config.models.max_tokens == 123
    assert config.audio == source.audio and config.audio is not source.audio
    assert config.runtime == source.runtime and config.runtime is not source.runtime
    assert not config.memory.enabled and not config.knowledge.enabled
    assert not config.logging.content_enabled and not config.admin.allow_source_editing
    assert not config.tools.enabled and not config.model_tools.enabled
    assert config.models.aws_secret_access_key is None
    assert config.models.bedrock_profile is None
    assert "private" not in config.prompts.soul_path.read_text()
    assert "text-to-speech" in config.prompts.soul_path.read_text()
    assert config.wake_word.phrases == default_config(target).wake_word.phrases
    for path in (config.prompts.soul_path, config.prompts.memory_path,
                 config.logging.directory, config.knowledge.directory, config.knowledge.index_directory):
        assert path.is_relative_to(target)


def test_cloud_profile_cannot_accidentally_make_paid_calls(tmp_path):
    source = default_config(tmp_path)
    source.models.chat_provider = "bedrock"
    with pytest.raises(ValueError, match="local chat provider"):
        validation.isolated_config(source, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_validation_lock_prevents_competing_hardware_runs_and_releases(tmp_path):
    with validation.validation_lock(tmp_path):
        with pytest.raises(ValueError, match="Another hardware validation"):
            with validation.validation_lock(tmp_path):
                pytest.fail("Concurrent lock was acquired")
    with validation.validation_lock(tmp_path):
        assert (tmp_path / "active.lock").stat().st_mode & 0o777 == 0o600


def test_resource_summary_excludes_startup_ramp(tmp_path):
    samples = [{"rss_mib": 5000, "ready": False}]
    samples += [{"rss_mib": value, "ready": True} for value in [100, 102, 103, 104]]
    result = validation.summarize_resources(samples)
    assert result["peak_rss_mib"] == 5000
    assert result["warm_start_rss_mib"] == 101
    assert result["warm_end_rss_mib"] == 103.5
    assert result["warm_rss_growth_mib"] == 2.5
    assert validation.summarize_resources([]) == {}


def test_event_summary_detects_content_without_copying_it(tmp_path):
    events = [{"event": "session_ended", "reason": "user_goodbye"},
              {"event": "assistant_audio_complete", "interrupted": True},
              {"event": "user_transcript", "text": "synthetic private phrase"}]
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
    (tmp_path / "events.jsonl.lock").write_text("")
    result = validation.summarize_events(tmp_path)
    assert result["voice_interruptions"] == 1
    assert result["session_end_reasons"] == {"user_goodbye": 1}
    assert result["content_fields_present"]
    assert "synthetic private phrase" not in json.dumps(result)


def test_private_fingerprint_detects_edits_without_modifying_files(tmp_path):
    (tmp_path / "config").mkdir()
    private = tmp_path / "config" / "SOUL.local.md"
    private.write_text("synthetic private persona")
    before = validation.protected_fingerprints(tmp_path)
    assert private.read_text() == "synthetic private persona"
    assert "synthetic private persona" not in json.dumps(before)
    private.write_text("edited synthetic persona")
    assert validation.protected_fingerprints(tmp_path) != before


@pytest.mark.parametrize("failure", [None, "turn", "heartbeat", "missing_heartbeat", "content", "status_content", "device", "duration", "broken_log"])
def test_live_gates_do_not_hide_failures(failure):
    status = {"lifecycle": "stopped", "last_error": None,
              "reliability": {"turn_failures": 0, "turn_timeouts": 0, "worker_restarts": 0,
                              "audio": {"input_overflows": 0}}}
    report = {"duration_completed": True, "raw_status_has_content": False,
              "events": {"content_fields_present": False, "invalid_records": 0}}
    samples = [{"ready": True, "heartbeat_age_seconds": 0.5}]
    if failure == "turn":
        status["reliability"]["turn_failures"] = 1
    elif failure == "heartbeat":
        samples[0]["heartbeat_age_seconds"] = 8
    elif failure == "missing_heartbeat":
        samples.append({"ready": True, "heartbeat_age_seconds": None})
    elif failure == "content":
        report["events"]["content_fields_present"] = True
    elif failure == "status_content":
        report["raw_status_has_content"] = True
    elif failure == "broken_log":
        report["events"]["invalid_records"] = 1
    elif failure == "device":
        status["reliability"]["audio"]["input_overflows"] = 1
    elif failure == "duration":
        report["duration_completed"] = False
    assert all(validation.live_gates(report, status, samples).values()) is (failure is None)


def test_partial_event_is_reported_as_failure_without_disclosing_text(tmp_path):
    (tmp_path / "events.jsonl").write_text('{"text":"synthetic private incomplete')
    report = validation.summarize_events(tmp_path)
    assert report["invalid_records"] == 1
    assert "synthetic private" not in json.dumps(report)


def test_playback_alignment_and_underflow_counts_are_preserved(tmp_path):
    rows = [dict(event='assistant_audio_complete', interrupted=True, alignment_resets=2, output_underflows=1),
            dict(event='assistant_audio_complete', interrupted=False, alignment_resets=1, output_underflows=0)]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    report = validation.summarize_events(tmp_path)
    assert report['alignment_resets'] == 3
    assert report['output_underflows'] == report['voice_interruptions'] == 1
    assert report['alignment_reset_reasons'] == {}
    assert report['capture_discard_reasons'] == {}
    assert report['output_timing']['playbacks'] == 0


def test_output_timing_summary_sums_counts_and_takes_maxima(tmp_path):
    rows = [dict(event='assistant_audio_complete', output_timing={
        'writes':count, 'gap_silence_writes':1, 'gap_silence_samples':480,
        'producer_wait_max_ms':delay}) for count,delay in [(4,100),(2,80)]]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    report = validation.summarize_events(tmp_path)['output_timing']
    assert report['playbacks'] == 2
    assert report['writes'] == 6
    assert report['gap_silence_writes'] == 2
    assert report['gap_silence_samples'] == 960
    assert report['producer_wait_max_ms'] == 100


def test_playback_reason_counters_are_aggregated(tmp_path):
    rows = [dict(event='assistant_audio_complete', alignment_resets=1,
                 alignment_reset_reasons={'timestamp_gap': 1},
                 capture_discard_reasons={'before_playback': 2}),
            dict(event='assistant_audio_complete', alignment_resets=2,
                 alignment_reset_reasons={'timestamp_gap': 1, 'output_underflow': 1},
                 capture_discard_reasons={'before_playback': 1})]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    report = validation.summarize_events(tmp_path)
    assert report['alignment_reset_reasons'] == {'timestamp_gap': 2, 'output_underflow': 1}
    assert report['capture_discard_reasons'] == {'before_playback': 3}


def test_cleanup_never_signals_an_unrelated_process_group(monkeypatch):
    process = Mock(pid=123)
    process.is_alive.side_effect = [True, False]
    kill, killpg = Mock(), Mock()
    monkeypatch.setattr(validation.os, "getpgid", lambda pid: 456)
    monkeypatch.setattr(validation.os, "kill", kill)
    monkeypatch.setattr(validation.os, "killpg", killpg)
    validation.stop_process(process)
    kill.assert_called_once_with(123, validation.signal.SIGINT)
    killpg.assert_not_called()


def test_tts_uses_remaining_turn_budget_and_closes_stream():
    worker = Mock(pid=1)
    runtime = SimpleNamespace(_turn_deadline=123.0, worker=worker,
                              config=SimpleNamespace(runtime=SimpleNamespace(turn_timeout_s=60)))
    runtime._infer = Mock(side_effect=[None, None])
    with pytest.raises(ValueError, match="no audio"):
        validation._synthesize(runtime, "synthetic")
    assert runtime._turn_deadline == 123.0
    worker.call.assert_called_once_with("tts_close", timeout_s=2)


def test_trial_overrides_leave_source_and_private_profile_untouched(tmp_path):
    source = default_config(tmp_path)
    before = asdict(source)
    target = validation.isolated_config(source, tmp_path / "trial")
    validation.apply_trial_overrides(target, "mlx", tmp_path / "kokoro-model")
    assert asdict(source) == before
    assert target.audio.tts_backend == "mlx"
    assert target.audio.mlx_kokoro_model == str(tmp_path / "kokoro-model")
    assert not target.memory.enabled and not target.tools.enabled


@pytest.mark.parametrize("failure", [None, "pid", "restart", "resident", "backend", "memory"])
def test_soak_rejects_worker_replacement_or_incomplete_model_evidence(failure):
    snapshot = dict(stt_loaded=True, chat_loaded=True, tts_loaded=True, endpoint_loaded=True,
                    tts_backend="mlx", peak_rss_mib=100, mlx_active_mib=200,
                    mlx_peak_mib=300, mlx_cache_mib=50)
    worker = SimpleNamespace(pid=42, restarts=0, call=Mock(return_value=snapshot))
    runtime = SimpleNamespace(worker=worker, config=SimpleNamespace(audio=SimpleNamespace(tts_backend="mlx")))
    if failure == "pid": worker.pid = 43
    if failure == "restart": worker.restarts = 1
    if failure == "resident": snapshot["chat_loaded"] = False
    if failure == "backend": snapshot["tts_backend"] = "onnx"
    if failure == "memory": snapshot["mlx_active_mib"] = float("nan")
    if failure:
        with pytest.raises(ValueError): validation.resident_sample(runtime, 0, 42)
    else:
        result = validation.resident_sample(runtime, 0, 42)
        assert result["worker_pid"] == 42 and result["mlx_active_mib"] == 200
