from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from bench import live_load_soak as soak


def test_case_order_is_balanced_in_each_cycle_and_varies():
    cycles = [[soak.case_for_turn(i + cycle * 10) for i in range(10)] for cycle in range(3)]
    assert all(sorted(cycle) == list(range(10)) for cycle in cycles)
    assert cycles[0] != cycles[1]


def make_proxy(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(soak.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(soak.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))
    physical = Mock(dropped_chunks=0, input_overflows=0, discarded_by_reason=Counter())
    runtime = SimpleNamespace(
        phase="load",
        row=None,
        seconds=60,
        load_started=10.0,
        report={"rows": []},
        session=SimpleNamespace(turn_count=0),
        config=SimpleNamespace(audio=SimpleNamespace(chunk_samples=512, sample_rate=16000)),
        fixtures=[np.ones(1024, dtype=np.float32) for _ in range(10)],
        check_control=Mock(),
        reset_session=Mock(),
        _open_session=Mock(),
        vad=Mock(),
    )
    return soak.SoakMicrophone(physical, runtime), runtime, physical, now


def test_replay_is_paced_but_player_uses_timestamped_physical_capture(monkeypatch):
    proxy, runtime, physical, now = make_proxy(monkeypatch)
    assert np.all(proxy.get() == 1)
    assert now[0] == pytest.approx(10.032)
    assert runtime.row["history_turns"] == 0
    physical.drain.assert_called_once()
    assert runtime.report["harness_discards"] == {"synthetic_input": 0}
    assert proxy.get_capture_nowait() is physical.get_capture_nowait.return_value
    assert proxy.get_nowait() is physical.get_nowait.return_value
    proxy.check_health()
    assert physical.check_health.call_count == 2
    proxy.set_capture_phase("playback")
    physical.set_capture_phase.assert_called_once_with("playback")
    assert hasattr(type(proxy), "get_capture_nowait")
    assert hasattr(type(proxy), "discard_pending")


def test_live_checks_use_real_input_without_synthetic_discard(monkeypatch):
    proxy, runtime, physical, _ = make_proxy(monkeypatch)
    runtime.phase = "live_before"
    assert proxy.get(timeout=0.1) is physical.get.return_value
    physical.discard_pending.assert_not_called()
    runtime.check_control.assert_called_once()


def test_replay_deadline_is_fatal_to_runtime_recovery_loop(monkeypatch):
    proxy, _, _, now = make_proxy(monkeypatch)
    proxy.get()
    now[0] += 16
    with pytest.raises(soak.SoakFailure, match="endpoint_deadline"):
        proxy.get()
    assert not issubclass(soak.SoakFailure, Exception)


def test_duration_waits_for_current_turn_then_stops_before_next(monkeypatch):
    proxy, runtime, _, now = make_proxy(monkeypatch)
    proxy.get()
    runtime.seconds = 0.01
    proxy.get()  # Current utterance continues beyond the duration boundary.
    runtime.report["rows"].append(runtime.row)
    proxy.advance()
    with pytest.raises(soak.PhaseComplete):
        proxy.get()


def test_idle_gap_does_not_open_new_turn_and_keeps_capture_drained(monkeypatch):
    proxy, runtime, physical, _ = make_proxy(monkeypatch)
    proxy.get()
    runtime.report["rows"].append(runtime.row)
    proxy.advance()
    assert proxy.get() is None
    assert runtime.row is None
    assert physical.drain.call_count == 2


def test_bounded_report_turns(monkeypatch):
    proxy, runtime, _, _ = make_proxy(monkeypatch)
    runtime.report["rows"] = [None] * soak.MAX_TURNS
    with pytest.raises(soak.SoakFailure, match="turn_limit"):
        proxy.get()


def test_markers_are_phase_scoped_and_abort_is_always_honored(tmp_path, monkeypatch):
    runtime = soak.LiveLoadRuntime.__new__(soak.LiveLoadRuntime)
    runtime.directory = tmp_path
    runtime.phase_started, runtime.check_seconds = 0, 60
    monkeypatch.setattr(soak.time, "monotonic", lambda: 1)
    (tmp_path / "finish").touch()
    runtime.phase = "live_before"
    runtime.check_control()
    (tmp_path / "start-load").touch()
    with pytest.raises(soak.PhaseComplete):
        runtime.check_control()
    runtime.phase = "load"
    runtime.check_control()
    (tmp_path / "abort").touch()
    with pytest.raises(soak.SoakFailure, match="operator_abort"):
        runtime.check_control()


def test_operator_window_has_a_deadline(tmp_path, monkeypatch):
    runtime = soak.LiveLoadRuntime.__new__(soak.LiveLoadRuntime)
    runtime.directory, runtime.phase = tmp_path, "live_after"
    runtime.phase_started, runtime.check_seconds = 0, 60
    monkeypatch.setattr(soak.time, "monotonic", lambda: 61)
    with pytest.raises(soak.SoakFailure, match="window_expired"):
        runtime.check_control()


def test_gates_require_duration_case_coverage_resources_and_clean_audio():
    report = dict(
        load_completed=True,
        rows=[dict(case_id=i, valid=True, resources={"worker_pid": 1}) for i in range(10)],
        events=dict(output_underflows=0, alignment_resets=0),
        reliability=dict(audio={"dropped_chunks": 0}, devices={"reconnect_attempts": 0}),
    )
    assert all(soak.soak_gates(report).values())
    report["rows"][2]["valid"] = False
    assert not soak.soak_gates(report)["valid_load_turns"]
    report["events"]["output_underflows"] = 1
    assert not soak.soak_gates(report)["no_output_underflows"]
    report["rows"].pop()
    assert not soak.soak_gates(report)["all_ten_cases"]
    assert not all(soak.soak_gates({}).values())


def test_test_discard_uses_real_microphone_api_and_separate_attribution():
    from arietta_voice.audio import CaptureChunk, MicrophoneStream

    microphone = MicrophoneStream(16000, 512, max_queue_chunks=4)
    for i in range(3):
        microphone._audio_q.put_nowait(CaptureChunk(np.ones(512, dtype=np.float32), float(i)))
    runtime = SimpleNamespace(report={})
    soak.discard_test_capture(runtime, microphone, "synthetic_input")
    assert microphone.statistics()["discarded_by_reason"] == {"turn_boundary": 3}
    assert runtime.report["harness_discards"] == {"synthetic_input": 3}
    assert microphone.empty() and microphone.dropped_chunks == 0


def test_paced_replay_keeps_real_capture_health_polling(monkeypatch):
    from arietta_voice.devices import CaptureHealth

    proxy, runtime, physical, now = make_proxy(monkeypatch)
    health = CaptureHealth(3, clock=lambda: now[0], wall=lambda: now[0])

    def check():
        health.captured()  # Simulate continuous physical device callbacks.
        health.check()

    physical.check_health.side_effect = check
    runtime.fixtures = [np.ones(16000 * 8, dtype=np.float32) for _ in range(10)]
    for _ in range(250):
        proxy.get()
    assert now[0] == pytest.approx(18.0)
    assert physical.check_health.call_count == 250


def test_resuming_after_separate_opening_checks_retains_closing_window():
    assert soak.phase_order(False, False) == ("live_before", "load", "live_after")
    assert soak.phase_order(False, True) == ("load", "live_after")
    assert soak.phase_order(True, False) == ("load",)


def test_resource_failure_is_fatal_instead_of_becoming_endpoint_timeout(monkeypatch):
    runtime = soak.LiveLoadRuntime.__new__(soak.LiveLoadRuntime)
    runtime.phase = "load"
    runtime.turn_failures = runtime.turn_cancellations = 0
    runtime.load_started, runtime.expected_pid = 0, 42
    runtime.row = dict(history_turns=0, playbacks=1, interrupted=False)
    runtime.session = SimpleNamespace(turn_count=1)
    runtime.report = {"rows": []}
    runtime.microphone = Mock(discarded_by_reason=Counter())
    runtime.save = Mock()
    monkeypatch.setattr(
        soak.AriettaVoiceRuntime,
        "process_utterance",
        lambda *args: SimpleNamespace(buffered_chunks=[], end_session_requested=False),
    )
    monkeypatch.setattr(soak, "resident_sample", Mock(side_effect=ValueError("Worker changed")))
    with pytest.raises(soak.SoakFailure, match="load_resource_validation"):
        runtime.process_utterance(np.zeros(512), Mock(), None)
    assert len(runtime.report["rows"]) == 1
    assert not runtime.report["rows"][0]["valid"]


def test_worker_continuity_is_established_after_opening_cancellation(monkeypatch):
    runtime = soak.LiveLoadRuntime.__new__(soak.LiveLoadRuntime)
    runtime.phase, runtime.row = "startup", None
    runtime.report, runtime.seconds = {"rows": []}, 0
    runtime.automated_only = runtime.skip_before = False
    runtime.worker = SimpleNamespace(pid=1, call=Mock())
    runtime.config = SimpleNamespace(runtime=SimpleNamespace(startup_timeout_s=60))
    runtime.reset_session = runtime.set_state = runtime.save = Mock()
    runtime.vad = runtime.event_logger = Mock()
    runtime.reliability_snapshot = Mock(return_value={})
    microphone = Mock(discarded_by_reason=Counter())
    identities = []

    def resident(model, started, expected):
        assert model.worker.pid == expected
        identities.append(expected)
        return {"worker_pid": expected}

    monkeypatch.setattr(soak, "resident_sample", resident)

    def listen(model, mic, keys):
        if model.phase == "live_before":
            model.worker.pid = 2
        if model.phase == "live_after":
            model.worker.pid = 3
        raise soak.PhaseComplete

    monkeypatch.setattr(soak.AriettaVoiceRuntime, "_listen", listen)
    runtime._listen(microphone, None)
    assert identities == [1, 2, 2]
    assert runtime.report["worker_pid_after_checks"] == 3
    assert runtime.report["load_initial_resources"]["worker_pid"] == 2
    assert runtime.report["load_end_resources"]["worker_pid"] == 2
    assert runtime.phase == "finished"
