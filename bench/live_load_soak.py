"""Isolated paced-input soak with real capture/AEC/output and live checks.

The disposable run directory contains start-load and finish marker files. Create
them after the before/after spoken checks. An abort marker stops any phase.
No marker contents are executed, and none are production app commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from arietta_voice.config import load_config
from arietta_voice.models import AssistantState
from arietta_voice.runtime import AriettaVoiceRuntime
from arietta_voice.runtime_status import read_runtime_status
from bench.hardware_validation import (
    apply_trial_overrides,
    collect_environment,
    isolated_config,
    live_gates,
    protected_fingerprints,
    resident_sample,
    rss_sample,
    stop_process,
    summarize_events,
    summarize_resources,
    validation_lock,
    write_report,
)
from bench.latency_baseline import QUESTIONS, SEED, load_shared_fixtures, quantiles

MAX_TURNS = 5000
IDLE_SECONDS = (1, 1, 2, 5, 1, 1, 2, 10, 1, 1)


class PhaseComplete(BaseException):
    """Escape the listening loop only at a between-turn boundary."""


class SoakFailure(BaseException):
    """Do not let the runtime's recoverable-error loop conceal test failures."""


def normalize(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def case_for_turn(index):
    order = list(range(len(QUESTIONS)))
    random.Random(SEED + index // len(QUESTIONS)).shuffle(order)
    return order[index % len(QUESTIONS)]


def discard_test_capture(runtime, microphone, reason):
    """Use the existing bounded drain; retain test attribution separately."""
    before = microphone.discarded_by_reason["turn_boundary"]
    microphone.drain()
    count = microphone.discarded_by_reason["turn_boundary"] - before
    counters = runtime.report.setdefault("harness_discards", {})
    counters[reason] = counters.get(reason, 0) + count


class SoakMicrophone:
    """Replay only listening input; the player always receives physical capture.

    Explicit methods preserve the player's type-based timestamp/health probes.
    Physical input during synthetic listening/rests is intentionally discarded,
    counted separately, and never merged with the synthetic utterance.
    """

    def __init__(self, physical, runtime):
        self.physical, self.runtime = physical, runtime
        self.offset = 0
        self.started = None
        self.next_at = time.monotonic()

    @property
    def dropped_chunks(self):
        return self.physical.dropped_chunks

    @property
    def input_overflows(self):
        return self.physical.input_overflows

    def get_capture_nowait(self):
        return self.physical.get_capture_nowait()

    def get_nowait(self):
        return self.physical.get_nowait()

    def drain(self):
        self.physical.drain()

    def discard_pending(self, reason):
        self.physical.discard_pending(reason)

    def set_capture_phase(self, phase):
        self.physical.set_capture_phase(phase)

    def check_health(self):
        self.physical.check_health()

    def statistics(self):
        return self.physical.statistics()

    def get(self, timeout=None):
        runtime = self.runtime
        runtime.check_control()
        if runtime.phase != "load":
            return self.physical.get(timeout=timeout)
        self.physical.check_health()
        # Poll at the normal audio cadence so capture remains bounded during
        # fixture replay and rests, without pretending this is live room input.
        discard_test_capture(runtime, self.physical, "synthetic_input")
        now = time.monotonic()
        if runtime.row is None:
            if now - runtime.load_started >= runtime.seconds:
                raise PhaseComplete
            if now < self.next_at:
                time.sleep(min(0.032, self.next_at - now))
                return None
            index = len(runtime.report["rows"])
            if index >= MAX_TURNS:
                raise SoakFailure("turn_limit")
            if index % 3 == 0:
                runtime.reset_session()
                runtime._open_session(None)
            runtime.vad.reset()
            self.started, self.offset = now, 0
            runtime.row = dict(
                turn=index + 1,
                case_id=case_for_turn(index),
                history_turns=runtime.session.turn_count,
                playbacks=0,
                interrupted=False,
                output_underflows=0,
                alignment_resets=0,
                stages_ms={},
                transcript_matches_fixture=False,
            )
        row = runtime.row
        source = runtime.fixtures[row["case_id"]]
        width, rate = runtime.config.audio.chunk_samples, runtime.config.audio.sample_rate
        deadline = self.started + (self.offset + width) / rate
        time.sleep(max(0, deadline - time.monotonic()))
        now = time.monotonic()
        row["max_replay_lag_ms"] = max(row.get("max_replay_lag_ms", 0), (now - deadline) * 1000)
        if now > self.started + len(source) / rate + 15:
            raise SoakFailure("fixture_endpoint_deadline")
        chunk = np.zeros(width, dtype=np.float32)
        piece = source[self.offset : self.offset + width]
        chunk[: len(piece)] = piece
        self.offset += width
        return chunk

    def advance(self):
        index = len(self.runtime.report["rows"]) - 1
        self.next_at = time.monotonic() + IDLE_SECONDS[index % len(IDLE_SECONDS)]
        self.started, self.offset = None, 0
        self.runtime.row = None


class LiveLoadRuntime(AriettaVoiceRuntime):
    def __init__(
        self,
        config,
        directory,
        fixtures,
        seconds,
        check_seconds,
        automated_only=False,
        skip_before=False,
    ):
        super().__init__(config)
        self.directory, self.fixtures = directory, fixtures
        self.seconds, self.check_seconds = seconds, check_seconds
        self.automated_only = automated_only
        self.skip_before = skip_before
        self.phase, self.row = "startup", None
        self.load_started = None
        self.report = dict(
            status="running",
            phase="startup",
            ready=False,
            rows=[],
            requested_load_seconds=seconds,
            automated_only=automated_only,
            skip_before=skip_before,
            private_context_loaded=False,
            recordings_saved=False,
            protocol="paced synthetic listening input; real capture/AEC/streamed speaker output",
        )

    def save(self):
        self.report.update(phase=self.phase, reliability=self.reliability_snapshot())
        write_report(self.directory / "report.json", self.report)

    def check_control(self):
        if (self.directory / "abort").exists():
            raise SoakFailure("operator_abort")
        if self.phase in ("live_before", "live_after"):
            marker = "start-load" if self.phase == "live_before" else "finish"
            if (self.directory / marker).exists():
                raise PhaseComplete
            if time.monotonic() - self.phase_started > self.check_seconds:
                raise SoakFailure("operator_window_expired")

    def _log_transcript(self, transcript, latency_ms):
        if self.row is not None:
            self.row["transcript_matches_fixture"] = normalize(transcript) == normalize(
                QUESTIONS[self.row["case_id"]]
            )
        super()._log_transcript(transcript, latency_ms)

    def _log_playback(self, playback):
        super()._log_playback(playback)
        if self.row is not None:
            self.row["playbacks"] += 1
            self.row["interrupted"] |= playback.interrupted
            self.row["output_underflows"] += playback.output_underflows
            self.row["alignment_resets"] += playback.alignment_resets

    def process_utterance(self, audio, microphone, keypress_monitor, **options):
        if self.phase != "load":
            return super().process_utterance(audio, microphone, keypress_monitor, **options)
        started, failures, cancellations = (
            time.monotonic(),
            self.turn_failures,
            self.turn_cancellations,
        )
        result = super().process_utterance(audio, microphone, keypress_monitor, **options)
        row = self.row
        row.update(
            processing_seconds=time.monotonic() - started,
            elapsed_seconds=time.monotonic() - self.load_started,
            valid=(
                self.turn_failures == failures
                and self.turn_cancellations == cancellations
                and not result.buffered_chunks
                and not result.end_session_requested
                and self.session.turn_count == row["history_turns"] + 1
                and row["playbacks"] > 0
                and not row["interrupted"]
            ),
        )
        discard_test_capture(self, self.microphone, "resource_sampling")
        try:
            row["resources"] = resident_sample(self, self.load_started, self.expected_pid)
        except Exception as exc:
            row.update(valid=False, resource_error_type=type(exc).__name__)
            self.report["rows"].append(row)
            self.save()
            raise SoakFailure("load_resource_validation") from exc
        discard_test_capture(self, self.microphone, "resource_sampling")
        self.report["rows"].append(row)
        self.report.update(
            completed_turns=len(self.report["rows"]), load_elapsed_seconds=row["elapsed_seconds"]
        )
        self.save()
        if not row["valid"]:
            raise SoakFailure("invalid_load_turn")
        microphone.advance()
        return result

    def _listen(self, microphone, keypress_monitor):
        # A reconnect must fail this continuous-worker/device run, not restart it.
        if self.phase != "startup":
            raise SoakFailure("audio_reconnected")
        proxy = SoakMicrophone(microphone, self)
        self.expected_pid = self.worker.pid
        self.report["initial_resources"] = resident_sample(
            self, time.monotonic(), self.expected_pid
        )
        original_call = self.worker.call

        def measured_call(operation, payload=None, **kwargs):
            started = time.monotonic()
            if operation in {"resource_snapshot", "warmup"}:

                def sampling_wait():
                    discard_test_capture(self, microphone, "resource_sampling")
                    self._playback_wait()

                kwargs.setdefault("on_wait", sampling_wait)
            value = original_call(operation, payload, **kwargs)
            if self.row is not None:
                stage = "chat" if operation in {"chat_start", "chat_next"} else operation
                stages = self.row["stages_ms"]
                stages[stage] = stages.get(stage, 0) + (time.monotonic() - started) * 1000
            return value

        self.worker.call = measured_call
        self.report["ready"] = True
        phases = phase_order(self.automated_only, self.skip_before)
        for phase in phases:
            self.phase, self.phase_started = phase, time.monotonic()
            self.reset_session()
            self.vad.reset()
            discard_test_capture(self, microphone, "soak_phase_boundary")
            self.set_state(AssistantState.WAITING_FOR_WAKE)
            if phase == "load":
                # Live cancellation may intentionally replace the worker without
                # a failure restart. Establish the continuous-load baseline only
                # after all four models have been warmed again.
                self.report["worker_pid_before_load_warmup"] = self.worker.pid
                self.worker.call("warmup", timeout_s=self.config.runtime.startup_timeout_s)
                self.expected_pid = self.worker.pid
                self.load_started = time.monotonic()
                self.report["load_initial_resources"] = resident_sample(
                    self, self.load_started, self.expected_pid
                )
                self.report["load_started_at"] = datetime.now(timezone.utc).isoformat()
            self.save()
            self.event_logger.log("live_load_phase", phase=phase)
            try:
                super()._listen(proxy, keypress_monitor)
            except PhaseComplete:
                pass
            if phase == "load":
                self.report["load_elapsed_seconds"] = time.monotonic() - self.load_started
                self.report["load_completed"] = self.report["load_elapsed_seconds"] >= self.seconds
                self.report["load_end_resources"] = resident_sample(
                    self, self.load_started, self.expected_pid
                )
            self.report[phase + "_reliability"] = self.reliability_snapshot()
        self.report["worker_pid_after_checks"] = self.worker.pid
        self.phase = "finished"
        self.report["status"] = "completed"
        self.save()


def phase_order(automated_only, skip_before):
    if automated_only:
        return ("load",)
    return ("load", "live_after") if skip_before else ("live_before", "load", "live_after")


def run_child(
    config, directory, fixtures_path, minutes, check_minutes, automated_only, skip_before
):
    os.setsid()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    runtime = None
    try:
        runtime = LiveLoadRuntime(
            config,
            directory,
            load_shared_fixtures(fixtures_path),
            minutes * 60,
            check_minutes * 60,
            automated_only,
            skip_before,
        )
        runtime.save()
        runtime.run()
        if runtime.phase != "finished":
            raise SoakFailure("stopped_before_completion")
    except BaseException as exc:
        if runtime is None:
            write_report(
                directory / "report.json", dict(status="failed", error_type=type(exc).__name__)
            )
        else:
            runtime.report.update(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, SoakFailure):
                runtime.report["failure_code"] = str(exc)
    finally:
        if runtime is not None:
            runtime.save()


def soak_gates(report):
    rows = report.get("rows", [])
    events = report.get("events", {})
    audio = report.get("reliability", {}).get("audio", {})
    return dict(
        duration_completed=report.get("load_completed") is True,
        valid_load_turns=bool(rows) and all(r.get("valid") for r in rows),
        all_ten_cases={r.get("case_id") for r in rows} == set(range(len(QUESTIONS))),
        resources_each_turn=bool(rows) and all(r.get("resources") for r in rows),
        no_output_underflows=events.get("output_underflows") == 0,
        no_alignment_resets=events.get("alignment_resets") == 0,
        no_capture_queue_drops=audio.get("dropped_chunks") == 0,
        no_device_reconnects=report.get("reliability", {})
        .get("devices", {})
        .get("reconnect_attempts")
        == 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--check-minutes", type=float, default=8)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--tts-backend", choices=["onnx", "mlx"])
    parser.add_argument("--mlx-kokoro-model", type=Path)
    parser.add_argument("--automated-only", action="store_true")
    parser.add_argument(
        "--skip-before",
        action="store_true",
        help="Reuse separately recorded opening checks; retain closing live window",
    )
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or not 0 < args.minutes <= 60:
        parser.error("minutes must be finite and in (0, 60]")
    if not math.isfinite(args.check_minutes) or not 0 < args.check_minutes <= 15:
        parser.error("check-minutes must be finite and in (0, 15]")
    load_shared_fixtures(args.fixtures)
    root = Path(__file__).resolve().parents[1]
    os.umask(0o077)
    with validation_lock(root / "tmp/hardware-validation"):
        source = load_config(args.config)
        if read_runtime_status(source)["is_running"]:
            parser.error("Stop the normal runtime before starting the soak")
        protected = protected_fingerprints(root)
        directory = (
            root / "tmp/live-load-soak" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        directory.mkdir(parents=True, mode=0o700)
        config = isolated_config(source, directory)
        apply_trial_overrides(config, args.tts_backend, args.mlx_kokoro_model)
        if not (
            config.audio.aec_enabled
            and config.audio.tts_enabled
            and config.audio.smart_turn_enabled
        ):
            parser.error("Live/load soak requires AEC, TTS and Smart Turn enabled")
        metadata = collect_environment(config, root)
        metadata.update(
            soak_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fixtures_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
            minutes=args.minutes,
            check_minutes=args.check_minutes,
            idle_seconds=IDLE_SECONDS,
            conversation_turns=3,
            seed=SEED,
        )
        write_report(directory / "environment.json", metadata)
        print(f"Soak directory: {directory}", flush=True)
        process = multiprocessing.get_context("spawn").Process(
            target=run_child,
            args=(
                config,
                directory,
                args.fixtures,
                args.minutes,
                args.check_minutes,
                args.automated_only,
                args.skip_before,
            ),
        )
        samples, host, error = [], [], None
        from bench.host_state import HostState

        monitor = HostState()
        started, last_phase, last_print = time.monotonic(), None, 0
        process.start()
        try:
            while process.is_alive():
                path = directory / "report.json"
                report = json.loads(path.read_text()) if path.exists() else {}
                status = read_runtime_status(config)
                elapsed = time.monotonic() - started
                phase = report.get("phase", "startup")
                sample = dict(
                    elapsed_seconds=elapsed,
                    phase=phase,
                    ready=report.get("ready", False),
                    heartbeat_age_seconds=status.get("heartbeat_age_seconds"),
                    **rss_sample(process.pid),
                )
                samples.append(sample)
                host.append(dict(monitor.sample(), phase=phase))
                write_report(directory / "resources.json", dict(samples=samples))
                write_report(directory / "host.json", host)
                if phase != last_phase or elapsed - last_print >= 60:
                    print(
                        json.dumps(
                            dict(
                                phase=phase,
                                elapsed_s=round(elapsed),
                                turns=report.get("completed_turns", 0),
                                rss_mib=sample["rss_mib"],
                            )
                        ),
                        flush=True,
                    )
                    last_phase, last_print = phase, elapsed
                if (
                    elapsed
                    > config.runtime.startup_timeout_s
                    + (args.minutes + 2 * args.check_minutes) * 60
                    + 180
                ):
                    raise TimeoutError("Soak outer deadline")
                process.join(timeout=5)
        except (Exception, KeyboardInterrupt) as exc:
            error = type(exc).__name__
        finally:
            stop_process(process)
        path = directory / "report.json"
        report = json.loads(path.read_text()) if path.exists() else dict(status="failed")
        status = read_runtime_status(config)
        report.update(
            events=summarize_events(config.logging.directory),
            reliability=status.get("reliability", {}),
            duration_completed=report.get("load_completed", False),
            protected_files_unchanged=protected == protected_fingerprints(root),
            human_checks="pending",
            resources_by_phase={
                p: summarize_resources([s for s in samples if s["phase"] == p])
                for p in sorted({s["phase"] for s in samples})
            },
        )
        raw = (
            json.loads((config.logging.directory / "runtime_status.json").read_text())
            if (config.logging.directory / "runtime_status.json").exists()
            else {}
        )
        report["raw_status_has_content"] = (
            bool(raw.get("last_user_transcript") or raw.get("last_assistant_text")) if raw else None
        )
        report["automated_gates"] = live_gates(report, status, samples)
        report["soak_gates"] = soak_gates(report)
        rows = report.get("rows", [])
        report["summary"] = dict(
            turns=len(rows),
            case_counts={
                str(i): sum(r["case_id"] == i for r in rows) for i in range(len(QUESTIONS))
            },
            processing_seconds=quantiles([r["processing_seconds"] for r in rows]),
            exact_transcript_matches=sum(r.get("transcript_matches_fixture", False) for r in rows),
        )
        if (
            error
            or process.exitcode != 0
            or report.get("status") != "completed"
            or not report["protected_files_unchanged"]
            or not all(report["automated_gates"].values())
            or not all(report["soak_gates"].values())
        ):
            report["status"] = "failed"
        if error:
            report["parent_error_type"] = error
        write_report(path, report)
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
        return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
