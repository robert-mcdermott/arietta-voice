"""Opt-in local hardware/model validation; never part of the synthetic CI suite.

uv run --locked scripts/hardware_validation.py --minutes 15
uv run --locked scripts/hardware_validation.py --mode live --minutes 5
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import fcntl
from importlib.metadata import version
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arietta_voice.config import default_config, load_config, validate_config


SYNTHETIC_PHRASE = "Bridget, what is two plus two?"
SYNTHETIC_PERSONA = (
    "You are Bridget, a voice assistant undergoing a synthetic hardware test. "
    "Your replies ARE spoken aloud by the application's text-to-speech engine. "
    "Answer directly in plain spoken language. Do not claim you cannot speak. "
    "If asked to count, write every requested number, without an introduction. "
    "Otherwise answer briefly in one sentence.\n"
)


@contextmanager
def validation_lock(directory):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / "active.lock").open("a") as handle:
        os.fchmod(handle.fileno(), 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another hardware validation is active") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def collect_environment(config, root):
    import sounddevice as sd
    from huggingface_hub import snapshot_download
    from arietta_voice.assets import cache_root
    metadata = {
        "app_version": version("arietta-voice"), "platform": platform.platform(),
        "python": platform.python_version(), "architecture": platform.machine(),
        "tts_backend": config.audio.tts_backend,
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "models": {"chat": config.models.chat_model, "stt_language": config.models.stt_language,
                   "voice": config.audio.voice, "max_tokens": config.models.max_tokens},
        "audio": {"sample_rate": config.audio.sample_rate, "chunk_samples": config.audio.chunk_samples,
                  "aec": config.audio.aec_enabled, "smart_turn": config.audio.smart_turn_enabled,
                  "tts": config.audio.tts_enabled,
                  "pre_roll_ms": config.audio.pre_roll_ms, "aec_delay_ms": config.audio.aec_delay_ms,
                  "tts_speed": config.audio.tts_speed, "queue_max_chunks": config.audio.queue_max_chunks},
        "overrides": ["synthetic persona and default wake phrases", "no knowledge, memory or tools",
                      "content logging off; no recordings", "temperature 0"],
        "packages": {name: version(name) for name in ("moonshine-voice", "mlx-vlm", "mlx", "kokoro-onnx", "onnxruntime", "torch", "silero-vad", "sounddevice")},
    }
    metadata["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    metadata["ram_gib"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30
    from arietta_voice.devices import select_devices
    selected_input, selected_output = select_devices(config.audio)
    metadata["devices"] = {kind: {key: row[key] for key in ("index", "name", "selector")}
                           for kind, row in (("input", selected_input), ("output", selected_output)) if row}
    metadata["cached_revisions"] = {}
    for model in (config.models.chat_model, "openai/whisper-tiny"):
        if Path(model).is_dir():
            metadata["cached_revisions"]["local_chat_path"] = "unversioned"
            continue
        try:
            metadata["cached_revisions"][model] = Path(snapshot_download(model, local_files_only=True)).name
        except Exception as exc:
            metadata["cached_revisions"][model] = type(exc).__name__
    from moonshine_voice.download import find_model_info, get_components_for_model_info
    from moonshine_voice.download_file import get_cache_dir
    stt = find_model_info(config.models.stt_language)
    stt_root = get_cache_dir() / stt["download_url"].removeprefix("https://")
    metadata["models"]["stt_model"] = stt["model_name"]
    metadata["stt_asset_sha256"] = {}
    for component in get_components_for_model_info(stt):
        asset = stt_root / component
        if asset.is_file():
            with asset.open("rb") as handle:
                metadata["stt_asset_sha256"][component] = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata["asset_sha256"] = {}
    for asset in (cache_root() / "kokoro" / "kokoro-v1.0.onnx",
                  cache_root() / "kokoro" / "voices-v1.0.bin",
                  Path(tempfile.gettempdir()) / "arietta-smart-turn" / "smart-turn-v3.2-cpu.onnx"):
        if asset.is_file():
            with asset.open("rb") as handle:
                metadata["asset_sha256"][asset.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return metadata


def apply_trial_overrides(config, backend=None, model=None):
    """Apply only to the disposable profile, never the source configuration."""
    if backend is not None:
        config.audio.tts_backend = backend
    if model is not None:
        config.audio.mlx_kokoro_model = str(model.resolve())
    validate_config(config)


def resident_sample(runtime, started, expected_pid):
    if runtime.worker.pid != expected_pid or runtime.worker.restarts:
        raise ValueError("Soak inference worker changed or restarted")
    sample = runtime.worker.call("resource_snapshot", timeout_s=5)
    if not all(sample.get(k) for k in ("stt_loaded", "chat_loaded", "tts_loaded", "endpoint_loaded")):
        raise ValueError("Soak requires all four models resident")
    if sample.get("tts_backend") != runtime.config.audio.tts_backend:
        raise ValueError("Unexpected soak TTS backend")
    for key in ("peak_rss_mib", "mlx_active_mib", "mlx_peak_mib", "mlx_cache_mib"):
        if sample.get(key) is None or not math.isfinite(sample[key]) or sample[key] < 0:
            raise ValueError("Missing or invalid model memory sample")
    return dict(sample, elapsed_seconds=time.monotonic() - started, worker_pid=expected_pid)


def summarize_events(directory):
    counts, reasons = Counter(), Counter()
    interruptions, content_fields, invalid_records = 0, False, 0
    alignment_resets, output_underflows = 0, 0
    for path in directory.glob("events.jsonl*"):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Invalid event")
            except ValueError:
                invalid_records += 1
                continue
            counts[event.get("event", "unknown")] += 1
            if event.get("event") == "session_ended":
                reasons[event.get("reason", "unknown")] += 1
            interruptions += event.get("event") == "assistant_audio_complete" and event.get("interrupted", False)
            if event.get("event") == "assistant_audio_complete":
                alignment_resets += event.get("alignment_resets", 0)
                output_underflows += event.get("output_underflows", 0)
            content_fields |= bool({"text", "transcript", "response", "prompt", "content"}.intersection(event))
    return {"event_counts": dict(counts), "session_end_reasons": dict(reasons),
            "voice_interruptions": interruptions, "content_fields_present": content_fields,
            "alignment_resets": alignment_resets, "output_underflows": output_underflows,
            "invalid_records": invalid_records}


def protected_fingerprints(root):
    """Compare local data without copying it into test reports or fixtures."""
    result = {}
    for name in ("config", "knowledge"):
        for path in (root / name).rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    result[str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def live_gates(report, status, samples):
    reliability = status.get("reliability", {})
    audio = reliability.get("audio", {})
    ages = [s["heartbeat_age_seconds"] for s in samples if s.get("ready") and s.get("heartbeat_age_seconds") is not None]
    return {
        "duration_completed": report.get("duration_completed", False),
        "clean_stop": status.get("lifecycle") == "stopped" and status.get("last_error") is None,
        "no_failed_turns": reliability.get("turn_failures") == 0,
        "no_timeouts": reliability.get("turn_timeouts") == 0,
        "no_worker_restarts": reliability.get("worker_restarts") == 0,
        "heartbeat_within_5_seconds": bool(ages) and max(ages) <= 5
        and len(ages) == sum(bool(s.get("ready")) for s in samples),
        "no_input_device_overflows": audio.get("input_overflows") == 0,
        "no_content_in_events": not report.get("events", {}).get("content_fields_present", True),
        "no_content_in_status": report.get("raw_status_has_content") is False
        and not status.get("last_user_transcript") and not status.get("last_assistant_text"),
        "valid_event_log": report.get("events", {}).get("invalid_records") == 0,
    }


def isolated_config(source, directory):
    """Copy only the model/audio settings needed for a local test, never secrets."""
    if source.models.chat_provider != "local":
        raise ValueError("Hardware validation requires a local chat provider; cloud calls are not part of this test")
    config = default_config(directory)
    config.audio = replace(source.audio)
    config.runtime = replace(source.runtime)
    config.models.chat_model = source.models.chat_model
    config.models.stt_language = source.models.stt_language
    config.models.max_tokens = source.models.max_tokens
    config.models.temperature = 0.0
    config.knowledge.enabled = False
    config.memory.enabled = False
    config.tools.enabled = ()
    config.model_tools.enabled = ()
    config.prompts.soul_path.parent.mkdir(parents=True, exist_ok=True)
    config.prompts.soul_path.write_text(SYNTHETIC_PERSONA)
    validate_config(config)
    return config


def write_report(path, report):
    temporary = path.with_suffix(".working")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def rss_sample(pid):
    """Aggregate resident memory/CPU for only our owned process tree."""
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss=,%cpu="], text=True, timeout=5)
    rows = [row.split() for row in output.splitlines()]
    descendants = {pid}
    while True:
        added = {int(row[0]) for row in rows if int(row[1]) in descendants} - descendants
        if not added:
            break
        descendants.update(added)
    selected = [row for row in rows if int(row[0]) in descendants]
    return {"rss_mib": round(sum(int(row[2]) for row in selected) / 1024, 1),
            "cpu_percent": round(sum(float(row[3]) for row in selected), 1),
            "processes": len(selected)}


def summarize_resources(samples):
    if not samples:
        return {}
    # Separate the startup ramp from the sustained portion; this is an observed
    # envelope, not proof that memory can never grow over a longer run.
    warm = [sample["rss_mib"] for sample in samples if sample.get("ready")]
    window = min(6, len(warm) // 2)
    return {"samples": len(samples), "peak_rss_mib": max(s["rss_mib"] for s in samples),
            "warm_start_rss_mib": statistics.median(warm[:window]) if window else None,
            "warm_end_rss_mib": statistics.median(warm[-window:]) if window else None,
            "warm_rss_growth_mib": round(statistics.median(warm[-window:]) - statistics.median(warm[:window]), 1)
            if window else None}


def _synthesize(runtime, text):
    import numpy as np
    chunks, count, rate = [], 0, None
    previous_deadline = runtime._turn_deadline
    runtime._turn_deadline = previous_deadline or time.monotonic() + runtime.config.runtime.turn_timeout_s
    try:
        runtime._infer("tts_start", text)
        while True:
            item = runtime._infer("tts_next")
            if item is None:
                break
            audio, sample_rate = item
            if rate is not None and rate != sample_rate:
                raise ValueError("TTS changed sample rate")
            rate = sample_rate
            count += len(audio)
            if count > rate * 30:
                raise ValueError("Synthetic TTS exceeded 30 seconds")
            chunks.append(audio)
        if not chunks:
            raise ValueError("TTS produced no audio")
        return np.concatenate(chunks), rate
    finally:
        runtime._turn_deadline = previous_deadline
        if runtime.worker.pid is not None:
            runtime.worker.call("tts_close", timeout_s=2)


def _components(config, report, path, minutes, microphone_seconds, playback,
                resource_metrics=False, fixtures_path=None, soak_interval=5):
    import numpy as np
    import sounddevice as sd
    from arietta_voice.audio import MicrophoneStream
    from arietta_voice.runtime import AriettaVoiceRuntime

    runtime = AriettaVoiceRuntime(config)
    runtime.status_publisher.start(runtime)
    timings = []
    try:
        from arietta_voice.devices import select_devices
        selected_input, selected_output = select_devices(config.audio)
        start = time.monotonic()
        runtime.warmup()
        report["warmup_seconds"] = round(time.monotonic() - start, 3)
        source_audio, source_rate = _synthesize(runtime, SYNTHETIC_PHRASE)
        audio = np.interp(
            np.arange(int(len(source_audio) * config.audio.sample_rate / source_rate)) * source_rate / config.audio.sample_rate,
            np.arange(len(source_audio)), source_audio,
        ).astype(np.float32)
        fixtures = None
        if fixtures_path is not None:
            from scripts.latency_baseline import load_shared_fixtures
            fixtures = load_shared_fixtures(fixtures_path)
            report["fixture_sha256"] = hashlib.sha256(fixtures_path.read_bytes()).hexdigest()
            report["fixture_count"] = len(fixtures)
        probabilities = [runtime.vad.probability(audio[i:i + config.audio.chunk_samples])
                         for i in range(0, len(audio) - config.audio.chunk_samples + 1, config.audio.chunk_samples)]
        runtime.vad.reset()
        report["vad_detects_synthetic_speech"] = bool(probabilities and max(probabilities) > 0.5)
        if not report["vad_detects_synthetic_speech"]:
            raise ValueError("VAD did not detect synthetic speech")

        if microphone_seconds:
            captured, square_sum, maximum = 0, 0.0, 0.0
            with MicrophoneStream(config.audio.sample_rate, config.audio.chunk_samples,
                                  device=selected_input['index'], stall_s=config.audio.device_stall_s,
                                  max_queue_chunks=config.audio.queue_max_chunks) as microphone:
                deadline = time.monotonic() + microphone_seconds
                while time.monotonic() < deadline:
                    chunk = microphone.get(timeout=0.1)
                    if chunk is not None:
                        captured += len(chunk)
                        square_sum += float(np.dot(chunk, chunk))
                        maximum = max(maximum, float(np.max(np.abs(chunk))))
                report["microphone"] = {**microphone.statistics(), "captured_samples": captured,
                                        "rms": math.sqrt(square_sum / captured) if captured else 0,
                                        "peak": maximum, "seconds": microphone_seconds,
                                        "audio_saved": False, "transcribed": False}
            if captured < config.audio.sample_rate * microphone_seconds * 0.5:
                raise ValueError("Insufficient microphone callbacks")
        if playback:
            # Fixed synthetic phrase, conservative gain; no system volume changes.
            with sd.OutputStream(samplerate=source_rate, channels=1, dtype="float32",
                                 device=selected_output['index'] if selected_output else None) as stream:
                underflow = stream.write(source_audio * 0.25)
            report["playback"] = {"seconds": round(len(source_audio) / source_rate, 3),
                                  "underflow": bool(underflow), "audibility": "requires_human_confirmation"}

        report["ready"] = True
        report["soak_started_at"] = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        expected_pid = runtime.worker.pid
        if resource_metrics:
            report["model_resources_after_warmup"] = resident_sample(runtime, started, expected_pid)
            report["model_resources"] = []
        report["soak_interval_seconds"] = soak_interval
        write_report(path, report)
        while True:
            turn = time.monotonic()
            runtime._turn_deadline = turn + config.runtime.turn_timeout_s
            case_id = len(timings) % len(fixtures) if fixtures else None
            turn_audio = fixtures[case_id] if fixtures else audio
            transcript = runtime._infer("transcribe", turn_audio)
            if not transcript.strip():
                raise ValueError("STT produced no transcript for synthetic speech")
            wake = runtime.wake_detector.match(transcript)
            if fixtures is None and (wake is None or not wake.prompt_text):
                raise ValueError("Synthetic wake command was not recognized")
            endpoint = runtime._infer("endpoint", turn_audio) if config.audio.smart_turn_enabled else None
            if endpoint is not None and (not math.isfinite(endpoint) or not 0 <= endpoint <= 1):
                raise ValueError("Invalid endpoint probability")
            response = runtime._respond(transcript if fixtures else wake.prompt_text)
            if not response.strip():
                raise ValueError("Chat produced an empty answer")
            answer_audio, answer_rate = _synthesize(runtime, response)
            runtime._turn_deadline = None
            if not np.isfinite(answer_audio).all():
                raise ValueError("TTS produced non-finite samples")
            timings.append(time.monotonic() - turn)
            if resource_metrics:
                report["model_resources"].append(dict(resident_sample(runtime, started, expected_pid),
                    turn=len(timings), case_id=case_id))
            report.update({"completed_turns": len(timings), "soak_elapsed_seconds": round(time.monotonic() - started, 1),
                           "endpoint_probability": endpoint, "worker_restarts": runtime.worker.restarts,
                           "synthetic_wake_recognized": True if fixtures is None else None,
                           "arithmetic_answer_contains_four": ("four" in response.lower() or "4" in response)
                           if fixtures is None else None,
                           "turn_latency_seconds": {"p50": round(float(np.percentile(timings, 50)), 3),
                                                    "p95": round(float(np.percentile(timings, 95)), 3)},
                           "last_answer_audio_seconds": round(len(answer_audio) / answer_rate, 3)})
            runtime.turn_stage = "idle"
            write_report(path, report)
            if time.monotonic() - started >= minutes * 60:
                break
            time.sleep(min(soak_interval, max(0, minutes * 60 - (time.monotonic() - started))))
        report["status"] = "passed"
    finally:
        runtime.memory_jobs.close()
        runtime.worker.close()
        runtime.chime_player.stop()
        runtime.status_publisher.stop(runtime)


def _run_child(config, directory, mode, minutes, microphone_seconds, playback,
               resource_metrics=False, fixtures_path=None, soak_interval=5):
    # Own a process group so even a native device/model hang can be stopped
    # without signalling another runtime or any of the user's processes.
    os.setsid()
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path = directory / "report.json"
    report = {"status": "running", "mode": mode, "requested_minutes": minutes,
              "tts_backend": config.audio.tts_backend,
              "started_at": datetime.now(timezone.utc).isoformat(), "ready": False,
              "private_context_loaded": False, "recordings_saved": False}
    write_report(path, report)
    try:
        if mode == "components":
            _components(config, report, path, minutes, microphone_seconds, playback,
                        resource_metrics, fixtures_path, soak_interval)
        else:
            from arietta_voice.runtime import AriettaVoiceRuntime
            AriettaVoiceRuntime(config).run()
            report["status"] = "stopped"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error_type"] = type(exc).__name__
        # Full traceback stays in the ignored local runner log. No transcript
        # or response is added to the summary report.
        import traceback
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report(path, report)


def stop_process(process):
    if process.pid is None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, sig)
            elif process.is_alive():
                os.kill(process.pid, sig)
        except ProcessLookupError:
            break
        process.join(timeout=3)
        if not process.is_alive():
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("components", "live"), default="components")
    parser.add_argument("--minutes", type=float, default=0)
    parser.add_argument("--microphone-seconds", type=float, default=0)
    parser.add_argument("--playback", action="store_true")
    parser.add_argument("--tts-backend", choices=("onnx", "mlx"))
    parser.add_argument("--mlx-kokoro-model", type=Path)
    parser.add_argument("--resource-metrics", action="store_true", help="Verify continuous four-model residency and sample worker allocations (components only)")
    parser.add_argument("--host-metrics", action="store_true", help="Read-only macOS host telemetry")
    parser.add_argument("--fixtures", type=Path, help="Cycle existing synthetic NPZ fixtures (components only)")
    parser.add_argument("--soak-interval", type=float, default=5, help="Pause between component turns, 0..30 seconds")
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or not 0 <= args.minutes <= 60:
        parser.error("minutes must be between 0 and 60")
    if not math.isfinite(args.microphone_seconds) or not 0 <= args.microphone_seconds <= 30:
        parser.error("microphone-seconds must be between 0 and 30")
    if args.mode == "live" and args.minutes <= 0:
        parser.error("live mode requires a positive minutes limit")
    if not math.isfinite(args.soak_interval) or not 0 <= args.soak_interval <= 30:
        parser.error("soak-interval must be between 0 and 30")
    if args.mode == "live" and (args.resource_metrics or args.fixtures):
        parser.error("resource-metrics and fixtures require components mode")
    root = Path(__file__).resolve().parents[1]
    os.umask(0o077)
    try:
        with validation_lock(root / "tmp" / "hardware-validation"):
            return run_profile(args, root)
    except ValueError as exc:
        parser.error(str(exc))


def run_profile(args, root):
    source = load_config(args.config)
    from arietta_voice.runtime_status import read_runtime_status
    if read_runtime_status(source)["is_running"]:
        raise ValueError("A voice runtime is active; stop it before hardware validation")
    protected = protected_fingerprints(root)
    directory = root / "tmp" / "hardware-validation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory.mkdir(parents=True, mode=0o700)
    config = isolated_config(source, directory)
    apply_trial_overrides(config, args.tts_backend, args.mlx_kokoro_model)
    if args.resource_metrics and not (config.audio.tts_enabled and config.audio.smart_turn_enabled):
        raise ValueError("Resource soak requires TTS and Smart Turn enabled")
    metadata = collect_environment(config, root)
    if config.audio.tts_backend == "mlx":
        model = Path(config.audio.mlx_kokoro_model)
        metadata["mlx_asset_sha256"] = {}
        for asset in (model/"kokoro-v1_0.safetensors", model/"voices"/f"{config.audio.voice}.safetensors"):
            with asset.open("rb") as handle:
                metadata["mlx_asset_sha256"][asset.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata["packages"].update({name: version(name) for name in ("mlx-audio", "misaki")})
    write_report(directory / "environment.json", metadata)
    print(f"Validation directory: {directory}", flush=True)
    process = multiprocessing.get_context("spawn").Process(
        target=_run_child, args=(config, directory, args.mode, args.minutes, args.microphone_seconds, args.playback,
                                 args.resource_metrics, args.fixtures, args.soak_interval))
    samples = []
    monitor, host = None, []
    if args.host_metrics:
        from scripts.host_state import HostState
        monitor = HostState()
    started, ready_at, last_print = time.monotonic(), None, 0
    duration_completed = False
    parent_error = None
    process.start()
    try:
        while process.is_alive():
            elapsed = time.monotonic() - started
            report_path = directory / "report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            status = read_runtime_status(config)
            ready = ready_at is not None or report.get("ready", False) or (
                args.mode == "live" and status["state"] in {"waiting_for_wake", "listening"})
            if ready and ready_at is None:
                ready_at = time.monotonic()
                print("READY: isolated voice runtime is listening." if args.mode == "live"
                      else "READY: model soak started.", flush=True)
            sample = {"elapsed_seconds": round(elapsed, 1), "ready": ready,
                      "heartbeat_age_seconds": status.get("heartbeat_age_seconds"),
                      "audio": status.get("reliability", {}).get("audio", {}), **rss_sample(process.pid)}
            samples.append(sample)
            if monitor is not None:
                host.append(dict(monitor.sample(), ready=ready))
                write_report(directory / "host.json", host)
            write_report(directory / "resources.json", {"summary": summarize_resources(samples), "samples": samples})
            if elapsed - last_print >= 30:
                print(json.dumps({"elapsed_s": round(elapsed), "ready": ready,
                                  "session_turns": status.get("session_turn_count", 0),
                                  "component_turns": report.get("completed_turns", 0), "rss_mib": sample["rss_mib"]}), flush=True)
                last_print = elapsed
            if args.mode == "live" and ready_at is not None and time.monotonic() - ready_at >= args.minutes * 60:
                duration_completed = True
                break
            if elapsed > config.runtime.startup_timeout_s + args.minutes * 60 + 180:
                raise TimeoutError("Hardware validation exceeded the outer deadline")
            process.join(timeout=5)
    except (Exception, KeyboardInterrupt) as exc:
        parent_error = type(exc).__name__
    finally:
        stop_process(process)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "failed"}
    report["resources"] = summarize_resources(samples)
    report["events"] = summarize_events(config.logging.directory)
    report["heartbeat_max_age_seconds"] = max(
        (sample["heartbeat_age_seconds"] for sample in samples if sample["ready"]
         and sample["heartbeat_age_seconds"] is not None), default=None)
    if args.mode == "live":
        report["duration_completed"] = duration_completed
        report["live_elapsed_seconds"] = round(time.monotonic() - ready_at, 1) if ready_at else 0
        status = read_runtime_status(config)
        try:
            raw_status = json.loads((config.logging.directory / "runtime_status.json").read_text())
            report["raw_status_has_content"] = bool(raw_status.get("last_user_transcript") or raw_status.get("last_assistant_text"))
        except (OSError, ValueError):
            report["raw_status_has_content"] = None
        report["reliability"] = status.get("reliability", {})
        report["human_checks"] = "pending"
        report["status"] = "completed" if duration_completed and report["status"] == "stopped" else "incomplete"
        report["automated_gates"] = live_gates(report, status, samples)
        if report["status"] == "completed" and not all(report["automated_gates"].values()):
            report["status"] = "failed"
    report["protected_files_unchanged"] = protected == protected_fingerprints(root)
    if args.mode == "components":
        report["automated_gates"] = {
            "duration_completed": report.get("soak_elapsed_seconds", -1) >= args.minutes * 60,
            "turns_completed": report.get("completed_turns", 0) > 0,
            "no_worker_restarts": report.get("worker_restarts") == 0,
            "no_content_in_events": not report["events"]["content_fields_present"],
            "valid_event_log": report["events"]["invalid_records"] == 0,
            "resource_samples_complete": not args.resource_metrics or
                len(report.get("model_resources", [])) == report.get("completed_turns", -1),
        }
        if not all(report["automated_gates"].values()):
            report["status"] = "failed"
    if not report["protected_files_unchanged"]:
        report["status"] = "failed"
    if parent_error or process.exitcode not in (None, 0):
        report["status"] = "interrupted" if parent_error == "KeyboardInterrupt" else "failed"
        report["error_type"] = parent_error or f"ChildExit{process.exitcode}"
    write_report(directory / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] in {"passed", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
