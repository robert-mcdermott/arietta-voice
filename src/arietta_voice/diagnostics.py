from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from arietta_voice.config import AppConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _check(name: str, ok: bool, detail: str, **metadata: Any) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "status": "ok" if ok else "error",
        "detail": detail,
        "metadata": metadata,
    }


def list_audio_devices() -> list[dict[str, Any]]:
    from arietta_voice.devices import device_inventory
    return device_inventory()


def run_startup_diagnostics(config: AppConfig, *, include_hardware: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    soul_exists = config.prompts.soul_path.exists()
    checks.append(
        _check(
            "prompt_soul",
            soul_exists,
            f"SOUL prompt {'found' if soul_exists else 'missing'} at {config.prompts.soul_path}",
            path=str(config.prompts.soul_path),
        )
    )

    checks.append(
        _check(
            "wake_word_phrases",
            bool(config.wake_word.phrases) or not config.wake_word.enabled,
            "Wake word phrases configured" if config.wake_word.enabled else "Wake word disabled",
            enabled=config.wake_word.enabled,
            phrases=list(config.wake_word.phrases),
            backend=config.wake_word.backend,
        )
    )

    memory_parent = config.prompts.memory_path.parent
    memory_parent.mkdir(parents=True, exist_ok=True)
    memory_writable = os.access(memory_parent, os.W_OK)
    checks.append(
        _check(
            "memory_path",
            memory_writable,
            f"Memory path parent {'is' if memory_writable else 'is not'} writable",
            path=str(config.prompts.memory_path),
        )
    )

    config.logging.directory.mkdir(parents=True, exist_ok=True)
    log_writable = os.access(config.logging.directory, os.W_OK)
    checks.append(
        _check(
            "logging_directory",
            log_writable,
            f"Logging directory {'is' if log_writable else 'is not'} writable",
            path=str(config.logging.directory),
        )
    )

    config.knowledge.directory.mkdir(parents=True, exist_ok=True)
    knowledge_readable = config.knowledge.directory.exists() and os.access(config.knowledge.directory, os.R_OK)
    checks.append(
        _check(
            "knowledge_directory",
            knowledge_readable,
            f"Knowledge directory {'is' if knowledge_readable else 'is not'} readable",
            path=str(config.knowledge.directory),
            backend=config.knowledge.backend,
        )
    )

    devices: list[dict[str, Any]] = []
    if include_hardware:
        try:
            devices = list_audio_devices()
            input_count = sum(1 for device in devices if device["max_input_channels"] > 0)
            output_count = sum(1 for device in devices if device["max_output_channels"] > 0)
            checks.append(
                _check(
                    "audio_devices",
                    input_count > 0 and (output_count > 0 or not (config.audio.tts_enabled or config.audio.chime_enabled)),
                    f"Found {input_count} input and {output_count} output audio devices",
                    input_count=input_count,
                    output_count=output_count,
                )
            )
            from arietta_voice.devices import select_devices
            selected_input, selected_output = select_devices(config.audio)
            checks.append(_check('audio_selection', True, 'Selected audio devices support the required formats',
                                 input=selected_input['selector'],
                                 output=selected_output['selector'] if selected_output else None))
        except Exception as exc:
            checks.append(
                _check(
                    "audio_devices",
                    False,
                    f"Audio device enumeration or selection failed: {type(exc).__name__}",
                    error=type(exc).__name__,
                )
            )

    overall_status = "ok" if all(check["ok"] for check in checks) else "degraded"
    return {
        "generated_at": _now_iso(),
        "overall_status": overall_status,
        "checks": checks,
        "audio_devices": devices,
    }
