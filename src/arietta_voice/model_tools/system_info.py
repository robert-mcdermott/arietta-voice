from __future__ import annotations

from pathlib import Path
import os
import platform
import shutil
import subprocess

from arietta_voice.model_tools.base import ModelToolContext, ModelToolExecution


class SystemInfoTool:
    name = "get_system_info"
    description = (
        "you need a short report about the current machine, such as model, RAM, free disk space, "
        "or load average"
    )
    arguments_example: dict[str, object] = {}

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def invoke(self, arguments: dict[str, object]) -> ModelToolExecution | None:
        del arguments
        report = build_system_report(self.project_root)
        return ModelToolExecution(
            followup_context=(
                "System information requested by the assistant.\n"
                f"Tool result: {report}\n"
                "Use this tool result directly when answering and do not invent extra hardware details."
            )
        )


def build_system_report(project_root: Path) -> str:
    model = _detect_model()
    ram = _detect_total_memory()
    free_disk = _detect_free_disk(project_root)
    load_avg = _detect_load_average()

    parts = [f"I'm running on {model}"]
    if ram:
        parts.append(f"{ram} RAM")
    if free_disk:
        parts.append(f"{free_disk} free disk space")
    if load_avg:
        parts.append(f"a 1-minute load average of {load_avg}")

    if len(parts) == 1:
        return f"{parts[0]}."
    if len(parts) == 2:
        return f"{parts[0]} with {parts[1]}."
    if len(parts) == 3:
        return f"{parts[0]} with {parts[1]} and {parts[2]}."
    return f"{parts[0]} with {parts[1]}, {parts[2]}, and {parts[3]}."


def _detect_model() -> str:
    model = _read_sysctl_value("hw.model")
    if model:
        return model
    system = platform.system() or "Unknown OS"
    machine = platform.machine() or "unknown architecture"
    return f"{system} ({machine})"


def _detect_total_memory() -> str | None:
    raw = _read_sysctl_value("hw.memsize")
    if raw and raw.isdigit():
        return _format_bytes(int(raw))

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
        if page_size > 0 and page_count > 0:
            return _format_bytes(page_size * page_count)
    return None


def _detect_free_disk(project_root: Path) -> str | None:
    try:
        usage = shutil.disk_usage(project_root)
    except OSError:
        return None
    return _format_bytes(usage.free)


def _detect_load_average() -> str | None:
    try:
        one_minute, _, _ = os.getloadavg()
    except OSError:
        return None
    return f"{one_minute:.2f}"


def _read_sysctl_value(key: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", key],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _format_bytes(value: int) -> str:
    gib = 1024**3
    tib = 1024**4
    if value >= tib:
        return f"{value / tib:.1f} TB"
    rounded_gb = round(value / gib)
    if rounded_gb >= 1:
        return f"{rounded_gb:.0f} GB"
    return f"{value / (1024**2):.0f} MB"


def create_tool(context: ModelToolContext) -> SystemInfoTool:
    return SystemInfoTool(context.config.project_root)
