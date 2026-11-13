"""Console application service: state, runtime control, knowledge, tools and chat, without HTTP."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arietta_voice import __version__
from arietta_voice.config import AppConfig, config_to_dict, load_config
from arietta_voice.conversation import ChatResult, ConversationService
from arietta_voice.diagnostics import run_startup_diagnostics
from arietta_voice.knowledge import KnowledgeRetriever, build_knowledge_index
from arietta_voice.llm import ChatModel, build_chat_model
from arietta_voice.logging_utils import EventLogger
from arietta_voice.prompts import PromptManager
from arietta_voice.runtime_status import read_runtime_status
from arietta_voice.runtime_supervisor import RuntimeSupervisor
from arietta_voice.tool_admin import ModelToolAdmin, ToolAdmin
from arietta_voice.turns import CancellationToken
from arietta_voice.web.schemas import ChatRequest, ModelToolCreateRequest, ToolCreateRequest
from arietta_voice.worker import (
    InferenceWorker,
    WorkerBusy,
    WorkerCancelled,
)

# Retain the existing API field names while reporting the application version.
API_VERSION = __version__
AUTH_REALM = "Arietta Admin"


class AriettaApiService:
    def __init__(self, config_path: Path | None = None, *, config: AppConfig | None = None) -> None:
        self.config_path = config_path.resolve() if config_path is not None else None
        self.config = config or load_config(config_path)
        self.started_at = _now_iso()
        self.started_monotonic = time.monotonic()
        self.event_logger = EventLogger.from_config(self.config.logging)
        self.prompt_manager = PromptManager(self.config)
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.runtime_supervisor = RuntimeSupervisor(self.config, config_path=self.config_path)
        self.tool_admin = ToolAdmin(
            self.config,
            config_path=self.config_path,
            knowledge_retriever=self.knowledge,
        )
        self.model_tool_admin = ModelToolAdmin(
            self.config,
            config_path=self.config_path,
            knowledge_retriever=self.knowledge,
        )
        self._llm: ChatModel | None = None
        self.worker = InferenceWorker(self.config)
        self._worker_model_pid: int | None = None
        self._chat_lock = threading.Lock()
        self._active_chat = None
        self._retired_chat_ids = deque(maxlen=128)
        self._conversation = ConversationService(
            self.config,
            prompt_manager=self.prompt_manager,
            knowledge=self.knowledge,
            generate=lambda *a, **kw: self._llm_generate(*a, **kw),
        )
        self.chat_turns = 0
        self.last_chat_at: str | None = None
        self.last_error: str | None = None

    def health(self) -> dict[str, Any]:
        try:
            diagnostics = run_startup_diagnostics(self.config, include_hardware=False)
            log_parent = self.config.logging.events_file.parent
            return {
                "status": diagnostics["overall_status"],
                "api_version": API_VERSION,
                "assistant": self.config.assistant.name,
                "generated_at": _now_iso(),
                "uptime_seconds": self.uptime_seconds(),
                "project_root": str(self.config.project_root),
                "events_file": str(self.config.logging.events_file),
                "events_file_exists": self.config.logging.events_file.exists(),
                "logs_writable": os.access(log_parent, os.W_OK),
                "runtime": read_runtime_status(self.config),
                "diagnostics": diagnostics,
            }
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "status": "error",
                "api_version": API_VERSION,
                "generated_at": _now_iso(),
                "error": self.last_error,
            }

    def state(self) -> dict[str, Any]:
        tools = self.inspect_tools()
        event_count = self._event_file_line_count()
        runtime_status = self.runtime_status()
        return {
            "api": {
                "version": API_VERSION,
                "started_at": self.started_at,
                "uptime_seconds": self.uptime_seconds(),
                "chat_turns": self.chat_turns,
                "last_chat_at": self.last_chat_at,
                "last_error": self.last_error,
                "inference": {
                    "operation": self.worker.operation,
                    "worker_pid": self.worker.pid,
                    "worker_restarts": self.worker.restarts,
                    "turn_timeout_s": self.config.runtime.turn_timeout_s,
                },
            },
            "runtime": runtime_status,
            "assistant": {
                "name": self.config.assistant.name,
                "wake_words": list(self.config.wake_word.phrases),
                "wake_enabled": self.config.wake_word.enabled,
            },
            "models": {
                "chat_provider": self.config.models.chat_provider,
                "chat_model": self.config.models.chat_model,
                "llm_loaded": self.worker.pid is not None
                and self.worker.pid == self._worker_model_pid,
                "stt_language": self.config.models.stt_language,
            },
            "knowledge": {
                "enabled": self.config.knowledge.enabled,
                "backend": self.config.knowledge.backend,
                "directory": str(self.config.knowledge.directory),
                "index_directory": str(self.config.knowledge.index_directory),
                "loaded": self.knowledge.is_loaded(),
            },
            "tools": {
                "enabled": list(self.config.tools.enabled),
                "loaded_count": None,
                "inventory_mode": "static",
                "error_count": len(tools["assistant_tools"]["errors"])
                + len(tools["model_tools"]["errors"]),
            },
            "logs": {
                "events_file": str(self.config.logging.events_file),
                "event_count": event_count,
            },
            "host": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }

    def config_payload(self) -> dict[str, Any]:
        return config_to_dict(self.config)

    def diagnostics(self, *, include_hardware: bool) -> dict[str, Any]:
        return run_startup_diagnostics(self.config, include_hardware=include_hardware)

    def runtime_status(self) -> dict[str, Any]:
        observed = read_runtime_status(self.config)
        supervisor_status = self.runtime_supervisor.status()
        if supervisor_status["managed_process_running"] and not observed["is_running"]:
            observed = {
                **observed,
                "state": "starting",
                "lifecycle": "starting",
                "note": "The API-managed voice runtime has started; waiting for its first heartbeat.",
            }
        return {
            **observed,
            **supervisor_status,
            "managed_by_api": bool(supervisor_status["managed_process_running"]),
        }

    def start_runtime(self) -> dict[str, Any]:
        payload = self.runtime_supervisor.start()
        self.event_logger.log(
            "api_runtime_start",
            changed=payload.get("changed"),
            message=payload.get("message"),
            process_id=payload.get("managed_process_id"),
        )
        return payload

    def stop_runtime(self) -> dict[str, Any]:
        payload = self.runtime_supervisor.stop()
        self.event_logger.log(
            "api_runtime_stop",
            changed=payload.get("changed"),
            message=payload.get("message"),
            process_id=payload.get("managed_process_id"),
            exit_code=payload.get("managed_exit_code"),
        )
        return payload

    def restart_runtime(self) -> dict[str, Any]:
        payload = self.runtime_supervisor.restart()
        self.event_logger.log(
            "api_runtime_restart",
            changed=payload.get("changed"),
            message=payload.get("message"),
            process_id=payload.get("managed_process_id"),
            exit_code=payload.get("managed_exit_code"),
        )
        return payload

    def read_events(
        self,
        *,
        limit: int,
        offset: int = 0,
        event: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        event_filter = None if event is None else event.strip()
        query_filter = "" if query is None else query.strip().lower()
        path = self.config.logging.events_file
        if not path.exists():
            return {
                "events_file": str(path),
                "items": [],
                "count": 0,
                "total": 0,
                "offset": offset,
                "limit": limit,
                "has_next": False,
                "has_previous": False,
                "event": event_filter,
                "query": query_filter,
            }

        matched: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"event": "unparsed_log_line", "line_number": line_number, "raw": raw}
                if event_filter and payload.get("event") != event_filter:
                    continue
                if query_filter and query_filter not in json.dumps(payload, sort_keys=True).lower():
                    continue
                matched.append(payload)
        ordered = list(reversed(matched))
        page = ordered[offset : offset + limit]
        total = len(ordered)
        return {
            "events_file": str(path),
            "items": page,
            "count": len(page),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_next": offset + limit < total,
            "has_previous": offset > 0,
            "event": event_filter,
            "query": query_filter,
        }

    def search_knowledge(self, query: str) -> dict[str, Any]:
        query = query.strip()
        result = self.knowledge.search(query)
        matches = (
            [] if result is None else [_knowledge_match_payload(match) for match in result.matches]
        )
        return {
            "query": query,
            "enabled": self.config.knowledge.enabled,
            "backend": self.config.knowledge.backend if result is None else result.backend,
            "directory": str(self.config.knowledge.directory),
            "matches": matches,
        }

    def build_index(self) -> dict[str, Any]:
        payload = build_knowledge_index(self.config.knowledge)
        self.worker.invalidate()
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.tool_admin.knowledge_retriever = self.knowledge
        self.model_tool_admin.knowledge_retriever = self.knowledge
        self.event_logger.log("api_knowledge_index", **payload)
        return payload

    def list_knowledge_files(self) -> dict[str, Any]:
        directory = self.config.knowledge.directory
        directory.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(directory)
            if not _is_supported_knowledge_path(relative_path):
                continue
            stat = path.stat()
            files.append(
                {
                    "path": relative_path.as_posix(),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "updated_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    )
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
                }
            )
        return {"directory": str(directory), "files": files}

    def read_knowledge_file(self, file_path: str) -> dict[str, Any]:
        path = _resolve_knowledge_file(self.config, file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(file_path)
        return {
            "path": path.relative_to(self.config.knowledge.directory).as_posix(),
            "content": path.read_text(encoding="utf-8"),
        }

    def write_knowledge_file(self, file_path: str, content: str) -> dict[str, Any]:
        path = _resolve_knowledge_file(self.config, file_path)
        created = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.worker.invalidate()
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.tool_admin.knowledge_retriever = self.knowledge
        self.model_tool_admin.knowledge_retriever = self.knowledge
        payload = {
            "path": path.relative_to(self.config.knowledge.directory).as_posix(),
            "created": created,
            "size_bytes": path.stat().st_size,
        }
        self.event_logger.log("api_knowledge_file_saved", **payload)
        return payload

    def delete_knowledge_file(self, file_path: str) -> dict[str, Any]:
        path = _resolve_knowledge_file(self.config, file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(file_path)
        relative_path = path.relative_to(self.config.knowledge.directory).as_posix()
        path.unlink()
        self.worker.invalidate()
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.tool_admin.knowledge_retriever = self.knowledge
        self.model_tool_admin.knowledge_retriever = self.knowledge
        payload = {"path": relative_path, "deleted": True}
        self.event_logger.log("api_knowledge_file_deleted", **payload)
        return payload

    def inspect_tools(self) -> dict[str, Any]:
        assistant_tools = self.tool_admin.inventory()
        return {
            "assistant_tools": assistant_tools,
            "model_tools": self.model_tool_admin.inventory(),
            "source_editing_enabled": self.config.admin.allow_source_editing,
        }

    def probe_tools(self, text: str) -> dict[str, Any]:
        text = text.strip()
        for tool_name in self.config.tools.enabled:
            probe = self.tool_admin.probe_tool(tool_name, text)
            if probe["matched"]:
                return probe
        return {
            "matched": False,
            "tool": None,
            "content": None,
            "bypass_model": None,
            "context": None,
        }

    def set_enabled_tools(self, enabled: list[str]) -> dict[str, Any]:
        payload = self.tool_admin.set_enabled(enabled)
        self.event_logger.log("api_tools_enabled_updated", enabled=payload["enabled"])
        return payload

    def probe_tool(self, tool_name: str, text: str) -> dict[str, Any]:
        return self.tool_admin.probe_tool(tool_name, text)

    def validate_tool(self, tool_name: str) -> dict[str, Any]:
        return self.tool_admin.validate_tool(tool_name)

    def create_tool(self, payload: ToolCreateRequest) -> dict[str, Any]:
        result = self.tool_admin.create_tool(
            module_name=payload.module_name,
            trigger_phrase=payload.trigger_phrase,
            response=payload.response,
            enable=payload.enable,
        )
        self.event_logger.log(
            "api_tool_created",
            module_name=result["module_name"],
            file=result["file"],
            enabled=result["enabled"],
        )
        return result

    def reload_tools(self) -> dict[str, Any]:
        self.worker.invalidate()
        payload = self.tool_admin.reload()
        self.event_logger.log("api_tools_reloaded", enabled=payload["enabled"])
        return payload

    def read_tool_source(self, tool_name: str) -> dict[str, Any]:
        return self.tool_admin.read_source(tool_name)

    def write_tool_source(self, tool_name: str, content: str) -> dict[str, Any]:
        payload = self.tool_admin.write_source(tool_name, content)
        self.worker.invalidate()
        self.event_logger.log(
            "api_tool_source_saved",
            configured_name=payload["configured_name"],
            file=payload["file"],
            valid=payload["tool"].get("valid"),
        )
        return payload

    def set_enabled_model_tools(self, enabled: list[str]) -> dict[str, Any]:
        payload = self.model_tool_admin.set_enabled(enabled)
        self.event_logger.log("api_model_tools_enabled_updated", enabled=payload["enabled"])
        return payload

    def invoke_model_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.model_tool_admin.invoke_tool(tool_name, arguments)

    def validate_model_tool(self, tool_name: str) -> dict[str, Any]:
        return self.model_tool_admin.validate_tool(tool_name)

    def create_model_tool(self, payload: ModelToolCreateRequest) -> dict[str, Any]:
        result = self.model_tool_admin.create_tool(
            module_name=payload.module_name,
            description=payload.description,
            arguments_example=payload.arguments_example,
            followup_context=payload.followup_context,
            enable=payload.enable,
        )
        self.event_logger.log(
            "api_model_tool_created",
            module_name=result["module_name"],
            file=result["file"],
            enabled=result["enabled"],
        )
        return result

    def reload_model_tools(self) -> dict[str, Any]:
        self.worker.invalidate()
        payload = self.model_tool_admin.reload()
        self.event_logger.log("api_model_tools_reloaded", enabled=payload["enabled"])
        return payload

    def read_model_tool_source(self, tool_name: str) -> dict[str, Any]:
        return self.model_tool_admin.read_source(tool_name)

    def write_model_tool_source(self, tool_name: str, content: str) -> dict[str, Any]:
        payload = self.model_tool_admin.write_source(tool_name, content)
        self.worker.invalidate()
        self.event_logger.log(
            "api_model_tool_source_saved",
            configured_name=payload["configured_name"],
            file=payload["file"],
            valid=payload["tool"].get("valid"),
        )
        return payload

    def model_tool_instructions(self) -> dict[str, Any]:
        return {
            "enabled": list(self.config.model_tools.enabled),
            "instruction_text": self.model_tool_admin.instruction_text(),
        }

    def chat(self, request: ChatRequest) -> ChatResult:
        token = CancellationToken()
        request_id = request.request_id or uuid.uuid4().hex
        with self._chat_lock:
            if request_id in self._retired_chat_ids:
                raise WorkerCancelled("Request ID already cancelled or completed; use a fresh ID")
            if self._active_chat is not None:
                raise WorkerBusy("Inference is busy")
            self._active_chat = (request_id, token)
        try:
            payload = request.model_dump(exclude={"request_id"})
            payload.update(
                enabled_tools=self.config.tools.enabled,
                enabled_model_tools=self.config.model_tools.enabled,
            )
            response = self.worker.call(
                "chat", payload, timeout_s=self.config.runtime.turn_timeout_s, cancellation=token
            )
            with self._chat_lock:
                token.check()
                self._worker_model_pid = (
                    self.worker.pid if response.pop("model_loaded", False) else None
                )
                result = ChatResult(**response)
                self.last_error = None
                self._record_chat(
                    request.message,
                    result.response,
                    source=result.source,
                    used_tool=result.used_tool,
                    model_tool=result.model_tool,
                )
                self._retired_chat_ids.append(request_id)
                self._active_chat = None
                return result
        finally:
            with self._chat_lock:
                if self._active_chat is not None and self._active_chat[1] is token:
                    self._retired_chat_ids.append(request_id)
                    self._active_chat = None

    def cancel_chat(self, request_id: str) -> bool:
        with self._chat_lock:
            if self._active_chat is not None and self._active_chat[0] == request_id:
                self._active_chat[1].cancel()
                return True
            if request_id in self._retired_chat_ids:
                return False
            # Remember early cancellation if the HTTP stop request overtakes chat.
            self._retired_chat_ids.append(request_id)
            return True

    def _chat_in_process(self, request: ChatRequest, *, record: bool = True) -> ChatResult:
        # In-process test/embedding entry point; production calls run in the worker.
        result = self._conversation.chat(**request.model_dump(exclude={"request_id"}))
        if record:
            self._record_chat(
                request.message,
                result.response,
                source=result.source,
                used_tool=result.used_tool,
                model_tool=result.model_tool,
            )
        return result

    def uptime_seconds(self) -> int:
        return int(time.monotonic() - self.started_monotonic)

    def _llm_generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        if self._llm is None:
            self._llm = build_chat_model(self.config.models)
        return self._llm.generate(
            messages,
            max_tokens=max_tokens or self.config.models.max_tokens,
            temperature=temperature if temperature is not None else self.config.models.temperature,
        )

    def _record_chat(
        self,
        message: str,
        response: str,
        *,
        source: str,
        used_tool: str | None,
        model_tool: str | None = None,
    ) -> None:
        self.chat_turns += 1
        self.last_chat_at = _now_iso()
        self.event_logger.log(
            "api_chat",
            source=source,
            used_tool=used_tool,
            model_tool=model_tool,
            message_chars=len(message),
            response_chars=len(response),
        )

    def _event_file_line_count(self) -> int:
        path = self.config.logging.events_file
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _knowledge_match_payload(match: Any) -> dict[str, Any]:
    return {
        "source": str(match.source_path),
        "title": match.title,
        "section": match.section_title,
        "score": match.score,
        "snippet": match.snippet,
    }


def _is_supported_knowledge_path(path: Path) -> bool:
    if any(part.startswith(".") for part in path.parts):
        return False
    return path.suffix.lower() in {".md", ".markdown", ".txt"}


def _resolve_knowledge_file(config: AppConfig, file_path: str) -> Path:
    normalized = Path(file_path)
    if not str(file_path).strip():
        raise ValueError("Knowledge file path is required.")
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("Knowledge file path must stay inside the knowledge directory.")
    if any(part.startswith(".") for part in normalized.parts):
        raise ValueError("Hidden knowledge files are not supported.")
    if normalized.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError("Knowledge files must use .md, .markdown, or .txt.")

    root = config.knowledge.directory.resolve()
    target = (root / normalized).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Knowledge file path must stay inside the knowledge directory.") from exc
    return target
