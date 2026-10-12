from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from arietta_voice import __version__
from arietta_voice.admin_auth import load_admin_credentials
from arietta_voice.config import AppConfig, config_to_dict, load_config
from arietta_voice.diagnostics import run_startup_diagnostics
from arietta_voice.knowledge import KnowledgeRetriever, build_knowledge_index
from arietta_voice.llm import ChatModel, build_chat_model
from arietta_voice.logging_utils import EventLogger
from arietta_voice.model_tooling import maybe_generate_with_model_tools
from arietta_voice.models import ConversationTurn
from arietta_voice.prompts import PromptManager, sanitize_for_speech
from arietta_voice.runtime_status import read_runtime_status
from arietta_voice.runtime_supervisor import RuntimeSupervisor
from arietta_voice.tool_admin import ModelToolAdmin, ToolAdmin
from arietta_voice.worker import InferenceWorker, WorkerBusy, WorkerFailure, WorkerTimeout


# Retain the existing API field names while reporting the application version.
API_VERSION = __version__
AUTH_REALM = "Arietta Admin"


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    history: list[ChatHistoryItem] = Field(default_factory=list, max_length=128)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    use_tools: bool = True


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class KnowledgeFileWriteRequest(BaseModel):
    content: str = Field(max_length=1_000_000)


class ToolProbeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ToolEnabledRequest(BaseModel):
    enabled: list[str] = Field(default_factory=list, max_length=200)


class ToolCreateRequest(BaseModel):
    module_name: str = Field(min_length=1, max_length=80)
    trigger_phrase: str = Field(default="example custom tool", min_length=1, max_length=240)
    response: str = Field(
        default="The custom tool is wired up. Replace this response with your real behavior.",
        min_length=1,
        max_length=4000,
    )
    enable: bool = True


class ToolSourceWriteRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1_000_000)


class ModelToolInvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelToolCreateRequest(BaseModel):
    module_name: str = Field(min_length=1, max_length=80)
    description: str = Field(
        default="you need a custom model-called tool result before answering",
        min_length=1,
        max_length=600,
    )
    arguments_example: dict[str, Any] = Field(default_factory=dict)
    followup_context: str = Field(
        default="Custom model tool result. Use this result when composing the final answer.",
        min_length=1,
        max_length=4000,
    )
    enable: bool = True


@dataclass(slots=True)
class ChatResult:
    response: str
    source: str
    used_tool: str | None
    model_tool: str | None


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
                "inference": {"operation": self.worker.operation, "worker_pid": self.worker.pid,
                              "worker_restarts": self.worker.restarts,
                              "turn_timeout_s": self.config.runtime.turn_timeout_s},
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
                "llm_loaded": self.worker.pid is not None and self.worker.pid == self._worker_model_pid,
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
        matches = [] if result is None else [_knowledge_match_payload(match) for match in result.matches]
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
                    ).isoformat(timespec="seconds").replace("+00:00", "Z"),
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
        return {"matched": False, "tool": None, "content": None, "bypass_model": None, "context": None}

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
        payload = request.model_dump()
        payload.update(enabled_tools=self.config.tools.enabled,
                       enabled_model_tools=self.config.model_tools.enabled)
        response = self.worker.call("chat", payload, timeout_s=self.config.runtime.turn_timeout_s)
        self._worker_model_pid = self.worker.pid if response.pop("model_loaded", False) else None
        result = ChatResult(**response)
        self.last_error = None
        self._record_chat(request.message, result.response, source=result.source,
                          used_tool=result.used_tool, model_tool=result.model_tool)
        return result

    def _chat_in_process(self, request: ChatRequest, *, record: bool = True) -> ChatResult:
        message = request.message.strip()
        history = [
            ConversationTurn(role=item.role, content=item.content.strip())
            for item in request.history
            if item.content.strip()
        ]
        used_tool: str | None = None
        model_tool: str | None = None

        tool_response = None
        if request.use_tools:
            probe = self.probe_tools(message)
            if probe["matched"]:
                used_tool = str(probe["tool"])
                if probe["bypass_model"]:
                    response = sanitize_for_speech(str(probe["content"]))
                    if record:
                        self._record_chat(message, response, source="tool", used_tool=used_tool)
                    return ChatResult(
                        response=response,
                        source="tool",
                        used_tool=used_tool,
                        model_tool=None,
                    )
                from arietta_voice.models import ToolResponse

                tool_response = ToolResponse(
                    name=str(probe["tool"]),
                    content=str(probe["content"] or ""),
                    bypass_model=False,
                    context=None if probe["context"] is None else str(probe["context"]),
                )

        messages = self.prompt_manager.build_messages(history, message, tool_response)

        def generate(model_messages: list[dict[str, str]]) -> str:
            return self._llm_generate(
                model_messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

        if request.use_tools:
            response, model_tool_use = maybe_generate_with_model_tools(
                messages,
                model_tool_registry=_AdHocModelToolRegistry(self.config, self.knowledge),
                llm_generate=generate,
            )
        else:
            # Do not even import/construct plugins when tools are disabled.
            response, model_tool_use = generate(messages), None
        if model_tool_use is not None:
            model_tool = model_tool_use.name
        clean_response = sanitize_for_speech(response)
        if record:
            self._record_chat(message, clean_response, source="model", used_tool=used_tool, model_tool=model_tool)
        return ChatResult(
            response=clean_response,
            source="model",
            used_tool=used_tool,
            model_tool=model_tool,
        )

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


class _AdHocModelToolRegistry:
    def __init__(self, config: AppConfig, knowledge: KnowledgeRetriever) -> None:
        from arietta_voice.model_tools import ModelToolRegistry

        self._registry = ModelToolRegistry(
            config,
            knowledge_retriever=knowledge if config.knowledge.enabled else None,
        )

    def instruction_text(self) -> str | None:
        return self._registry.instruction_text()

    def execute(self, name: str, arguments: dict[str, object]):
        return self._registry.execute(name, arguments)


def create_app(
    config_path: Path | None = None,
    *,
    service: AriettaApiService | None = None,
) -> FastAPI:
    service = service or AriettaApiService(config_path)
    base_path = service.config_path or service.config.project_root / "config" / "arietta_voice.toml"
    admin_credentials = load_admin_credentials(base_path)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            service.worker.close()

    app = FastAPI(
        title="Arietta Voice API",
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.arietta_service = service
    allowed_hosts = ["localhost", "127.0.0.1", "[::1]"]
    extra_hosts = os.environ.get("ARIETTA_ADMIN_ALLOWED_HOSTS", "")
    if "*" in extra_hosts:
        raise ValueError("ARIETTA_ADMIN_ALLOWED_HOSTS must list exact hostnames without wildcards")
    allowed_hosts.extend(host.strip() for host in extra_hosts.split(",") if host.strip())
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def protect_browser_requests(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if (origin is not None and origin != expected_origin) or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-origin admin actions are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
    security_scheme = HTTPBasic(auto_error=False)

    def require_admin(credentials: HTTPBasicCredentials | None = Depends(security_scheme)) -> str:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": f'Basic realm="{AUTH_REALM}"'},
            )
        if admin_credentials.verify(credentials.username, credentials.password):
            return credentials.username
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": f'Basic realm="{AUTH_REALM}"'},
        )

    def svc() -> AriettaApiService:
        return app.state.arietta_service

    def require_source_editing(_user: str = Depends(require_admin)) -> str:
        if not svc().config.admin.allow_source_editing:
            raise HTTPException(status_code=403, detail="Source editing is disabled. Set admin.allow_source_editing=true in private config and restart the API to enable it.")
        return _user

    @app.get("/health")
    def public_health() -> dict[str, Any]:
        # Liveness only: no runtime file, diagnostics, identity, or exception
        # details. Authenticated /api/health provides readiness diagnostics.
        return {"status": "ok", "api_version": API_VERSION}

    @app.get("/api/health")
    def api_health(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().health()

    @app.get("/api/state")
    def api_state(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().state()

    @app.get("/api/runtime")
    def api_runtime(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().runtime_status()

    @app.post("/api/runtime/start")
    def api_runtime_start(_user: str = Depends(require_admin)) -> dict[str, Any]:
        try:
            return svc().start_runtime()
        except Exception as exc:
            service_obj = svc()
            service_obj.last_error = f"{type(exc).__name__}: {exc}"
            service_obj.event_logger.log("api_error", route="/api/runtime/start", error=service_obj.last_error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=service_obj.last_error,
            ) from exc

    @app.post("/api/runtime/stop")
    def api_runtime_stop(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().stop_runtime()

    @app.post("/api/runtime/restart")
    def api_runtime_restart(_user: str = Depends(require_admin)) -> dict[str, Any]:
        try:
            return svc().restart_runtime()
        except Exception as exc:
            service_obj = svc()
            service_obj.last_error = f"{type(exc).__name__}: {exc}"
            service_obj.event_logger.log("api_error", route="/api/runtime/restart", error=service_obj.last_error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=service_obj.last_error,
            ) from exc

    @app.get("/api/config")
    def api_config(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().config_payload()

    @app.get("/api/diagnostics")
    def api_diagnostics(
        _user: str = Depends(require_admin),
        include_hardware: bool = Query(default=False),
    ) -> dict[str, Any]:
        return svc().diagnostics(include_hardware=include_hardware)

    @app.get("/api/logs")
    def api_logs(
        _user: str = Depends(require_admin),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        event: str | None = Query(default=None),
        q: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return svc().read_events(limit=limit, offset=offset, event=event, query=q)

    @app.get("/api/tools")
    def api_tools(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().inspect_tools()

    @app.post("/api/tools/probe")
    def api_probe_tools(
        payload: ToolProbeRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        return svc().probe_tools(payload.text)

    @app.post("/api/tools/enabled")
    def api_tools_enabled(
        payload: ToolEnabledRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().set_enabled_tools(payload.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/tools/create")
    def api_tool_create(
        payload: ToolCreateRequest,
        _user: str = Depends(require_source_editing),
    ) -> dict[str, Any]:
        try:
            return svc().create_tool(payload)
        except FileExistsError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/tools/reload")
    def api_tools_reload(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().reload_tools()

    @app.get("/api/tools/{tool_name}/source")
    def api_tool_source_read(
        tool_name: str,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().read_tool_source(tool_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.put("/api/tools/{tool_name}/source")
    def api_tool_source_write(
        tool_name: str,
        payload: ToolSourceWriteRequest,
        _user: str = Depends(require_source_editing),
    ) -> dict[str, Any]:
        try:
            return svc().write_tool_source(tool_name, payload.content)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/tools/{tool_name}/probe")
    def api_probe_tool(
        tool_name: str,
        payload: ToolProbeRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().probe_tool(tool_name, payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/tools/{tool_name}/validate")
    def api_validate_tool(
        tool_name: str,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        return svc().validate_tool(tool_name)

    @app.post("/api/model-tools/enabled")
    def api_model_tools_enabled(
        payload: ToolEnabledRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().set_enabled_model_tools(payload.enabled)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/model-tools/create")
    def api_model_tool_create(
        payload: ModelToolCreateRequest,
        _user: str = Depends(require_source_editing),
    ) -> dict[str, Any]:
        try:
            return svc().create_model_tool(payload)
        except FileExistsError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/model-tools/reload")
    def api_model_tools_reload(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().reload_model_tools()

    @app.get("/api/model-tools/{tool_name}/source")
    def api_model_tool_source_read(
        tool_name: str,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().read_model_tool_source(tool_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.put("/api/model-tools/{tool_name}/source")
    def api_model_tool_source_write(
        tool_name: str,
        payload: ToolSourceWriteRequest,
        _user: str = Depends(require_source_editing),
    ) -> dict[str, Any]:
        try:
            return svc().write_model_tool_source(tool_name, payload.content)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/model-tools/instructions")
    def api_model_tool_instructions(_user: str = Depends(require_admin)) -> dict[str, Any]:
        try:
            return svc().model_tool_instructions()
        except Exception as exc:
            service_obj = svc()
            service_obj.last_error = f"{type(exc).__name__}: {exc}"
            service_obj.event_logger.log(
                "api_error",
                route="/api/model-tools/instructions",
                error=service_obj.last_error,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=service_obj.last_error,
            ) from exc

    @app.post("/api/model-tools/{tool_name}/invoke")
    def api_invoke_model_tool(
        tool_name: str,
        payload: ModelToolInvokeRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().invoke_model_tool(tool_name, payload.arguments)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/model-tools/{tool_name}/validate")
    def api_validate_model_tool(
        tool_name: str,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        return svc().validate_model_tool(tool_name)

    @app.post("/api/knowledge/search")
    def api_knowledge_search(
        payload: KnowledgeSearchRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        return svc().search_knowledge(payload.query)

    @app.post("/api/knowledge/index")
    def api_knowledge_index(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().build_index()

    @app.get("/api/knowledge/files")
    def api_knowledge_files(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().list_knowledge_files()

    @app.get("/api/knowledge/files/{file_path:path}")
    def api_knowledge_file_read(
        file_path: str,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().read_knowledge_file(file_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge file not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.put("/api/knowledge/files/{file_path:path}")
    def api_knowledge_file_write(
        file_path: str,
        payload: KnowledgeFileWriteRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().write_knowledge_file(file_path, payload.content)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.delete("/api/knowledge/files/{file_path:path}")
    def api_knowledge_file_delete(
        file_path: str,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().delete_knowledge_file(file_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge file not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/chat")
    def api_chat(payload: ChatRequest, _user: str = Depends(require_admin)) -> dict[str, Any]:
        try:
            result = svc().chat(payload)
        except Exception as exc:
            service_obj = svc()
            service_obj.last_error = type(exc).__name__
            service_obj.event_logger.log("api_error", route="/api/chat", error_type=service_obj.last_error)
            code, detail = 502, "The model or tool failed. The next request will use a fresh worker."
            if isinstance(exc, WorkerBusy):
                code, detail = 503, "Inference is busy. Wait for the current request to finish."
            elif isinstance(exc, WorkerTimeout):
                code, detail = 504, "The turn timed out and was stopped. No automatic retry was made; check any external action before repeating it."
            raise HTTPException(
                status_code=code,
                detail=detail,
            ) from exc
        return {
            "response": result.response,
            "source": result.source,
            "used_tool": result.used_tool,
            "model_tool": result.model_tool,
        }

    @app.get("/api/openapi.json")
    def api_openapi(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return app.openapi()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin_console(_user: str = Depends(require_admin)) -> HTMLResponse:
        return HTMLResponse(ADMIN_HTML)

    return app


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


ADMIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Arietta Voice Admin</title>
  <link rel="icon" href="data:,">
  <script>
    document.documentElement.dataset.theme = localStorage.getItem("arietta-theme") || "light";
  </script>
  <style>
    :root {
      --bg: #f7f8f5;
      --panel: #ffffff;
      --panel-strong: #f0f4f2;
      --sidebar: #fbfcfa;
      --field: #ffffff;
      --active-bg: #e7f2ef;
      --active-border: #9bc5bd;
      --active-ink: #064d47;
      --code-bg: #20241f;
      --code-ink: #f3f7ef;
      --focus: rgba(8, 127, 117, 0.12);
      --ink: #222520;
      --muted: #667065;
      --line: #d9dfd7;
      --accent: #087f75;
      --accent-strong: #05665d;
      --button-info: #2f5f78;
      --button-info-strong: #244b5f;
      --warn: #a76300;
      --bad: #b42318;
      --good: #197344;
      --disabled-bg: #eef1ed;
      --disabled-ink: #8b9489;
      --shadow: 0 3px 12px rgba(31, 37, 28, 0.04);
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    :root[data-theme="dark"] {
      --bg: #161914;
      --panel: #20261f;
      --panel-strong: #2a3128;
      --sidebar: #1b2019;
      --field: #151a14;
      --active-bg: #123f3b;
      --active-border: #2f9d90;
      --active-ink: #d8fff8;
      --code-bg: #0f130e;
      --code-ink: #edf5ea;
      --focus: rgba(45, 170, 155, 0.22);
      --ink: #eef3ec;
      --muted: #a9b5a6;
      --line: #3b4538;
      --accent: #11a99c;
      --accent-strong: #35c1b4;
      --button-info: #4a8fb1;
      --button-info-strong: #61a6c8;
      --warn: #d58b21;
      --bad: #de4f45;
      --good: #34a56b;
      --disabled-bg: #30382d;
      --disabled-ink: #788475;
      --shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
      color-scheme: dark;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }
    .layout {
      display: grid;
      grid-template-columns: 248px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      background: var(--sidebar);
      border-right: 1px solid var(--line);
      padding: 24px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .brand {
      display: grid;
      gap: 4px;
      margin-bottom: 26px;
    }
    .brand h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
      font-weight: 750;
    }
    .brand span {
      color: var(--muted);
      font-size: 13px;
    }
    nav {
      display: grid;
      gap: 6px;
    }
    .nav-button {
      appearance: none;
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink);
      border-radius: 7px;
      padding: 10px 12px;
      text-align: left;
      font: inherit;
      font-size: 14px;
      cursor: pointer;
    }
    .nav-button:hover { background: var(--panel-strong); }
    .nav-button.active {
      border-color: var(--active-border);
      background: var(--active-bg);
      color: var(--active-ink);
      font-weight: 650;
    }
    .sidebar-footer {
      display: grid;
      gap: 10px;
      margin-top: auto;
    }
    .sidebar-link {
      display: block;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 12px;
      color: var(--muted);
      text-decoration: none;
      font-size: 14px;
    }
    .sidebar-link:hover {
      border-color: #aab5a7;
      background: var(--panel-strong);
      color: var(--ink);
    }
    .theme-toggle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 12px;
      color: var(--muted);
      font-size: 14px;
      cursor: pointer;
      user-select: none;
    }
    .theme-toggle:hover {
      border-color: #aab5a7;
      background: var(--panel-strong);
      color: var(--ink);
    }
    .theme-toggle input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .theme-slider {
      position: relative;
      width: 42px;
      height: 24px;
      border-radius: 999px;
      background: #c5cec2;
      flex: 0 0 auto;
      transition: background 160ms ease;
    }
    .theme-slider::after {
      content: "";
      position: absolute;
      top: 3px;
      left: 3px;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #ffffff;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
      transition: transform 160ms ease;
    }
    .theme-toggle input:checked + .theme-slider {
      background: var(--accent);
    }
    .theme-toggle input:checked + .theme-slider::after {
      transform: translateX(18px);
    }
    main {
      padding: 28px;
      min-width: 0;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }
    .topbar h2 {
      margin: 0;
      font-size: 24px;
      line-height: 1.15;
    }
    .status-row {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 0 3px rgba(167, 99, 0, 0.12);
    }
    .dot.ok {
      background: var(--good);
      box-shadow: 0 0 0 3px rgba(25, 115, 68, 0.12);
    }
    .dot.bad {
      background: var(--bad);
      box-shadow: 0 0 0 3px rgba(180, 35, 24, 0.12);
    }
    .button {
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 7px;
      min-height: 38px;
      padding: 8px 12px;
      font: inherit;
      font-size: 14px;
      cursor: pointer;
    }
    .button:hover { border-color: #aab5a7; }
    .button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      font-weight: 650;
    }
    .button.primary:hover { background: var(--accent-strong); }
    .button.success {
      background: var(--good);
      border-color: var(--good);
      color: #ffffff;
      font-weight: 650;
    }
    .button.success:hover { background: #115d35; }
    .button.danger {
      background: var(--bad);
      border-color: var(--bad);
      color: #ffffff;
      font-weight: 650;
    }
    .button.danger:hover { background: #8f1d14; }
    .button.warn {
      background: var(--warn);
      border-color: var(--warn);
      color: #ffffff;
      font-weight: 650;
    }
    .button.warn:hover { background: #835000; }
    .button.info {
      background: var(--button-info);
      border-color: var(--button-info);
      color: #ffffff;
      font-weight: 650;
    }
    .button.info:hover { background: var(--button-info-strong); }
    .button:disabled,
    .button:disabled:hover {
      background: var(--disabled-bg);
      border-color: var(--line);
      color: var(--disabled-ink);
      cursor: not-allowed;
      opacity: 0.8;
      font-weight: 500;
    }
    .view { display: none; }
    .view.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }
    .metric, .panel, .row-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    .metric {
      padding: 16px;
      min-height: 118px;
      display: grid;
      align-content: space-between;
      gap: 14px;
    }
    .metric .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0;
    }
    .metric .value {
      font-size: 21px;
      line-height: 1.1;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .metric .sub {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .panel {
      padding: 18px;
      margin-bottom: 18px;
    }
    .eyebrow { color: var(--accent); font-size: 11px; font-weight: 750; letter-spacing: 1.4px; text-transform: uppercase; margin-bottom: 7px; }
    .runtime-summary { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; color: var(--muted); font-size: 13px; }
    .runtime-badge { color: var(--active-ink); background: var(--active-bg); border: 1px solid var(--active-border); padding: 6px 12px; border-radius: 99px; font-weight: 650; }
    .runtime-badge.offline { color: var(--muted); background: var(--panel-strong); border-color: var(--line); }
    .reliability-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 16px 0; }
    .reliability-card { background: var(--panel-strong); padding: 16px; border-radius: 10px; min-width: 0; }
    .reliability-card h4 { margin: 0 0 10px; font-size: 12px; color: var(--muted); font-weight: 600; }
    .reliability-value { font-size: 22px; font-weight: 650; margin-bottom: 8px; overflow-wrap: anywhere; }
    .reliability-note { color: var(--muted); font-size: 12px; line-height: 1.7; }
    progress { width: 100%; height: 6px; accent-color: var(--accent); display: block; margin: 12px 0; }
    .notice { border: 1px solid var(--warn); border-left-width: 4px; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; line-height: 1.6; overflow-wrap: anywhere; }
    .notice.error-notice { border-color: var(--bad); }
    .notice button { float: right; margin-left: 12px; }
    details.runtime-details { border-top: 1px solid var(--line); margin-top: 20px; padding-top: 14px; }
    details.runtime-details summary { cursor: pointer; font-size: 13px; color: var(--muted); margin-bottom: 10px; }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
    @media (max-width: 1100px) { .reliability-grid { grid-template-columns: 1fr; } }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .panel h3 {
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }
    .form-row {
      display: flex;
      gap: 10px;
      align-items: stretch;
      margin-bottom: 14px;
    }
    .compact-row {
      align-items: center;
      flex-wrap: wrap;
    }
    .compact-row input {
      min-width: 180px;
      flex: 1 1 220px;
    }
    .compact-row .button {
      flex: 0 0 auto;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--field);
      color: var(--ink);
      font: inherit;
      padding: 10px 12px;
      outline: none;
    }
    input[type="checkbox"] {
      width: auto;
      accent-color: var(--accent);
    }
    textarea {
      min-height: 112px;
      resize: vertical;
      line-height: 1.45;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--focus);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 750;
      background: var(--panel-strong);
    }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: 0;
    }
    pre {
      margin: 0;
      padding: 14px;
      background: var(--code-bg);
      color: var(--code-ink);
      border-radius: 8px;
      overflow: auto;
      max-height: 620px;
      font-size: 12px;
      line-height: 1.5;
    }
    .split {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 420px);
      gap: 18px;
      align-items: start;
    }
    .knowledge-layout {
      grid-template-columns: minmax(300px, 420px) minmax(0, 1fr);
    }
    .tools-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 400px);
      gap: 18px;
      align-items: start;
    }
    .tool-toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .tool-list {
      display: grid;
      gap: 10px;
    }
    .tool-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-strong);
      padding: 12px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
    }
    .tool-item.enabled {
      border-color: var(--active-border);
      background: var(--active-bg);
    }
    .tool-item.invalid {
      border-color: rgba(180, 35, 24, 0.45);
    }
    .tool-title {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 5px;
    }
    .tool-title strong {
      font-size: 14px;
    }
    .tool-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .tool-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .tool-toggle {
      display: inline-flex;
      min-height: 38px;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
      padding: 8px 10px;
      font-size: 13px;
      color: var(--ink);
    }
    .tool-error {
      margin-top: 10px;
      border: 1px solid rgba(180, 35, 24, 0.35);
      border-radius: 7px;
      padding: 10px;
      color: var(--bad);
      background: rgba(180, 35, 24, 0.08);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .status-pill.ok {
      border-color: rgba(25, 115, 68, 0.35);
      color: var(--good);
    }
    .status-pill.bad {
      border-color: rgba(180, 35, 24, 0.35);
      color: var(--bad);
    }
    .status-pill.warn {
      border-color: rgba(167, 99, 0, 0.35);
      color: var(--warn);
    }
    .tool-create-form {
      display: grid;
      gap: 12px;
    }
    .tool-create-form label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    #tools-create-response,
    #model-tools-create-context,
    #model-tools-create-args,
    #model-tools-arguments {
      min-height: 140px;
    }
    #tools-result,
    #model-tools-result {
      margin-top: 8px;
      margin-bottom: 14px;
      max-height: 260px;
    }
    #tools-result-title,
    #model-tools-result-title {
      margin-top: 12px;
      min-height: 22px;
    }
    .source-editor {
      min-height: 520px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      tab-size: 4;
    }
    .source-test-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 420px);
      gap: 14px;
      align-items: start;
      margin-bottom: 14px;
    }
    .source-test-grid label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .source-test-grid textarea {
      min-height: 90px;
    }
    .chat-log {
      display: grid;
      gap: 10px;
      max-height: 420px;
      overflow: auto;
      padding-right: 4px;
    }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--panel-strong);
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .bubble strong {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
    }
    .result-list {
      display: grid;
      gap: 10px;
    }
    .knowledge-editor {
      min-height: 460px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }
    .file-list {
      display: grid;
      gap: 8px;
    }
    .file-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-strong);
    }
    .file-item strong {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 14px;
    }
    .file-item .meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .row-card {
      padding: 12px;
      box-shadow: none;
    }
    .row-card h4 {
      margin: 0 0 6px;
      font-size: 14px;
    }
    .row-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      color: var(--muted);
      background: var(--panel-strong);
      margin-right: 6px;
      margin-bottom: 4px;
    }
    .muted { color: var(--muted); }
    .error { color: var(--bad); }
    #poll-status { color: var(--muted); }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      aside {
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      nav { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .split { grid-template-columns: 1fr; }
      .tools-layout { grid-template-columns: 1fr; }
      .source-test-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
      main { padding: 18px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      nav { grid-template-columns: 1fr 1fr; }
      .grid { grid-template-columns: 1fr; }
      .form-row { flex-direction: column; }
      .runtime-actions { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; }
      .tool-item { grid-template-columns: 1fr; }
      .tool-actions,
      .tool-toolbar { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="brand">
        <h1>Arietta Voice</h1>
        <span>Admin Console</span>
      </div>
      <nav>
        <button class="nav-button active" data-view="dashboard">Dashboard</button>
        <button class="nav-button" data-view="chat">Chat</button>
        <button class="nav-button" data-view="knowledge">Knowledge</button>
        <button class="nav-button" data-view="tools">Tools</button>
        <button class="nav-button" data-view="logs">Logs</button>
        <button class="nav-button" data-view="config">Config</button>
      </nav>
      <div class="sidebar-footer">
        <label class="theme-toggle" for="theme-toggle">
          <span>Dark Mode</span>
          <input id="theme-toggle" type="checkbox" role="switch" aria-label="Toggle dark mode">
          <span class="theme-slider" aria-hidden="true"></span>
        </label>
        <a class="sidebar-link" href="https://github.com/robert-mcdermott/arietta-voice" target="_blank" rel="noopener noreferrer">GitHub Repository</a>
      </div>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <div class="eyebrow">Arietta Voice <span id="app-version"></span></div>
          <h2 id="view-title">Dashboard</h2>
          <div class="status-row"><span id="health-dot" class="dot"></span><span id="health-text">Checking</span><span id="poll-status">Auto-refresh on</span></div>
        </div>
        <button class="button info" id="refresh">Refresh</button>
      </div>

      <div id="error-notice" class="notice error-notice" role="alert" hidden><button id="error-dismiss" class="button">Dismiss</button><span id="error-message"></span></div>

      <section id="dashboard" class="view active">
        <div class="grid">
          <div class="metric"><div class="label">Assistant</div><div id="metric-assistant" class="value">--</div><div id="metric-wake" class="sub">--</div></div>
          <div class="metric"><div class="label">Runtime</div><div id="metric-runtime" class="value">--</div><div id="metric-uptime" class="sub">--</div></div>
          <div class="metric"><div class="label">Model</div><div id="metric-model" class="value">--</div><div id="metric-provider" class="sub">--</div></div>
          <div class="metric"><div class="label">Knowledge</div><div id="metric-knowledge" class="value">--</div><div id="metric-index" class="sub">--</div></div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <h3>Voice Runtime</h3>
            <div class="form-row runtime-actions" style="margin:0">
              <button class="button success" id="runtime-start">Start</button>
              <button class="button danger" id="runtime-stop">Stop</button>
              <button class="button warn" id="runtime-restart">Restart</button>
              <button class="button info" id="runtime-refresh">Refresh</button>
            </div>
          </div>
          <div class="runtime-summary"><span id="runtime-badge" class="runtime-badge">Checking</span><span id="runtime-summary"></span></div>
          <div id="runtime-warning" class="notice" role="status" hidden></div>
          <div id="runtime-reliability" class="reliability-grid"></div>
          <div id="api-inference-note" class="reliability-note"></div>
          <details class="runtime-details"><summary>Process, session, and file details</summary><div id="runtime-detail"></div></details>
        </div>
        <div class="split">
          <div class="panel">
            <div class="panel-head"><h3>Diagnostics</h3><button class="button" id="diagnostics-run">Run</button></div>
            <div id="diagnostics-table"></div>
          </div>
          <div class="panel">
            <div class="panel-head"><h3>Recent Events</h3><button class="button" data-view-target="logs">Open</button></div>
            <div id="recent-events"></div>
          </div>
        </div>
      </section>

      <section id="chat" class="view">
        <div class="split">
          <div class="panel">
            <div class="panel-head"><h3>Chat Test</h3><button class="button" id="chat-clear">Clear</button></div>
            <div id="chat-log" class="chat-log"></div>
          </div>
          <div class="panel">
            <label for="chat-message" class="muted">Message · ⌘/Ctrl + Enter to send</label>
            <textarea id="chat-message" maxlength="12000" placeholder="Type a message for the assistant"></textarea>
            <div class="form-row">
              <button class="button primary" id="chat-send">Send</button>
              <button class="button" id="tool-probe">Probe Tools</button>
            </div>
            <div id="chat-status" class="reliability-note" role="status">Ready</div>
            <pre id="chat-meta">{}</pre>
          </div>
        </div>
      </section>

      <section id="knowledge" class="view">
        <div class="split knowledge-layout">
          <div class="panel">
            <div class="panel-head"><h3>Knowledge Files</h3><button class="button" id="knowledge-files-refresh">Refresh</button></div>
            <div class="form-row">
              <input id="knowledge-file-path" placeholder="notes/example.md">
              <button class="button primary" id="knowledge-file-new">New</button>
            </div>
            <div id="knowledge-file-list"></div>
          </div>
          <div class="panel">
            <div class="panel-head">
              <h3>Editor</h3>
              <div class="form-row" style="margin:0">
                <button class="button primary" id="knowledge-file-save">Save</button>
                <button class="button danger" id="knowledge-file-delete">Delete</button>
              </div>
            </div>
            <textarea id="knowledge-file-content" class="knowledge-editor" placeholder="Open or create a knowledge file"></textarea>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head"><h3>Knowledge Search</h3><button class="button" id="knowledge-index">Build Index</button></div>
          <div class="form-row">
            <input id="knowledge-query" placeholder="Search local knowledge">
            <button class="button primary" id="knowledge-search">Search</button>
          </div>
          <div id="knowledge-results" class="result-list"></div>
        </div>
        <div class="panel"><pre id="knowledge-meta">{}</pre></div>
      </section>

      <section id="tools" class="view">
        <div class="tools-layout">
          <div class="panel">
            <div class="panel-head">
              <h3>Tool Registry</h3>
              <div class="tool-toolbar">
                <button class="button primary" id="tools-save-enabled">Save Enabled</button>
                <button class="button info" id="tools-reload">Reload API</button>
                <button class="button warn" id="tools-restart-runtime">Restart Runtime</button>
                <button class="button" id="tools-refresh">Refresh</button>
              </div>
            </div>
            <div class="status-row" id="tools-summary">No registry loaded.</div>
            <div class="form-row" style="margin-top:14px">
              <input id="tools-probe-text" placeholder="Sample text to probe against a tool">
              <button class="button primary" id="tools-probe-enabled">Probe Enabled</button>
            </div>
            <div class="status-row" id="tools-result-title">No action result yet.</div>
            <pre id="tools-result">{}</pre>
            <div id="tools-content"></div>
          </div>
          <div class="panel">
            <div class="panel-head"><h3>New Tool</h3><button class="button primary" id="tools-create" disabled>Create</button></div>
            <div class="tool-create-form">
              <label>
                Module Name
                <input id="tools-create-name" placeholder="my_custom_tool">
              </label>
              <label>
                Trigger Phrase
                <input id="tools-create-trigger" value="example custom tool">
              </label>
              <label>
                Response
                <textarea id="tools-create-response">The custom tool is wired up. Replace this response with your real behavior.</textarea>
              </label>
              <label class="tool-toggle" style="justify-content:flex-start;text-transform:none;font-weight:500;color:var(--ink)">
                <input id="tools-create-enable" type="checkbox" checked>
                Enable after creation
              </label>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="tools-layout">
            <div>
              <div class="panel-head">
                <h3>Model-Called Tools</h3>
                <div class="tool-toolbar">
                  <button class="button primary" id="model-tools-save-enabled">Save Enabled</button>
                  <button class="button info" id="model-tools-reload">Reload API</button>
                  <button class="button" id="model-tools-instructions">Preview Instructions</button>
                </div>
              </div>
              <div class="status-row" id="model-tools-summary">No model tool registry loaded.</div>
              <label class="tool-create-form" style="margin-top:14px">
                <span>Invocation Arguments JSON</span>
                <textarea id="model-tools-arguments">{}</textarea>
              </label>
              <div class="status-row" id="model-tools-result-title">No model tool action result yet.</div>
              <pre id="model-tools-result">{}</pre>
              <div id="tools-model-content"></div>
            </div>
            <div>
              <div class="panel-head"><h3>New Model Tool</h3><button class="button primary" id="model-tools-create" disabled>Create</button></div>
              <div class="tool-create-form">
                <label>
                  Module Name
                  <input id="model-tools-create-name" placeholder="my_model_tool">
                </label>
                <label>
                  Description
                  <input id="model-tools-create-description" value="you need a custom model-called tool result before answering">
                </label>
                <label>
                  Arguments Example JSON
                  <textarea id="model-tools-create-args">{"topic":"<short topic>"}</textarea>
                </label>
                <label>
                  Follow-up Context
                  <textarea id="model-tools-create-context">Custom model tool result. Use this result when composing the final answer.</textarea>
                </label>
                <label class="tool-toggle" style="justify-content:flex-start;text-transform:none;font-weight:500;color:var(--ink)">
                  <input id="model-tools-create-enable" type="checkbox" checked>
                  Enable after creation
                </label>
              </div>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <h3>Tool Source</h3>
            <div class="tool-toolbar">
              <button class="button primary" id="tool-source-save">Save Source</button>
              <button class="button info" id="tool-source-validate">Validate</button>
              <button class="button" id="tool-source-test">Probe / Invoke</button>
              <button class="button info" id="tool-source-reload">Reload API</button>
              <button class="button warn" id="tool-source-restart-runtime">Restart Runtime</button>
            </div>
          </div>
          <p class="muted" id="source-editing-policy">Source editing is disabled.</p>
          <div class="status-row" id="tool-source-meta">Open a tool source file from a tool card.</div>
          <div class="source-test-grid">
            <label>
              Deterministic Probe Text
              <input id="tool-source-test-text" placeholder="Sample text for deterministic tools">
            </label>
            <label>
              Model Tool Arguments JSON
              <textarea id="tool-source-test-args">{}</textarea>
            </label>
          </div>
          <textarea id="tool-source-content" class="source-editor" spellcheck="false" readonly placeholder="Click View Source on a deterministic or model-called tool."></textarea>
          <div class="status-row" id="tool-source-result-title">No source action result yet.</div>
          <pre id="tool-source-result">{}</pre>
        </div>
      </section>

      <section id="logs" class="view">
        <div class="panel">
          <div class="panel-head"><h3>Event Log</h3><button class="button info" id="logs-refresh">Refresh</button></div>
          <div class="form-row compact-row">
            <input id="logs-search" placeholder="Search logs">
            <input id="logs-event" placeholder="Filter by event">
            <button class="button primary" id="logs-apply">Apply</button>
            <button class="button" id="logs-clear">Clear</button>
          </div>
          <div class="form-row compact-row">
            <button class="button" id="logs-prev">Previous</button>
            <button class="button" id="logs-next">Next</button>
            <span class="muted" id="logs-page-info">No log page loaded.</span>
          </div>
          <div id="logs-content"></div>
        </div>
      </section>

      <section id="config" class="view">
        <div class="panel">
          <div class="panel-head"><h3>Resolved Config</h3><button class="button" id="config-refresh">Refresh</button></div>
          <pre id="config-json">{}</pre>
        </div>
      </section>
    </main>
  </div>
  <script>
    const state = {
      chatHistory: [],
      chatPending: false,
      knowledgeDirty: false,
      currentKnowledgeFile: null,
      currentView: "dashboard",
      dashboardLoading: false,
      activeViewLoading: false,
      runtimeActionPending: null,
      logsOffset: 0,
      logsLimit: 50,
      logsTotal: 0,
      lastToolsPayload: null,
      sourceEditingEnabled: false,
      sourceEditor: {
        kind: null,
        tool: null,
        file: null,
        dirty: false
      },
      lastDashboardRefreshAt: null
    };
    const DASHBOARD_POLL_MS = 3000;
    const ACTIVE_VIEW_POLL_MS = 9000;
    const viewTitles = {
      dashboard: "Dashboard",
      chat: "Chat",
      knowledge: "Knowledge",
      tools: "Tools",
      logs: "Logs",
      config: "Config"
    };
    const $ = (selector) => document.querySelector(selector);
    const $$ = (selector) => [...document.querySelectorAll(selector)];

    async function api(path, options = {}) {
      const headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
      const response = await fetch(path, Object.assign({}, options, { headers }));
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const detail = typeof body.detail === "string" ? body.detail : response.statusText;
        throw new Error(response.status === 401 ? "Your admin session needs authentication. Reload this page to sign in." : `${detail} (${response.status})`);
      }
      return response.json();
    }

    function setHealth(payload) {
      const dot = $("#health-dot");
      const text = $("#health-text");
      dot.classList.remove("ok", "bad");
      if (payload.status === "ok") dot.classList.add("ok");
      if (payload.status === "error") dot.classList.add("bad");
      text.textContent = `${payload.status || "unknown"} · ${payload.assistant || "Arietta"}`;
    }

    function setError(message) {
      $("#health-dot").classList.add("bad");
      $("#health-text").textContent = message;
      $("#error-message").textContent = message;
      $("#error-notice").hidden = false;
    }

    function setPollStatus(message) {
      $("#poll-status").textContent = message;
    }

    function currentTimeLabel() {
      return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function formatSeconds(seconds) {
      const value = Number(seconds || 0);
      if (value < 90) return `${value}s`;
      if (value < 3600) return `${Math.floor(value / 60)}m ${value % 60}s`;
      return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
    }

    function table(headers, rows) {
      const head = headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("");
      const body = rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("");
      return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[char]);
    }

    function applyTheme(theme) {
      const normalized = theme === "dark" ? "dark" : "light";
      document.documentElement.dataset.theme = normalized;
      localStorage.setItem("arietta-theme", normalized);
      $("#theme-toggle").checked = normalized === "dark";
    }

    function initThemeToggle() {
      const toggle = $("#theme-toggle");
      applyTheme(document.documentElement.dataset.theme || "light");
      toggle.addEventListener("change", () => applyTheme(toggle.checked ? "dark" : "light"));
    }

    async function loadDashboard(options = {}) {
      if (state.dashboardLoading) return;
      state.dashboardLoading = true;
      const silent = Boolean(options.silent);
      if (!silent) setPollStatus("Refreshing...");
      try {
        const [health, currentState, diagnostics, logs] = await Promise.all([
          api("/api/health"),
          api("/api/state"),
          api("/api/diagnostics"),
          api("/api/logs?limit=6")
        ]);
        setHealth(health);
        $("#app-version").textContent = `· v${currentState.api.version}`;
        $("#metric-assistant").textContent = currentState.assistant.name;
        $("#metric-wake").textContent = currentState.assistant.wake_enabled ? currentState.assistant.wake_words.join(", ") : "wake disabled";
        const runtime = currentState.runtime;
        $("#metric-runtime").textContent = String(runtime.state).replaceAll("_", " ");
        $("#metric-uptime").textContent = runtime.is_running
          ? `pid ${runtime.process_id || "--"} · heartbeat ${runtime.heartbeat_age_seconds ?? 0}s`
          : runtime.note;
        $("#metric-model").textContent = currentState.models.chat_model.split("/").pop();
        $("#metric-model").title = currentState.models.chat_model;
        $("#metric-provider").textContent = currentState.models.chat_provider;
        $("#metric-knowledge").textContent = currentState.knowledge.enabled ? currentState.knowledge.backend : "Disabled";
        $("#metric-index").textContent = !currentState.knowledge.enabled ? "Retrieval is turned off" : currentState.knowledge.loaded ? "Loaded" : "Loads on demand";
        $("#metric-index").title = currentState.knowledge.directory;
        renderRuntime(runtime);
        renderReliability(runtime, currentState.api);
        renderDiagnostics(diagnostics);
        renderRecentEvents(logs.items || []);
        state.lastDashboardRefreshAt = new Date();
        setPollStatus(`Updated ${currentTimeLabel()}`);
      } finally {
        state.dashboardLoading = false;
      }
    }

    function renderRuntime(runtime) {
      updateRuntimeControls(runtime);
      const rows = [
        ["State", `<strong>${escapeHtml(runtime.state)}</strong>`],
        ["Lifecycle", escapeHtml(runtime.lifecycle)],
        ["Running", runtime.is_running ? "yes" : "no"],
        ["Managed by API", runtime.managed_process_running ? "yes" : "no"],
        ["Managed logs", `<code>${escapeHtml(runtime.managed_stdout_log || "--")}</code>`],
        ["Last heartbeat", escapeHtml(runtime.last_heartbeat_at || "--")],
        ["Heartbeat age", runtime.heartbeat_age_seconds === null ? "--" : `${escapeHtml(runtime.heartbeat_age_seconds)}s`],
        ["Process", escapeHtml(runtime.process_id || "--")],
        ["Session", runtime.session_active ? `active · ${escapeHtml(runtime.session_turn_count || 0)} turns` : "inactive"],
        ["Last user", escapeHtml(runtime.last_user_transcript || "--")],
        ["Last assistant", escapeHtml(runtime.last_assistant_text || "--")],
        ["Status file", `<code>${escapeHtml(runtime.status_file)}</code>`],
        ["Note", escapeHtml(runtime.note)]
      ];
      $("#runtime-detail").innerHTML = table(["Field", "Value"], rows);
    }

    function renderReliability(runtime, apiState) {
      const data = runtime.reliability || {};
      const audio = data.audio || {};
      const active = Boolean(runtime.is_running);
      const badge = $("#runtime-badge");
      badge.textContent = String(runtime.state || "unknown").replaceAll("_", " ");
      badge.classList.toggle("offline", !active);
      $("#runtime-summary").textContent = active
        ? `Heartbeat ${runtime.heartbeat_age_seconds ?? 0}s ago · ${runtime.session_active ? "Conversation active" : "Ready for wake word"}`
        : runtime.note;
      const elapsed = Number(data.turn_elapsed_s || 0);
      const limit = Number(data.stage_timeout_s || data.turn_timeout_s || 60);
      const stageNames = { idle: "Ready", transcribe: "Transcribing", endpoint: "Checking turn end", chat: "Thinking", tts_start: "Preparing speech", tts_next: "Speaking", memory: "Updating memory", loading_models: "Loading models", stopped: "Stopped" };
      const stage = active ? (stageNames[data.turn_stage] || "Starting") : "Offline";
      const queued = Math.max(0, Number(audio.queued_chunks) || 0);
      const capacity = Math.max(0, Number(audio.queue_max_chunks) || 0);
      $("#runtime-reliability").innerHTML = `
        <div class="reliability-card"><h4>Current turn</h4><div class="reliability-value">${escapeHtml(stage)}</div>
          <div class="reliability-note">${elapsed ? `${elapsed}s of ${limit}s budget` : `Time limit ${limit}s`}</div>
          <progress aria-label="Turn time budget used" max="${limit}" value="${Math.min(elapsed, limit)}"></progress>
          <div class="reliability-note">Last turn ${data.last_turn_duration_ms == null ? "—" : `${escapeHtml(data.last_turn_duration_ms)} ms`}</div></div>
        <div class="reliability-card"><h4>Microphone buffer</h4><div class="reliability-value">${capacity ? `${queued} / ${capacity}` : "—"}</div>
          <div class="reliability-note">Chunks queued · bounded capture</div>
          <progress aria-label="Microphone buffer usage" max="${capacity || 1}" value="${queued}"></progress>
          <div class="reliability-note">Dropped ${escapeHtml(audio.dropped_chunks ?? 0)} · Input overruns ${escapeHtml(audio.input_overflows ?? 0)}</div></div>
        <div class="reliability-card"><h4>Recovery · this runtime</h4><div class="reliability-value">${escapeHtml(data.turn_failures ?? 0)} failed turn${data.turn_failures === 1 ? "" : "s"}</div>
          <div class="reliability-note">Timeouts ${escapeHtml(data.turn_timeouts ?? 0)} · Worker resets ${escapeHtml(data.worker_restarts ?? 0)}<br>
          Oversized utterances ${escapeHtml(data.utterances_discarded ?? 0)}<br>Recording drops ${escapeHtml(audio.recording_dropped_chunks ?? 0)}</div></div>`;
      const warnings = [];
      if (runtime.last_error) warnings.push(`Last error: ${runtime.last_error}. No automatic retry was made. Check any external action before repeating it.`);
      if (audio.recording_limit_reached) warnings.push("Recording reached its duration limit; voice interaction can continue.");
      if (audio.recording_error) warnings.push(`Recording stopped: ${audio.recording_error}. The recording may be incomplete.`);
      if (audio.recording_dropped_chunks) warnings.push("Recording lost audio under load; the saved file has gaps.");
      $("#runtime-warning").hidden = !warnings.length;
      $("#runtime-warning").textContent = warnings.join(" ");
      const inference = apiState.inference || {};
      $("#api-inference-note").textContent = `Web chat: ${inference.operation ? "processing a request" : "ready"} · ${inference.turn_timeout_s || limit}s turn limit.`;
    }

    function setButtonTone(button, tone) {
      button.className = `button ${tone || ""}`.trim();
    }

    function updateRuntimeControls(runtime) {
      const start = $("#runtime-start");
      const stop = $("#runtime-stop");
      const restart = $("#runtime-restart");
      const refresh = $("#runtime-refresh");
      if (state.runtimeActionPending) {
        setButtonTone(start, "");
        setButtonTone(stop, "");
        setButtonTone(restart, "");
        setButtonTone(refresh, "info");
        start.disabled = true;
        stop.disabled = true;
        restart.disabled = true;
        refresh.disabled = false;
        start.textContent = state.runtimeActionPending === "start" ? "Starting..." : "Start";
        stop.textContent = state.runtimeActionPending === "stop" ? "Stopping..." : "Stop";
        restart.textContent = state.runtimeActionPending === "restart" ? "Restarting..." : "Restart";
        return;
      }
      const managedRunning = Boolean(runtime.managed_process_running);
      const starting = runtime.lifecycle === "starting" || (managedRunning && !runtime.is_running);
      const canStart = !managedRunning && (!runtime.is_running || runtime.is_stale || runtime.lifecycle === "stopped");
      const canStop = managedRunning;
      const canRestart = managedRunning && !starting;

      setButtonTone(start, canStart ? "success" : "");
      setButtonTone(stop, canStop ? "danger" : "");
      setButtonTone(restart, canRestart ? "warn" : "");
      setButtonTone(refresh, "info");

      start.disabled = !canStart;
      stop.disabled = !canStop;
      restart.disabled = !canRestart;
      refresh.disabled = false;

      start.textContent = starting ? "Starting" : "Start";
      stop.textContent = managedRunning ? "Stop" : "Stop";
      restart.textContent = "Restart";

      start.title = canStart ? "Start an API-managed voice runtime" : "A voice runtime is already running";
      stop.title = canStop ? "Stop the API-managed voice runtime" : "Only API-managed runtimes can be stopped here";
      restart.title = canRestart ? "Restart the API-managed voice runtime" : "Restart is available after the API starts the runtime";
      refresh.title = "Refresh runtime status now";
    }

    async function runtimeAction(action) {
      state.runtimeActionPending = action;
      ["runtime-start", "runtime-stop", "runtime-restart"].forEach((id) => { $("#" + id).disabled = true; });
      setPollStatus(`${action} requested...`);
      try {
        const payload = await api(`/api/runtime/${action}`, { method: "POST" });
        $("#health-text").textContent = payload.message || `Runtime ${action} requested`;
      } finally {
        state.runtimeActionPending = null;
        await loadDashboard({ silent: true });
      }
    }

    function renderDiagnostics(payload) {
      const rows = (payload.checks || []).map((check) => [
        `<strong>${escapeHtml(check.name)}</strong>`,
        `<span class="${check.ok ? "" : "error"}">${escapeHtml(check.status)}</span>`,
        escapeHtml(check.detail)
      ]);
      $("#diagnostics-table").innerHTML = table(["Check", "Status", "Detail"], rows);
    }

    function renderRecentEvents(items) {
      if (!items.length) {
        $("#recent-events").innerHTML = `<p class="muted">No events recorded.</p>`;
        return;
      }
      $("#recent-events").innerHTML = items.map((item) => `
        <div class="row-card">
          <h4>${escapeHtml(item.event || "event")}</h4>
          <p>${escapeHtml(item.ts || "")}</p>
        </div>
      `).join("");
    }

    async function loadConfig() {
      $("#config-json").textContent = JSON.stringify(await api("/api/config"), null, 2);
    }

    function logsQueryString() {
      const params = new URLSearchParams();
      params.set("limit", String(state.logsLimit));
      params.set("offset", String(state.logsOffset));
      const event = $("#logs-event").value.trim();
      const query = $("#logs-search").value.trim();
      if (event) params.set("event", event);
      if (query) params.set("q", query);
      return params.toString();
    }

    async function loadLogs(options = {}) {
      if (options.reset) state.logsOffset = 0;
      const payload = await api(`/api/logs?${logsQueryString()}`);
      state.logsTotal = payload.total || 0;
      const rows = (payload.items || []).map((item) => [
        escapeHtml(item.ts || ""),
        `<strong>${escapeHtml(item.event || "")}</strong>`,
        `<code>${escapeHtml(JSON.stringify(item))}</code>`
      ]);
      $("#logs-content").innerHTML = rows.length ? table(["Time", "Event", "Payload"], rows) : `<p class="muted">No events recorded.</p>`;
      const start = payload.total ? payload.offset + 1 : 0;
      const end = payload.offset + payload.count;
      $("#logs-page-info").textContent = `${start}-${end} of ${payload.total} matching events`;
      $("#logs-prev").disabled = !payload.has_previous;
      $("#logs-next").disabled = !payload.has_next;
    }

    async function loadTools() {
      const payload = await api("/api/tools");
      renderTools(payload);
    }

    function renderTools(payload) {
      state.lastToolsPayload = payload;
      state.sourceEditingEnabled = Boolean(payload.source_editing_enabled);
      $("#tools-create").disabled = !state.sourceEditingEnabled;
      $("#model-tools-create").disabled = !state.sourceEditingEnabled;
      $("#source-editing-policy").textContent = state.sourceEditingEnabled
        ? "Source editing is enabled. Saving Python source can execute code on this machine."
        : "Source editing is disabled. You can view source and explicitly validate or test existing tools.";
      updateSourceControls();
      const assistant = payload.assistant_tools || { enabled: [], available: [], tools: [], load_errors: [] };
      const records = assistant.tools || [];
      const enabledCount = (assistant.enabled || []).length;
      const availableCount = (assistant.available || []).length;
      const invalidCount = (assistant.load_errors || []).length;
      $("#tools-summary").innerHTML = `
        <span class="status-pill ok">${enabledCount} enabled</span>
        <span class="status-pill">${availableCount} available</span>
        <span class="status-pill ${invalidCount ? "bad" : "ok"}">${invalidCount} inspection errors</span>
        <span class="muted">Directory: <code>${escapeHtml(assistant.tools_directory || "")}</code></span>
      `;

      $("#tools-content").innerHTML = records.length
        ? `<div class="tool-list">${records.map(renderToolCard).join("")}</div>`
        : `<p class="muted">No deterministic tools found.</p>`;

      renderModelTools(payload.model_tools || { enabled: [], available: [], tools: [], load_errors: [] });
      markToolsClean();
    }

    function renderToolCard(tool) {
      const statusClass = tool.error ? "bad" : "warn";
      const enabled = Boolean(tool.enabled);
      const checkboxDisabled = tool.error && !enabled ? "disabled" : "";
      const statusText = tool.error ? "inspection error" : "not validated";
      const availableText = tool.available ? "available" : "configured only";
      const errorHtml = tool.error ? `<div class="tool-error">${escapeHtml(tool.error)}</div>` : "";
      return `
        <div class="tool-item ${enabled ? "enabled" : ""} ${tool.error ? "invalid" : ""}">
          <div>
            <div class="tool-title">
              <strong>${escapeHtml(tool.configured_name)}</strong>
              <span class="status-pill ${statusClass}">${statusText}</span>
              <span class="status-pill ${enabled ? "ok" : ""}">${enabled ? "enabled" : "disabled"}</span>
              <span class="status-pill">${escapeHtml(availableText)}</span>
            </div>
            <div class="tool-meta">
              Runtime: <code>${escapeHtml(tool.tool_name || "--")}</code><br>
              Module: <code>${escapeHtml(tool.module || "--")}</code><br>
              Class: <code>${escapeHtml(tool.class || "--")}</code>
            </div>
            ${errorHtml}
          </div>
          <div class="tool-actions">
            <label class="tool-toggle">
              <input class="tool-enabled" type="checkbox" data-tool="${escapeHtml(tool.configured_name)}" ${enabled ? "checked" : ""} ${checkboxDisabled}>
              <span>${enabled ? "Enabled" : "Disabled"}</span>
            </label>
            <button class="button" data-tool-action="probe" data-tool-name="${escapeHtml(tool.configured_name)}">Probe</button>
            <button class="button info" data-tool-action="validate" data-tool-name="${escapeHtml(tool.configured_name)}">Validate</button>
            <button class="button" data-tool-action="edit-source" data-tool-name="${escapeHtml(tool.configured_name)}" ${tool.source === "local" && tool.available ? "" : "disabled"}>View Source</button>
          </div>
        </div>
      `;
    }

    function renderModelTools(modelTools) {
      const records = modelTools.tools || [];
      const enabledCount = (modelTools.enabled || []).length;
      const availableCount = (modelTools.available || []).length;
      const invalidCount = (modelTools.load_errors || []).length;
      $("#model-tools-summary").innerHTML = `
        <span class="status-pill ok">${enabledCount} enabled</span>
        <span class="status-pill">${availableCount} available</span>
        <span class="status-pill ${invalidCount ? "bad" : "ok"}">${invalidCount} inspection errors</span>
        <span class="muted">Directory: <code>${escapeHtml(modelTools.tools_directory || "")}</code></span>
      `;
      $("#tools-model-content").innerHTML = records.length
        ? `<div class="tool-list">${records.map(renderModelToolCard).join("")}</div>`
        : `<p class="muted">No model-called tools found.</p>`;
      markModelToolsClean();
    }

    function renderModelToolCard(tool) {
      const statusClass = tool.error ? "bad" : "warn";
      const enabled = Boolean(tool.enabled);
      const checkboxDisabled = tool.error && !enabled ? "disabled" : "";
      const statusText = tool.error ? "inspection error" : "not validated";
      const availableText = tool.available ? "available" : "configured only";
      const errorHtml = tool.error ? `<div class="tool-error">${escapeHtml(tool.error)}</div>` : "";
      const args = tool.arguments_example ? JSON.stringify(tool.arguments_example) : "{}";
      return `
        <div class="tool-item ${enabled ? "enabled" : ""} ${tool.error ? "invalid" : ""}">
          <div>
            <div class="tool-title">
              <strong>${escapeHtml(tool.configured_name)}</strong>
              <span class="status-pill ${statusClass}">${statusText}</span>
              <span class="status-pill ${enabled ? "ok" : ""}">${enabled ? "enabled" : "disabled"}</span>
              <span class="status-pill">${escapeHtml(availableText)}</span>
            </div>
            <div class="tool-meta">
              Runtime: <code>${escapeHtml(tool.tool_name || "--")}</code><br>
              Module: <code>${escapeHtml(tool.module || "--")}</code><br>
              Class: <code>${escapeHtml(tool.class || "--")}</code><br>
              Expected Args: <code>${escapeHtml(args)}</code><br>
              ${escapeHtml(tool.description || "No description available.")}
            </div>
            ${errorHtml}
          </div>
          <div class="tool-actions">
            <label class="tool-toggle">
              <input class="model-tool-enabled" type="checkbox" data-tool="${escapeHtml(tool.configured_name)}" ${enabled ? "checked" : ""} ${checkboxDisabled}>
              <span>${enabled ? "Enabled" : "Disabled"}</span>
            </label>
            <button class="button" data-model-tool-action="use-args" data-tool-name="${escapeHtml(tool.configured_name)}">Use Args</button>
            <button class="button" data-model-tool-action="invoke" data-tool-name="${escapeHtml(tool.configured_name)}" ${tool.active ? "" : "disabled"}>Invoke</button>
            <button class="button info" data-model-tool-action="validate" data-tool-name="${escapeHtml(tool.configured_name)}">Validate</button>
            <button class="button" data-model-tool-action="edit-source" data-tool-name="${escapeHtml(tool.configured_name)}" ${tool.source === "local" && tool.available ? "" : "disabled"}>View Source</button>
          </div>
        </div>
      `;
    }

    function markToolsDirty() {
      const button = $("#tools-save-enabled");
      button.classList.remove("primary");
      button.classList.add("warn");
      button.textContent = "Save Changes";
    }

    function markToolsClean() {
      const button = $("#tools-save-enabled");
      button.classList.remove("warn");
      button.classList.add("primary");
      button.textContent = "Save Enabled";
    }

    function markModelToolsDirty() {
      const button = $("#model-tools-save-enabled");
      button.classList.remove("primary");
      button.classList.add("warn");
      button.textContent = "Save Changes";
    }

    function markModelToolsClean() {
      const button = $("#model-tools-save-enabled");
      button.classList.remove("warn");
      button.classList.add("primary");
      button.textContent = "Save Enabled";
    }

    function toolPathName(name) {
      return encodeURIComponent(name);
    }

    function modelToolRecord(name) {
      const records = state.lastToolsPayload?.model_tools?.tools || [];
      return records.find((tool) => tool.configured_name === name) || null;
    }

    function setToolsResult(label, payload) {
      $("#tools-result-title").textContent = label;
      $("#tools-result").textContent = JSON.stringify(payload, null, 2);
    }

    function setModelToolsResult(label, payload) {
      $("#model-tools-result-title").textContent = label;
      $("#model-tools-result").textContent = typeof payload === "string"
        ? payload
        : JSON.stringify(payload, null, 2);
    }

    function readJsonTextarea(selector) {
      const text = $(selector).value.trim();
      if (!text) return {};
      return JSON.parse(text);
    }

    function writeJsonTextarea(selector, value) {
      $(selector).value = JSON.stringify(value || {}, null, 2);
    }

    function sourceEndpoint(kind, tool, suffix = "source") {
      const base = kind === "model" ? "/api/model-tools" : "/api/tools";
      return `${base}/${toolPathName(tool)}/${suffix}`;
    }

    function setSourceResult(label, payload) {
      $("#tool-source-result-title").textContent = label;
      $("#tool-source-result").textContent = typeof payload === "string"
        ? payload
        : JSON.stringify(payload, null, 2);
    }

    function updateSourceControls() {
      const hasSource = Boolean(state.sourceEditor.kind && state.sourceEditor.tool);
      $("#tool-source-save").disabled = !hasSource || !state.sourceEditingEnabled;
      $("#tool-source-content").readOnly = !state.sourceEditingEnabled;
      $("#tool-source-validate").disabled = !hasSource;
      $("#tool-source-test").disabled = !hasSource;
      $("#tool-source-reload").disabled = !hasSource;
      $("#tool-source-restart-runtime").disabled = !hasSource;
      const label = hasSource
        ? `${state.sourceEditor.kind === "model" ? "Model-called" : "Deterministic"} tool: ${state.sourceEditor.tool}`
        : "Open a tool source file from a tool card.";
      const dirty = state.sourceEditor.dirty ? " · unsaved changes" : "";
      const file = state.sourceEditor.file ? ` · ${state.sourceEditor.file}` : "";
      $("#tool-source-meta").innerHTML = `${escapeHtml(label)}${escapeHtml(dirty)}${file ? ` · <code>${escapeHtml(state.sourceEditor.file)}</code>` : ""}`;
    }

    async function openToolSource(kind, name) {
      const payload = await api(sourceEndpoint(kind, name));
      state.sourceEditor = {
        kind,
        tool: name,
        file: payload.file,
        dirty: false
      };
      $("#tool-source-content").value = payload.content;
      if (kind === "tool") {
        $("#tool-source-test-text").value = $("#tools-probe-text").value || "";
      } else {
        writeJsonTextarea("#tool-source-test-args", payload.tool?.arguments_example || {});
      }
      updateSourceControls();
      setSourceResult(`Source loaded: ${name}`, {
        configured_name: payload.configured_name,
        enabled: payload.enabled,
        file: payload.file,
        valid: payload.tool?.valid,
        active: payload.tool?.active,
        error: payload.tool?.error
      });
      $("#tool-source-content").focus();
    }

    async function saveToolSource() {
      const editor = state.sourceEditor;
      if (!editor.kind || !editor.tool) return;
      const payload = await api(sourceEndpoint(editor.kind, editor.tool), {
        method: "PUT",
        body: JSON.stringify({ content: $("#tool-source-content").value })
      });
      state.sourceEditor.dirty = false;
      updateSourceControls();
      setSourceResult(`Source saved: ${editor.tool}`, payload);
      await loadTools();
    }

    async function validateToolSource() {
      const editor = state.sourceEditor;
      if (!editor.kind || !editor.tool) return;
      const endpoint = editor.kind === "model"
        ? `/api/model-tools/${toolPathName(editor.tool)}/validate`
        : `/api/tools/${toolPathName(editor.tool)}/validate`;
      const payload = await api(endpoint, { method: "POST" });
      setSourceResult(`Validation result: ${editor.tool}`, payload);
      await loadTools();
    }

    async function testToolSource() {
      const editor = state.sourceEditor;
      if (!editor.kind || !editor.tool) return;
      if (editor.kind === "model") {
        let argumentsPayload;
        try {
          argumentsPayload = readJsonTextarea("#tool-source-test-args");
        } catch (error) {
          setSourceResult("Invoke result", { error: `Invalid arguments JSON: ${error.message}`, tool: editor.tool });
          return;
        }
        const tool = modelToolRecord(editor.tool);
        const mismatch = argumentKeyMismatch(tool, argumentsPayload);
        if (mismatch) {
          setSourceResult(`Invoke check: ${editor.tool}`, mismatch);
          return;
        }
        const payload = await api(`/api/model-tools/${toolPathName(editor.tool)}/invoke`, {
          method: "POST",
          body: JSON.stringify({ arguments: argumentsPayload })
        });
        setSourceResult(`Invoke result: ${editor.tool}`, payload);
        return;
      }

      const text = $("#tool-source-test-text").value.trim();
      if (!text) {
        setSourceResult("Probe result", { error: "Enter sample text before probing.", tool: editor.tool });
        return;
      }
      const payload = await api(`/api/tools/${toolPathName(editor.tool)}/probe`, {
        method: "POST",
        body: JSON.stringify({ text })
      });
      setSourceResult(`Probe result: ${editor.tool}`, payload);
    }

    async function reloadSourceToolFamily() {
      const editor = state.sourceEditor;
      if (!editor.kind || !editor.tool) return;
      const payload = editor.kind === "model"
        ? await api("/api/model-tools/reload", { method: "POST" })
        : await api("/api/tools/reload", { method: "POST" });
      setSourceResult(`Reload result: ${editor.kind === "model" ? "model-called tools" : "deterministic tools"}`, payload);
      await loadTools();
    }

    function argumentKeyMismatch(tool, argumentsPayload) {
      const expected = tool?.arguments_example || {};
      const expectedKeys = Object.keys(expected);
      const providedKeys = Object.keys(argumentsPayload || {});
      if (!expectedKeys.length && providedKeys.length) {
        return {
          tool: tool.configured_name,
          expected_arguments: expected,
          provided_arguments: argumentsPayload,
          error: "This tool expects no arguments. Click Use Args to load {} before invoking."
        };
      }
      const missingKeys = expectedKeys.filter((key) => !Object.prototype.hasOwnProperty.call(argumentsPayload, key));
      if (missingKeys.length) {
        return {
          tool: tool.configured_name,
          expected_arguments: expected,
          provided_arguments: argumentsPayload,
          missing_keys: missingKeys,
          error: `This tool expects ${missingKeys.join(", ")} in the arguments JSON. Click Use Args to load the expected shape.`
        };
      }
      return null;
    }

    async function saveToolEnabled() {
      const enabled = $$(".tool-enabled").filter((input) => input.checked).map((input) => input.dataset.tool);
      const button = $("#tools-save-enabled");
      button.disabled = true;
      button.textContent = "Saving...";
      try {
        const payload = await api("/api/tools/enabled", {
          method: "POST",
          body: JSON.stringify({ enabled })
        });
        setToolsResult("Save result", { saved: true, enabled: payload.enabled });
        await loadTools();
        await loadDashboard({ silent: true });
      } finally {
        button.disabled = false;
        if (button.textContent === "Saving...") markToolsDirty();
      }
    }

    async function probeEnabledToolsFromRegistry() {
      const text = $("#tools-probe-text").value.trim();
      if (!text) {
        setToolsResult("Probe result", { error: "Enter sample text before probing." });
        return;
      }
      const payload = await api("/api/tools/probe", {
        method: "POST",
        body: JSON.stringify({ text })
      });
      setToolsResult("Probe enabled tools result", payload);
    }

    async function probeSingleTool(name) {
      const text = $("#tools-probe-text").value.trim();
      if (!text) {
        setToolsResult("Probe result", { error: "Enter sample text before probing.", tool: name });
        return;
      }
      const payload = await api(`/api/tools/${toolPathName(name)}/probe`, {
        method: "POST",
        body: JSON.stringify({ text })
      });
      setToolsResult(`Probe result: ${name}`, payload);
    }

    async function validateSingleTool(name) {
      const payload = await api(`/api/tools/${toolPathName(name)}/validate`, { method: "POST" });
      setToolsResult(`Validation result: ${name}`, payload);
      await loadTools();
    }

    async function createTool() {
      const button = $("#tools-create");
      button.disabled = true;
      button.textContent = "Creating...";
      try {
        const payload = await api("/api/tools/create", {
          method: "POST",
          body: JSON.stringify({
            module_name: $("#tools-create-name").value.trim(),
            trigger_phrase: $("#tools-create-trigger").value.trim(),
            response: $("#tools-create-response").value,
            enable: $("#tools-create-enable").checked
          })
        });
        setToolsResult(`Create result: ${payload.module_name}`, payload);
        $("#tools-create-name").value = "";
        await loadTools();
        await loadDashboard({ silent: true });
      } finally {
        button.disabled = !state.sourceEditingEnabled;
        button.textContent = "Create";
      }
    }

    async function reloadTools() {
      const payload = await api("/api/tools/reload", { method: "POST" });
      setToolsResult("Reload result", payload);
      await loadTools();
    }

    async function saveModelToolEnabled() {
      const enabled = $$(".model-tool-enabled").filter((input) => input.checked).map((input) => input.dataset.tool);
      const button = $("#model-tools-save-enabled");
      button.disabled = true;
      button.textContent = "Saving...";
      try {
        const payload = await api("/api/model-tools/enabled", {
          method: "POST",
          body: JSON.stringify({ enabled })
        });
        setModelToolsResult("Save result", { saved: true, enabled: payload.enabled });
        await loadTools();
        await loadDashboard({ silent: true });
      } finally {
        button.disabled = false;
        if (button.textContent === "Saving...") markModelToolsDirty();
      }
    }

    async function invokeModelTool(name) {
      let argumentsPayload;
      try {
        argumentsPayload = readJsonTextarea("#model-tools-arguments");
      } catch (error) {
        setModelToolsResult("Invoke result", { error: `Invalid arguments JSON: ${error.message}`, tool: name });
        return;
      }
      const tool = modelToolRecord(name);
      const mismatch = argumentKeyMismatch(tool, argumentsPayload);
      if (mismatch) {
        setModelToolsResult(`Invoke check: ${name}`, mismatch);
        return;
      }
      const payload = await api(`/api/model-tools/${toolPathName(name)}/invoke`, {
        method: "POST",
        body: JSON.stringify({ arguments: argumentsPayload })
      });
      setModelToolsResult(`Invoke result: ${name}`, payload);
    }

    function useModelToolArgs(name) {
      const tool = modelToolRecord(name);
      const args = tool?.arguments_example || {};
      writeJsonTextarea("#model-tools-arguments", args);
      setModelToolsResult(`Arguments loaded: ${name}`, {
        tool: name,
        arguments: args
      });
    }

    async function validateModelTool(name) {
      const payload = await api(`/api/model-tools/${toolPathName(name)}/validate`, { method: "POST" });
      setModelToolsResult(`Validation result: ${name}`, payload);
      await loadTools();
    }

    async function createModelTool() {
      const button = $("#model-tools-create");
      button.disabled = true;
      button.textContent = "Creating...";
      try {
        let argumentsExample;
        try {
          argumentsExample = readJsonTextarea("#model-tools-create-args");
        } catch (error) {
          setModelToolsResult("Create result", { error: `Invalid arguments example JSON: ${error.message}` });
          return;
        }
        const payload = await api("/api/model-tools/create", {
          method: "POST",
          body: JSON.stringify({
            module_name: $("#model-tools-create-name").value.trim(),
            description: $("#model-tools-create-description").value.trim(),
            arguments_example: argumentsExample,
            followup_context: $("#model-tools-create-context").value,
            enable: $("#model-tools-create-enable").checked
          })
        });
        setModelToolsResult(`Create result: ${payload.module_name}`, payload);
        $("#model-tools-create-name").value = "";
        await loadTools();
        await loadDashboard({ silent: true });
      } finally {
        button.disabled = !state.sourceEditingEnabled;
        button.textContent = "Create";
      }
    }

    async function reloadModelTools() {
      const payload = await api("/api/model-tools/reload", { method: "POST" });
      setModelToolsResult("Reload result", payload);
      await loadTools();
    }

    async function previewModelToolInstructions() {
      const payload = await api("/api/model-tools/instructions", { method: "POST" });
      setModelToolsResult("Instruction preview", payload.instruction_text || "No model-called tools are enabled.");
    }

    async function searchKnowledge() {
      const query = $("#knowledge-query").value.trim();
      if (!query) return;
      const payload = await api("/api/knowledge/search", {
        method: "POST",
        body: JSON.stringify({ query })
      });
      $("#knowledge-meta").textContent = JSON.stringify({
        query: payload.query,
        backend: payload.backend,
        directory: payload.directory,
        matches: payload.matches.length
      }, null, 2);
      $("#knowledge-results").innerHTML = payload.matches.length ? payload.matches.map((match) => `
        <div class="row-card">
          <h4>${escapeHtml(match.title)}</h4>
          <div><span class="pill">${escapeHtml(match.source)}</span><span class="pill">score ${escapeHtml(match.score)}</span></div>
          <p>${escapeHtml(match.snippet)}</p>
        </div>
      `).join("") : `<p class="muted">No matches.</p>`;
    }

    async function buildKnowledgeIndex() {
      $("#knowledge-meta").textContent = "Building index...";
      const payload = await api("/api/knowledge/index", { method: "POST" });
      $("#knowledge-meta").textContent = JSON.stringify(payload, null, 2);
      await loadDashboard();
    }

    function knowledgeFileUrl(path) {
      return `/api/knowledge/files/${path.split("/").map(encodeURIComponent).join("/")}`;
    }

    async function loadKnowledgeFiles() {
      const payload = await api("/api/knowledge/files");
      const items = (payload.files || []).map((file) => `
        <div class="file-item">
          <div>
            <strong title="${escapeHtml(file.path)}">${escapeHtml(file.path)}</strong>
            <div class="meta">${escapeHtml(file.size_bytes)} bytes · ${escapeHtml(file.updated_at)}</div>
          </div>
          <button class="button" data-knowledge-file="${escapeHtml(file.path)}">Open</button>
        </div>
      `).join("");
      $("#knowledge-file-list").innerHTML = items
        ? `<div class="file-list">${items}</div>`
        : `<p class="muted">No knowledge files yet.</p>`;
    }

    async function openKnowledgeFile(path) {
      if (state.knowledgeDirty && !confirm("Discard unsaved knowledge edits and open another file?")) return;
      const payload = await api(knowledgeFileUrl(path));
      state.currentKnowledgeFile = payload.path;
      $("#knowledge-file-path").value = payload.path;
      $("#knowledge-file-content").value = payload.content;
      state.knowledgeDirty = false;
      $("#knowledge-meta").textContent = JSON.stringify({ opened: payload.path }, null, 2);
    }

    function newKnowledgeFile() {
      const path = $("#knowledge-file-path").value.trim();
      if (!path) {
        $("#knowledge-meta").textContent = JSON.stringify({ error: "Enter a file path first." }, null, 2);
        return;
      }
      state.currentKnowledgeFile = path;
      if (!$("#knowledge-file-content").value.trim()) {
        const title = path.split("/").pop().replace(/\\.(md|markdown|txt)$/i, "").replace(/[-_]+/g, " ");
        $("#knowledge-file-content").value = `---\\ntitle: ${title}\\ntags: \\n---\\n\\n## Notes\\n\\n`;
      }
      $("#knowledge-meta").textContent = JSON.stringify({ editing: path }, null, 2);
    }

    async function saveKnowledgeFile() {
      const path = ($("#knowledge-file-path").value || state.currentKnowledgeFile || "").trim();
      if (!path) return;
      const payload = await api(knowledgeFileUrl(path), {
        method: "PUT",
        body: JSON.stringify({ content: $("#knowledge-file-content").value })
      });
      state.currentKnowledgeFile = payload.path;
      state.knowledgeDirty = false;
      $("#knowledge-file-path").value = payload.path;
      $("#knowledge-meta").textContent = JSON.stringify(payload, null, 2);
      await loadKnowledgeFiles();
      await loadDashboard();
    }

    async function deleteKnowledgeFile() {
      const path = ($("#knowledge-file-path").value || state.currentKnowledgeFile || "").trim();
      if (!path) return;
      if (!confirm(`Delete ${path}? This cannot be undone here.`)) return;
      const payload = await api(knowledgeFileUrl(path), { method: "DELETE" });
      state.currentKnowledgeFile = null;
      state.knowledgeDirty = false;
      $("#knowledge-file-path").value = "";
      $("#knowledge-file-content").value = "";
      $("#knowledge-meta").textContent = JSON.stringify(payload, null, 2);
      await loadKnowledgeFiles();
      await loadDashboard();
    }

    function renderChat() {
      $("#chat-log").innerHTML = state.chatHistory.map((message) => `
        <div class="bubble"><strong>${escapeHtml(message.role)}</strong>${escapeHtml(message.content)}</div>
      `).join("");
    }

    async function sendChat() {
      const textarea = $("#chat-message");
      const message = textarea.value.trim();
      if (!message || state.chatPending) return;
      const history = state.chatHistory.slice(-10);
      state.chatPending = true;
      ["chat-send", "chat-clear", "tool-probe"].forEach((id) => { $("#" + id).disabled = true; });
      textarea.disabled = true;
      $("#chat-send").textContent = "Thinking…";
      $("#chat-status").textContent = "Processing one turn. The server enforces the configured time limit.";
      state.chatHistory.push({ role: "user", content: message });
      textarea.value = "";
      renderChat();
      try {
        const payload = await api("/api/chat", {
          method: "POST", body: JSON.stringify({ message, history })
        });
        state.chatHistory.push({ role: "assistant", content: payload.response });
        state.chatHistory = state.chatHistory.slice(-40);
        $("#chat-meta").textContent = JSON.stringify(payload, null, 2);
        $("#chat-status").textContent = "Response received";
      } catch (error) {
        state.chatHistory.pop();
        textarea.value = message;
        $("#chat-status").textContent = error.message;
        throw error;
      } finally {
        state.chatPending = false;
        ["chat-send", "chat-clear", "tool-probe"].forEach((id) => { $("#" + id).disabled = false; });
        textarea.disabled = false;
        $("#chat-send").textContent = "Send";
        renderChat();
        textarea.focus();
      }
      await loadDashboard({ silent: true });
    }

    async function probeTools() {
      const text = $("#chat-message").value.trim();
      if (!text) return;
      $("#chat-meta").textContent = JSON.stringify(await api("/api/tools/probe", {
        method: "POST",
        body: JSON.stringify({ text })
      }), null, 2);
    }

    function switchView(view) {
      state.currentView = view;
      $$(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
      $$(".view").forEach((section) => section.classList.toggle("active", section.id === view));
      $("#view-title").textContent = viewTitles[view] || "Dashboard";
      if (view === "config") loadConfig().catch((error) => setError(error.message));
      if (view === "logs") loadLogs({ reset: true }).catch((error) => setError(error.message));
      if (view === "tools") loadTools().catch((error) => setError(error.message));
      if (view === "knowledge") loadKnowledgeFiles().catch((error) => setError(error.message));
    }

    async function refreshActiveView() {
      if (state.activeViewLoading) return;
      state.activeViewLoading = true;
      try {
        if (state.currentView === "logs") await loadLogs();
        if (state.currentView === "tools") await loadTools();
        if (state.currentView === "config") await loadConfig();
        if (state.currentView === "knowledge" && !state.currentKnowledgeFile) await loadKnowledgeFiles();
      } catch (error) {
        setError(error.message);
      } finally {
        state.activeViewLoading = false;
      }
    }

    $$(".nav-button").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
    $$("[data-view-target]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.viewTarget)));
    $("#refresh").addEventListener("click", () => loadDashboard().catch((error) => setError(error.message)));
    $("#runtime-refresh").addEventListener("click", () => loadDashboard().catch((error) => setError(error.message)));
    $("#runtime-start").addEventListener("click", () => runtimeAction("start").catch((error) => setError(error.message)));
    $("#runtime-stop").addEventListener("click", () => runtimeAction("stop").catch((error) => setError(error.message)));
    $("#runtime-restart").addEventListener("click", () => runtimeAction("restart").catch((error) => setError(error.message)));
    $("#diagnostics-run").addEventListener("click", () => api("/api/diagnostics?include_hardware=true").then(renderDiagnostics).catch((error) => setError(error.message)));
    $("#config-refresh").addEventListener("click", () => loadConfig().catch((error) => setError(error.message)));
    $("#logs-refresh").addEventListener("click", () => loadLogs().catch((error) => setError(error.message)));
    $("#logs-apply").addEventListener("click", () => loadLogs({ reset: true }).catch((error) => setError(error.message)));
    $("#logs-clear").addEventListener("click", () => {
      $("#logs-search").value = "";
      $("#logs-event").value = "";
      loadLogs({ reset: true }).catch((error) => setError(error.message));
    });
    $("#logs-prev").addEventListener("click", () => {
      state.logsOffset = Math.max(0, state.logsOffset - state.logsLimit);
      loadLogs().catch((error) => setError(error.message));
    });
    $("#logs-next").addEventListener("click", () => {
      state.logsOffset += state.logsLimit;
      loadLogs().catch((error) => setError(error.message));
    });
    $("#logs-search").addEventListener("keydown", (event) => { if (event.key === "Enter") loadLogs({ reset: true }).catch((error) => setError(error.message)); });
    $("#logs-event").addEventListener("keydown", (event) => { if (event.key === "Enter") loadLogs({ reset: true }).catch((error) => setError(error.message)); });
    $("#tools-refresh").addEventListener("click", () => loadTools().catch((error) => setError(error.message)));
    $("#tools-save-enabled").addEventListener("click", () => saveToolEnabled().catch((error) => setError(error.message)));
    $("#tools-reload").addEventListener("click", () => reloadTools().catch((error) => setError(error.message)));
    $("#tools-restart-runtime").addEventListener("click", () => runtimeAction("restart").catch((error) => setError(error.message)));
    $("#tools-probe-enabled").addEventListener("click", () => probeEnabledToolsFromRegistry().catch((error) => setError(error.message)));
    $("#tools-create").addEventListener("click", () => createTool().catch((error) => setError(error.message)));
    $("#tools-probe-text").addEventListener("keydown", (event) => { if (event.key === "Enter") probeEnabledToolsFromRegistry().catch((error) => setError(error.message)); });
    $("#tools-content").addEventListener("change", (event) => {
      if (event.target.classList.contains("tool-enabled")) markToolsDirty();
    });
    $("#tools-content").addEventListener("click", (event) => {
      const target = event.target.closest("[data-tool-action]");
      if (!target) return;
      if (target.dataset.toolAction === "probe") probeSingleTool(target.dataset.toolName).catch((error) => setError(error.message));
      if (target.dataset.toolAction === "validate") validateSingleTool(target.dataset.toolName).catch((error) => setError(error.message));
      if (target.dataset.toolAction === "edit-source") openToolSource("tool", target.dataset.toolName).catch((error) => setError(error.message));
    });
    $("#model-tools-save-enabled").addEventListener("click", () => saveModelToolEnabled().catch((error) => setError(error.message)));
    $("#model-tools-reload").addEventListener("click", () => reloadModelTools().catch((error) => setError(error.message)));
    $("#model-tools-instructions").addEventListener("click", () => previewModelToolInstructions().catch((error) => setError(error.message)));
    $("#model-tools-create").addEventListener("click", () => createModelTool().catch((error) => setError(error.message)));
    $("#tools-model-content").addEventListener("change", (event) => {
      if (event.target.classList.contains("model-tool-enabled")) markModelToolsDirty();
    });
    $("#tools-model-content").addEventListener("click", (event) => {
      const target = event.target.closest("[data-model-tool-action]");
      if (!target) return;
      if (target.dataset.modelToolAction === "use-args") useModelToolArgs(target.dataset.toolName);
      if (target.dataset.modelToolAction === "invoke") invokeModelTool(target.dataset.toolName).catch((error) => setError(error.message));
      if (target.dataset.modelToolAction === "validate") validateModelTool(target.dataset.toolName).catch((error) => setError(error.message));
      if (target.dataset.modelToolAction === "edit-source") openToolSource("model", target.dataset.toolName).catch((error) => setError(error.message));
    });
    $("#tool-source-content").addEventListener("input", () => {
      if (!state.sourceEditor.kind) return;
      state.sourceEditor.dirty = true;
      updateSourceControls();
    });
    $("#tool-source-save").addEventListener("click", () => saveToolSource().catch((error) => setError(error.message)));
    $("#tool-source-validate").addEventListener("click", () => validateToolSource().catch((error) => setError(error.message)));
    $("#tool-source-test").addEventListener("click", () => testToolSource().catch((error) => setError(error.message)));
    $("#tool-source-reload").addEventListener("click", () => reloadSourceToolFamily().catch((error) => setError(error.message)));
    $("#tool-source-restart-runtime").addEventListener("click", () => runtimeAction("restart").catch((error) => setError(error.message)));
    $("#knowledge-search").addEventListener("click", () => searchKnowledge().catch((error) => setError(error.message)));
    $("#knowledge-query").addEventListener("keydown", (event) => { if (event.key === "Enter") searchKnowledge().catch((error) => setError(error.message)); });
    $("#knowledge-index").addEventListener("click", () => buildKnowledgeIndex().catch((error) => setError(error.message)));
    $("#knowledge-files-refresh").addEventListener("click", () => loadKnowledgeFiles().catch((error) => setError(error.message)));
    $("#knowledge-file-new").addEventListener("click", newKnowledgeFile);
    $("#knowledge-file-save").addEventListener("click", () => saveKnowledgeFile().catch((error) => setError(error.message)));
    $("#knowledge-file-content").addEventListener("input", () => { state.knowledgeDirty = true; });
    $("#knowledge-file-delete").addEventListener("click", () => deleteKnowledgeFile().catch((error) => setError(error.message)));
    $("#knowledge-file-list").addEventListener("click", (event) => {
      const target = event.target.closest("[data-knowledge-file]");
      if (target) openKnowledgeFile(target.dataset.knowledgeFile).catch((error) => setError(error.message));
    });
    $("#chat-send").addEventListener("click", () => sendChat().catch((error) => setError(error.message)));
    $("#tool-probe").addEventListener("click", () => probeTools().catch((error) => setError(error.message)));
    $("#chat-clear").addEventListener("click", () => { state.chatHistory = []; renderChat(); $("#chat-meta").textContent = "{}"; });
    $("#chat-message").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) sendChat().catch((error) => setError(error.message));
    });

    setInterval(() => {
      if (!document.hidden) loadDashboard({ silent: true }).catch((error) => setError(error.message));
    }, DASHBOARD_POLL_MS);
    setInterval(() => {
      if (!document.hidden) refreshActiveView();
    }, ACTIVE_VIEW_POLL_MS);
    $("#error-dismiss").addEventListener("click", () => { $("#error-notice").hidden = true; });
    window.addEventListener("beforeunload", (event) => {
      if (state.sourceEditor.dirty || state.knowledgeDirty || state.chatPending) {
        event.preventDefault();
        event.returnValue = "";
      }
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        loadDashboard({ silent: true }).catch((error) => setError(error.message));
        refreshActiveView();
      }
    });

    initThemeToggle();
    updateSourceControls();
    loadDashboard().catch((error) => setError(error.message));
  </script>
</body>
</html>
"""
