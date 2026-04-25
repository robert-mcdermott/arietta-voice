from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import secrets
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from arietta_voice.config import AppConfig, config_to_dict, load_config
from arietta_voice.diagnostics import run_startup_diagnostics
from arietta_voice.knowledge import KnowledgeRetriever, build_knowledge_index
from arietta_voice.llm import ChatModel, build_chat_model
from arietta_voice.logging_utils import EventLogger
from arietta_voice.model_tooling import maybe_generate_with_model_tools
from arietta_voice.model_tools.base import ModelToolContext
from arietta_voice.model_tools.registry import _tool_module_name as _model_tool_module_name
from arietta_voice.models import ConversationTurn
from arietta_voice.prompts import PromptManager, sanitize_for_speech
from arietta_voice.runtime_status import read_runtime_status
from arietta_voice.runtime_supervisor import RuntimeSupervisor
from arietta_voice.tools.base import ToolContext
from arietta_voice.tools.registry import _tool_module_name as _assistant_tool_module_name


API_VERSION = "0.1.0"
AUTH_REALM = "Arietta Admin"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "password"


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    use_tools: bool = True


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class KnowledgeFileWriteRequest(BaseModel):
    content: str = Field(max_length=1_000_000)


class ToolProbeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@dataclass(slots=True)
class ChatResult:
    response: str
    source: str
    used_tool: str | None
    model_tool: str | None


class AriettaApiService:
    def __init__(self, config_path: Path | None = None, *, config: AppConfig | None = None) -> None:
        self.config_path = config_path
        self.config = config or load_config(config_path)
        self.started_at = _now_iso()
        self.started_monotonic = time.monotonic()
        self.event_logger = EventLogger(self.config.logging.events_file)
        self.prompt_manager = PromptManager(self.config)
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        self.runtime_supervisor = RuntimeSupervisor(self.config, config_path=config_path)
        self._llm: ChatModel | None = None
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
                "llm_loaded": self._llm is not None and self._llm.is_loaded(),
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
                "loaded_count": len(tools["assistant_tools"]["loaded"]),
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
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
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
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
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
        self.knowledge = KnowledgeRetriever(self.config.knowledge)
        payload = {"path": relative_path, "deleted": True}
        self.event_logger.log("api_knowledge_file_deleted", **payload)
        return payload

    def inspect_tools(self) -> dict[str, Any]:
        assistant_loaded: list[dict[str, Any]] = []
        assistant_errors: list[dict[str, str]] = []
        assistant_context = ToolContext(
            config=self.config,
            knowledge_retriever=self.knowledge if self.config.knowledge.enabled else None,
        )
        for tool_name in self.config.tools.enabled:
            try:
                module_name = _assistant_tool_module_name(tool_name)
                module = importlib.import_module(module_name)
                factory = getattr(module, "create_tool")
                tool = factory(assistant_context)
                assistant_loaded.append(
                    {
                        "configured_name": tool_name,
                        "name": tool.name,
                        "module": module_name,
                        "class": type(tool).__name__,
                    }
                )
            except Exception as exc:
                assistant_errors.append(
                    {
                        "configured_name": tool_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        model_loaded: list[dict[str, Any]] = []
        model_errors: list[dict[str, str]] = []
        model_context = ModelToolContext(
            config=self.config,
            knowledge_retriever=self.knowledge if self.config.knowledge.enabled else None,
        )
        for tool_name in self.config.model_tools.enabled:
            try:
                module_name = _model_tool_module_name(tool_name)
                module = importlib.import_module(module_name)
                factory = getattr(module, "create_tool")
                tool = factory(model_context)
                if tool is None:
                    continue
                model_loaded.append(
                    {
                        "configured_name": tool_name,
                        "name": tool.name,
                        "module": module_name,
                        "class": type(tool).__name__,
                        "description": tool.description,
                        "arguments_example": tool.arguments_example,
                    }
                )
            except Exception as exc:
                model_errors.append(
                    {
                        "configured_name": tool_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return {
            "assistant_tools": {
                "enabled": list(self.config.tools.enabled),
                "loaded": assistant_loaded,
                "errors": assistant_errors,
            },
            "model_tools": {
                "enabled": list(self.config.model_tools.enabled),
                "loaded": model_loaded,
                "errors": model_errors,
            },
        }

    def probe_tools(self, text: str) -> dict[str, Any]:
        text = text.strip()
        assistant_context = ToolContext(
            config=self.config,
            knowledge_retriever=self.knowledge if self.config.knowledge.enabled else None,
        )
        for tool_name in self.config.tools.enabled:
            module_name = _assistant_tool_module_name(tool_name)
            module = importlib.import_module(module_name)
            tool = getattr(module, "create_tool")(assistant_context)
            response = tool.maybe_handle(text)
            if response is None:
                continue
            return {
                "matched": True,
                "tool": response.name,
                "content": response.content,
                "bypass_model": response.bypass_model,
                "context": response.context,
            }
        return {"matched": False, "tool": None, "content": None, "bypass_model": None, "context": None}

    def chat(self, request: ChatRequest) -> ChatResult:
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
        response, model_tool_use = maybe_generate_with_model_tools(
            messages,
            model_tool_registry=_AdHocModelToolRegistry(self.config, self.knowledge),
            llm_generate=lambda model_messages: self._llm_generate(
                model_messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ),
        )
        if model_tool_use is not None:
            model_tool = model_tool_use.name
        clean_response = sanitize_for_speech(response)
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
    require_authentication: bool = True,
) -> FastAPI:
    app = FastAPI(
        title="Arietta Voice API",
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.arietta_service = service or AriettaApiService(config_path)
    security_scheme = HTTPBasic(auto_error=False)

    def require_admin(credentials: HTTPBasicCredentials | None = Depends(security_scheme)) -> str:
        if not require_authentication:
            return DEFAULT_ADMIN_USERNAME
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": f'Basic realm="{AUTH_REALM}"'},
            )
        expected_username = os.environ.get("ARIETTA_ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME)
        expected_password = os.environ.get("ARIETTA_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
        username_ok = secrets.compare_digest(credentials.username, expected_username)
        password_ok = secrets.compare_digest(credentials.password, expected_password)
        if username_ok and password_ok:
            return credentials.username
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": f'Basic realm="{AUTH_REALM}"'},
        )

    def svc() -> AriettaApiService:
        return app.state.arietta_service

    @app.get("/health")
    def public_health() -> dict[str, Any]:
        return svc().health()

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
            service_obj.last_error = f"{type(exc).__name__}: {exc}"
            service_obj.event_logger.log("api_error", route="/api/chat", error=service_obj.last_error)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=service_obj.last_error,
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
  <style>
    :root {
      --bg: #f7f8f5;
      --panel: #ffffff;
      --panel-strong: #f0f4f2;
      --ink: #222520;
      --muted: #667065;
      --line: #d9dfd7;
      --accent: #087f75;
      --accent-strong: #05665d;
      --warn: #a76300;
      --bad: #b42318;
      --good: #197344;
      --shadow: 0 16px 34px rgba(31, 37, 28, 0.10);
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
      background: #fbfcfa;
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
      border-color: #9bc5bd;
      background: #e7f2ef;
      color: #064d47;
      font-weight: 650;
    }
    .sidebar-link {
      display: block;
      margin-top: auto;
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
      background: #2f5f78;
      border-color: #2f5f78;
      color: #ffffff;
      font-weight: 650;
    }
    .button.info:hover { background: #244b5f; }
    .button:disabled,
    .button:disabled:hover {
      background: #eef1ed;
      border-color: var(--line);
      color: #8b9489;
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
      border-radius: 8px;
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
      font-size: 24px;
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
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
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
      background: #ffffff;
      color: var(--ink);
      font: inherit;
      padding: 10px 12px;
      outline: none;
    }
    textarea {
      min-height: 112px;
      resize: vertical;
      line-height: 1.45;
    }
    input:focus, textarea:focus {
      border-color: #67aaa0;
      box-shadow: 0 0 0 3px rgba(8, 127, 117, 0.12);
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
      background: #fafbf8;
    }
    code, pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: 0;
    }
    pre {
      margin: 0;
      padding: 14px;
      background: #20241f;
      color: #f3f7ef;
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
      background: #fbfcfa;
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
      background: #fbfcfa;
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
      background: #fafbf8;
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
    }
    @media (max-width: 620px) {
      main { padding: 18px; }
      .topbar { align-items: flex-start; flex-direction: column; }
      nav { grid-template-columns: 1fr 1fr; }
      .grid { grid-template-columns: 1fr; }
      .form-row { flex-direction: column; }
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
      <a class="sidebar-link" href="https://github.com/robert-mcdermott/arietta-voice" target="_blank" rel="noopener noreferrer">GitHub Repository</a>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h2 id="view-title">Dashboard</h2>
          <div class="status-row"><span id="health-dot" class="dot"></span><span id="health-text">Checking</span><span id="poll-status">Auto-refresh on</span></div>
        </div>
        <button class="button info" id="refresh">Refresh</button>
      </div>

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
            <div class="form-row" style="margin:0">
              <button class="button success" id="runtime-start">Start</button>
              <button class="button danger" id="runtime-stop">Stop</button>
              <button class="button warn" id="runtime-restart">Restart</button>
              <button class="button info" id="runtime-refresh">Refresh</button>
            </div>
          </div>
          <div id="runtime-detail"></div>
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
            <textarea id="chat-message" placeholder="Type a message for the assistant"></textarea>
            <div class="form-row">
              <button class="button primary" id="chat-send">Send</button>
              <button class="button" id="tool-probe">Probe Tools</button>
            </div>
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
        <div class="panel">
          <div class="panel-head"><h3>Registered Tools</h3><button class="button" id="tools-refresh">Refresh</button></div>
          <div id="tools-content"></div>
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
      currentKnowledgeFile: null,
      currentView: "dashboard",
      dashboardLoading: false,
      activeViewLoading: false,
      runtimeActionPending: null,
      logsOffset: 0,
      logsLimit: 50,
      logsTotal: 0,
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
        const body = await response.text();
        throw new Error(`${response.status} ${response.statusText}: ${body}`);
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
        $("#metric-assistant").textContent = currentState.assistant.name;
        $("#metric-wake").textContent = currentState.assistant.wake_enabled ? currentState.assistant.wake_words.join(", ") : "wake disabled";
        const runtime = currentState.runtime;
        $("#metric-runtime").textContent = runtime.state;
        $("#metric-uptime").textContent = runtime.is_running
          ? `pid ${runtime.process_id || "--"} · heartbeat ${runtime.heartbeat_age_seconds ?? 0}s`
          : runtime.note;
        $("#metric-model").textContent = currentState.models.chat_model;
        $("#metric-provider").textContent = currentState.models.chat_provider;
        $("#metric-knowledge").textContent = currentState.knowledge.backend;
        $("#metric-index").textContent = currentState.knowledge.loaded ? "loaded" : currentState.knowledge.directory;
        renderRuntime(runtime);
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
      const assistantRows = payload.assistant_tools.loaded.map((tool) => [
        `<strong>${escapeHtml(tool.name)}</strong>`,
        escapeHtml(tool.configured_name),
        `<code>${escapeHtml(tool.module)}</code>`
      ]);
      const modelRows = payload.model_tools.loaded.map((tool) => [
        `<strong>${escapeHtml(tool.name)}</strong>`,
        escapeHtml(tool.configured_name),
        escapeHtml(tool.description)
      ]);
      const errors = [...payload.assistant_tools.errors, ...payload.model_tools.errors];
      const errorHtml = errors.length ? `<h3>Errors</h3>` + table(["Tool", "Error"], errors.map((err) => [
        escapeHtml(err.configured_name), `<span class="error">${escapeHtml(err.error)}</span>`
      ])) : "";
      $("#tools-content").innerHTML = `
        <h3>Deterministic</h3>
        ${assistantRows.length ? table(["Name", "Config", "Module"], assistantRows) : `<p class="muted">No deterministic tools loaded.</p>`}
        <h3 style="margin-top:18px">Model Requested</h3>
        ${modelRows.length ? table(["Name", "Config", "Description"], modelRows) : `<p class="muted">No model-requested tools loaded.</p>`}
        ${errorHtml}
      `;
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
      const payload = await api(knowledgeFileUrl(path));
      state.currentKnowledgeFile = payload.path;
      $("#knowledge-file-path").value = payload.path;
      $("#knowledge-file-content").value = payload.content;
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
      $("#knowledge-file-path").value = payload.path;
      $("#knowledge-meta").textContent = JSON.stringify(payload, null, 2);
      await loadKnowledgeFiles();
      await loadDashboard();
    }

    async function deleteKnowledgeFile() {
      const path = ($("#knowledge-file-path").value || state.currentKnowledgeFile || "").trim();
      if (!path) return;
      const payload = await api(knowledgeFileUrl(path), { method: "DELETE" });
      state.currentKnowledgeFile = null;
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
      if (!message) return;
      state.chatHistory.push({ role: "user", content: message });
      textarea.value = "";
      renderChat();
      const payload = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          message,
          history: state.chatHistory.filter((item) => item.role === "user" || item.role === "assistant").slice(-10)
        })
      });
      state.chatHistory.push({ role: "assistant", content: payload.response });
      $("#chat-meta").textContent = JSON.stringify(payload, null, 2);
      renderChat();
      await loadDashboard();
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
    $("#knowledge-search").addEventListener("click", () => searchKnowledge().catch((error) => setError(error.message)));
    $("#knowledge-query").addEventListener("keydown", (event) => { if (event.key === "Enter") searchKnowledge().catch((error) => setError(error.message)); });
    $("#knowledge-index").addEventListener("click", () => buildKnowledgeIndex().catch((error) => setError(error.message)));
    $("#knowledge-files-refresh").addEventListener("click", () => loadKnowledgeFiles().catch((error) => setError(error.message)));
    $("#knowledge-file-new").addEventListener("click", newKnowledgeFile);
    $("#knowledge-file-save").addEventListener("click", () => saveKnowledgeFile().catch((error) => setError(error.message)));
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
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        loadDashboard({ silent: true }).catch((error) => setError(error.message));
        refreshActiveView();
      }
    });

    loadDashboard().catch((error) => setError(error.message));
  </script>
</body>
</html>
"""
