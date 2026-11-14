"""HTTP routes for the console API and the packaged console frontend."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware

from arietta_voice.admin_auth import load_admin_credentials
from arietta_voice.web.schemas import (
    ChatRequest,
    KnowledgeFileWriteRequest,
    KnowledgeSearchRequest,
    SettingsUpdateRequest,
    TimerCreateRequest,
    ToolCreateRequest,
    ToolEnabledRequest,
    ToolProbeRequest,
    ToolSourceWriteRequest,
    ToolTestRequest,
)
from arietta_voice.web.service import API_VERSION, AUTH_REALM, AriettaApiService
from arietta_voice.worker import (
    WorkerBusy,
    WorkerCancelled,
    WorkerTimeout,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
# The console ships no inline script or style, so nothing else may run or load.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


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
            if (origin is not None and origin != expected_origin) or request.headers.get(
                "sec-fetch-site"
            ) == "cross-site":
                return JSONResponse(
                    {"detail": "Cross-origin admin actions are not allowed"}, status_code=403
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
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
            raise HTTPException(
                status_code=403,
                detail="Source editing is disabled. Set admin.allow_source_editing=true in private config and restart the API to enable it.",
            )
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
            service_obj.event_logger.log(
                "api_error", route="/api/runtime/start", error=service_obj.last_error
            )
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
            service_obj.event_logger.log(
                "api_error", route="/api/runtime/restart", error=service_obj.last_error
            )
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

    @app.post("/api/tools/instructions")
    def api_tool_instructions(_user: str = Depends(require_admin)) -> dict[str, Any]:
        try:
            return svc().tool_instructions()
        except Exception as exc:
            service_obj = svc()
            service_obj.last_error = f"{type(exc).__name__}: {exc}"
            service_obj.event_logger.log(
                "api_error", route="/api/tools/instructions", error=service_obj.last_error
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=service_obj.last_error
            ) from exc

    @app.post("/api/tools/{tool_name}/test")
    def api_test_tool(
        tool_name: str,
        payload: ToolTestRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        if (payload.text is None) == (payload.arguments is None):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Send either text or arguments",
            )
        try:
            return svc().test_tool(tool_name, text=payload.text, arguments=payload.arguments)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/tools/{tool_name}/validate")
    def api_validate_tool(
        tool_name: str,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        return svc().validate_tool(tool_name)

    @app.get("/api/settings")
    def api_settings(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().settings()

    @app.put("/api/settings")
    def api_update_settings(
        payload: SettingsUpdateRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().update_settings(payload.changes)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/api/timers")
    def api_timers(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().timers()

    @app.post("/api/timers")
    def api_add_timer(
        payload: TimerCreateRequest,
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().add_timer(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/timers/stop")
    def api_stop_timers(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return svc().stop_ringing()

    @app.post("/api/timers/{alert_id}/cancel")
    def api_cancel_timer(
        alert_id: str = ApiPath(pattern=r"^[0-9a-f]{8}$"),
        _user: str = Depends(require_admin),
    ) -> dict[str, Any]:
        try:
            return svc().cancel_timer(alert_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="No active timer or reminder"
            ) from exc

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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge file not found"
            ) from exc
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge file not found"
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/api/chat/{request_id}/cancel")
    def cancel_api_chat(
        request_id: str = ApiPath(pattern=r"^[A-Za-z0-9_-]{1,64}$"),
        _user: str = Depends(require_admin),
    ) -> dict:
        return {"cancelled": svc().cancel_chat(request_id)}

    def chat_failure(exc: Exception, route: str) -> tuple[int, str]:
        service_obj = svc()
        service_obj.last_error = type(exc).__name__
        service_obj.event_logger.log("api_error", route=route, error_type=service_obj.last_error)
        if isinstance(exc, WorkerBusy):
            return 503, "Inference is busy. Wait for the current request to finish."
        if isinstance(exc, WorkerCancelled):
            return 409, "Turn cancelled. Any external action already completed is not undone."
        if isinstance(exc, WorkerTimeout):
            return (
                504,
                "The turn timed out and was stopped. No automatic retry was made; check any "
                "external action before repeating it.",
            )
        return 502, "The model or tool failed. The next request will use a fresh worker."

    @app.post("/api/chat/stream")
    def api_chat_stream(payload: ChatRequest, _user: str = Depends(require_admin)):
        """The answer as server-sent events: ``clause`` events, then ``done`` or ``error``."""
        try:
            events = svc().chat_stream(payload)
        except Exception as exc:
            code, detail = chat_failure(exc, "/api/chat/stream")
            raise HTTPException(status_code=code, detail=detail) from exc

        def sse():
            try:
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                code, detail = chat_failure(exc, "/api/chat/stream")
                yield f"data: {json.dumps({'kind': 'error', 'status': code, 'detail': detail})}\n\n"
            finally:
                events.close()

        # The background task frees the turn even if the client left before the
        # first event, when sse() never starts and its cleanup never runs.
        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
            background=BackgroundTask(events.close),
        )

    @app.post("/api/chat")
    def api_chat(payload: ChatRequest, _user: str = Depends(require_admin)) -> dict[str, Any]:
        try:
            result = svc().chat(payload)
        except Exception as exc:
            code, detail = chat_failure(exc, "/api/chat")
            raise HTTPException(status_code=code, detail=detail) from exc
        return {
            "response": result.response,
            "source": result.source,
            "used_tool": result.used_tool,
            "model_tool": result.model_tool,
        }

    @app.get("/api/openapi.json")
    def api_openapi(_user: str = Depends(require_admin)) -> dict[str, Any]:
        return app.openapi()

    @app.get("/", include_in_schema=False)
    @app.get("/admin", include_in_schema=False)
    def admin_console(_user: str = Depends(require_admin)) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/static/{asset_path:path}", include_in_schema=False)
    def static_asset(asset_path: str, _user: str = Depends(require_admin)) -> FileResponse:
        target = (STATIC_DIR / asset_path).resolve()
        media_type = STATIC_MEDIA_TYPES.get(target.suffix)
        if media_type is None or not target.is_relative_to(STATIC_DIR) or not target.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
        return FileResponse(target, media_type=media_type)

    return app
