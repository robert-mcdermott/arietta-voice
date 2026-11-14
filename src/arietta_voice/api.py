"""Compatibility import path. The console API lives in :mod:`arietta_voice.web`."""

from arietta_voice.web.app import create_app
from arietta_voice.web.schemas import (
    ChatHistoryItem,
    ChatRequest,
    KnowledgeFileWriteRequest,
    KnowledgeSearchRequest,
    ToolCreateRequest,
    ToolEnabledRequest,
    ToolProbeRequest,
    ToolSourceWriteRequest,
    ToolTestRequest,
)
from arietta_voice.web.service import API_VERSION, AUTH_REALM, AriettaApiService

__all__ = [
    "API_VERSION",
    "AUTH_REALM",
    "AriettaApiService",
    "ChatHistoryItem",
    "ChatRequest",
    "KnowledgeFileWriteRequest",
    "KnowledgeSearchRequest",
    "ToolCreateRequest",
    "ToolEnabledRequest",
    "ToolProbeRequest",
    "ToolSourceWriteRequest",
    "ToolTestRequest",
    "create_app",
]
