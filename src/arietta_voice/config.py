from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import tomllib


DEFAULT_GREETINGS = (
    "Yes?",
    "I'm here.",
    "How can I help?",
)

DEFAULT_GOODBYES = (
    "Goodbye.",
    "See you later.",
    "Talk to you soon.",
)

DEFAULT_WAKE_WORDS = ("bridget",)

DEFAULT_ENABLED_TOOLS = (
    "local_time",
    "assistant_identity",
    "knowledge_retrieval",
)

DEFAULT_ENABLED_MODEL_TOOLS = (
    "search_knowledge",
    "system_info",
)


def locate_project_root(start: Path | None = None) -> Path:
    if start is not None:
        seed = Path(start).expanduser()
        return (seed if seed.is_dir() else seed.parent).resolve()

    for seed in (Path.cwd(), Path(__file__).resolve()):
        current = seed if seed.is_dir() else seed.parent
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").exists():
                return candidate
    return Path.cwd()


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _merge_tuple(
    values: list[str] | tuple[str, ...] | None,
    fallback: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if values is None:
        return fallback
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if normalized or allow_empty:
        return normalized
    return fallback


@dataclass(slots=True)
class AssistantProfile:
    name: str = "Bridget"
    identity: str = (
        "I'm Bridget, a local voice-first assistant built with Arietta Voice. "
        "I can answer questions from my local knowledge base, use tools, and optionally "
        "route chat turns through AWS Bedrock."
    )
    greeting_variants: tuple[str, ...] = field(default_factory=lambda: DEFAULT_GREETINGS)
    goodbye_variants: tuple[str, ...] = field(default_factory=lambda: DEFAULT_GOODBYES)
    max_history_turns: int = 8


@dataclass(slots=True)
class WakeWordConfig:
    enabled: bool = True
    backend: str = "transcript"
    phrases: tuple[str, ...] = field(default_factory=lambda: DEFAULT_WAKE_WORDS)
    acknowledgement: str = "Yes?"
    session_timeout_s: float = 20.0
    allow_inline_commands: bool = True


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    chunk_samples: int = 512
    silence_ms: int = 700
    tts_enabled: bool = True
    smart_turn_enabled: bool = True
    aec_enabled: bool = True
    chime_enabled: bool = True
    voice: str = "af_heart"
    tts_speed: float = 1.0
    barge_in_threshold_chunks: int = 5


@dataclass(slots=True)
class ModelConfig:
    chat_provider: str = "local"
    chat_model: str = "mlx-community/gemma-4-E4B-it-4bit"
    bedrock_region: str | None = None
    bedrock_profile: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    stt_language: str = "en"
    temperature: float = 0.7
    max_tokens: int = 220


@dataclass(slots=True)
class PromptConfig:
    soul_path: Path
    memory_path: Path


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = False
    consolidate_every_turns: int = 5


@dataclass(slots=True)
class LoggingConfig:
    directory: Path
    events_file: Path


@dataclass(slots=True)
class KnowledgeConfig:
    enabled: bool = True
    backend: str = "keyword"
    directory: Path = field(default_factory=lambda: Path("knowledge"))
    index_directory: Path = field(default_factory=lambda: Path(".knowledge_index"))
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_matches: int = 3
    snippet_char_limit: int = 420
    min_score: float = 6.0
    hybrid_alpha: float = 0.55
    min_vector_score: float = 0.3


@dataclass(slots=True)
class ToolsConfig:
    enabled: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ENABLED_TOOLS)


@dataclass(slots=True)
class ModelToolsConfig:
    enabled: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ENABLED_MODEL_TOOLS)


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    assistant: AssistantProfile
    wake_word: WakeWordConfig
    audio: AudioConfig
    models: ModelConfig
    prompts: PromptConfig
    memory: MemoryConfig
    logging: LoggingConfig
    knowledge: KnowledgeConfig
    tools: ToolsConfig
    model_tools: ModelToolsConfig


def default_config(project_root: Path | None = None) -> AppConfig:
    root = locate_project_root(project_root)
    config_dir = root / "config"
    log_dir = root / "logs"
    return AppConfig(
        project_root=root,
        assistant=AssistantProfile(),
        wake_word=WakeWordConfig(),
        audio=AudioConfig(),
        models=ModelConfig(),
        prompts=PromptConfig(
            soul_path=config_dir / "SOUL.md",
            memory_path=config_dir / "MEMORY.md",
        ),
        memory=MemoryConfig(),
        logging=LoggingConfig(
            directory=log_dir,
            events_file=log_dir / "events.jsonl",
        ),
        knowledge=KnowledgeConfig(
            directory=root / "knowledge",
            index_directory=root / ".knowledge_index",
        ),
        tools=ToolsConfig(),
        model_tools=ModelToolsConfig(),
    )


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        base = default_config()
        candidate = base.project_root / "config" / "arietta_voice.toml"
        if not candidate.exists():
            return base
        path = candidate
    else:
        path = path.resolve()
        root_hint = path.parent.parent if path.parent.name == "config" else path.parent
        base = default_config(root_hint)

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    assistant = raw.get("assistant", {})
    wake_word = raw.get("wake_word", {})
    audio = raw.get("audio", {})
    models = raw.get("models", {})
    prompts = raw.get("prompts", {})
    memory = raw.get("memory", {})
    logging_cfg = raw.get("logging", {})
    knowledge = raw.get("knowledge", {})
    tools = raw.get("tools", {})
    model_tools = raw.get("model_tools", {})

    return AppConfig(
        project_root=base.project_root,
        assistant=AssistantProfile(
            name=assistant.get("name", base.assistant.name),
            identity=assistant.get("identity", base.assistant.identity),
            greeting_variants=_merge_tuple(
                assistant.get("greeting_variants"),
                base.assistant.greeting_variants,
            ),
            goodbye_variants=_merge_tuple(
                assistant.get("goodbye_variants"),
                base.assistant.goodbye_variants,
            ),
            max_history_turns=int(assistant.get("max_history_turns", base.assistant.max_history_turns)),
        ),
        wake_word=WakeWordConfig(
            enabled=bool(wake_word.get("enabled", base.wake_word.enabled)),
            backend=str(wake_word.get("backend", base.wake_word.backend)).strip()
            or base.wake_word.backend,
            phrases=_merge_tuple(
                wake_word.get("phrases"),
                base.wake_word.phrases,
            ),
            acknowledgement=str(
                wake_word.get("acknowledgement", base.wake_word.acknowledgement)
            ),
            session_timeout_s=float(
                wake_word.get("session_timeout_s", base.wake_word.session_timeout_s)
            ),
            allow_inline_commands=bool(
                wake_word.get("allow_inline_commands", base.wake_word.allow_inline_commands)
            ),
        ),
        audio=AudioConfig(
            sample_rate=int(audio.get("sample_rate", base.audio.sample_rate)),
            chunk_samples=int(audio.get("chunk_samples", base.audio.chunk_samples)),
            silence_ms=int(audio.get("silence_ms", base.audio.silence_ms)),
            tts_enabled=bool(audio.get("tts_enabled", base.audio.tts_enabled)),
            smart_turn_enabled=bool(audio.get("smart_turn_enabled", base.audio.smart_turn_enabled)),
            aec_enabled=bool(audio.get("aec_enabled", base.audio.aec_enabled)),
            chime_enabled=bool(audio.get("chime_enabled", base.audio.chime_enabled)),
            voice=str(audio.get("voice", base.audio.voice)),
            tts_speed=float(audio.get("tts_speed", base.audio.tts_speed)),
            barge_in_threshold_chunks=int(
                audio.get("barge_in_threshold_chunks", base.audio.barge_in_threshold_chunks)
            ),
        ),
        models=ModelConfig(
            chat_provider=str(models.get("chat_provider", base.models.chat_provider)).strip()
            or base.models.chat_provider,
            chat_model=str(models.get("chat_model", base.models.chat_model)),
            bedrock_region=str(models.get("bedrock_region", "")).strip() or base.models.bedrock_region,
            bedrock_profile=str(models.get("bedrock_profile", "")).strip()
            or base.models.bedrock_profile,
            aws_access_key_id=str(models.get("aws_access_key_id", "")).strip()
            or base.models.aws_access_key_id,
            aws_secret_access_key=str(models.get("aws_secret_access_key", "")).strip()
            or base.models.aws_secret_access_key,
            aws_session_token=str(models.get("aws_session_token", "")).strip()
            or base.models.aws_session_token,
            stt_language=str(models.get("stt_language", base.models.stt_language)),
            temperature=float(models.get("temperature", base.models.temperature)),
            max_tokens=int(models.get("max_tokens", base.models.max_tokens)),
        ),
        prompts=PromptConfig(
            soul_path=_resolve_path(base_dir, prompts.get("soul_path", base.prompts.soul_path)),
            memory_path=_resolve_path(base_dir, prompts.get("memory_path", base.prompts.memory_path)),
        ),
        memory=MemoryConfig(
            enabled=bool(memory.get("enabled", base.memory.enabled)),
            consolidate_every_turns=int(
                memory.get("consolidate_every_turns", base.memory.consolidate_every_turns)
            ),
        ),
        logging=LoggingConfig(
            directory=_resolve_path(base_dir, logging_cfg.get("directory", base.logging.directory)),
            events_file=_resolve_path(base_dir, logging_cfg.get("events_file", base.logging.events_file)),
        ),
        knowledge=KnowledgeConfig(
            enabled=bool(knowledge.get("enabled", base.knowledge.enabled)),
            backend=str(knowledge.get("backend", base.knowledge.backend)).strip()
            or base.knowledge.backend,
            directory=_resolve_path(base_dir, knowledge.get("directory", base.knowledge.directory)),
            index_directory=_resolve_path(
                base_dir,
                knowledge.get("index_directory", base.knowledge.index_directory),
            ),
            embedding_model=str(
                knowledge.get("embedding_model", base.knowledge.embedding_model)
            ).strip()
            or base.knowledge.embedding_model,
            max_matches=int(knowledge.get("max_matches", base.knowledge.max_matches)),
            snippet_char_limit=int(
                knowledge.get("snippet_char_limit", base.knowledge.snippet_char_limit)
            ),
            min_score=float(knowledge.get("min_score", base.knowledge.min_score)),
            hybrid_alpha=float(knowledge.get("hybrid_alpha", base.knowledge.hybrid_alpha)),
            min_vector_score=float(
                knowledge.get("min_vector_score", base.knowledge.min_vector_score)
            ),
        ),
        tools=ToolsConfig(
            enabled=_merge_tuple(
                tools.get("enabled"),
                base.tools.enabled,
                allow_empty=True,
            ),
        ),
        model_tools=ModelToolsConfig(
            enabled=_merge_tuple(
                model_tools.get("enabled"),
                base.model_tools.enabled,
                allow_empty=True,
            ),
        ),
    )


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    data = _stringify_paths(asdict(config))
    return _redact_sensitive_values(data)


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_stringify_paths(item) for item in value]
    return value


def _redact_sensitive_values(value: Any) -> Any:
    secret_keys = {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"}
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in secret_keys and item:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_sensitive_values(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    return value
