from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
from typing import Any
import tomllib

from arietta_voice.private_config import local_config_path


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
    content_enabled: bool = False
    max_bytes: int = 5_000_000
    backup_count: int = 3


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
class AdminConfig:
    allow_source_editing: bool = False


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
    admin: AdminConfig = field(default_factory=AdminConfig)


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
        if not candidate.exists() and not local_config_path(candidate).exists():
            return base
        path = candidate
    else:
        path = path.resolve()
        root_hint = path.parent.parent if path.parent.name == "config" else path.parent
        base = default_config(root_hint)

    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not path.exists() and not local_config_path(path).exists():
        raise FileNotFoundError(path)
    _validate_toml_types(raw, base)
    override_path = local_config_path(path)
    if override_path != path and override_path.exists():
        overrides = tomllib.loads(override_path.read_text(encoding="utf-8"))
        _validate_toml_types(overrides, base)
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
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
    admin = raw.get("admin", {})

    config = AppConfig(
        project_root=base.project_root,
        admin=AdminConfig(allow_source_editing=admin.get("allow_source_editing", False)),
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
            backend=str(wake_word.get("backend", base.wake_word.backend)).strip(),
            phrases=_merge_tuple(
                wake_word.get("phrases"),
                base.wake_word.phrases,
                allow_empty=True,
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
            chat_provider=str(models.get("chat_provider", base.models.chat_provider)).strip(),
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
            content_enabled=logging_cfg.get("content_enabled", base.logging.content_enabled),
            max_bytes=logging_cfg.get("max_bytes", base.logging.max_bytes),
            backup_count=logging_cfg.get("backup_count", base.logging.backup_count),
        ),
        knowledge=KnowledgeConfig(
            enabled=bool(knowledge.get("enabled", base.knowledge.enabled)),
            backend=str(knowledge.get("backend", base.knowledge.backend)).strip(),
            directory=_resolve_path(base_dir, knowledge.get("directory", base.knowledge.directory)),
            index_directory=_resolve_path(
                base_dir,
                knowledge.get("index_directory", base.knowledge.index_directory),
            ),
            embedding_model=str(
                knowledge.get("embedding_model", base.knowledge.embedding_model)
            ).strip(),
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
    return validate_config(config)


def _validate_toml_types(raw: dict[str, Any], defaults: AppConfig) -> None:
    """Reject coercions such as bool('false') and int(1.5) before loading."""
    sections = {item.name for item in fields(defaults)} - {"project_root"}
    for section, values in raw.items():
        if section not in sections:
            raise ValueError(f"Unknown configuration section: {section}")
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a TOML table")
        section_defaults = getattr(defaults, section)
        keys = {item.name for item in fields(section_defaults)}
        for key, value in values.items():
            name = f"{section}.{key}"
            if key not in keys:
                raise ValueError(f"Unknown configuration setting: {name}")
            default = getattr(section_defaults, key)
            if isinstance(default, bool):
                valid, expected = type(value) is bool, "a boolean"
            elif isinstance(default, int):
                valid, expected = type(value) is int, "an integer"
            elif isinstance(default, float):
                valid, expected = type(value) in (int, float), "a number"
            elif isinstance(default, tuple):
                valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
                expected = "an array of strings"
            else:
                valid, expected = isinstance(value, str), "a string"
            if not valid:
                raise ValueError(f"{name} must be {expected}")


def validate_config(config: AppConfig) -> AppConfig:
    """Validate runtime assumptions, including values supplied by CLI overrides."""
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def number(name: str, value: float, minimum: float, maximum: float = math.inf) -> None:
        require(math.isfinite(value) and minimum <= value <= maximum,
                f"{name} must be finite and between {minimum:g} and {maximum:g}")

    require(config.assistant.max_history_turns >= 0, "assistant.max_history_turns must be >= 0")
    require(bool(config.assistant.name.strip()), "assistant.name must not be blank")
    require(config.wake_word.backend == "transcript", "wake_word.backend must be 'transcript'")
    require(not config.wake_word.enabled or bool(config.wake_word.phrases),
            "wake_word.phrases must not be empty when wake word detection is enabled")
    require(all(any(char.isalnum() for char in phrase) for phrase in config.wake_word.phrases),
            "wake_word.phrases must contain words or numbers")
    number("wake_word.session_timeout_s", config.wake_word.session_timeout_s, 0.001)
    # The current VAD, Smart Turn, and AEC paths share this input format.
    require(config.audio.sample_rate == 16000, "audio.sample_rate must be 16000")
    require(config.audio.chunk_samples == 512, "audio.chunk_samples must be 512")
    require(config.audio.silence_ms > 0, "audio.silence_ms must be > 0")
    require(config.audio.barge_in_threshold_chunks > 0, "audio.barge_in_threshold_chunks must be > 0")
    number("audio.tts_speed", config.audio.tts_speed, 0.001)
    require(bool(config.audio.voice.strip()), "audio.voice must not be blank")
    require(config.models.chat_provider in {"local", "bedrock"},
            "models.chat_provider must be 'local' or 'bedrock'")
    require(bool(config.models.chat_model.strip()), "models.chat_model must not be blank")
    require(bool(config.models.stt_language.strip()), "models.stt_language must not be blank")
    number("models.temperature", config.models.temperature, 0, 2)
    require(config.models.max_tokens > 0, "models.max_tokens must be > 0")
    require(config.memory.consolidate_every_turns > 0, "memory.consolidate_every_turns must be > 0")
    require(config.logging.max_bytes >= 1024, "logging.max_bytes must be >= 1024")
    require(config.logging.backup_count >= 0, "logging.backup_count must be >= 0")
    require(config.knowledge.backend in {"keyword", "vector", "hybrid"},
            "knowledge.backend must be 'keyword', 'vector', or 'hybrid'")
    require(bool(config.knowledge.embedding_model.strip()), "knowledge.embedding_model must not be blank")
    require(config.knowledge.max_matches > 0, "knowledge.max_matches must be > 0")
    require(config.knowledge.snippet_char_limit > 0, "knowledge.snippet_char_limit must be > 0")
    number("knowledge.min_score", config.knowledge.min_score, 0)
    number("knowledge.hybrid_alpha", config.knowledge.hybrid_alpha, 0, 1)
    number("knowledge.min_vector_score", config.knowledge.min_vector_score, -1, 1)
    return config


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
