from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from arietta_voice.config import config_to_dict, load_config
from arietta_voice.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arietta Voice")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the voice assistant")
    run_parser.add_argument("--config", type=Path, help="Path to a TOML config file")
    run_parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--tts", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--aec", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--smart-turn", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--chime", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--wake", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--voice", help="Kokoro voice id")
    run_parser.add_argument(
        "--model",
        help="Chat model id. Use an MLX model for local chat or a Bedrock model id when chat_provider=bedrock.",
    )
    run_parser.add_argument("--chat-provider", choices=("local", "bedrock"), help="Chat provider")
    run_parser.add_argument(
        "--wake-word",
        action="append",
        dest="wake_words",
        help="Override the configured wake word phrase. Repeat to allow multiple phrases.",
    )
    run_parser.add_argument("--session-timeout", type=float, help="Seconds to keep a session open after activity")
    run_parser.add_argument("--record", nargs="?", const="", metavar="FILE")

    config_parser = subparsers.add_parser("config", help="Print the resolved config")
    config_parser.add_argument("--config", type=Path, help="Path to a TOML config file")

    doctor_parser = subparsers.add_parser("doctor", help="Run startup diagnostics")
    doctor_parser.add_argument("--config", type=Path, help="Path to a TOML config file")
    doctor_parser.add_argument("--json", action="store_true", help="Print JSON output")

    knowledge_parser = subparsers.add_parser("knowledge-search", help="Search the local knowledge base")
    knowledge_parser.add_argument("query", nargs="+", help="Query text to search for")
    knowledge_parser.add_argument("--config", type=Path, help="Path to a TOML config file")
    knowledge_parser.add_argument("--json", action="store_true", help="Print JSON output")

    knowledge_index_parser = subparsers.add_parser(
        "knowledge-index",
        help="Build or refresh the local semantic knowledge index",
    )
    knowledge_index_parser.add_argument("--config", type=Path, help="Path to a TOML config file")
    knowledge_index_parser.add_argument("--json", action="store_true", help="Print JSON output")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP API and admin console")
    serve_parser.add_argument("--config", type=Path, help="Path to a TOML config file")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8765, help="Bind port")
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Uvicorn log level",
    )

    subparsers.add_parser("devices", help="List audio devices")
    return parser


def _apply_runtime_overrides(config, args):
    if args.memory is not None:
        config.memory.enabled = args.memory
    if args.tts is not None:
        config.audio.tts_enabled = args.tts
    if args.aec is not None:
        config.audio.aec_enabled = args.aec
    if args.smart_turn is not None:
        config.audio.smart_turn_enabled = args.smart_turn
    if args.chime is not None:
        config.audio.chime_enabled = args.chime
    if args.wake is not None:
        config.wake_word.enabled = args.wake
    if args.voice:
        config.audio.voice = args.voice
    if args.model:
        config.models.chat_model = args.model
    if args.chat_provider:
        config.models.chat_provider = args.chat_provider
    if getattr(args, "wake_words", None):
        config.wake_word.phrases = tuple(word.strip() for word in args.wake_words if word.strip())
    if getattr(args, "session_timeout", None) is not None:
        config.wake_word.session_timeout_s = float(args.session_timeout)
    return config


def _resolve_record_path(value: str | None) -> Path | None:
    if value is None:
        return None
    if value == "":
        return Path("tmp") / "arietta-recording.wav"
    return Path(value)


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    if argv and argv[0] not in {
        "run",
        "config",
        "doctor",
        "knowledge-search",
        "knowledge-index",
        "serve",
        "devices",
        "-h",
        "--help",
    }:
        argv = ["run", *argv]
    elif not argv:
        argv = ["run"]
    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "devices":
        from arietta_voice.diagnostics import list_audio_devices

        print(json.dumps(list_audio_devices(), indent=2))
        return 0

    if args.command == "config":
        config = load_config(args.config)
        print(json.dumps(config_to_dict(config), indent=2))
        return 0

    if args.command == "doctor":
        from arietta_voice.diagnostics import run_startup_diagnostics

        config = load_config(args.config)
        report = run_startup_diagnostics(config)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "knowledge-search":
        from arietta_voice.knowledge import KnowledgeRetriever

        config = load_config(args.config)
        retriever = KnowledgeRetriever(config.knowledge)
        result = retriever.search(" ".join(args.query))
        payload = (
            {
                "enabled": config.knowledge.enabled,
                "backend": config.knowledge.backend,
                "directory": str(config.knowledge.directory),
                "matches": [],
            }
            if result is None
            else {
                "enabled": config.knowledge.enabled,
                "backend": result.backend,
                "directory": str(config.knowledge.directory),
                "matches": [
                    {
                        "source": str(match.source_path),
                        "title": match.title,
                        "section": match.section_title,
                        "score": match.score,
                        "snippet": match.snippet,
                    }
                    for match in result.matches
                ],
            }
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "knowledge-index":
        from arietta_voice.knowledge import build_knowledge_index

        config = load_config(args.config)
        payload = build_knowledge_index(config.knowledge)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        from arietta_voice.api import create_app

        app = create_app(args.config)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return 0

    from arietta_voice.runtime import AriettaVoiceRuntime, RuntimeOptions

    config = load_config(args.config if hasattr(args, "config") else None)
    config = _apply_runtime_overrides(config, args)
    runtime = AriettaVoiceRuntime(
        config,
        RuntimeOptions(record_path=_resolve_record_path(getattr(args, "record", None))),
    )
    runtime.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
