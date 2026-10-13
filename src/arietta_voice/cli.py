from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys
import warnings

from arietta_voice import __version__
from arietta_voice.config import config_to_dict, load_config, validate_config
from arietta_voice.logging_utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arietta Voice")
    parser.add_argument("--version", action="version", version=f"arietta-voice {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the voice assistant")
    run_parser.add_argument("--version", action="version", version=f"arietta-voice {__version__}")
    run_parser.add_argument("--config", type=Path, help="Path to a TOML config file")
    run_parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--tts", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--aec", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--smart-turn", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--chime", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--wake", action=argparse.BooleanOptionalAction, default=None)
    run_parser.add_argument("--voice", help="Kokoro voice id")
    run_parser.add_argument("--input-device", help="Exact device name or HOST::NAME from devices; empty uses system default")
    run_parser.add_argument("--output-device", help="Exact device name or HOST::NAME from devices; empty uses system default")
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
    private_parser = subparsers.add_parser("config-init", help="Create private config/persona copies without changing originals")
    private_parser.add_argument("--config", type=Path, help="Base TOML configuration to preserve")
    admin_parser = subparsers.add_parser("admin-setup", help="Set private admin credentials")
    admin_parser.add_argument("--config", type=Path, help="Base TOML configuration used by serve")
    admin_parser.add_argument("--username", default="admin")
    admin_parser.add_argument("--replace", action="store_true", help="Rotate existing credentials")
    return parser


def _apply_runtime_overrides(config, args):
    for name in ('input_device', 'output_device'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config.audio, name, value.strip() or None)
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
    return validate_config(config)


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
        "config-init",
        "admin-setup",
        "-h",
        "--help",
        "--version",
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

    try:
        config = load_config(args.config)
        if args.command == "run":
            config = _apply_runtime_overrides(config, args)
    except (ValueError, OSError) as exc:
        parser.error(f"Invalid configuration: {exc}")

    if args.command == "config":
        print(json.dumps(config_to_dict(config), indent=2))
        return 0

    if args.command in {"config-init", "admin-setup"}:
        base_path = args.config or config.project_root / "config" / "arietta_voice.toml"
        try:
            if args.command == "config-init":
                from arietta_voice.private_config import initialize_private_config

                print(json.dumps(initialize_private_config(base_path), indent=2))
            else:
                from arietta_voice.admin_auth import save_admin_credentials

                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    password = getpass.getpass("New admin password (at least 12 characters): ")
                    repeated = getpass.getpass("Repeat password: ")
                if password != repeated:
                    parser.error("Passwords do not match; credentials were not changed")
                saved = save_admin_credentials(base_path, args.username, password, replace=args.replace)
                print(f"Admin credentials saved to {saved}. Restart the API to apply them.")
        except (EOFError, getpass.GetPassWarning):
            parser.error("Run admin-setup in an interactive terminal with hidden password input")
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        return 0

    if args.command == "doctor":
        from arietta_voice.diagnostics import run_startup_diagnostics

        report = run_startup_diagnostics(config)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "knowledge-search":
        from arietta_voice.knowledge import KnowledgeRetriever

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

        payload = build_knowledge_index(config.knowledge)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "serve":
        import uvicorn

        from arietta_voice.api import create_app

        try:
            app = create_app(args.config)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return 0

    from arietta_voice.runtime import AriettaVoiceRuntime, RuntimeOptions

    try:
        runtime = AriettaVoiceRuntime(
            config,
            RuntimeOptions(record_path=_resolve_record_path(getattr(args, "record", None))),
        )
        runtime.run()
    except Exception as exc:
        # Model/plugin exceptions can contain prompts or tool arguments. Keep
        # those out of redirected process logs unless content was opted in.
        logger = configure_logging()
        if config.logging.content_enabled:
            logger.exception("Voice runtime failed")
        else:
            logger.error(
                "Voice runtime failed (%s). Enable logging.content_enabled for error details.",
                type(exc).__name__,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
