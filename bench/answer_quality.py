"""Answer quality with the real chat model: refusals, wrong facts, and first-word time.

Asks every question in evals/answer_questions.json through the conversation
service (the same path as voice and console chat, tools included) and reports
how many answers decline ("I don't have that information"), how many facts
with a known answer are right, and the median time to the first spoken clause.
Silent, and nothing leaves the machine with the default local model.

By default it runs an isolated copy of the shipped setup (the tracked persona,
the two public starter notes and the default tools) and never reads private
settings or knowledge. Pass --config to measure a particular setup, such as
your own private one.

Each question is asked on its own unless --conversation groups them, carrying
the history as voice does. With made-up household notes, that measures answers
from notes in the middle of a conversation:

uv run --locked bench/answer_quality.py --rounds 2
uv run --locked bench/answer_quality.py --config config/arietta_voice.toml --questions mine.json
uv run --locked bench/answer_quality.py --notes evals/fixture_notes \\
    --questions evals/notes_questions.json --conversation 6 --rounds 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "answer_questions.json"
PUBLIC_NOTES = tuple(ROOT / "knowledge" / name for name in ("README.md", "getting-started.md"))


def is_refusal(answer: str) -> bool:
    from arietta_voice.answers import declines

    return declines(answer)


def is_correct(answer: str, expected: str | None) -> bool | None:
    """True or False for a question with a known answer (a regular expression), else None."""
    if expected is None:
        return None
    return bool(re.search(expected, answer, re.IGNORECASE))


def _median(values) -> float | None:
    values = [value for value in values if value is not None]
    return round(statistics.median(values), 2) if values else None


def summarize(rows: list[dict]) -> dict:
    facts = [row for row in rows if row["correct"] is not None]
    summary = {
        "answers": len(rows),
        "refused": sum(row["refused"] for row in rows),
        "facts": len(facts),
        "facts_right": sum(row["correct"] for row in facts),
        "median_first_s": _median(row["first_s"] for row in rows),
        "smarter_answers": sum(1 for row in rows if row.get("smarter")),
    }
    if any(row.get("notes") for row in rows):
        summary["median_first_with_notes_s"] = _median(
            row["first_s"] for row in rows if row.get("notes")
        )
        summary["median_first_without_notes_s"] = _median(
            row["first_s"] for row in rows if not row.get("notes")
        )
    return summary


def isolated_config(directory: Path, notes=PUBLIC_NOTES):
    """The shipped setup in a scratch folder: tracked persona and public or made-up notes."""
    from arietta_voice.config import default_config, validate_config

    config = default_config(directory)
    config.prompts.soul_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "config" / "SOUL.md", config.prompts.soul_path)
    config.knowledge.directory.mkdir(parents=True, exist_ok=True)
    for path in notes:
        shutil.copy(path, config.knowledge.directory / path.name)
    config.memory.enabled = False
    config.logging.content_enabled = False
    return validate_config(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--json", type=Path, help="write every answer here")
    parser.add_argument(
        "--smarter-model", help="also enable this smarter model, such as gemma4:31b-cloud"
    )
    parser.add_argument("--smarter-provider", default="ollama", choices=("ollama", "bedrock"))
    parser.add_argument("--smarter-when", default="unsure", choices=("unsure", "always"))
    parser.add_argument(
        "--notes", type=Path, help="a folder of Markdown notes to use instead of the public ones"
    )
    parser.add_argument(
        "--conversation",
        type=int,
        default=1,
        help="ask this many questions in a row in one conversation, with history",
    )
    args = parser.parse_args()
    if args.notes is not None and args.config is not None:
        parser.error("--notes applies only to the isolated setup, not with --config")
    if not 1 <= args.conversation <= 20:
        parser.error("--conversation must be 1..20")

    from arietta_voice.config import load_config
    from arietta_voice.conversation import ConversationService

    scratch = None
    if args.config is None:
        scratch = tempfile.TemporaryDirectory(prefix="answer-quality-")
        notes = PUBLIC_NOTES if args.notes is None else tuple(sorted(args.notes.glob("*.md")))
        config = isolated_config(Path(scratch.name), notes)
    else:
        config = load_config(args.config)
    config.mcp.enabled = False  # Measure the model, not whatever servers are connected.
    if args.smarter_model:
        config.smarter_model.enabled = True
        config.smarter_model.model = args.smarter_model
        config.smarter_model.provider = args.smarter_provider
        config.smarter_model.when = args.smarter_when
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    service = ConversationService(config)
    service.knowledge.build_index_if_needed()  # As the assistant does when it starts.
    list(service.stream(message="Hello", use_tools=not args.no_tools))  # Load and warm up.
    rows = []
    for _ in range(args.rounds):
        history: list[dict] = []
        for index, item in enumerate(questions):
            if index % args.conversation == 0:
                history = []  # A new conversation.
            started, first, events = time.monotonic(), None, []
            stream = service.stream(message=item["q"], history=history, use_tools=not args.no_tools)
            for event in stream:
                if event["kind"] == "clause" and first is None:
                    first = round(time.monotonic() - started, 2)
                events.append(event)
            result = events[-1]["result"]
            answer = result["response"]
            rows.append(
                {
                    "question": item["q"],
                    "answer": answer,
                    "refused": is_refusal(answer),
                    "correct": is_correct(answer, item.get("a")),
                    "first_s": first,
                    "tool": result.get("model_tool") or result.get("used_tool"),
                    "notes": result.get("used_tool") == "search_knowledge",
                    "history_turns": len(history) // 2,
                    "smarter": result.get("smarter"),
                }
            )
            history += [
                {"role": "user", "content": item["q"]},
                {"role": "assistant", "content": answer},
            ]
    summary = summarize(rows)
    print(json.dumps(summary))
    for row in rows:
        if row["refused"] or row["correct"] is False:
            label = "declined" if row["refused"] else "wrong"
            print(f"  {label:8} {row['question']!r} -> {row['answer'][:100]!r}")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "answers": rows}, indent=1))
    if scratch is not None:
        scratch.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
