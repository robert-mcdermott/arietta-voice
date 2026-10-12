# Synthetic evaluations and CI

Run the same checks as CI on an Apple Silicon Mac with Python 3.13 and `uv`:

```bash
uv run --no-project --python 3.13 scripts/check_release.py
uv sync --locked --group dev
uv run --locked pytest tests --junitxml=tmp/ci/unit.xml
uv run --locked pytest evals --junitxml=tmp/ci/evaluations.xml
uv build --out-dir tmp/ci/dist
uv run --no-project --python 3.13 scripts/check_release.py --artifacts tmp/ci/dist
uv venv --python 3.13 tmp/ci/smoke
uv pip install --python tmp/ci/smoke/bin/python --no-deps tmp/ci/dist/*.whl
uv run --no-project --python tmp/ci/smoke/bin/python python -I -m arietta_voice.cli --version
```

Use a fresh/empty artifact directory when building a different version: the
checker requires exactly one wheel and one sdist. Use a fresh smoke environment
as well. The final command checks the installed wheel in an isolated interpreter,
without loading the source checkout or installing model dependencies.
`uv run --locked pytest` runs
both suites. To investigate one case, use e.g.
`uv run --locked pytest evals -k tool-failure-no-retry -v`.

## What these checks establish

[`cases.json`](cases.json) contains 43 stable, named cases using invented text
and documents. Every case is a blocking assertion; the pass target is 100%.
There are no live API calls, microphone recordings, model downloads, API keys,
user knowledge files, or performance thresholds in this corpus.

| Area | Assertions |
| --- | --- |
| Wake (8 cases) | Prefix-only activation; original punctuation, case and Unicode preserved |
| Prompts (6) | Persona, memory opt-in, context ordering, complete history pairs, current message once |
| Memory (8) | NONE/empty/non-bullet output, durable facts, exact sentinel matching, disabled writes |
| Tool policy (10) | Malformed/unknown/disabled requests cannot execute; one-call limit; no failure retries |
| Sessions (5 sequences) | Ambient speech, acknowledgement, inline commands, follow-ups, direct farewells, failure recovery |
| Keyword retrieval (6) | Expected first source and no match for synthetic absent facts/small talk |

The unit suite adds API `use_tools=false` enforcement, auth/privacy controls,
buffer bounds, consolidation cadence and failure handling, 48 corrupt-index
fallback/rebuild cases, an empty index, and version/packaging/privacy gates.
Real spawned processes exercise timeout termination, crash recovery, rejected
concurrent jobs, shutdown, config snapshot invalidation, and concurrent log
writes. These are orchestration checks using synthetic handlers, not model
inference tests. Process tests run serially; do not add `pytest-xdist` without
reviewing resource ownership and timing assumptions.

The root [`conftest.py`](../conftest.py) requires explicit temporary config roots,
clears ambient admin/AWS credentials, redirects AWS/Hugging Face settings to
temporary paths, and blocks pytest-process network connections, live model
loading and audio-device entry points. Individual tests replace those entry
points with fakes. Spawned processes do not inherit monkeypatches: their tests
must continue to use synthetic handlers or explicit temporary configs with
deterministic tools. Never use implicit repository config discovery in tests.

To add a case, use a unique descriptive `id`, entirely invented input and a
specific expected result in the relevant JSON section. Add the harness assertion
first when introducing a new behavior. Keep failure text and fixture data safe
to include in CI reports. The `schema_version` describes this fixture format;
it is separate from the app version and knowledge index format.

## CI and privacy boundaries

[`ci.yml`](../.github/workflows/ci.yml) runs on pushes, pull requests and manual
dispatch. It uses the standard `macos-15` ARM64 runner and Python 3.13, matching
the supported locked environment. Actions are pinned to commit hashes and `uv`
to a tested version. Dependency installation is locked; permission is limited
to reading repository content. Only the two synthetic JUnit XML reports are
uploaded, with seven-day retention. App logs, recordings, config, model caches,
and distributions are not uploaded. Package download caching contains `uv`
dependencies, not application data. No publishing, release tags, or deployment
is automated.

The tracked-path gate rejects force-added knowledge except the two starter
articles, config except the shipped defaults/examples, and known runtime/private
paths. It checks the Git index, so newly staged files are included. It never
unstages or deletes files. A temporary Git repository regression test proves
that `git add --force` cannot bypass the gate. Package inspection checks
wheel/sdist metadata against `pyproject.toml` and `uv.lock`, rejects known private
paths and archive links, and does not extract archive members. The source
distribution includes the synthetic suites and excludes local config/knowledge.

These are path checks, **not a secret scanner**. Personal data pasted into an
allowed public source, fixture, README or default config still needs review.
Custom data directories also need explicit ignore/check rules. Review staged
changes before committing; ignore rules alone cannot remove earlier commits.

Hosted CI was verified on October 11, 2026 (America/Los_Angeles):

| Commit | Hosted run | Unit/process tests | Synthetic evaluations |
| --- | --- | --- | --- |
| `6023154` | [First CI run: passed](https://github.com/robert-mcdermott/arietta-voice/actions/runs/38199350677) | 308 passed | 43 passed |
| `fe04096` | [Hardware-validation commit: passed](https://github.com/robert-mcdermott/arietta-voice/actions/runs/38201095806) | 326 passed | 43 passed |

Both runs completed every step, including locked dependency installation,
tracked-path/version checks, wheel/sdist build and inspection, the installed
wheel's dependency-free version smoke test, and synthetic report upload. No CI
fix was needed. A future release must still pass CI on its exact release commit;
these runs establish the Phase 1 gate, not an exemption for later changes.

## Remaining release gates

These synthetic checks do not measure speech recognition, generated-answer
quality, real BGE retrieval, GPU performance, microphone behavior, or long-run
stability. Keep the Phase 1 manual hardware smoke and soak gates in
[`HARDWARE_VALIDATION.md`](HARDWARE_VALIDATION.md) and
[`ROADMAP.md`](../ROADMAP.md), using the current locked model stack. Phase 3's
representative retrieval benchmark remains separate; the six keyword cases here
are regression fixtures and do not establish a recall/accuracy score for a model.

CI configuration references:
[GitHub hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/).
