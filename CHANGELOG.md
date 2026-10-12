# Changelog

User-visible changes to Arietta Voice. See [RELEASING.md](RELEASING.md) for
version policy and the release procedure.

## Unreleased

### Added

- `arietta-voice --version` and `arietta-voice run --version` without loading
  configuration, models, or audio hardware.
- Project review, phased roadmap, and release/versioning instructions.

### Changed

- Package and API version reporting now read installed metadata derived from
  `pyproject.toml`, removing duplicate hardcoded version declarations.
- New knowledge files and subdirectories are ignored by default. Only
  `knowledge/README.md` and `knowledge/getting-started.md` are allowed by the
  repository's ignore rules.
- The knowledge article template moved from `knowledge/_template.md` to
  `examples/knowledge-template.md`; customization documentation follows it.
- macOS `.DS_Store` files are ignored.
- Public `/health` now returns only liveness and version. Authenticated
  `/api/health` retains detailed diagnostics.
- Active transcript/response logging and runtime status text now require
  `logging.content_enabled = true`. Passive wake misses never log text.
  Interactive text-only answers remain visible; redirected responses follow
  the content setting.
- JSONL event logs rotate with configurable `max_bytes` (default 5,000,000)
  and `backup_count` (default 3), with process-safe appends/rotation and
  owner-only file permissions. Oversized events are replaced with markers.
- Config loading rejects unknown keys, invalid types, non-finite values, and
  unsupported audio/retrieval/generation settings before model startup.

### Fixed

- API `use_tools=false` bypasses both deterministic and model-requested tools,
  including model-tool registry construction.
- History limits consistently count user/assistant pairs; zero clears history.
- Inline wake commands retain original case, punctuation, numbers, and Unicode.
- Mentions of goodbye inside a question no longer end the session.
- Runtime status omits conversation text by default, including when reading
  older status files, and session reset clears previous conversation text.
- Voice-runtime failures suppress potentially private exception details unless
  content logging is enabled, while preserving a nonzero exit status.
- BGE v1.5 uses normalized CLS embeddings instead of mean pooling. Index
  format 2 invalidates incompatible cached embeddings.
- Knowledge retrieval excludes links outside its root and hidden directories.

### Migration notes

- Existing custom articles remain in place. If an article was previously
  tracked in your checkout, remove that specific path from the Git index with
  `git rm --cached -- path`; this retains its local file, but notifies Git to
  remove it in the next commit. Earlier commits still retain their content.
- Keep private knowledge in new files rather than the two tracked starter files.
  Knowledge directories configured elsewhere need their own protection.
- Run `uv sync --locked` after a version change to refresh installed metadata.
- Rebuild an existing semantic index with `uv run arietta-voice knowledge-index`,
  then restart the runtime/API. Until rebuilt, hybrid retrieval falls back to
  keywords and vector-only retrieval returns no matches.
- New logging defaults apply when settings are absent; no local config edits
  are required. Restart both runtime and API to apply them. Old logs, recordings,
  and memory files are not erased or scrubbed by this change. JSONL rotation
  does not manage launchd stdout/stderr logs. See README's privacy section.

## 0.1.0 baseline

The repository already declared `0.1.0` before this changelog was introduced.
This entry records that baseline, not a newly published or dated release.
It includes local wake-word voice interaction, Moonshine STT, Kokoro TTS,
Silero/Smart Turn endpointing, MLX/Bedrock chat, knowledge retrieval, two tool
paths, optional memory, and the web admin/API with runtime supervision.
