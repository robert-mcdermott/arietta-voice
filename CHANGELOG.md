# Changelog

User-visible changes to Arietta Voice. See [RELEASING.md](RELEASING.md) for
version policy and the release procedure.

## Unreleased

### Added

- Apple Silicon CI for locked installs, isolated unit/process tests, synthetic
  evaluations, distribution builds and version/private-file checks. Actions are
  pinned; only synthetic JUnit reports are uploaded.
- A 43-case synthetic corpus for wake/session behavior, prompts, memory, tool
  policy and keyword retrieval; corrupt-index and config snapshot regressions.
- Source distributions include tests/evaluations and exclude local config and
  knowledge. Tracked-file checks reject force-added private paths.

- Configurable microphone, utterance, recording, and echo-reference limits;
  streamed WAV recording with pressure/error reporting.
- Serialized inference processes with turn, endpoint, and startup deadlines;
  model/tool failure recovery and an independent runtime heartbeat.
- Admin reliability cards, recording/recovery warnings, expandable runtime
  details, clearer errors, and unsaved source/knowledge editor warnings.

- `config-init` creates ignored, owner-only private config/persona snapshots and
  backups while retaining original files. Sibling `.local.toml` settings override
  the selected base config; CLI runtime overrides take precedence.
- `admin-setup` creates a private salted PBKDF2-SHA256 password hash. Credentials
  are now required; explicit environment credentials remain supported.
- Admin Host/origin checks and non-cacheable responses.

- `arietta-voice --version` and `arietta-voice run --version` without loading
  configuration, models, or audio hardware.
- Project review, phased roadmap, and release/versioning instructions.

### Changed

- Removed the built-in admin password and authentication bypass. Credential
  setup must complete before starting the API.
- Tool creation/source writes require `admin.allow_source_editing = true`.
  Inventory and source reads inspect syntax/literal metadata without loading
  plugins. Runtime validation remains an explicit authenticated action.
- Tool enablement changes persist atomically in the private config instead of
  changing tracked defaults. Instruction preview now uses POST rather than GET.

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

- Malformed, incomplete or incompatible semantic index files fall back to keyword
  search in hybrid mode (no result in vector mode); explicit builds repair corrupt
  caches instead of treating them as current. Source documents remain untouched.
- Memory extraction treats `NONE` as an exact sentinel, preserving facts that
  contain that substring in a name or word.

- Failed/timed-out inference is stopped before the next job; failed turns are
  not stored or automatically retried. Playback cleanup closes resources on
  generation/device failures. Memory updates use private atomic replacements.
- Web chat prevents duplicate submissions and duplicate current-message history,
  keeps failed drafts, and returns safe busy/timeout/failure responses.

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

- New audio/runtime limits apply through defaults without changing private
  settings. Restart the runtime and API. Slow first-run model downloads may need
  a larger `runtime.startup_timeout_s`; see README for limits and recovery scope.

- Run `uv run arietta-voice config-init` once to preserve existing settings and
  persona in private files, then customize those files. The command never resets
  tracked originals; review the copies before restoring tracked defaults.
- Run `uv run arietta-voice admin-setup` once with the same base `--config` used
  by `serve`. Restart the API after credential changes. Old default credentials
  no longer work; remove obsolete password environment entries to use the hash
  file. Source editing stays disabled unless explicitly enabled in config.
- Local settings, persona copies, credential hashes, migration backups, and
  `.working` config copies under `config/` are now ignored. Custom directories
  require their own ignore rules; already tracked private files remain tracked.

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
