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

### Migration notes

- Existing custom articles remain in place. If an article was previously
  tracked in your checkout, remove that specific path from the Git index with
  `git rm --cached -- path`; this retains its local file, but notifies Git to
  remove it in the next commit. Earlier commits still retain their content.
- Keep private knowledge in new files rather than the two tracked starter files.
  Knowledge directories configured elsewhere need their own protection.
- Run `uv sync --locked` after a version change to refresh installed metadata.

## 0.1.0 baseline

The repository already declared `0.1.0` before this changelog was introduced.
This entry records that baseline, not a newly published or dated release.
It includes local wake-word voice interaction, Moonshine STT, Kokoro TTS,
Silero/Smart Turn endpointing, MLX/Bedrock chat, knowledge retrieval, two tool
paths, optional memory, and the web admin/API with runtime supervision.
