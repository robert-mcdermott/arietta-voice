# Arietta Voice Versioning and Releases

The existing application baseline is **0.1.0**. The only version declaration is
`[project].version` in [pyproject.toml](pyproject.toml). Runtime code reads the
installed distribution metadata through `arietta_voice.__version__`; the CLI,
FastAPI metadata, and existing `api_version` / `api.version` response fields use
that value. These API fields report the application build, not a separately
versioned HTTP protocol.

```bash
uv version --short
uv run arietta-voice --version
```

The first command reads project metadata. The second reports the installed app.
After editing the version manually, run `uv lock` and `uv sync --locked` so the
lockfile and installed metadata agree. `uv version` normally updates and syncs
them as part of a version change. A source-only import without installation
reports `0+unknown`; it does not guess a release number.

## Version policy

Use `MAJOR.MINOR.PATCH` in package metadata and `vMAJOR.MINOR.PATCH` for Git tags.

- **Patch**, for example `0.1.0` to `0.1.1`: compatible bug fixes, privacy fixes,
  documentation, and small operational improvements.
- **Minor**, for example `0.1.1` to `0.2.0`: a coherent new capability or a
  deliberate compatibility change during the pre-1.0 period. Document migration
  steps explicitly; do not hide breaking config/tool changes in patch releases.
- **Major**, starting with `1.0.0`: a stable compatibility promise for documented
  CLI behavior, config schema, HTTP contracts, and tool interfaces. Later breaking
  changes require a new major version.
- Use Python-compatible prereleases such as `0.3.0rc1` for release candidates,
  with a matching Git tag such as `v0.3.0rc1`.

Keep application version, config schema version, knowledge `INDEX_VERSION`, and
model revisions separate. A model/library upgrade warrants an app release, but
does not automatically require a major version; user-visible compatibility does.

## Preparing a release

1. Review the working tree and the proposed release scope. Keep personal config,
   knowledge, logs, and recordings out of the commit. The knowledge allowlist
   permits only `knowledge/README.md` and `knowledge/getting-started.md`; ensure
   those files still contain generic content. Ignore rules do not protect files
   already tracked or explicitly force-added.
2. Choose the next version, then run one of:

   ```bash
   uv version --bump patch
   # Or, for a planned feature release:
   uv version 0.2.0
   ```

3. Move the appropriate entries from `Unreleased` in [CHANGELOG.md](CHANGELOG.md)
   into a version section with the actual release date. Include config/index
   migrations, changed models, and rollback notes where applicable.
4. Run the release checks:

   ```bash
   uv sync --locked
   uv run --locked pytest
   uv run --locked arietta-voice --version
   uv build
   git diff --check
   git diff --stat
   git ls-files knowledge
   ```

   Check that wheel/sdist metadata matches the intended version and that no
   personal data is packaged. For model/audio/runtime changes, also run the
   hardware and evaluation gates in [ROADMAP.md](ROADMAP.md). Passing unit tests
   alone does not validate microphone operation or new model compatibility.
5. Stage the intended files explicitly, inspect `git diff --cached`, and commit
   the version, lockfile, changelog, and release changes together. Tag the reviewed
   commit with the matching annotated tag. For example, **only when releasing
   0.1.1**:

   ```bash
   git tag -a v0.1.1 -m "Arietta Voice 0.1.1"
   ```

6. Push the reviewed commit/tag and create a GitHub release using the changelog
   section. Attach built distributions if desired. Publishing to PyPI is a
   separate decision: starter assets and first-run setup need packaging work
   before a standalone end-user installation is fully supported.

No tag or release is created by the current review. No historical release date
is inferred from the pre-existing `0.1.0` declaration. The next proposed patch
is `0.1.1`; the roadmap's feature milestones begin at `0.2.0`.

## Release automation to add

CI should validate locked installs, tests, builds, version consistency, and the
tracked-knowledge allowlist on pull requests. A tag-triggered release workflow
should reject a tag that differs from package metadata and build from that exact
commit. Keep publication separately authorized and retain a manual Apple Silicon
audio smoke gate until suitable CI hardware exists. Never move an existing
published version tag to a different commit; issue another patch release.
