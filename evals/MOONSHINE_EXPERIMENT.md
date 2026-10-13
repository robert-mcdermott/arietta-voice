# Isolated Moonshine streaming upgrade experiment

October 12, 2026, Apple M4 Mac. **Keep production on locked Moonshine 0.0.54.**
The isolated 0.1.5 candidate successfully transcribes the synthetic corpus, but
does not yet justify a production migration. No app dependency, lockfile, STT
adapter, private configuration, persona, knowledge or local setting was changed.

## Candidate and compatibility

Moonshine 0.1.5 is the selected non-prerelease candidate; its PyPI classifier is
still alpha. It adds split frontend assets, revised language defaults and
terminology biasing APIs. This is an evaluation candidate, not a claim of
production stability. Sources: [pinned PyPI release](https://pypi.org/project/moonshine-voice/0.1.5/)
and [upstream release notes](https://raw.githubusercontent.com/moonshine-ai/moonshine/main/CHANGELOGS.md).
The details below were also checked against the installed pinned packages.

Both releases already expose `create_stream`, and English already uses
`MEDIUM_STREAMING`. Arietta currently calls `transcribe_without_streaming` on a
completed utterance. Comparing batch and streaming on both versions separates
the API change from the package/model change, although the revised weights mean
this is not a library-only performance comparison.

| Property | 0.0.54 | 0.1.5 |
| --- | --- | --- |
| English architecture | `MEDIUM_STREAMING` | `MEDIUM_STREAMING` |
| English asset URL suffix | `medium-streaming-en/quantized` | `medium-streaming-en/quantized_26_08_21` |
| Frontend | `frontend.ort` | `frontend.model.ort` + `frontend.weights.ort` |
| Downloaded model directory size | 428.6 MiB | 256.7 MiB |
| Decoder attention asset | `decoder_kv_with_attention.ort` included | Not in required component list |
| Explicit keyterms / context | Unavailable | `set_keyterms` / `set_context` |

Both directories use ORT assets. Do not copy an old cache into the new model
directory or hand-maintain a list of component filenames; use the pinned
package's component manifest. The experiment records SHA-256 hashes and sizes
for assets and fixtures. It does not download optional spelling or diarization
models and explicitly disables speaker identification in both versions.

Catalog checks resolve `en` and `English` to the same architecture and reject an
unsupported language. These are metadata checks, not recognition tests:

| Language | Old default | Candidate default |
| --- | --- | --- |
| English | Medium streaming | Medium streaming |
| Spanish, Japanese | Base | Small streaming |
| Arabic, Vietnamese, Chinese | Base | Tiny streaming |
| Korean | Tiny | Tiny |
| Ukrainian | Base | Base |
| German | Unavailable | Small streaming |
| Tagalog | Unavailable | Tiny streaming |

Never silently assume every language streams. Validate each chosen language,
model architecture and its applicable license before offering it in production.
The candidate's model metadata no longer contains `model_name`, which the
existing hardware-report helper expects. That helper needs adaptation as part
of an eventual migration; the production environment remains compatible today.

## Protocol and timing boundaries

`scripts/moonshine_experiment.py` generates ten fixed Kokoro `af_heart` fixtures:
wake word, inline command, ambient name mention, a question containing goodbye,
farewell, three sentences, product names, technical jargon, numbers and silence.
Fixtures are mono PCM16 at 16 kHz, with identical 250 ms pre-roll and one second
of post-roll. No microphone, speaker, assistant model, tool, personal memory or
private configuration is opened. Fixture synthesis can fetch public Kokoro
assets if absent; decoder runs use explicit existing model paths without
automatic downloads. Candidate assets and its environment live under ignored
`tmp/`; baseline assets are read from the existing cache.

Each row below is a fresh process/model owner over the same ten files, with one
pass per condition in sequential order. Python 3.13.3 and NumPy 2.4.4 are shared.
Model load is separate; the first inference is included, not discarded as
warmup. Streaming receives 512-sample chunks (32 ms) paced against a monotonic
clock, requesting updates every 250 ms. A new stream is closed after each file;
the transcriber is reused within each condition. No inference runs concurrently
with another condition. Earlier pilot runs established harness compatibility.

First-partial time starts at fixture playback start, including pre-roll.
Stop-flush time starts **after the complete file, including its post-roll**, is
fed. These small flush times are not speech-end latency or an endpoint detector
measurement. Feed lag is lateness *before* submitting the next chunk; the harness
catches up without dropping samples. A microphone would need bounded queues and
an explicit overload policy. CPU time sums native threads; peak RSS is the whole
decoder process, including imports and model allocations. Cached load timings
are not cold-boot measurements.

## Results

All six conditions completed. Each produced 8/10 normalized exact matches and
the same **5 edits / 58 reference words = 8.62% literal WER**. The two mismatches
are numeric formatting: `two plus two` became `2 plus 2`, and `minus five point
two` became `minus 5.2`. Case and punctuation are ignored; numbers are deliberately
not canonicalized. These differences were semantically correct on inspection,
not evidence of five recognition mistakes. Silence stayed empty in every run.

Medians below cover the nine spoken fixtures; RSS covers the full process.
These are descriptive summaries of a tiny corpus, not representative latency
percentiles or evidence that Phase 2 performance targets have been met.

| Condition | Cached load ms | Batch decode median ms | First partial median ms | Stop flush median ms | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0.54 batch | 282.5 | 242.5 | — | — | 1360.9 |
| 0.1.5 batch | 109.3 | 313.9 | — | — | 1056.7 |
| 0.0.54 stream | 268.4 | — | 1167.2 | 4.4 | 1356.4 |
| 0.1.5 stream | 118.1 | — | 1147.6 | 6.6 | 1541.4 |
| 0.1.5 stream + keyterms | 128.2 | — | 1159.2 | 5.6 | 1629.5 |
| 0.1.5 stream + context | 113.5 | — | 1151.0 | 5.2 | 1559.5 |

Candidate batch decoding was about 29% slower at the median while loading faster
and using less peak memory. That memory improvement did **not** carry through
to streaming. Unhinted first-partial times differed by only about 20 ms; pilots
varied, so this does not establish a speed improvement. Median CPU time per
spoken file was 4194.2 ms for baseline streaming and 3808.9 ms for candidate
streaming, including native threads; this is not total app CPU consumption.

Maximum chunk-feed lag was 327.7 ms for baseline streaming and 385.7 ms for
candidate streaming (251.7/259.9 ms with keyterms/context). All samples were
eventually fed. Those stalls are far longer than a 32 ms capture chunk and must
be handled before connecting this API to live capture. Strict finalization
succeeded in all four streaming conditions; the unhinted runs each revised
partial text 27 times across the corpus. Both hint modes restored `Moonshine`
and `Kokoro` capitalization without changing normalized word accuracy, and both
post-clear wake streams returned `Bridget` successfully.

Final texts were also checked against the app's existing wake/farewell policies
in the baseline environment: wake-only and inline activation, preserved inline
prompt, no activation for an ambient name mention, a question about goodbye
remaining active, direct goodbye matching, and empty silence. Every condition
passed. This checks final text policy only; no live session or action ran.

Raw reports are local in `tmp/moonshine-final-results/`. For matching the recorded
inputs, SHA-256 of `json.dumps(mapping, sort_keys=True).encode()` is:

- Fixture filename-independent ID → hash mapping:
  `a6ee638babc9a82ae8148af1ef40ef5f3ad7173347b88cdd22528f5075598047`
- Baseline asset filename → size/hash mapping:
  `e491dd236f57700ad77d3118b5afb0ccbda5ead845518208bb7abbe20e28e783`
- Candidate asset filename → size/hash mapping:
  `87525d0131566817a5f55c6a76add4e8cf29c27b7d14a380c8043d8e8d80ba67`

The project suite passed **439 tests**, including 13 new synthetic experiment
tests. Before/after fingerprints matched for config and knowledge; the production
package remained 0.0.54. These checks do not establish recognition quality on
human speech, other voices/languages, noisy rooms, sustained capture, acoustic
interruption, or resource contention with the rest of the assistant.

## Partial/final handling and hints

Partial events replace a snapshot keyed by `line_id`; they never append commands
or trigger wake/farewell/tool actions. A single native update may emit multiple
events for one line. The final transcript returned by `stop()` is authoritative,
deduplicated by line ID and ordered by start time. Some native errors arrive as
`Error` callbacks and a missing result rather than an exception. The harness
fails on either, on incomplete final lines, or on event/text/line limits, and
always closes its stream. The outer spawned process has a 180-second deadline
(configurable from 30 to 600 seconds), including native calls; timeout/crash
produces a failed local report and the owned child is reaped.

Keyterms are a small, explicit synthetic list: Bridget, Arietta, Moonshine,
Kokoro, Kubernetes, PostgreSQL and Ceph. The context condition uses a short
invented software sentence with `max_terms=10`. These APIs replace the previous
term list. Both clearing APIs are exercised before a final fresh wake stream on
the same transcriber. Successful reset transcription shows continued operation,
not proof that all internal bias state is absent. Hint accuracy and false
positives need a broader corpus before any automatic knowledge-derived hints.

## Repeat locally

Run from the repository with a quiet Mac and the normal voice runtime stopped
for comparable measurements. Choose fresh report/fixture paths on a repeat;
existing completed outputs are rejected. Downloads need internet access.

```bash
# Preserve the project environment; install the candidate only under tmp/.
uv venv --python 3.13.3 tmp/moonshine-0.1.5
uv pip install --python tmp/moonshine-0.1.5/bin/python \
  'moonshine-voice==0.1.5' 'numpy==2.4.4'
uv run --locked scripts/moonshine_experiment.py fixtures --output tmp/moonshine-fixtures

# Resolve the existing baseline cache without downloading or changing it.
uv run --locked scripts/moonshine_experiment.py prepare \
  --output tmp/moonshine-baseline-model.json
# Explicit candidate download goes only into this separate cache.
uv run --no-project --python tmp/moonshine-0.1.5/bin/python python \
  scripts/moonshine_experiment.py prepare --download \
  --cache tmp/moonshine-assets-0.1.5 --output tmp/moonshine-candidate-model.json

uv run --locked scripts/moonshine_experiment.py run \
  --model tmp/moonshine-baseline-model.json --fixtures tmp/moonshine-fixtures \
  --mode batch --output tmp/moonshine-results/baseline-batch.json
uv run --locked scripts/moonshine_experiment.py run \
  --model tmp/moonshine-baseline-model.json --fixtures tmp/moonshine-fixtures \
  --mode stream --paced --output tmp/moonshine-results/baseline-stream.json

uv run --no-project --python tmp/moonshine-0.1.5/bin/python python \
  scripts/moonshine_experiment.py run --model tmp/moonshine-candidate-model.json \
  --fixtures tmp/moonshine-fixtures --mode batch \
  --output tmp/moonshine-results/candidate-batch.json
uv run --no-project --python tmp/moonshine-0.1.5/bin/python python \
  scripts/moonshine_experiment.py run --model tmp/moonshine-candidate-model.json \
  --fixtures tmp/moonshine-fixtures --mode stream --paced \
  --output tmp/moonshine-results/candidate-stream.json
```

Repeat the last command with `--hints keyterms` and then `--hints context`, each
with a distinct output JSON. If baseline assets are missing, `prepare --download
--cache tmp/moonshine-assets-0.0.54` explicitly downloads them into a separate
cache too. Neither preparation nor transcription changes the production model
selection. Keep WAVs, package inventories, manifests, logs and raw reports in
ignored local storage; commit the harness, tests and this reviewed summary only.

## Promotion gates

1. Expand paired, repeated measurements to real voices, accents, noise, silence,
   wake/farewell ambiguity, long sessions and each supported language. Measure
   semantic number accuracy separately from literal word-error rate.
2. Integrate streaming only behind an explicit experimental adapter/profile with
   a batch fallback. Preserve cancellation/deadlines and the current worker owner;
   do not run native inference on the capture callback or main listening loop.
3. Validate audio backlog, bounded queues, cancellation and native failures with
   live capture and playback, including Smart Turn endpoint coordination. Prove
   provisional text cannot dispatch tools or end a session.
4. Repeat the full response-latency and interruption baselines, including resource
   contention with generation and TTS. Meet Phase 2 gates without unexplained
   accuracy regressions before changing the dependency pin or default adapter.
5. Adapt model metadata/reporting, pin language/model revisions, validate clean
   cache installation and rollback, then run hosted CI and a hardware soak on
   the proposed production profile. No version bump or promotion occurs here.
