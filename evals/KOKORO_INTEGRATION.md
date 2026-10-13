# Opt-in MLX Kokoro integration trial

October 12, 2026, Apple M4, 32 GiB RAM, macOS 15.7.9, Python 3.13.3.
This trial adds an optional MLX adapter to the normal shared inference worker.
ONNX remains the default. The production environment, dependency manifest and
lockfile are unchanged. Private settings, persona, credentials, knowledge and
memory are preserved; synthetic test profiles and raw artifacts live under
ignored `tmp/`.

## Scope and configuration

The adapter uses the existing sentence/clause speech pipeline, bounded queues,
4096-sample IPC frames, 30-second backend segment limit, turn cancellation and
killable inference process. Gemma generation and MLX synthesis share one owner;
the adapter materializes each GPU result as CPU float32 before playback. This
does not add synthesis streaming within a Kokoro segment.

| Setting | Default | Trial behavior |
| --- | --- | --- |
| `audio.tts_backend` | `"onnx"` | Explicit `"mlx"` selects the experimental adapter. |
| `audio.mlx_kokoro_model` | `""` | Required local model directory for MLX; relative paths resolve beside the config. |
| `audio.voice` | `"af_heart"` | Trial accepts named American/British English voices with an existing local voice file. |
| `audio.tts_speed` | `1.0` | Trial range 0.5–2.0. Only `af_heart` at 1.0 was measured. |

The model directory must contain `kokoro` in its name (the pinned upstream
loader uses the name to identify the architecture), `config.json`,
`kokoro-v1_0.safetensors`, and `voices/<voice>.safetensors`. Startup checks reject
missing assets/dependencies; there is no automatic backend fallback. Preinstall
the English spaCy model to avoid Misaki's automatic installation path.

## Separate installation and rollback

Run from the repository root. This environment contains the full application,
not just the TTS component. Do not reuse the earlier component experiment's
environment, which had different versions of the other model dependencies.

```bash
uv venv --python 3.13 tmp/kokoro-integration/venv
uv pip install --python tmp/kokoro-integration/venv/bin/python \
  -r evals/kokoro_integration_requirements.txt
uv pip install --no-deps --python tmp/kokoro-integration/venv/bin/python -e .
```

Use the already downloaded model from the [component experiment](KOKORO_EXPERIMENT.md),
or follow that document's pinned asset download instructions. The measured
revision is `a71e4d38b236d968966a2002c4c895dbd12b1c3c` of
`mlx-community/Kokoro-82M-bf16`; the pinned loader actually loads float32
parameters. Both backends use the same public `af_heart` voice embeddings.

For a temporary interactive opt-in, after preparing the environment/assets:

```bash
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  -m arietta_voice.cli run --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16
```

This command uses your normal private persona/settings for an interactive
session, but writes no configuration override. The benchmark commands below
instead create synthetic isolated profiles. To return to the production stack,
stop the trial and use `uv run --locked arietta-voice run --tts-backend onnx`.
No package downgrade or private config edit is needed.

The trial pins MLX Audio 0.5.8, MLX 0.31.1, Misaki 0.9.4 and the English spaCy
model 3.8.0. Integrating MLX Audio requires newer Transformers 5.17.0,
tokenizers 0.23.3, Hugging Face Hub 1.16.1 and safetensors 0.8.0. Gemma's
MLX-VLM 0.4.4, Torch 2.11.0, Moonshine 0.0.54 and ONNX Runtime 1.24.4 stay at
production versions. The [full trial freeze](kokoro_integration_requirements.txt)
is separate from `uv.lock`.

## Full-conversation protocol

Production ONNX reference blocks bracket four matched trial blocks:
ONNX → MLX → MLX → ONNX. Each fresh process has a separately reported initial
turn, then ten warm turns grouped into two five-turn conversations. Conversation
history persists within each group. Total: 66 completed turns, including 20
warm samples per trial backend and 20 production reference samples. The ten
short no-tool questions use seed 1729 and exactly the same input waveform hashes
in all conditions. Pilots are excluded.

The runtime includes paced 16 kHz synthetic input, normal VAD/endpointing, batch
Moonshine STT, Gemma 4 E4B 4-bit generation, sentence/clause synthesis, AEC reference
bookkeeping and actual output-device writes. Output goes to the Jabra SPEAK 410
USB with playback muted. No physical microphone is opened. All four models
(STT, chat, endpoint detector, TTS) are loaded together and checked after each
turn. Knowledge, memory, tools and chime are disabled; temperature is zero.

Primary latency is waveform end to first output write, not measured acoustic
onset. Full turn duration includes paced playback, so different speech lengths
also affect it. Process startup and initial turns are separate from warm
percentiles. Worker memory is sampled after each turn, outside its timing;
the existing parent sampler records RSS/CPU across the owned process tree.
RSS and shared MLX allocator metrics overlap and must **not** be added together.
Allocator peak includes Gemma as well as MLX TTS. These short runs cannot
establish long-term memory stability or the Phase 2 100-turn acceptance target.

## Reproduce the comparison

Prepare identical synthetic input once using the production ONNX adapter:

```bash
uv run --locked scripts/kokoro_integration.py prepare \
  --output tmp/kokoro-integration/fixtures-new.npz
```

Both Kokoro ONNX assets must already be cached; preparation does not download
them. The NPZ has a matching JSON corpus/hash manifest. For each trial block:

```bash
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/latency_baseline.py --turns 10 --idle-seconds 5 \
  --fixtures tmp/kokoro-integration/fixtures-new.npz \
  --conversation-turns 5 --resource-metrics --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16
```

Select `--tts-backend onnx` for the matched ONNX blocks. Use `uv run --locked`
instead of the trial environment for a production ONNX reference. Omit
`--audible` to retain muted output. Optional `--config` reads another source
profile; the harness copies only model/audio/runtime settings into a fresh
synthetic profile. Keep voice, speed, chime, devices and model settings identical
across blocks. Run serially on an otherwise quiet Mac. Each invocation prints a
fresh directory under `tmp/latency-baseline/` containing report and environment
metadata. The existing benchmark timeout and hardware-validation lock apply.

Summarize the four matched trial reports, excluding the production reference:

```bash
uv run --locked scripts/kokoro_integration.py summarize \
  PATH_TO_ONNX_1/report.json PATH_TO_MLX_1/report.json \
  PATH_TO_MLX_2/report.json PATH_TO_ONNX_2/report.json \
  --output tmp/kokoro-integration/summary-new.json
```

The summary rejects incomplete/failed gates, unbalanced backends, mismatched
fixtures or conversation cases, and missing resident models. Check environment
metadata and dependency versions before comparing reports from different runs.

Run the separate native cancellation/recovery check without opening audio:

```bash
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/kokoro_integration.py recovery \
  --model tmp/kokoro-experiment/Kokoro-82M-bf16 \
  --output tmp/kokoro-integration/recovery-new.json
```

It cancels while `tts_next` is waiting on native synthesis, verifies the old
worker stopped, reloads all models, and generates a fresh synthetic answer and
bounded audio frames. Cancellation-to-worker-stop is a software lifecycle
measurement, not interruption-to-silence at the speaker.

## Results and decision

All six blocks passed their completion, failure, output-underflow, worker-restart,
content-logging and private-file gates. All four models remained loaded. Response
hashes matched at every corresponding turn across all six blocks. Three of the
eleven transcripts per block did not match the fixture's normalized wording;
the same cases were flagged in every condition. The comparison therefore shows
no differential transcript mismatch, but does not claim perfect STT accuracy.

| Metric | Production ONNX | Trial ONNX | Trial MLX |
| --- | ---: | ---: | ---: |
| Warm turns | 20 | 20 | 20 |
| Waveform end → first write p50 | 2,607 ms | 2,952 ms | 2,308 ms |
| Waveform end → first write p95 | 4,019 ms | 4,383 ms | 3,438 ms |
| Waveform end → first signal write p50 | 2,607 ms | 2,952 ms | 2,520 ms |
| Waveform end → first signal write p95 | 4,019 ms | 4,383 ms | 3,652 ms |
| First TTS chunk p50 | 596 ms | 667 ms | 228 ms |
| Full turn processing/playback p50 | 3,479 ms | 3,701 ms | 3,958 ms |
| Cached startup, range | 7.19–7.28 s | 6.98–7.33 s | 6.55–6.57 s |
| Initial turn first write, range | 3.50–4.15 s | 3.15–3.41 s | 3.48–3.77 s |
| Worker high-water RSS, range | 3.97–4.83 GiB | 4.28–5.13 GiB | 4.22–5.30 GiB |
| Shared MLX allocator peak | 5.23 GiB | 5.23 GiB | 6.43 GiB |
| MLX active allocation after each turn | 4.86 GiB | 4.86 GiB | 5.17 GiB |

Within the matched trial environment, MLX improved median first-write latency
by **21.8%**, descriptive p95 by **21.6%**, and median first signal-containing
write by **14.6%**. Its initial PCM includes some silence; first signal write
uses the existing harness's amplitude threshold and is still a software metric.
Median full turn duration increased by 6.9%, consistent with longer generated
speech. Faster synthesis does not imply shorter playback.

Against the bracketed production reference, the median first-write improvement
is **11.5%**, and first signal write improves only **3.3%**. Trial ONNX was
13.2% slower at the median than production ONNX, despite identical response
hashes. The dependency changes, run variability and individual stage outliers
need further controlled investigation; this experiment cannot attribute that
difference to one package. The production reference was added specifically to
avoid presenting the matched-trial gain as the expected production gain.

Follow-up: the [ONNX environment investigation](ONNX_ENVIRONMENT_INVESTIGATION.md)
reversed the comparison order and found trial ONNX 9.1% faster, rather than
13.2% slower. Resident-model TTS probes differed by only 0.8% and produced
identical audio. The original gap is not a reproducible dependency penalty;
retain the numbers above as observations from this round, not a correction
factor for future MLX measurements.

MLX adds about **313 MiB** of steady active MLX allocation and **1,226 MiB** to
the shared allocator peak. Active allocations were stable after the first turn
within these short blocks. Worker RSS high-water values varied by process;
five-second whole-tree samples also missed brief peaks and reflected changing
resident pages. Neither measure establishes lower total memory pressure. Keep
RSS, allocator peak and active allocation separate; do not sum them or infer
long-term leak freedom from these short runs.

The separate native recovery check passed: cancellation stopped the blocked
worker in **295 ms**, then a fresh worker loaded all four models and generated
a new answer plus 32,400 samples of bounded 24 kHz audio. This single software
check does not satisfy the physical p95 interruption target.

**Decision:** retain the explicit opt-in trial; keep ONNX as the production
default. Before proposing migration, investigate the dependency/control gap,
run at least 100 warm turns per condition plus a longer shared-model soak, and
check live audible playback, echo and interruption with the MLX adapter. Repeat
the [listening comparison](KOKORO_EXPERIMENT.md) for realistic conversation
answers and any additional voices. The earlier review preferred MLX's currency
pronunciation and otherwise found similar quality; no new listening claim is
made from muted output. Phase 2's 30% median reduction and physical interruption
criteria remain open.

Raw local evidence: `tmp/kokoro-integration/final/runs.json` maps the six blocks
to `tmp/latency-baseline/` reports. Trial and production summaries are separate
JSON files there; `tmp/kokoro-integration/recovery.json` records recovery. These
artifacts, the synthetic profile, model assets and environment remain ignored.

## Validation and evidence identifiers

The locked synthetic suite passed **530 tests**. New coverage checks explicit
backend selection, local assets/dependencies, temporary CLI overrides, supported
voice constraints, materialized/finite/bounded PCM, stream cleanup, retained
conversation history, shared fixture validation and comparison rejection gates.
Offline wheel/sdist builds and release privacy/version checks passed. All 29
protected config/knowledge files outside the two intentionally updated public
config templates matched their original hashes. Production package versions,
`pyproject.toml` and `uv.lock` are unchanged.

SHA-256 identifiers for the measured artifacts:

- Shared input NPZ: `5d51f83f8d4808c9bc93c0403b02f6cbf5c0e24a04b8e07017e7cebffc154cf0`.
- Measured latency harness: `4b41df645d3093fea4ebc178623906e4945f411b83671f2f6fc57e729c268f8b`.
- Trial dependency freeze: `d51af5925d2faf0175b7a1fe29ca3088faffb48d4412ecc04c5d787c00907fee`.
- MLX weights: `4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8`.
- MLX voice: `2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094`.
- Gemma revision: `cc3b666c01c20395e0dcebd53854504c7d9821f9`.

Environment reports also retain production lock, ONNX asset, Smart Turn and
Moonshine asset hashes. The local `final/evidence.json` adds MLX asset and trial
freeze hashes. No private content is included in this tracked report.
