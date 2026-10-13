# ONNX latency: production versus MLX trial environment

October 12, 2026, Apple M4 / 32 GiB, macOS 15.7.9, Python 3.13.3.

**Finding:** the earlier 13.2% median ONNX slowdown in the MLX trial environment
did not reproduce. Reversing the order of the conversation blocks made the trial
environment 9.1% faster. Identical-input TTS probes with all other models loaded
measured only a 0.8% median difference. These results do not support treating the
original gap as a fixed dependency regression or rolling back a particular
package. They also do not establish statistical equivalence.

Production dependencies and runtime defaults remain unchanged. The investigation
uses the existing separate trial environment, cached public models and synthetic
profiles. No package installation, private config/persona edit, microphone capture
or audible playback was needed.

## What differs between environments

The production and trial environments have four changed existing dependency pins:

| Package | Production | Trial |
| --- | --- | --- |
| Transformers | 5.5.4 | 5.17.0 |
| tokenizers | 0.22.2 | 0.23.3 |
| Hugging Face Hub | 1.11.0 | 1.16.1 |
| safetensors | 0.7.0 | 0.8.0 |

The trial also contains MLX Audio, Misaki, spaCy and their additional dependencies,
including SciPy. Kokoro ONNX 0.5.0, ONNX Runtime 1.24.4, NumPy 2.4.4,
phonemizer-fork and espeakng-loader have identical installed Python/native-library
SHA-256 manifests across the environments. This verifies implementation files,
not just version strings. The model and voice assets are shared cached files.

Both use `CPUExecutionProvider`, sequential execution, all graph optimizations,
and ONNX session intra/inter-op thread settings of `0` (runtime-selected defaults,
not zero worker threads). Selected provider/thread environment overrides matched.
Torch reported four threads in both resident-model probes. MLX Audio and spaCy
were not imported by the ONNX path. SciPy was imported in the trial's resident
process, but not production's; its presence did not produce a material slowdown
in the resident TTS probe. This observation is not a causal test of SciPy alone.

## Conversation repeat

The original [integration trial](KOKORO_INTEGRATION.md) used production ONNX
references around trial ONNX/MLX blocks. Considering only ONNX, its order was
production → trial → trial → production, with MLX runs between the trial blocks.
This investigation reversed the ONNX order: **trial → production → production →
trial**, without intervening MLX blocks.

Each fresh process ran one initial turn followed by ten warm turns in two
five-turn conversations. All conditions used the same existing input NPZ,
question order/seed, synthetic persona, chat model/revision, model assets, voice,
speed, history policy, audio settings and unchanged latency harness. Gemma,
Moonshine, Smart Turn and ONNX Kokoro were loaded together. Output was muted
through the Jabra SPEAK 410; there was no physical microphone input. Knowledge,
memory, tools and chimes were disabled. No other model benchmark or test suite
ran concurrently. The normal voice runtime was confirmed inactive.

All **44 new turns** passed the existing completion, failure, restart, output
underflow, log privacy and protected-file gates. Corresponding response hashes
matched across all original and repeated ONNX blocks. Transcript match flags
were unchanged; the earlier normalized-text mismatch cases remain, rather than
being counted as perfect transcription.

| Waveform end → first output write | Production ONNX | Trial ONNX | Trial median difference |
| --- | ---: | ---: | ---: |
| Original round, 20 warm turns each, p50 | 2,607 ms | 2,952 ms | +13.2% |
| Reversed repeat, 20 warm turns each, p50 | 2,635 ms | 2,394 ms | −9.1% |
| Both rounds, 40 warm turns each, p50 | 2,607 ms | 2,663 ms | +2.1% |
| Original round, p95 | 4,019 ms | 4,383 ms | — |
| Reversed repeat, p95 | 4,161 ms | 3,833 ms | — |
| Both rounds, p95 | 4,161 ms | 4,348 ms | — |

The pooled row is descriptive, not a corrected environment-effect estimate.
The block medians show why a single pooled number is inadequate:

| Round | Production block medians | Trial block medians |
| --- | --- | --- |
| Original | 2,434 / 2,657 ms | 3,045 / 2,923 ms |
| Reversed repeat | 2,712 / 2,566 ms | 2,356 / 2,435 ms |

The original trial's largest STT call took 1,361 ms and endpoint check 389 ms;
the repeat's maxima were 364 and 36 ms. Variation was not confined to speech
synthesis. In the repeat, STT, chat and first TTS chunk medians all favored the
trial environment. Stage totals include work after first playback and their
medians cannot be added or subtracted to explain the end-to-end median.

## Component isolation

The new `scripts/onnx_environment.py probe` instruments the existing ONNX
adapter inside an isolated child process. It measures phonemization, native
`InferenceSession.run`, and silence trimming without changing their arguments
or outputs. There is a 240-second outer bound. Probes require already cached
assets and use offline model loading.

For each of two modes, four fresh processes ran in trial → production → production
→ trial order. Each process synthesized a separately reported initial utterance,
then five repetitions of three fixed public texts (acknowledgement, short answer,
numbers/currency). That is **30 warm utterances per environment per mode**, 120
warm utterances overall, plus eight initial utterances. The modes are:

- TTS alone, without loading the chat/STT/endpoint models.
- TTS with all four models loaded and one synthetic Gemma warmup answer completed
  first. This holds model residency constant; it does not reproduce interleaved
  token generation/playback or live microphone/AEC work.

| Warm first materialized audio | Production ONNX | Trial ONNX |
| --- | ---: | ---: |
| TTS alone p50 | 922.2 ms | 878.3 ms |
| TTS alone p95 | 1,604.4 ms | 1,367.0 ms |
| Other models loaded p50 | 882.3 ms | 889.6 ms |
| Other models loaded p95 | 1,320.8 ms | 1,351.0 ms |
| Other models loaded: native ONNX inference p50 | 881.0 ms | 888.0 ms |
| Other models loaded: phonemization p50 | 0.210 ms | 0.233 ms |
| Other models loaded: trimming p50 | 0.638 ms | 0.666 ms |

Native ONNX inference accounted for more than **99.8%** of measured synthesis
wall time in all four conditions. Warm phonemization benefits from repeated
texts; these sub-millisecond figures are not cold text-processing measurements.
All corresponding phoneme hashes and float32 output audio hashes matched exactly
across all eight blocks. The four resident-model checks confirmed all models
loaded. No silence, clipping or invalid PCM was reported.

The resident median difference is **7.3 ms (0.8%)**. Even with identical
standalone code and outputs, production's two standalone block medians differed
by 75 ms, and trial's by 86 ms. This supports treating short-run timing variation
as a substantial confound. Host scheduling, thermal/power state, memory pressure
and native thread-pool interactions were not independently controlled or
measured here, so none can be named as the proven root cause.

## Decision and next measurement

The planned longer run is now complete; see the
[100-turn counterbalanced comparison](KOKORO_100_TURN_COMPARISON.md) for six
execution orders, production/trial/MLX results and measured host context. The
original findings below remain observations from this shorter investigation.

Keep the production environment and ONNX default as they are. Keep the four
updated pins confined to the optional MLX trial; they are part of its dependency
requirements. There is no demonstrated ONNX compatibility or deterministic-output
regression, and no repeatable 13.2% latency penalty to fix by changing a pin.

Do not use the old production/trial ONNX gap as a correction factor for MLX's
benefit, or reuse a baseline from a different round to claim a new MLX speedup.
The earlier MLX latency numbers remain observations from that experiment.
Before migration, run at least 100 warm turns per condition with multiple short,
counterbalanced blocks that include production ONNX, trial ONNX and trial MLX.
Record host load/power/thermal context, retain initial turns separately, report
block variability and paired cases, and repeat live playback/interruption and
the longer shared-model soak. No Phase 2 acceptance target is closed by this
investigation.

## Reproduction

Use the existing trial environment and shared fixtures from
[the integration setup](KOKORO_INTEGRATION.md). Run only one benchmark at a time,
with the normal voice runtime stopped. From the repository root:

```bash
uv run --locked scripts/onnx_environment.py probe --label production \
  --repeats 5 --output tmp/onnx-environment/new-production-alone.json

uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/onnx_environment.py probe --label trial --repeats 5 \
  --output tmp/onnx-environment/new-trial-alone.json
```

Add `--resident` and fresh output filenames for the resident-model probe. Optional
`--config` supplies model/audio settings; the probe copies them into a synthetic
profile with private context disabled, fixes ONNX `af_heart`/speed 1.0, and enables
STT/chat/endpoint/TTS loading. It writes no config override. Repeat the four-block
order above for each mode. Package/code fingerprints are collected after
measurement, and all results stay under ignored `tmp/`.

For conversation runs, use the existing `scripts/latency_baseline.py` commands
with `--tts-backend onnx --conversation-turns 5 --resource-metrics`, identical
`--fixtures` and source profile for every block. The recorded experiment used
the prior synthetic profile at `tmp/kokoro-integration/profile/app.toml` with
chimes disabled, rather than changing the user's private config. Run the trial
environment first, then production twice, then trial again, with fresh processes.

Create a local JSON manifest listing each block's environment and the report
directory printed by the benchmark, for example:

```json
[
  {"environment": "trial", "directory": "tmp/latency-baseline/TRIAL_RUN"},
  {"environment": "production", "directory": "tmp/latency-baseline/PRODUCTION_RUN"}
]
```

Include all four blocks for the measured protocol, then compare:

```bash
uv run --locked scripts/onnx_environment.py compare \
  tmp/onnx-environment/new-runs.json \
  --output tmp/onnx-environment/new-summary.json
```

The comparison rejects failed/incomplete blocks, unbalanced environments,
different fixtures/history/response hashes, and mismatched model/audio/benchmark
metadata. It reports pooled and per-block timings plus descriptive paired
trial-minus-production differences; these are not confidence intervals.

Local evidence is under `tmp/onnx-environment/repeat/` and `probes/`, each with a
`runs.json` manifest. Probe reports contain actual dependency versions, code/native
hashes, session settings, stage timings, PCM/phoneme hashes and resident-model
flags. No model assets, environments, private data or raw local reports belong
in the commit.

## Validation and evidence identifiers

The locked test/evaluation suite passed **542 tests**. New regression coverage
rejects invalid comparison gates, incomplete/unbalanced blocks, changed responses,
fixtures, history or model metadata; it also checks initial-turn exclusion and
that stage timing preserves return values and failures. Offline wheel/sdist and
release privacy/version checks passed.

All **31 config/knowledge files** matched their original fingerprints, with no
additions or removals. Production and trial dependency pins are unchanged.
There are no application-source, config-template, manifest or lockfile changes.

SHA-256 identifiers:

- Measured probe harness: `d2bdcd96eba660bb5bbbbf4a9c3ad86ade3e461a57d2d22859d01eab6f5eb057`.
- Identical core implementation manifest (sorted JSON of package/file hashes):
  `5ec015e4f2c5735cd7129ad8fc47bc888468914f8a0e2e29ccc079b2d8c4a0f6`.
- Shared input NPZ: `5d51f83f8d4808c9bc93c0403b02f6cbf5c0e24a04b8e07017e7cebffc154cf0`.
- Unchanged conversation harness: `4b41df645d3093fea4ebc178623906e4945f411b83671f2f6fc57e729c268f8b`.
