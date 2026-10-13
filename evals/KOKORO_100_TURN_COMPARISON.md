# Counterbalanced Kokoro conversation comparison

October 12, 2026, Apple M4 / 32 GiB, macOS 15.7.9, Python 3.13.3.

**Result:** all 318 turns passed. Against production ONNX, MLX reduced the warm
median first-write latency by 12.5%, but the median first signal-containing write
by only 3.4%. Its first-signal p95 improved by 12.4%, while shared MLX allocator
peak increased from 5.23 to 6.43 GiB. Keep ONNX as the default and MLX opt-in.
The measurements support a synthesis benefit, not an unconditional migration.

This experiment compares production ONNX, trial ONNX and trial MLX with 100
warm conversation turns per condition. All activity uses synthetic context and
the existing separate environments; production settings and dependencies stay
unchanged. Local raw data lives under ignored `tmp/kokoro-comparison/`.

## Protocol

Conditions are A = production ONNX, B = trial ONNX, C = trial MLX. Six rounds
cover all six orderings. Each condition appears twice in each ordinal position.
Every block starts a fresh process, reports one initial turn separately, and
then runs its warm turns in five-turn conversations. There are 300 warm turns
and 18 initial turns overall.
The initial turn follows the normal runtime warmup. Process startup is measured
separately with already cached assets; neither measure is a first-install or
uncached-download test.

| Round | Order | Warm turns per condition | Corpus offset |
| --- | --- | ---: | ---: |
| 1 | A → B → C | 20 | 0 |
| 2 | B → C → A | 20 | 0 |
| 3 | C → B → A | 15 | 0 |
| 4 | A → C → B | 15 | 5 |
| 5 | B → A → C | 15 | 0 |
| 6 | C → A → B | 15 | 5 |

Offsets rotate the ten-question corpus before the existing seeded shuffle.
Every question appears exactly ten times per condition, and each history depth
0–4 appears twenty times. All three conditions within a round receive identical
input waveforms and history order. Warm exposure by ordinal position is 30/35/35
turns in varying order; perfect exposure equality is impossible with 100 turns
and three positions. Block counts and within-round condition ordering are balanced.

The setup uses the same Apple M4 Mac with 32 GiB RAM, Gemma 4 E4B 4-bit,
Moonshine 0.0.54, Smart Turn and `af_heart` at speed 1.0. All four models remain
loaded together. The production and trial dependency sets are those in
[the integration trial](KOKORO_INTEGRATION.md), with no package changes. The shared
synthetic input NPZ comes from that trial. Temperature is zero; chimes,
knowledge, memory and tools are disabled in an isolated profile. Private
persona/context is never loaded into the benchmark conversation.

The existing runtime handles paced synthetic input, endpointing, transcription,
generation, clause synthesis and actual output-device writes. Jabra output is
muted, and no physical microphone is opened. No other model benchmark or test
suite runs concurrently. Each block includes five seconds of idle time and
there is a three-second pause between blocks. The outer runner stops on a failed
block and retains its evidence; it never silently replaces or discards failures.

## Timing and host-state boundaries

The primary metrics are waveform end to first output write and first
signal-containing write. They are software boundaries, not acoustic onset at
the speaker. A signal-containing write has a sample magnitude above `1e-4`
before muting. Initial turns are excluded from warm percentiles. Full turn time
includes paced output playback; longer generated speech can increase it even
when synthesis is faster. Speech-end and turn-end monotonic timestamps support
alignment with the host samples.

The read-only host monitor samples approximately every five seconds:

- CPU utilization from differences in Mach host CPU tick counters, plus system
  load averages. CPU utilization is normalized across the host to 0–100%.
- `NSProcessInfo` thermal state: 0 nominal, 1 fair, 2 serious, 3 critical; and
  the low-power-mode flag when available.
- Power source, VM free/compressed/wired memory, cumulative swap-in/out counters,
  swap usage and the kernel's memory-pressure level.
- Sample timestamps, collection duration and explicit collection errors.

No process names, command lines, application content, battery identifiers or
arbitrary environment variables are collected by the monitor. Unknown readings
remain unknown. The first CPU sample has no prior tick interval and is null.
Host metrics describe the whole Mac, including unrelated background activity;
they are not attributed solely to Arietta. Sampling is coarse and can miss
short spikes. A thermal or pressure reading is context, not proof of a latency
cause. Swap already present before the experiment is not counted as application
memory growth.

The existing benchmark separately samples its owned process-tree RSS/CPU.
Worker high-water RSS and shared MLX allocator peak/active memory are recorded
outside turn timing. These overlapping memory measures must not be added
together. Restarted blocks measure repeatability across processes; they do not
replace a continuous long-running soak.

## Results

All 18 planned blocks completed in about 42.4 minutes. Every block passed the
completion, valid-turn, failure, worker-restart, output-underflow, log-privacy,
stopped-status and protected-file gates. All corresponding response hashes
matched within every round. All four models were resident for every turn.
Normalized transcript checks matched the fixture text on 80/100 warm turns in
each condition, with identical match flags for corresponding inputs. These
inherited fixture mismatches remain visible; this is not a claim of perfect
transcription accuracy or a fresh live speech-quality test.

Warm latency in milliseconds, with 100 turns per column:

| Metric | Production ONNX | Trial ONNX | Trial MLX |
| --- | ---: | ---: | ---: |
| Waveform end → first write, p50 | 2,447 | 2,588 | 2,142 |
| Waveform end → first write, p95 | 3,557 | 3,764 | 2,906 |
| Waveform end → first signal-containing write, p50 | 2,447 | 2,588 | 2,364 |
| Waveform end → first signal-containing write, p95 | 3,557 | 3,764 | 3,116 |
| First TTS chunk, p50 | 560 | 585 | 200 |
| First TTS chunk, p95 | 1,347 | 1,355 | 533 |
| Processing through playback completion, p50 | 3,637 | 3,687 | 3,866 |
| Processing through playback completion, p95 | 8,563 | 8,668 | 8,618 |

MLX's first signal-containing write follows its first write by a median **212 ms**
per turn (p95 259 ms). Against trial ONNX in the same environment, MLX improves
the pooled first-write median by 17.3% and first-signal median by 8.7%. Against
production ONNX, those gains are 12.5% and 3.4%. Its first-signal p95 improves by
17.2% against trial ONNX and 12.4% against production. Full playback completion
does not improve at the median; the two engines generate different waveforms
and durations even for identical text.

The first-chunk benefit persists with all models loaded. Warm stage medians for
production ONNX / trial ONNX / MLX were 197 / 215 / 210 ms for transcription and
908 / 939 / 907 ms for chat calls. Waveform-end to processing remained about
712 ms, and opening output about 110–112 ms. Stage totals may include work after
playback begins; their percentiles cannot be added to reconstruct end-to-end
percentiles.

Six separately reported initial turns per condition had first-write medians of
3,215 / 3,174 / 3,617 ms. Cached process-startup medians were 6.92 / 7.27 / 7.30 s.
At the median, MLX's initial conversational turn remains slower despite the
warm-turn benefit.

### Block variation and paired inputs

First-write block medians in milliseconds:

| Round | Production ONNX | Trial ONNX | Trial MLX |
| --- | ---: | ---: | ---: |
| 1 | 2,618 | 2,692 | 2,252 |
| 2 | 2,546 | 2,534 | 2,099 |
| 3 | 2,493 | 2,364 | 2,224 |
| 4 | 2,452 | 2,954 | 2,199 |
| 5 | 2,376 | 2,478 | 2,088 |
| 6 | 2,377 | 2,424 | 2,093 |

For each input/history position, subtract production's first-signal latency
from the candidate's latency before taking the round's median. Negative means
the candidate was faster:

| Round | Trial ONNX − production, ms | MLX − production, ms |
| --- | ---: | ---: |
| 1 | +75 | −170 |
| 2 | +47 | −143 |
| 3 | −24 | −96 |
| 4 | +375 | −173 |
| 5 | +151 | −125 |
| 6 | −2 | −129 |

MLX's paired first-signal median favored it in every round, and 87/100 paired
turns were faster than production. The pooled paired median difference was
−147 ms; this differs from subtracting the two pooled medians (−83 ms), because
these are different statistics. Trial ONNX's pooled median was 5.8% slower than
production, but its paired round differences changed sign and ranged from
−24 to +375 ms. Treat these as descriptive observations from one Mac/session,
not independent-trial confidence intervals or proof of an environment effect.

### Host state and memory

All **501 host samples** showed nominal thermal state, AC power and low-power
mode off. No telemetry collection errors occurred. The largest sample gap was
8.0 seconds in both wall and monotonic time, including block transitions.
The monitor's collection duration was 58 ms median and 75 ms p95, versus the
approximately five-second sample interval; its cost is included in this run.

Samples taken during the turn phase, including the separate initial turns:

| Host metric | Production ONNX | Trial ONNX | Trial MLX |
| --- | ---: | ---: | ---: |
| Samples | 146 | 146 | 152 |
| CPU busy, median | 22.8% | 26.3% | 21.4% |
| CPU busy, p95 | 35.7% | 58.3% | 35.8% |
| Memory-pressure level 2 samples | 4/146 | 12/146 | 31/152 |

The busiest trial ONNX block was round four: host CPU median 50.5%, p95 82.7%,
maximum 88.0%, versus production's 25.2% and MLX's 21.2% medians that round.
Its first-write p95 was 5,123 ms, versus 3,642 / 2,831 ms for production / MLX.
These coincident readings identify a confound; they do not establish whether
the backend, background applications or another cause produced the load. No
busy block or individual outlier was removed or replaced, and no host-load
correction was applied. Power and thermal readings alone do not make the host
otherwise controlled.

Pressure levels were 1 in 439 samples and 2 in 62 samples across the full run.
MLX had more level-2 samples during turns, especially its first block. Host swap
was already 11,068 MiB before model startup and was 11,528 MiB at the last sample;
compressed memory changed from 5,066 to 8,095 MiB. These whole-host changes
include unrelated activity and the last sample still belonged to a turn phase.
They are not an application leak estimate or an after-shutdown measurement.

| Model resource metric | Production ONNX | Trial ONNX | Trial MLX |
| --- | ---: | ---: | ---: |
| Maximum worker high-water RSS | 5.61 GiB | 5.31 GiB | 5.12 GiB |
| Shared MLX allocator peak | 5.23 GiB | 5.23 GiB | 6.43 GiB |
| Shared MLX active memory at block end | 4.86 GiB | 4.86 GiB | 5.17 GiB |

End-of-block active allocations were effectively identical across the six
restarts of each condition. MLX's lower worker RSS does not cancel its additional
1.19 GiB allocator peak: these overlapping measures describe different aspects
of memory use. A continuous shared-model soak is still required.

## Decision and remaining gates

Follow-up: the [MLX live/soak validation](MLX_LIVE_SOAK.md) passed initial
operator playback/AEC/interruption checks and a 15-minute, 390-turn continuous
shared-model soak. Acoustic measurements and longer representative live/load
validation remain open. The comparison results below retain their original scope.

Keep production ONNX and its dependencies unchanged. Retain the isolated MLX
option: it consistently reduces synthesis time and paired response onset in
this workload, but its pooled median first-signal benefit against production
is modest and comes with more shared allocator memory and slower initial turns.

Do not roll back trial dependencies based on the pooled 5.8% ONNX gap alone.
The round variation, busy-host block and prior identical native-code/PCM probes
do not establish a fixed dependency regression. Conversely, this experiment
does not prove the environments equivalent. Do not reuse an old environment
gap as a correction factor for MLX.

Before making MLX the default, validate audible playback/AEC/interruption with
the integrated backend and run a longer continuous shared-model soak. A focused
next latency experiment could evaluate MLX's leading silence while guarding
against clipped initial phonemes and voice/language regressions; no trimming
or audio behavior was changed here. The earlier listening comparison remains
the speech-quality evidence. This muted synthetic run does not close acoustic
onset, live interruption, device recovery or the roadmap's matched Phase 1
baseline acceptance criteria.

## Reproduction

Prepare the [separate trial environment and shared fixtures](KOKORO_INTEGRATION.md).
Stop the normal voice runtime, keep the intended output device connected, and run
from the repository root:

```bash
uv run --locked scripts/kokoro_comparison.py \
  --fixtures tmp/kokoro-integration/fixtures.npz \
  --model tmp/kokoro-experiment/Kokoro-82M-bf16 \
  --output tmp/kokoro-comparison/new-100-turn-run
```

The command requires a fresh output directory, existing model assets and existing
production/trial environments. It does not install packages, enable MLX in your
private config or change power settings. Optional `--config` supplies source
model/audio settings; the runner creates its own synthetic profile and fixes
the protocol's voice/speed, history limit and disabled context features.
`--production-python` and `--trial-python` can select explicit existing virtual
environments. The default trial environment is under `tmp/kokoro-integration/venv`.

The directory contains the schedule, public dependency inventories, source/asset
hashes, per-block logs, report-directory manifest, host samples, summary and
final protected-file/dependency checks. Each block retains its full report and
environment metadata under `tmp/latency-baseline/`. All these local artifacts
remain ignored. Blocks have a ten-minute outer deadline with bounded cleanup.

The summary requires the complete planned schedule, successful block gates,
identical fixtures/model/audio settings, correct case/history order and all
four resident models. It reports response-hash differences explicitly rather
than hiding changed work. Paired differences compare conditions within each
round; they are descriptive rather than confidence intervals.

## Validation and evidence

The locked unit/synthetic evaluation suite passed **549 tests**. New coverage
checks schedule/order/case/history balance, telemetry parsing and CPU counter
wrap, incomplete/invalid evidence rejection, and explicit response mismatches.
Offline wheel/sdist builds and release privacy/version checks passed for 0.1.0.

All **31 config/knowledge files** retained their original fingerprints, with no
additions or removals. Both production and trial dependency inventories matched
their starting values. No application implementation, config template, persona,
credential, package manifest or lockfile was changed. The latency harness gained
corpus rotation and monotonic turn timestamps; its default ordering is unchanged.

Local evidence is under `tmp/kokoro-comparison/20261012-100-turns-v2/`, including
`runs.json`, `host.json`, `summary.json`, `preservation.json` and source/asset
identifiers. The manifest links all 18 full reports under `tmp/latency-baseline/`.
Raw reports, generated profiles, logs, model assets and environments stay ignored.

Measured SHA-256 identifiers:

- Comparison runner: `5c260f2d36f2650561037b8aba02175d3c113991bdd67a19d026073994757882`.
- Host monitor: `d90a5d3755280a4f33a596d33cc6eae0cb18873fb2a6bb5fd2290bac10ee6518`.
- Latency harness: `46f94bae6ef64c1623fcf185dfc7dcbc3705a0ae72745b970293939f1a71a1b3`.
- Shared input NPZ: `5d51f83f8d4808c9bc93c0403b02f6cbf5c0e24a04b8e07017e7cebffc154cf0`.
