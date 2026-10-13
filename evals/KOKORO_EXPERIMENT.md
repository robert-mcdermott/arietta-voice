# Isolated ONNX / MLX Kokoro comparison

October 12, 2026, Apple M4 Mac. Production remains on ONNX Kokoro. This experiment
does not load private config, persona, knowledge, memory or tools, and does not
open microphone or output devices. It writes synthetic WAVs and measurements
under ignored `tmp/kokoro-experiment/`. Candidate dependencies have a separate
virtual environment; the app dependency manifest and lockfile are unchanged.

## Candidates and reproducibility

- Baseline: locked `kokoro-onnx==0.5.0`, `onnxruntime==1.24.4`, v1.0 ONNX and voice
  assets already cached by Arietta. The harness calls the same `create_stream`
  API and eSpeak setup as the production adapter, with default CPU execution.
- Candidate: `mlx-audio==0.5.8`, `mlx==0.31.1`, `misaki[en]==0.9.4` and
  `en_core_web_sm==3.8.0`, with all resolved packages captured in
  [candidate requirements](kokoro_candidate_requirements.txt). Both environments
  use Python 3.13.3 and NumPy 2.4.4. The English spaCy model is installed explicitly
  before running, preventing Misaki's automatic package installation path.
- Model: `mlx-community/Kokoro-82M-bf16`, revision
  `a71e4d38b236d968966a2002c4c895dbd12b1c3c`. Despite that repository name, the pinned
  loader reports **float32 parameters**. This is not a quantized-model experiment.
  Both adapters use the public `af_heart` voice at speed 1.0 and 24 kHz output.
  Their voice tensors were compared directly: shape `(510, 1, 256)`, bit-identical
  values and maximum absolute difference zero. Asset SHA-256 and sizes are
  recorded in each report.

Sources: [ONNX implementation](https://github.com/thewh1teagle/kokoro-onnx),
[MLX Audio release](https://pypi.org/project/mlx-audio/0.5.8/),
[pinned model](https://huggingface.co/mlx-community/Kokoro-82M-bf16/tree/a71e4d38b236d968966a2002c4c895dbd12b1c3c),
and [MLX Kokoro implementation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/kokoro/kokoro.py).
The installed release source was inspected as well: Kokoro yields complete
segments and ignores generic `stream`/`streaming_interval` kwargs. Do not infer
subsentence streaming support from MLX Audio's generic streaming guide.

## Protocol and measurement boundaries

Eight fixed English cases cover acknowledgements, a short answer, numbers and
currency, dates/times, product names, technical abbreviations, a question and a
paragraph. Every fresh process first synthesizes the same separate warmup text.
Each measured block then renders all eight cases three times in a deterministic
shuffled order. Final order is ONNX → MLX → MLX → ONNX, with a three-second pause
between blocks: 48 warm utterances per backend, plus two first inferences each.
Pilots are excluded. No other test suite is run concurrently with measurement.

Timing starts at the synthesis request, including text processing. First-audio
latency stops when a nonempty chunk is materialized as CPU-readable samples;
MLX evaluation and conversion happen before the timestamp. Full synthesis ends
after generator exhaustion. WAV writes and signal diagnostics after synthesis
are outside timing. RTF is synthesis wall time divided by generated duration;
lower is better. These are TTS component timings, not microphone-to-speaker or
full-conversation latency. Different pronunciation/segmentation pipelines can
produce different audio lengths, so first-chunk timing and RTF must be read
together with the per-case results and listening assessment.

Imports/model loading and first inference are reported separately from warm
runs. Assets are predownloaded and hashed before timing, warming filesystem
caches: these are fresh-process, cached-asset starts, not cold boots/downloads.
CPU time sums all process threads. Process peak RSS includes imports and native
allocations; MLX active/cache/peak allocator memory is reported separately and
must not be added to RSS on unified memory. These measurements do not quantify
energy, GPU utilization, or contention with Arietta's LLM/STT models.

Waveform checks reject empty, nonfinite, oversized and incorrectly shaped/rated
output, and report silence, clipping, peak, RMS and segment boundaries. They do
not score naturalness or intelligibility. Human listening uses blinded A/B
labels; raw audio is retained separately from any listening-level adjustment.
Per-run reports retain the corpus/harness/asset hashes and chunk arrival times.
The whole synthesis process has a deadline, including native hangs; failed or
incomplete runs cannot be aggregated as successful results.

## Running the experiment

All commands use `uv`. Create the candidate environment without activating it
or installing into Arietta's `.venv`:

```bash
uv venv --python 3.13 tmp/kokoro-experiment/mlx-venv
uv pip install --python tmp/kokoro-experiment/mlx-venv/bin/python \
  -r evals/kokoro_candidate_requirements.txt
```

Download only `config.json`, `kokoro-v1_0.safetensors` and
`voices/af_heart.safetensors` from the pinned revision into
`tmp/kokoro-experiment/Kokoro-82M-bf16`. Retain the model-family directory name:
the upstream loader uses it to infer architecture when config lacks model type.
Use explicit local files for both engines. The harness forces Hugging Face
offline mode during synthesis and passes a local voice path to MLX.

```bash
uv run --locked - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'mlx-community/Kokoro-82M-bf16',
    revision='a71e4d38b236d968966a2002c4c895dbd12b1c3c',
    allow_patterns=['config.json', 'kokoro-v1_0.safetensors', 'voices/af_heart.safetensors'],
    local_dir='tmp/kokoro-experiment/Kokoro-82M-bf16',
    cache_dir='tmp/kokoro-experiment/hf-cache', token=False,
)
PY
```

```bash
uv run --locked scripts/kokoro_experiment.py --backend onnx \
  --model /absolute/path/to/kokoro-v1.0.onnx \
  --voices /absolute/path/to/voices-v1.0.bin \
  --output tmp/kokoro-experiment/1-onnx.json --repeats 3

uv run --no-project --python tmp/kokoro-experiment/mlx-venv/bin/python \
  scripts/kokoro_experiment.py --backend mlx \
  --model tmp/kokoro-experiment/Kokoro-82M-bf16 \
  --output tmp/kokoro-experiment/2-mlx.json --repeats 3
```

Repeat MLX then ONNX with fresh output names to reproduce the four-block order.
Run only one model benchmark at a time on an otherwise quiet Mac. Existing
reports are never overwritten by the CLI. Generated WAVs and environments stay
local; only source, synthetic tests, dependency pins and this report are tracked.

## Results and decision

All four final blocks completed and all 100 utterances (96 warm, four initial)
passed the signal checks, with no empty, silent, nonfinite or clipped output.
Every measured utterance produced one chunk on both backends. MLX materialized
audio earlier, but this corpus demonstrates faster whole-segment synthesis,
not incremental within-segment speech generation.

| Metric | ONNX CPU | MLX Metal |
| --- | ---: | ---: |
| Warm first-audio median | 1,175.4 ms | 421.8 ms |
| Warm first-audio p95 | 2,140.2 ms | 777.1 ms |
| Warm full-synthesis median | 1,175.5 ms | 436.4 ms |
| Warm synthesis RTF median | 0.249 | 0.080 |
| Warm CPU time median | 5,320.2 ms | 217.2 ms |
| Generated audio duration median | 4.77 s | 5.50 s |
| Cached import/load, two processes | 588.8–591.5 ms | 784.9–786.2 ms |
| First inference, two processes | 1,078.9–1,219.6 ms | 1,619.5–1,727.4 ms |
| Process peak RSS, two processes | 1,120.9–1,121.5 MiB | 791.5–794.1 MiB |
| MLX allocator peak | — | 1,942.3 MiB |
| MLX allocator active after run | — | 312.6 MiB |

MLX reduced median first-audio latency by **64.1%** and the descriptive p95 by
**63.7%** on this corpus. It was faster in every paired case. Its first inference
and import/load were slower, and it produced longer speech despite matching
speed and voice settings. These are 48 repeated warm samples over eight texts,
not a representative user-workload p95 or a statistical confidence interval.

| Case | ONNX first-audio median ms | MLX first-audio median ms | ONNX audio s | MLX audio s |
| --- | ---: | ---: | ---: | ---: |
| Acknowledgement | 606.7 | 198.9 | 1.94 | 2.35 |
| Short answer | 809.7 | 285.1 | 3.05 | 3.58 |
| Numbers/currency | 1,220.2 | 466.2 | 4.99 | 6.10 |
| Date/time | 1,342.8 | 443.7 | 5.61 | 5.83 |
| Product names | 1,128.4 | 395.1 | 4.54 | 5.17 |
| Technical words | 1,303.2 | 452.9 | 5.50 | 6.00 |
| Question | 957.7 | 344.7 | 3.67 | 4.25 |
| Paragraph | 2,135.8 | 776.9 | 10.01 | 10.65 |

The lower MLX CPU time reflects work on the GPU; it is not an energy-saving
measurement. Lower process RSS also does **not** establish lower total memory
pressure: MLX recorded a 1.90 GiB allocator peak, which RSS does not fully capture.
Do not add the two measurements or treat their difference as a memory saving.
Co-residency with the app's MLX language model still needs measurement.

Local final reports, raw WAVs and environment metadata are in
`tmp/kokoro-experiment/final/`. Rebuild the blinded listening pairs with:

```bash
uv run --locked -m scripts.kokoro_listening \
  --onnx tmp/kokoro-experiment/final/1-onnx.json \
  --mlx tmp/kokoro-experiment/final/2-mlx.json \
  --output tmp/kokoro-experiment/listening-new
```

Pairs cover numbers, technical terms and the paragraph. Overall RMS is matched
to -25 dBFS with a 0.95 peak cap; pauses are retained. This is not LUFS matching.
The same hidden A/B mapping applies throughout, stored separately in the local
answer key. After listening, the user preferred B's rendering of the money amount
in pair 1 and reported that the engines otherwise sounded similar across the
three pairs. Revealing the key: **A = ONNX, B = MLX**. This gives MLX a narrow
observed advantage for currency pronunciation, with otherwise similar perceived
quality in this review. It is one listener and three excerpts, not a general
quality ranking or evidence of equivalence across voices/languages. Objective
signal checks alone cannot establish a speech-quality winner.

**Decision:** proceed next with an opt-in MLX integration experiment, supported
by its large warm TTS latency improvement and this listening review's similar
perceived quality, with more natural currency pronunciation. Keep production
ONNX unchanged for now. Before a migration, measure full-turn latency and peak memory
with the LLM/STT resident, and cover cancellation, bounded queues, playback/AEC,
speaker interruption, supported voices/languages and rollback. The Phase 2
end-to-end latency and physical interruption exit criteria are not met by this
component benchmark alone.

## Validation and evidence identifiers

The full locked test/evaluation suite passed **506 tests**. New synthetic tests
exercise deterministic case ordering, materialized-chunk timing, invalid/empty
PCM, bounded output, process deadlines, incomplete or mismatched aggregation,
WAV encoding, listening-level matching and source-file tampering/path escape.
Offline wheel/source builds and release privacy/version checks passed. Hashes
of all 31 existing config/knowledge files matched before and after; no private
settings, persona or knowledge were changed. Production package versions,
`pyproject.toml` and `uv.lock` stayed unchanged.

Recorded identifiers (SHA-256):

- Corpus: `b763322a1568b2f936abfa951cea32d9d0621e624ed76ab9c133040d7d9238ab`.
- Measured harness: `3a6daafa0d374eb01724bc59e3f50951273d384abe895db37bcc04718fd4693e`.
- ONNX asset manifest: `5f283d8e030e494183b406fc01036ca26b5b31c67138922553662cf3d0c9b177`.
- MLX asset manifest: `e52a33522276fef31038b6ae03a1c2f913a7467e27b067988ec554cc575d18cc`.

Manifest digests hash `json.dumps(report['assets'], sort_keys=True).encode()`;
each manifest records filenames, byte sizes and individual asset hashes. ONNX
model plus bundled voices total 337.4 MiB; selected MLX weights/config/one voice
total 312.5 MiB. These disk sizes cover different voice inventories and are not
a comparison of complete installation sizes.
