# Benchmarks

Opt-in harnesses that exercise real models and audio hardware. They use an
isolated synthetic persona and never read or change private settings or
knowledge. Run them from the repository root.

| Harness | Purpose | Guide |
| --- | --- | --- |
| `latency_baseline.py` | Stage timings over 100 warm turns. The standing benchmark: rerun it after every pipeline change. | [Latency baseline](../evals/LATENCY_BASELINE.md) |
| `hardware_validation.py` | Real-model smoke test, live microphone/speaker checks and a bounded soak. Includes the forced-gap playback mode from `forced_gap_validation.py`. | [Hardware validation](../evals/HARDWARE_VALIDATION.md) |
| `live_load_soak.py` | Long synthetic-input, live-output soak with audio fault gates. | [Live/load soak](../evals/LIVE_LOAD_SOAK.md) |
| `host_state.py` | Read-only CPU, thermal, power and memory-pressure sampling shared by the harnesses. | — |

```bash
uv run --locked bench/latency_baseline.py --turns 100 --audible
```

Tests for these harnesses run with the main suite under `tests/`.

## Archived experiments

`archive/` holds harnesses for experiments that have reached a decision. They
are kept runnable for reference but are outside the default test run and CI.
Their reports are in
[docs/archive/experiments/](../docs/archive/experiments/).

```bash
uv run --locked pytest bench/archive/tests
```

| Experiment | Decision |
| --- | --- |
| Kokoro ONNX vs MLX (`kokoro_*.py`, `onnx_environment.py`) | ONNX stays the default; MLX stays opt-in. Reopen only if the latency benchmark shows TTS is the largest remaining stage. |
| Moonshine streaming (`moonshine_experiment.py`) | No accuracy gain in batch use; streaming promotion is scheduled in the roadmap's latency milestone. |
| Output underflow probe (`output_underflow_probe.py`) | Forced-gap underflows fixed; the trigger for the remaining MLX-path underflows is unproven. |

New experiments are time-boxed to one working day with a decision rule written
down first; see [ROADMAP.md](../ROADMAP.md).
