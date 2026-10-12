# Initial hardware baseline

Recorded October 11, 2026 (America/Los_Angeles; October 12 UTC), using app
**0.1.0** and the existing locked dependencies. Real-model component smoke and
the requested **15-minute live microphone soak passed**. This is an idle-heavy
initial baseline with guided speech, not a sustained generation benchmark.
See [HARDWARE_VALIDATION.md](HARDWARE_VALIDATION.md) for repeatable commands.

## Environment and isolation

| Setting | Recorded value |
| --- | --- |
| Mac | Apple M4, 32 GiB RAM, arm64 |
| OS / Python | macOS 15.7.9 / Python 3.13.3 |
| Input / output | Jabra SPEAK 410 USB |
| Capture | 16 kHz, 512-sample chunks, queue capacity 64 |
| Speech output | Kokoro `af_heart`, speed 1.0 |
| Echo cancellation / Smart Turn | Both enabled |
| Chat | `mlx-community/gemma-4-E4B-it-4bit` |
| Chat revision | `cc3b666c01c20395e0dcebd53854504c7d9821f9` |
| STT | Moonshine `medium-streaming-en`, through the existing batch interface |
| Packages | moonshine-voice 0.0.54; mlx-vlm 0.4.4; mlx 0.31.1; kokoro-onnx 0.5.0; onnxruntime 1.24.4; silero-vad 6.2.1; torch 2.11.0; sounddevice 0.5.5 |
| Application source commit | `60231545680bd7795a71892c52bab2b6a0082de7` plus the uncommitted validation harness/docs |
| Lock SHA-256 | `66e417fd303b79a832585a55a08b4fe1c0b97fef14c661e764404dd868543f0c` |

Tests used an isolated synthetic persona, default Bridget wake phrases,
temperature zero, and disabled knowledge, memory, custom tools, and content
logging. No recordings were saved. All 24 ignored private files under config
and knowledge matched their before-test hashes; tracked settings and the
lockfile were unchanged. The final smoke also passed the harness's full
config/knowledge fingerprint comparison.

The initial smoke and live run used a 64-token response cap. The final harness
preserves the configured output limit; its final component smoke used 220.
The live baseline therefore does not establish behavior at the full configured
limit over 15 minutes. Missing Smart Turn assets were obtained by the normal
loader; dependencies were not upgraded.

## Measured results

| Check | Result |
| --- | --- |
| Initial component smoke | Passed real VAD, synthetic STT/wake, endpoint, chat and TTS |
| Initial warmup / single round trip | 9.302 s / 2.705 s |
| Final harness component smoke | Passed; 6.686 s warmup / 1.756 s single round trip |
| Synthetic endpoint probability | 0.9854 |
| Live interval | 03:28–03:43 UTC; 905.7 s after readiness including shutdown |
| Failed turns / timeouts / worker restarts | 0 / 0 / 0 |
| Native input-device overflows | 0 |
| Bounded capture queue drops | 17 old chunks discarded; capacity remained 64 |
| Maximum sampled heartbeat age | 2.3 s |
| Sessions / session ends | 3 / 3: two idle timeouts and one explicit user goodbye |
| Response/audio-completion events | 18 / 18, including acknowledgements and farewells |
| Recorded voice interruptions | 1 |
| Process-tree RSS peak | 3386.7 MiB |
| First / last warm RSS window medians | 2289.1 / 1056.45 MiB |
| Resource samples | 182, approximately five seconds apart |
| Shutdown | Clean; worker closed, no last error |
| Event/status privacy | No conversation content fields; no malformed event records |

The final event/privacy/heartbeat gates were assessed from the completed run's
artifacts using the finished harness. The live process had started with an
earlier harness revision, which did not yet emit these aggregate gates itself.
Raw reports, asset fingerprints and device metadata remain in ignored local
`tmp/hardware-validation/` directories; only reviewed aggregate results are
recorded here.

## Human observations

The operator confirmed audible wake acknowledgement, a clear answer of four,
and that a question mentioning goodbye kept the conversation active. A separate
farewell check produced a goodbye and remained silent for approximately 25
seconds; the log confirmed `session_ended` with `user_goodbye`.

The first counting request exposed an incomplete synthetic persona: the model
claimed it could not speak. Only the isolated test persona was corrected to
explain the application's TTS output. On retry, the operator confirmed counting
aloud, interruption of the count, and a response of six to the follow-up. The
private persona was never edited. The corrected persona is part of the harness.

## Limits and next gates

The queue discarded old audio while the runtime was busy. No loss was reported
in the confirmed checks, but the drops warrant observation under heavier speech
load. Device overflow and queue discard are different measurements.

RSS did not show net growth between the sampled warm windows. RSS is not total
Metal/unified memory, and falling residency does not prove absence of a leak.
The two single-turn timings are functional smoke observations, not meaningful
p50/p95 estimates or speech-end-to-first-audio measurements. Response events are
not a count of model turns, and wake-miss events do not establish a false-wake
rate without labeled audio.

Still separate: sustained repeated-model load, representative quality tests,
100-turn stage/first-audio latency measurements, measured interruption latency,
sleep/device reconnect, the first hosted CI run, and the Phase 4/5 eight-hour
and 24-hour soak gates. This baseline does not mark those checks complete.
