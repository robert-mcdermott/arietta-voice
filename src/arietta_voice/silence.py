"""Even out the silence speech engines put around each synthesised clip.

Kokoro on ONNX starts a clip after about 35 ms and ends it about 90 ms after the
last sound. Kokoro on MLX pads every clip with about 300 ms before and 420 ms
after. Before the first clip that padding is dead time before the assistant is
heard, and between clauses it becomes a pause of most of a second. Trimming both
engines to the same short margins removes the delay and keeps the rhythm alike.
"""

from __future__ import annotations

import numpy as np

# Below about -60 dBFS: inaudible next to Kokoro's speech, which peaks near 0.2-0.5.
SILENCE_LEVEL = 1e-3
KEEP_LEADING_S = 0.04
KEEP_TRAILING_S = 0.10


def trim_silence(samples: np.ndarray, rate: int) -> np.ndarray:
    """Return samples with at most the standard margins of silence at each end.

    Clips that are entirely silent shrink to the leading margin rather than
    disappearing, so callers always receive some audio.
    """
    keep_leading = max(1, round(KEEP_LEADING_S * rate))
    keep_trailing = round(KEEP_TRAILING_S * rate)
    loud = np.flatnonzero(np.abs(samples) > SILENCE_LEVEL)
    if not len(loud):
        return samples[:keep_leading]
    start = max(0, loud[0] - keep_leading)
    end = min(len(samples), loud[-1] + 1 + keep_trailing)
    return samples[start:end]
