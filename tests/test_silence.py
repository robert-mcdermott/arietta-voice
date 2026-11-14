import numpy as np
import pytest

from arietta_voice.silence import trim_silence

RATE = 24000


def clip(lead_s, speech_s, trail_s, level=0.3):
    return np.concatenate(
        [
            np.zeros(round(lead_s * RATE), dtype=np.float32),
            np.full(round(speech_s * RATE), level, dtype=np.float32),
            np.zeros(round(trail_s * RATE), dtype=np.float32),
        ]
    )


def edges(samples):
    loud = np.flatnonzero(np.abs(samples) > 1e-3)
    return loud[0] / RATE, (len(samples) - loud[-1] - 1) / RATE


def test_long_padding_is_cut_to_the_standard_margins():
    trimmed = trim_silence(clip(0.30, 1.0, 0.42), RATE)
    lead, trail = edges(trimmed)
    assert lead == pytest.approx(0.04, abs=1 / RATE)
    assert trail == pytest.approx(0.10, abs=1 / RATE)
    assert np.count_nonzero(trimmed) == RATE  # Every speech sample is kept.


def test_short_padding_and_quiet_speech_are_left_alone():
    original = clip(0.03, 0.5, 0.08)
    np.testing.assert_array_equal(trim_silence(original, RATE), original)
    # A soft onset just above the silence level counts as sound, not padding.
    soft = clip(0.30, 0.5, 0.0, level=0.002)
    assert edges(trim_silence(soft, RATE))[0] == pytest.approx(0.04, abs=1 / RATE)


def test_silence_inside_speech_is_kept():
    gap = np.concatenate([clip(0.3, 0.2, 0.5), clip(0.0, 0.2, 0.3)])
    trimmed = trim_silence(gap, RATE)
    assert len(trimmed) == round((0.04 + 0.2 + 0.5 + 0.2 + 0.1) * RATE)


def test_an_entirely_silent_clip_shrinks_but_never_vanishes():
    trimmed = trim_silence(np.zeros(RATE, dtype=np.float32), RATE)
    assert len(trimmed) == round(0.04 * RATE)
    assert len(trim_silence(np.zeros(3, dtype=np.float32), RATE)) == 3
