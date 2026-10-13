import asyncio

import numpy as np
import pytest

from scripts import output_underflow_probe as probe


@pytest.mark.parametrize('gap_s', [0., .6])
def test_control_schedule_keeps_pcm_exact_and_exposes_gap_polls(monkeypatch, gap_s):
    now = [10.]
    monkeypatch.setattr(probe.time, 'monotonic', lambda:now[0])
    async def sleep(seconds):
        now[0] += seconds
    monkeypatch.setattr(probe.asyncio, 'sleep', sleep)
    pcm = np.arange(5000,dtype=np.float32)
    async def collect():
        frames, polls = [], 0
        async for samples, rate in probe.scheduled_pcm(pcm,gap_s=gap_s):
            if samples is None:
                polls += 1
            else:
                assert rate == 24000 and len(samples) <= 4096
                assert not np.shares_memory(samples,pcm)
                frames.append(samples)
        np.testing.assert_array_equal(np.concatenate(frames),np.tile(pcm,2))
        assert bool(polls) == bool(gap_s)
    asyncio.run(collect())
    assert now[0]-10 == pytest.approx(gap_s)
