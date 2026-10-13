"""Device identity and recovery primitives for the voice process's audio owner."""
from __future__ import annotations

import time

import sounddevice as sd


class AudioDeviceError(RuntimeError):
    """A recoverable route/capture discontinuity; never retry the interrupted turn."""


DEVICE_ERRORS = (AudioDeviceError, sd.PortAudioError)


def device_inventory() -> list[dict]:
    hosts = sd.query_hostapis()
    defaults = sd.default.device
    return [dict(index=index, name=row['name'], host_api=hosts[row['hostapi']]['name'],
                 selector=f"{hosts[row['hostapi']]['name']}::{row['name']}",
                 max_input_channels=int(row['max_input_channels']),
                 max_output_channels=int(row['max_output_channels']),
                 default_samplerate=float(row['default_samplerate']),
                 default_input=index == defaults[0], default_output=index == defaults[1])
            for index, row in enumerate(sd.query_devices())]


def resolve_device(selector: str | None, kind: str) -> dict:
    """Match exact names, optionally host-qualified. Never silently fall back."""
    candidates = [row for row in device_inventory() if row[f'max_{kind}_channels'] > 0]
    if not selector:
        matches = [row for row in candidates if row[f'default_{kind}']]
    else:
        name = selector.strip().casefold()
        matches = [row for row in candidates if name in (row['name'].casefold(), row['selector'].casefold())]
    if not matches:
        raise AudioDeviceError(f'{kind.title()}Unavailable')
    if len(matches) != 1:
        raise AudioDeviceError(f'{kind.title()}Ambiguous')
    return matches[0]


def refresh_audio_backend() -> None:
    """Only call after the voice process has closed *all* its audio streams.

    The pinned sounddevice has no public refresh API. A balanced reinitialization
    refreshes PortAudio's cached indices/defaults; never persist those indices.
    A failed initialize is retried without a second unmatched terminate.
    """
    if sd._initialized:
        if sd._initialized != 1:
            raise AudioDeviceError('SharedAudioBackend')
        sd._terminate()
    sd._initialize()


def select_devices(audio) -> tuple[dict, dict | None]:
    source = resolve_device(audio.input_device, 'input')
    sd.check_input_settings(device=source['index'], channels=1, dtype='float32', samplerate=audio.sample_rate)
    output = None
    if audio.tts_enabled or audio.chime_enabled:
        output = resolve_device(audio.output_device, 'output')
        sd.check_output_settings(device=output['index'], channels=1, dtype='float32', samplerate=24000)
    return source, output


class CaptureHealth:
    def __init__(self, stall_s: float, *, clock=time.monotonic, wall=time.time):
        self.stall_s, self.clock, self.wall = stall_s, clock, wall
        self.started = self.last_poll = clock()
        self.last_wall = wall()
        self.last_callback = self.started

    def captured(self):
        self.last_callback = self.clock()

    def check(self, active=True):
        now, wall = self.clock(), self.wall()
        elapsed, wall_elapsed = now-self.last_poll, wall-self.last_wall
        self.last_poll, self.last_wall = now, wall
        if elapsed > self.stall_s or wall_elapsed-elapsed > self.stall_s:
            raise AudioDeviceError('SleepOrSchedulingGap')
        if active is False:
            raise AudioDeviceError('InputStreamInactive')
        if now-self.last_callback > self.stall_s:
            raise AudioDeviceError('CaptureStalled')
