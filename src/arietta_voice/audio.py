from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import os
from pathlib import Path
import queue
import select
import sys
import termios
import time
import tty
from typing import Callable
import wave

import numpy as np
import sounddevice as sd

sd.default.latency = "high"


CHIME_SR = 24_000


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes())


def _fade_tone(freq: float, duration: float, amplitude: float = 0.6) -> np.ndarray:
    sample_count = int(duration * CHIME_SR)
    timeline = np.linspace(0, duration, sample_count, dtype=np.float32)
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.arange(sample_count) / (sample_count - 1)))
    return amplitude * np.sin(2 * np.pi * freq * timeline) * envelope


def _silence(duration: float) -> np.ndarray:
    return np.zeros(int(duration * CHIME_SR), dtype=np.float32)


def make_chime(duration: float = 30.0, tick_every: float = 1.5) -> np.ndarray:
    head = np.concatenate([_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)])
    tick = _fade_tone(550, 0.04, amplitude=0.18)
    total = int(duration * CHIME_SR)
    buffer = np.zeros(total, dtype=np.float32)
    buffer[: len(head)] = head
    step = int(tick_every * CHIME_SR)
    for offset in range(len(head), total, step):
        end = min(offset + len(tick), total)
        buffer[offset:end] = tick[: end - offset]
    return buffer


class ChimePlayer:
    def __init__(self, enabled: bool) -> None:
        self.sound = make_chime() if enabled else None
        self.started_at = 0.0

    def start(self) -> None:
        if self.sound is None:
            return
        sd.play(self.sound, CHIME_SR)
        self.started_at = time.monotonic()

    def stop_at_gap(self) -> None:
        if self.sound is None or self.started_at == 0:
            return
        chime_head = 0.22
        tick_duration = 0.04
        tick_every = 1.5
        elapsed = time.monotonic() - self.started_at
        if elapsed < chime_head:
            time.sleep(chime_head - elapsed)
        else:
            phase = (elapsed - chime_head) % tick_every
            if phase < tick_duration:
                time.sleep(tick_duration - phase + 0.005)
        sd.stop()
        self.started_at = 0.0


class MicrophoneStream(AbstractContextManager["MicrophoneStream"]):
    def __init__(
        self,
        sample_rate: int,
        chunk_samples: int,
        *,
        channels: int = 1,
        record_path: Path | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.channels = channels
        self.record_path = record_path
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._record_chunks: list[np.ndarray] | None = [] if record_path else None
        self._stream = None

    def __enter__(self) -> "MicrophoneStream":
        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                print(status, file=sys.stderr)
            chunk = indata[:, 0].copy()
            if self._record_chunks is not None:
                self._record_chunks.append(chunk)
            self._audio_q.put(chunk)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self.record_path and self._record_chunks:
            save_wav(self.record_path, np.concatenate(self._record_chunks), self.sample_rate)

    def get(self, timeout: float | None = None) -> np.ndarray | None:
        try:
            return self._audio_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_nowait(self) -> np.ndarray:
        return self._audio_q.get_nowait()

    def empty(self) -> bool:
        return self._audio_q.empty()

    def drain(self) -> None:
        while not self._audio_q.empty():
            self._audio_q.get_nowait()


class RawKeypressMonitor(AbstractContextManager["RawKeypressMonitor"]):
    def __init__(self) -> None:
        self.enabled = os.name == "posix" and sys.stdin.isatty()
        self._old_term = None

    def __enter__(self) -> "RawKeypressMonitor":
        if self.enabled:
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.enabled and self._old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)

    def pressed(self) -> bool:
        if not self.enabled:
            return False
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
            return True
        return False


def make_aec_processor(sample_rate: int):
    from livekit.rtc import AudioFrame
    from livekit.rtc.apm import AudioProcessingModule

    frame_width = 160

    def to_i16(chunk: np.ndarray) -> np.ndarray:
        scaled = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
        if len(scaled) < frame_width:
            return np.pad(scaled, (0, frame_width - len(scaled)))
        return scaled

    def to_frame(chunk: np.ndarray) -> AudioFrame:
        return AudioFrame(
            chunk.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=frame_width,
        )

    apm = AudioProcessingModule(echo_cancellation=True, noise_suppression=True)

    def process(mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        cleaned = np.zeros_like(mic)
        for offset in range(0, len(mic), frame_width):
            mic_piece = mic[offset : offset + frame_width]
            ref_piece = reference[offset : offset + frame_width]
            mic_frame = to_frame(to_i16(mic_piece))
            apm.process_reverse_stream(to_frame(to_i16(ref_piece)))
            apm.process_stream(mic_frame)
            cleaned[offset : offset + len(mic_piece)] = (
                np.frombuffer(bytes(mic_frame.data), dtype=np.int16).astype(np.float32) / 32767
            )[: len(mic_piece)]
        return cleaned

    return process


def _get_reference_segment(tts_audio: np.ndarray, offset: int, length: int) -> np.ndarray:
    if offset >= len(tts_audio):
        return np.zeros(length, dtype=np.float32)
    segment = tts_audio[offset : offset + length]
    if len(segment) < length:
        return np.concatenate([segment, np.zeros(length - len(segment), dtype=np.float32)])
    return segment


@dataclass(slots=True)
class PlaybackResult:
    interrupted: bool
    buffered_chunks: list[np.ndarray] = field(default_factory=list)


def _playback_level_from_chunk(playback_chunk: np.ndarray) -> float:
    flattened = playback_chunk[:, 0]
    rms = float(np.sqrt(np.mean(np.square(flattened, dtype=np.float32)))) if len(flattened) else 0.0
    return min(1.0, rms * 7.0)


def play_tts_stream_passive(
    *,
    tts_stream,
    chime_player: ChimePlayer,
    playback_level_callback: Callable[[float, bool], None] | None = None,
) -> None:
    async def _play() -> None:
        output_stream = None
        async for chunk_samples, sr in tts_stream:
            if output_stream is None:
                chime_player.stop_at_gap()
                output_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                output_stream.start()

            frame = chunk_samples.reshape(-1, 1)
            for offset in range(0, len(frame), 1024):
                playback_chunk = frame[offset : offset + 1024]
                if playback_level_callback is not None:
                    playback_level_callback(_playback_level_from_chunk(playback_chunk), True)
                output_stream.write(playback_chunk)

        if output_stream is not None:
            output_stream.stop()
            output_stream.close()
        if playback_level_callback is not None:
            playback_level_callback(0.0, False)

    asyncio.run(_play())


def play_tts_stream(
    *,
    tts_stream,
    microphone: MicrophoneStream,
    vad,
    keypress_monitor: RawKeypressMonitor,
    chime_player: ChimePlayer,
    sample_rate: int,
    use_aec: bool,
    barge_in_threshold_chunks: int,
    playback_level_callback: Callable[[float, bool], None] | None = None,
) -> PlaybackResult:
    microphone.drain()
    aec_process = make_aec_processor(sample_rate) if use_aec else None
    output_stream = None
    tts_reference_chunks: list[np.ndarray] = []
    interrupted = False
    state = {
        "playback_started": None,
        "consecutive_speech": 0,
        "mic_offset": 0,
        "speech_chunks": [],
    }

    def check_barge_in() -> tuple[bool, list[np.ndarray]]:
        if aec_process is None or state["playback_started"] is None or not tts_reference_chunks:
            return False, []
        if time.monotonic() - state["playback_started"] < 0.5:
            return False, []

        reference_audio = np.concatenate(tts_reference_chunks)
        while not microphone.empty():
            mic_chunk = microphone.get_nowait()
            if len(mic_chunk) == 0:
                continue
            reference = _get_reference_segment(reference_audio, state["mic_offset"], len(mic_chunk))
            state["mic_offset"] += len(mic_chunk)
            cleaned = aec_process(mic_chunk, reference)
            if vad.probability(cleaned.astype(np.float32)) > 0.8:
                state["consecutive_speech"] += 1
                state["speech_chunks"].append(mic_chunk)
                if state["consecutive_speech"] >= barge_in_threshold_chunks:
                    return True, list(state["speech_chunks"])
            else:
                state["consecutive_speech"] = 0
                state["speech_chunks"].clear()
        return False, []

    async def _play() -> PlaybackResult:
        nonlocal interrupted, output_stream
        buffered_chunks: list[np.ndarray] = []
        async for chunk_samples, sr in tts_stream:
            if output_stream is None:
                chime_player.stop_at_gap()
                output_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                output_stream.start()
                microphone.drain()
                vad.reset()
                state["playback_started"] = time.monotonic()

            if aec_process is not None:
                if sr == sample_rate:
                    tts_reference_chunks.append(chunk_samples.astype(np.float32))
                else:
                    positions = np.arange(0, len(chunk_samples), sr / sample_rate)
                    resampled = np.interp(positions, np.arange(len(chunk_samples)), chunk_samples)
                    tts_reference_chunks.append(resampled.astype(np.float32))

            frame = chunk_samples.reshape(-1, 1)
            for offset in range(0, len(frame), 1024):
                if keypress_monitor.pressed():
                    interrupted = True
                    break
                voice_interrupt, buffered_chunks = check_barge_in()
                if voice_interrupt:
                    interrupted = True
                    break
                playback_chunk = frame[offset : offset + 1024]
                if playback_level_callback is not None:
                    playback_level_callback(_playback_level_from_chunk(playback_chunk), True)
                output_stream.write(playback_chunk)
            if interrupted:
                break

        if output_stream is not None:
            output_stream.stop()
            output_stream.close()
        if playback_level_callback is not None:
            playback_level_callback(0.0, False)
        microphone.drain()
        vad.reset()
        return PlaybackResult(interrupted=interrupted, buffered_chunks=buffered_chunks)

    return asyncio.run(_play())
