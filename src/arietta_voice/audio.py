from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from collections import deque
import threading
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

    def stop(self) -> None:
        if self.started_at:
            sd.stop()
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


class RecordingWriter:
    """A bounded nonblocking callback queue; the writer alone owns the WAV file."""
    def __init__(self, path: Path, sample_rate: int, max_chunks: int, max_seconds: float) -> None:
        if max_chunks < 1 or max_seconds <= 0:
            raise ValueError("Recording limits must be positive")
        self.path, self.sample_rate = path, sample_rate
        self.queue: queue.Queue = queue.Queue(maxsize=max_chunks)
        self.max_samples = int(max_seconds * sample_rate)
        self.accepted_samples = 0
        self.dropped_chunks = 0
        self.limit_reached = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._write, name="arietta-recording", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def append(self, chunk: np.ndarray) -> None:
        if self.error or self.limit_reached or self._stop.is_set():
            return
        remaining = self.max_samples - self.accepted_samples
        piece = chunk[:remaining].copy()
        try:
            self.queue.put_nowait(piece)
            self.accepted_samples += len(piece)
        except queue.Full:
            self.dropped_chunks += 1
        self.limit_reached = self.accepted_samples >= self.max_samples

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with os.fdopen(os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "wb") as file:
                os.fchmod(file.fileno(), 0o600)
                with wave.open(file, "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(self.sample_rate)
                    while not self._stop.is_set() or not self.queue.empty():
                        try:
                            chunk = self.queue.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        handle.writeframes((chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
        except Exception as exc:
            self.error = type(exc).__name__
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

    def close(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
        if self._thread.is_alive():
            self.error = "RecordingCloseTimeout"


class MicrophoneStream(AbstractContextManager["MicrophoneStream"]):
    def __init__(self, sample_rate: int, chunk_samples: int, *, channels: int = 1,
                 record_path: Path | None = None, max_queue_chunks: int = 64,
                 recording_queue_max_chunks: int = 256, max_recording_s: float = 3600.0) -> None:
        if max_queue_chunks < 1:
            raise ValueError("Microphone queue limit must be positive")
        self.sample_rate, self.chunk_samples, self.channels = sample_rate, chunk_samples, channels
        self.record_path = record_path
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_chunks)
        self.dropped_chunks = 0
        self.input_overflows = 0
        self._recorder = RecordingWriter(record_path, sample_rate, recording_queue_max_chunks, max_recording_s) if record_path else None
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            self.input_overflows += 1
        chunk = indata[:, 0].copy()
        if self._recorder is not None:
            self._recorder.append(chunk)
        try:
            self._audio_q.put_nowait(chunk)
        except queue.Full:
            # Preserve recent audio without blocking PortAudio's callback.
            try:
                self._audio_q.get_nowait()
                self.dropped_chunks += 1
            except queue.Empty:
                pass
            try:
                self._audio_q.put_nowait(chunk)
            except queue.Full:
                self.dropped_chunks += 1

    def __enter__(self) -> "MicrophoneStream":
        if self._recorder is not None:
            self._recorder.start()
        try:
            self._stream = sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                                          dtype="float32", blocksize=self.chunk_samples, callback=self._callback)
            self._stream.start()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                finally:
                    self._stream.close()
        finally:
            self._stream = None
            if self._recorder is not None:
                self._recorder.close()
            self.drain()

    def statistics(self) -> dict:
        recorder = self._recorder
        return {"queued_chunks": self._audio_q.qsize(), "queue_max_chunks": self._audio_q.maxsize,
                "dropped_chunks": self.dropped_chunks, "input_overflows": self.input_overflows,
                "recording_dropped_chunks": recorder.dropped_chunks if recorder else 0,
                "recording_limit_reached": recorder.limit_reached if recorder else False,
                "recording_error": recorder.error if recorder else None}

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
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                return


class UtteranceBuffer:
    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self.chunks: list[np.ndarray] = []
        self.samples = 0

    def append(self, chunk: np.ndarray) -> bool:
        if self.samples + len(chunk) > self.max_samples:
            return False
        self.chunks.append(chunk)
        self.samples += len(chunk)
        return True

    def clear(self) -> None:
        self.chunks.clear()
        self.samples = 0

    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)


class AudioReferenceBuffer:
    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self.samples = np.zeros(0, dtype=np.float32)
        self.end = 0

    def append(self, samples: np.ndarray) -> None:
        self.end += len(samples)
        self.samples = np.concatenate((self.samples, samples))[-self.max_samples:].copy()

    def read(self, offset: int, length: int) -> np.ndarray:
        result = np.zeros(length, dtype=np.float32)
        start = self.end - len(self.samples)
        first, last = max(offset, start), min(offset + length, self.end)
        if last > first:
            result[first - offset:last - offset] = self.samples[first - start:last - start]
        return result


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


def play_tts_stream_passive(*, tts_stream, chime_player: ChimePlayer,
                            playback_level_callback=None, deadline: float | None = None, cancellation=None) -> None:
    play_tts_stream(tts_stream=tts_stream, microphone=None, vad=None, keypress_monitor=None,
                    chime_player=chime_player, sample_rate=16000, use_aec=False,
                    barge_in_threshold_chunks=5, playback_level_callback=playback_level_callback,
                    deadline=deadline, cancellation=cancellation)


def play_tts_stream(*, tts_stream, microphone, vad, keypress_monitor, chime_player: ChimePlayer,
                    sample_rate: int, use_aec: bool, barge_in_threshold_chunks: int,
                    playback_level_callback=None, reference_seconds: float = 4.0,
                    deadline: float | None = None, cancellation=None) -> PlaybackResult:
    from contextlib import ExitStack
    from arietta_voice.worker import WorkerTimeout

    if microphone is not None:
        microphone.drain()
    aec_process = make_aec_processor(sample_rate) if use_aec else None
    reference = AudioReferenceBuffer(int(reference_seconds * sample_rate))
    speech_chunks = deque(maxlen=barge_in_threshold_chunks)
    playback_started = None
    mic_offset = 0
    observed_drops = None

    def check_barge_in() -> tuple[bool, list[np.ndarray]]:
        nonlocal mic_offset, observed_drops
        if aec_process is None or playback_started is None or time.monotonic() - playback_started < 0.5:
            return False, []
        drops = microphone.dropped_chunks + microphone.input_overflows
        if drops != observed_drops:
            observed_drops = drops
            microphone.drain()
            speech_chunks.clear()
            mic_offset = int((time.monotonic() - playback_started) * sample_rate)
            return False, []
        while microphone is not None:
            try:
                chunk = microphone.get_nowait()
            except queue.Empty:
                break
            cleaned = aec_process(chunk, reference.read(mic_offset, len(chunk)))
            mic_offset += len(chunk)
            if vad.probability(cleaned.astype(np.float32)) > 0.8:
                speech_chunks.append(chunk)
                if len(speech_chunks) >= barge_in_threshold_chunks:
                    return True, list(speech_chunks)
            else:
                speech_chunks.clear()
        return False, []

    async def _play() -> PlaybackResult:
        nonlocal playback_started, observed_drops
        output_stream = None
        buffered_chunks: list[np.ndarray] = []
        interrupted = False
        completed = False
        try:
            async for chunk_samples, sr in tts_stream:
                if cancellation is not None:
                    cancellation.check()
                if deadline is not None and time.monotonic() >= deadline:
                    raise WorkerTimeout("Speech playback exceeded the turn time limit")
                if output_stream is None:
                    chime_player.stop_at_gap()
                    output_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                    output_stream.start()
                    if microphone is not None:
                        microphone.drain()
                        observed_drops = microphone.dropped_chunks + microphone.input_overflows
                    if vad is not None:
                        vad.reset()
                    playback_started = time.monotonic()
                if aec_process is not None:
                    positions = np.arange(0, len(chunk_samples), sr / sample_rate)
                    reference.append(np.interp(positions, np.arange(len(chunk_samples)), chunk_samples).astype(np.float32))
                frame = chunk_samples.reshape(-1, 1)
                for offset in range(0, len(frame), 1024):
                    if cancellation is not None:
                        cancellation.check()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise WorkerTimeout("Speech playback exceeded the turn time limit")
                    if keypress_monitor is not None and keypress_monitor.pressed():
                        interrupted = True
                        if cancellation is not None:
                            cancellation.cancel()
                        break
                    voice_interrupt, buffered_chunks = check_barge_in()
                    if voice_interrupt:
                        interrupted = True
                        if cancellation is not None:
                            cancellation.cancel()
                        break
                    playback_chunk = frame[offset:offset + 1024]
                    if playback_level_callback is not None:
                        playback_level_callback(_playback_level_from_chunk(playback_chunk), True)
                    output_stream.write(playback_chunk)
                if interrupted:
                    break
            completed = True
            return PlaybackResult(interrupted=interrupted, buffered_chunks=buffered_chunks)
        finally:
            # Release/abort the device before waiting for inference cancellation.
            # Generator cleanup still runs even if a device cleanup raises.
            try:
                with ExitStack() as cleanup:
                    if playback_level_callback is not None:
                        cleanup.callback(playback_level_callback, 0.0, False)
                    cleanup.callback(chime_player.stop)
                    if microphone is not None and not buffered_chunks:
                        cleanup.callback(microphone.drain)
                    if vad is not None:
                        cleanup.callback(vad.reset)
                    if output_stream is not None:
                        cleanup.callback(output_stream.close)
                        abort = not completed or interrupted or (cancellation is not None and cancellation.cancelled)
                        cleanup.callback(output_stream.abort if abort else output_stream.stop)
            finally:
                if hasattr(tts_stream, "aclose"):
                    await tts_stream.aclose()

    return asyncio.run(_play())
