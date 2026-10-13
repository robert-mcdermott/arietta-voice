from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from collections import Counter, deque
import threading
import math
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
from arietta_voice.devices import CaptureHealth

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
    def __init__(self, enabled: bool, device=None) -> None:
        self.sound = make_chime() if enabled else None
        self.started_at = 0.0
        self.device = device

    def stop(self) -> None:
        if self.started_at:
            try:
                try:
                    sd.stop()
                finally:
                    sd.get_stream().close()
            finally:
                self.started_at = 0.0

    def start(self) -> None:
        if self.sound is None:
            return
        self.started_at = time.monotonic()
        try:
            sd.play(self.sound, CHIME_SR, device=self.device)
        except BaseException:
            # A convenience stream can be created before its start fails.
            # Release it before the runtime reinitializes PortAudio.
            try:
                self.stop()
            except Exception:
                pass
            raise

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
        self.stop()


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


@dataclass(slots=True)
class CaptureChunk:
    samples: np.ndarray
    captured_at: float


class MicrophoneStream(AbstractContextManager["MicrophoneStream"]):
    def __init__(self, sample_rate: int, chunk_samples: int, *, channels: int = 1,
                 record_path: Path | None = None, max_queue_chunks: int = 64,
                 recording_queue_max_chunks: int = 256, max_recording_s: float = 3600.0,
                 device=None, stall_s: float = 3.0, recorder: RecordingWriter | None = None) -> None:
        if max_queue_chunks < 1:
            raise ValueError("Microphone queue limit must be positive")
        self.sample_rate, self.chunk_samples, self.channels = sample_rate, chunk_samples, channels
        self.record_path = record_path
        self.device, self.stall_s = device, stall_s
        self._health = None
        self._audio_q: queue.Queue[CaptureChunk] = queue.Queue(maxsize=max_queue_chunks)
        self.dropped_chunks = 0
        self.input_overflows = 0
        self.capture_phase = "listening"
        self.drops_by_phase = Counter()
        self.discarded_by_reason = Counter()
        self._owns_recorder = recorder is None
        self._recorder = recorder or (RecordingWriter(record_path, sample_rate, recording_queue_max_chunks, max_recording_s) if record_path else None)
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if self._health is not None:
            self._health.captured()
        if status:
            self.input_overflows += 1
        chunk = indata[:, 0].copy()
        now = time.monotonic()
        # PortAudio times share a device clock; translate its ADC age into the
        # monotonic domain used by playback. Never infer capture time at dequeue.
        captured_at = now - len(chunk) / self.sample_rate
        if time_info is not None:
            age = float(time_info.currentTime - time_info.inputBufferAdcTime)
            if math.isfinite(age) and 0 <= age <= 10:
                captured_at = now - age
        packet = CaptureChunk(chunk, captured_at)
        if self._recorder is not None:
            self._recorder.append(chunk)
        try:
            self._audio_q.put_nowait(packet)
        except queue.Full:
            # Preserve recent audio without blocking PortAudio's callback.
            try:
                self._audio_q.get_nowait()
                self._capture_drop()
            except queue.Empty:
                pass
            try:
                self._audio_q.put_nowait(packet)
            except queue.Full:
                self._capture_drop()

    def _capture_drop(self):
        self.dropped_chunks += 1
        self.drops_by_phase[self.capture_phase] += 1

    def set_capture_phase(self, phase):
        if phase not in {"listening", "endpoint", "transcribe", "awaiting_output", "playback", "playback_cleanup"}:
            raise ValueError("Unknown capture phase")
        self.capture_phase = phase

    def __enter__(self) -> "MicrophoneStream":
        if self._recorder is not None and self._owns_recorder:
            self._recorder.start()
        try:
            self._health = CaptureHealth(self.stall_s)
            self._stream = sd.InputStream(samplerate=self.sample_rate, channels=self.channels,
                                          dtype="float32", blocksize=self.chunk_samples, callback=self._callback,
                                          device=self.device)
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
            if self._recorder is not None and self._owns_recorder:
                self._recorder.close()
            self.drain()

    def check_health(self) -> None:
        if self._health is not None and self._stream is not None:
            self._health.check(self._stream.active)

    def statistics(self) -> dict:
        recorder = self._recorder
        return {"queued_chunks": self._audio_q.qsize(), "queue_max_chunks": self._audio_q.maxsize,
                "dropped_chunks": self.dropped_chunks, "input_overflows": self.input_overflows,
                "capture_phase": self.capture_phase, "drops_by_phase": dict(self.drops_by_phase),
                "discarded_chunks": sum(self.discarded_by_reason.values()),
                "discarded_by_reason": dict(self.discarded_by_reason),
                "recording_dropped_chunks": recorder.dropped_chunks if recorder else 0,
                "recording_limit_reached": recorder.limit_reached if recorder else False,
                "recording_error": recorder.error if recorder else None}

    def get(self, timeout: float | None = None) -> np.ndarray | None:
        self.check_health()
        try:
            return self._audio_q.get(timeout=timeout).samples
        except queue.Empty:
            return None

    def get_nowait(self) -> np.ndarray:
        return self.get_capture_nowait().samples

    def get_capture_nowait(self) -> CaptureChunk:
        self.check_health()
        return self._audio_q.get_nowait()

    def empty(self) -> bool:
        return self._audio_q.empty()

    def drain(self) -> None:
        self.discard_pending("turn_boundary")

    def discard_pending(self, reason: str) -> None:
        if reason not in {"turn_boundary", "transcribing", "before_output", "aec_disabled", "capture_gap"}:
            raise ValueError("Unknown capture discard reason")
        # Snapshot the bounded queue: never chase newly arriving callbacks forever.
        for _ in range(self._audio_q.qsize()):
            try:
                self._audio_q.get_nowait()
                self.discarded_by_reason[reason] += 1
            except queue.Empty:
                return


def discard_capture(microphone, reason):
    if microphone is not None:
        if hasattr(type(microphone), "discard_pending"):
            microphone.discard_pending(reason)
        else:
            microphone.drain()


def capture_phase(microphone, phase):
    if microphone is not None and hasattr(type(microphone), "set_capture_phase"):
        microphone.set_capture_phase(phase)


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


class TimedAudioReferenceBuffer:
    """Only submitted output, on a monotonic timeline; gaps remain silence."""
    def __init__(self, max_seconds: float) -> None:
        self.max_seconds = max_seconds
        self.blocks = deque()

    def clear(self) -> None:
        self.blocks.clear()

    def append(self, samples: np.ndarray, rate: int, started_at: float) -> None:
        end = started_at + len(samples) / rate
        # Bound even a single oversized producer block.
        samples = samples[-max(1, int(self.max_seconds * rate)):].copy()
        started_at = end - len(samples) / rate
        self.blocks.append((started_at, rate, samples))
        while self.blocks and self.blocks[0][0] < end - self.max_seconds:
            start, old_rate, old = self.blocks.popleft()
            skip = max(0, math.ceil((end - self.max_seconds - start) * old_rate))
            if skip < len(old):
                self.blocks.appendleft((start + skip / old_rate, old_rate, old[skip:].copy()))
                break

    def read(self, captured_at: float, length: int, sample_rate: int) -> np.ndarray:
        times = captured_at + np.arange(length) / sample_rate
        result = np.zeros(length, dtype=np.float32)
        for start, rate, samples in self.blocks:
            selected = (times >= start) & (times < start + len(samples) / rate)
            if selected.any():
                result[selected] = np.interp((times[selected] - start) * rate,
                                             np.arange(len(samples)), samples)
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

    if sample_rate != 16000:
        raise ValueError("AEC requires 16 kHz input")
    frame_width = sample_rate // 100

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
    pending_mic = np.zeros(0, dtype=np.float32)
    pending_ref = np.zeros(0, dtype=np.float32)
    # One fixed frame of algorithmic delay, rather than padding each 32ms chunk
    # and incorrectly advancing the native AEC clock by 40ms.
    output = np.zeros(frame_width, dtype=np.float32)

    def process_frame(mic, ref):
        mic_frame = to_frame(to_i16(mic))
        apm.process_reverse_stream(to_frame(to_i16(ref)))
        # Reference was already aligned to capture time before entering APM.
        apm.set_stream_delay_ms(0)
        apm.process_stream(mic_frame)
        return np.frombuffer(bytes(mic_frame.data), dtype=np.int16).astype(np.float32) / 32767

    def process(mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
        nonlocal pending_mic, pending_ref, output
        if len(mic) != len(reference):
            raise ValueError("AEC inputs must have matching lengths")
        pending_mic = np.concatenate((pending_mic, mic))
        pending_ref = np.concatenate((pending_ref, reference))
        frames = len(pending_mic) // frame_width
        pieces = [output]
        for offset in range(0, frames * frame_width, frame_width):
            pieces.append(process_frame(pending_mic[offset:offset+frame_width],
                                        pending_ref[offset:offset+frame_width]))
        pending_mic = pending_mic[frames*frame_width:].copy()
        pending_ref = pending_ref[frames*frame_width:].copy()
        output = np.concatenate(pieces)
        cleaned, output = output[:len(mic)].copy(), output[len(mic):].copy()
        return cleaned

    def flush():
        nonlocal pending_mic, pending_ref, output
        if len(pending_mic):
            output = np.concatenate((output, process_frame(pending_mic, pending_ref)[:len(pending_mic)]))
        result = output.copy()
        pending_mic = pending_ref = output = np.zeros(0, dtype=np.float32)
        return result

    process.flush = flush

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
    alignment_resets: int = 0
    output_underflows: int = 0
    alignment_reset_reasons: dict[str, int] = field(default_factory=dict)
    capture_discard_reasons: dict[str, int] = field(default_factory=dict)


def _playback_level_from_chunk(playback_chunk: np.ndarray) -> float:
    flattened = playback_chunk[:, 0]
    rms = float(np.sqrt(np.mean(np.square(flattened, dtype=np.float32)))) if len(flattened) else 0.0
    return min(1.0, rms * 7.0)


def play_tts_stream_passive(*, tts_stream, chime_player: ChimePlayer,
                            playback_level_callback=None, deadline: float | None = None, cancellation=None,
                            keypress_monitor=None, output_device=None, health_check=None) -> None:
    play_tts_stream(tts_stream=tts_stream, microphone=None, vad=None, keypress_monitor=keypress_monitor,
                    chime_player=chime_player, sample_rate=16000, use_aec=False,
                    barge_in_threshold_chunks=5, playback_level_callback=playback_level_callback,
                    deadline=deadline, cancellation=cancellation, output_device=output_device,
                    health_check=health_check)


def play_tts_stream(*, tts_stream, microphone, vad, keypress_monitor, chime_player: ChimePlayer,
                    sample_rate: int, use_aec: bool, barge_in_threshold_chunks: int,
                    playback_level_callback=None, reference_seconds: float = 4.0,
                    deadline: float | None = None, cancellation=None,
                    pre_roll_ms: float = 0, aec_delay_ms: float = 0,
                    output_device=None, health_check=None) -> PlaybackResult:
    from contextlib import ExitStack
    from arietta_voice.worker import WorkerTimeout

    capture_phase(microphone, "awaiting_output")
    discard_capture(microphone, "before_output")
    aec_process = make_aec_processor(sample_rate) if use_aec else None
    reference = TimedAudioReferenceBuffer(reference_seconds)
    speech_chunks = deque(maxlen=barge_in_threshold_chunks + math.ceil(pre_roll_ms * sample_rate / 512000))
    consecutive_speech = 0
    playback_started = None
    observed_drops = None
    capture_end = None
    output_end = None
    alignment_resets = 0
    output_underflows = 0
    reset_reasons, discard_reasons = Counter(), Counter()

    def reset_capture(reason):
        nonlocal aec_process, consecutive_speech, capture_end, alignment_resets
        alignment_resets += 1
        reset_reasons[reason] += 1
        speech_chunks.clear()
        consecutive_speech = 0
        capture_end = None
        if vad is not None:
            vad.reset()
        if use_aec:
            aec_process = make_aec_processor(sample_rate)

    def check_barge_in() -> tuple[bool, list[np.ndarray]]:
        nonlocal observed_drops, capture_end, consecutive_speech
        if keypress_monitor is not None and keypress_monitor.pressed():
            return True, []
        if playback_started is None:
            # Capture before speech output is not a supported barge-in window.
            # It was already drained at stream startup; retire it during model
            # waits instead of letting that unused backlog overrun the queue.
            discard_capture(microphone, "before_output")
            return False, []
        if aec_process is None or microphone is None:
            discard_capture(microphone, "aec_disabled")
            return False, []
        drops = microphone.dropped_chunks + microphone.input_overflows
        if drops != observed_drops:
            observed_drops = drops
            discard_capture(microphone, "capture_gap")
            reset_capture("capture_overflow")
            return False, []
        # Bound work per poll even when capture continues while VAD runs.
        for _ in range(64):
            try:
                if hasattr(type(microphone), 'get_capture_nowait'):
                    packet = microphone.get_capture_nowait()
                    chunk, captured_at = packet.samples, packet.captured_at
                else:
                    chunk = microphone.get_nowait()
                    captured_at = time.monotonic() - len(chunk) / sample_rate
            except queue.Empty:
                break
            if len(chunk) != 512:
                discard_reasons["malformed_capture"] += 1
                reset_capture("malformed_capture")
                continue
            if captured_at < playback_started:
                # ADC packets arriving after OutputStream.start may precede its
                # boundary. A fresh AEC has no alignment to invalidate yet.
                discard_reasons["before_playback"] += 1
                if capture_end is not None:
                    reset_capture("timestamp_gap")
                continue
            if capture_end is not None and abs(captured_at - capture_end) > .02:
                reset_capture("timestamp_gap")
            capture_end = captured_at + len(chunk) / sample_rate
            if time.monotonic() - capture_end > reference_seconds:
                discard_reasons["stale_capture"] += 1
                reset_capture("stale_capture")
                continue
            cleaned = aec_process(chunk, reference.read(captured_at - aec_delay_ms / 1000,
                                                       len(chunk), sample_rate))
            speech_chunks.append(cleaned.copy())
            if vad.probability(cleaned.astype(np.float32)) > 0.8:
                consecutive_speech += 1
                if consecutive_speech >= barge_in_threshold_chunks:
                    buffered = list(speech_chunks)
                    if hasattr(aec_process, 'flush'):
                        buffered.append(aec_process.flush())
                    return True, buffered
            else:
                consecutive_speech = 0
        return False, []

    async def _play() -> PlaybackResult:
        nonlocal playback_started, observed_drops, output_end, output_underflows
        output_stream = None
        buffered_chunks: list[np.ndarray] = []
        interrupted = False
        completed = False
        try:
            async for chunk_samples, sr in tts_stream:
                if health_check is not None:
                    health_check()
                if cancellation is not None:
                    cancellation.check()
                if deadline is not None and time.monotonic() >= deadline:
                    raise WorkerTimeout("Speech playback exceeded the turn time limit")
                if isinstance(chunk_samples, Exception):
                    raise chunk_samples
                if chunk_samples is None:
                    voice_interrupt, buffered_chunks = check_barge_in()
                    if voice_interrupt:
                        interrupted = True
                        if cancellation is not None:
                            cancellation.cancel()
                        break
                    continue
                if output_stream is None:
                    chime_player.stop_at_gap()
                    output_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", device=output_device)
                    output_stream.start()
                    if microphone is not None:
                        discard_capture(microphone, "before_output")
                        observed_drops = microphone.dropped_chunks + microphone.input_overflows
                    if vad is not None:
                        vad.reset()
                    playback_started = time.monotonic()
                    capture_phase(microphone, "playback")
                elif sr != output_rate:
                    raise ValueError("Sample rate changed during playback")
                output_rate = sr
                frame = chunk_samples.reshape(-1, 1)
                for offset in range(0, len(frame), 1024):
                    if health_check is not None:
                        health_check()
                    if cancellation is not None:
                        cancellation.check()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise WorkerTimeout("Speech playback exceeded the turn time limit")
                    voice_interrupt, buffered_chunks = check_barge_in()
                    if voice_interrupt:
                        interrupted = True
                        if cancellation is not None:
                            cancellation.cancel()
                        break
                    playback_chunk = frame[offset:offset + 1024]
                    if playback_level_callback is not None:
                        playback_level_callback(_playback_level_from_chunk(playback_chunk), True)
                    # Blocking PortAudio does not expose DAC timestamps. Estimate
                    # presentation from reported latency and queued sample duration,
                    # and preserve real gaps between clauses. Never stage an entire
                    # future TTS chunk in the rolling echo reference.
                    latency = output_stream.latency
                    latency = float(latency) if isinstance(latency, (int, float)) else 0.0
                    if not math.isfinite(latency) or latency < 0:
                        raise ValueError("Invalid output latency")
                    presented_at = max(time.monotonic() + latency, output_end or 0)
                    underflow = output_stream.write(playback_chunk)
                    output_end = presented_at + len(playback_chunk) / sr
                    if underflow is True:
                        output_underflows += 1
                        reference.clear()
                        reset_capture("output_underflow")
                        output_end = None
                    elif aec_process is not None:
                        reference.append(playback_chunk[:, 0], sr, presented_at)
                if interrupted:
                    break
            # Remain interruptible while the final submitted samples are still
            # queued at the device; stop() itself would otherwise block this poll.
            while not interrupted and output_end is not None and time.monotonic() < output_end:
                if health_check is not None:
                    health_check()
                if cancellation is not None:
                    cancellation.check()
                if deadline is not None and time.monotonic() >= deadline:
                    raise WorkerTimeout("Speech playback exceeded the turn time limit")
                interrupted, buffered_chunks = check_barge_in()
                if interrupted:
                    if cancellation is not None:
                        cancellation.cancel()
                    break
                time.sleep(.01)
            completed = True
            return PlaybackResult(interrupted=interrupted, buffered_chunks=buffered_chunks,
                                  alignment_resets=alignment_resets, output_underflows=output_underflows,
                                  alignment_reset_reasons=dict(reset_reasons),
                                  capture_discard_reasons=dict(discard_reasons))
        finally:
            capture_phase(microphone, "playback_cleanup")
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
                try:
                    if hasattr(tts_stream, "aclose"):
                        await tts_stream.aclose()
                    # Generator shutdown may take long enough for capture to overflow.
                    # Never splice a preserved prefix onto discontinuous later audio.
                    if buffered_chunks and microphone is not None:
                        if microphone.dropped_chunks + microphone.input_overflows != observed_drops:
                            buffered_chunks.clear()
                            microphone.drain()
                finally:
                    capture_phase(microphone, "listening")

    return asyncio.run(_play())
