"""
VAANI-SHIELD — Live Microphone Streaming Engine.

Captures continuous audio from the local microphone, accumulates incoming samples,
converts to mono float32, resamples to 16,000 Hz if necessary, and yields exact
1.0-second audio chunks (16,000 samples) matching the model's training specification.

Usage (standalone test)
-----------------------
    python -m src.live_mic --seconds 5
    python -m src.live_mic --list-devices
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.signal
import sounddevice as sd

from .audio import is_silent
from .config import VaaniConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)


def list_input_devices() -> List[dict]:
    """Return a list of available audio input devices."""
    devices = []
    try:
        raw_devices = sd.query_devices()
        for idx, dev in enumerate(raw_devices):
            if dev.get("max_input_channels", 0) > 0:
                devices.append({
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                })
    except Exception as exc:
        log.warning("Could not query sound devices: %s", exc)
    return devices


class LiveMicrophoneStream:
    """
    Continuous audio streamer from microphone into fixed 1-second chunks.

    Parameters
    ----------
    config         : VaaniConfig instance
    device_index   : int or None (None = default input device)
    chunk_callback : optional callable invoked as chunk_callback(chunk_16k, timestamp)
    target_sr      : target sample rate for the model (default: 16000 Hz)
    """

    def __init__(
        self,
        config: VaaniConfig = DEFAULT_CONFIG,
        device_index: Optional[int] = None,
        chunk_callback: Optional[Callable[[np.ndarray, float], None]] = None,
        target_sr: int = 16_000,
    ) -> None:
        self.config = config
        self.target_sr = target_sr
        self.device_index = device_index
        self.chunk_callback = chunk_callback

        self.chunk_samples = int(self.target_sr * self.config.chunk_seconds)  # 16,000

        self._queue: queue.Queue[Tuple[np.ndarray, float]] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        self._lock = threading.Lock()

        self._capture_sr = self.target_sr
        self._channels = 1
        self._buffer = np.array([], dtype=np.float32)

        self._determine_capture_settings()

    def _determine_capture_settings(self) -> None:
        """Determine whether direct 16kHz capture or native resampling is needed."""
        try:
            sd.check_input_settings(
                device=self.device_index,
                samplerate=self.target_sr,
                channels=1,
            )
            self._capture_sr = self.target_sr
            self._channels = 1
            log.info("Direct capture supported at %d Hz, 1 channel", self.target_sr)
        except Exception as exc:
            log.info("Direct 16kHz check failed (%s). Falling back to device default SR.", exc)
            try:
                dev_info = sd.query_devices(self.device_index, "input")
                self._capture_sr = int(dev_info["default_samplerate"])
                self._channels = min(2, int(dev_info["max_input_channels"]))
                log.info(
                    "Capturing at native device rate: %d Hz, %d channels with polyphase resampling.",
                    self._capture_sr, self._channels,
                )
            except Exception as e:
                log.warning("Could not query input device info: %s. Defaulting to 44100 Hz.", e)
                self._capture_sr = 44_100
                self._channels = 1

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Stream callback executed in sounddevice audio thread."""
        if status:
            log.warning("Sounddevice status flag: %s", status)

        if not self._is_running:
            return

        # 1. Convert to mono float32
        data = indata.astype(np.float32)
        if data.ndim > 1 and data.shape[1] > 1:
            mono = np.mean(data, axis=1)
        elif data.ndim > 1:
            mono = data[:, 0]
        else:
            mono = data

        with self._lock:
            # Accumulate raw input
            self._buffer = np.concatenate((self._buffer, mono))

            # Required samples at capture_sr for exactly 1.0 second of audio
            needed_samples = int(self._capture_sr * self.config.chunk_seconds)

            while len(self._buffer) >= needed_samples:
                raw_chunk = self._buffer[:needed_samples]
                self._buffer = self._buffer[needed_samples:]

                # 2. Resample to 16,000 Hz if needed
                if self._capture_sr == self.target_sr:
                    chunk_16k = raw_chunk
                else:
                    gcd = math.gcd(self.target_sr, self._capture_sr)
                    up = self.target_sr // gcd
                    down = self._capture_sr // gcd
                    chunk_16k = scipy.signal.resample_poly(raw_chunk, up, down).astype(np.float32)

                # Ensure exact chunk length
                if len(chunk_16k) < self.chunk_samples:
                    pad = np.zeros(self.chunk_samples - len(chunk_16k), dtype=np.float32)
                    chunk_16k = np.concatenate((chunk_16k, pad))
                elif len(chunk_16k) > self.chunk_samples:
                    chunk_16k = chunk_16k[:self.chunk_samples]

                ts = time.time()
                self._queue.put((chunk_16k, ts))

                if self.chunk_callback is not None:
                    try:
                        self.chunk_callback(chunk_16k, ts)
                    except Exception as e:
                        log.error("Error in chunk callback: %s", e)

    def start(self) -> None:
        """Start the audio stream."""
        if self._is_running:
            return

        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            self._stream = sd.InputStream(
                samplerate=self._capture_sr,
                channels=self._channels,
                device=self.device_index,
                callback=self._audio_callback,
                blocksize=int(self._capture_sr * 0.1),  # 100 ms blocks
                dtype="float32",
            )
            self._is_running = True
            self._stream.start()
            log.info("Microphone stream started (capture_sr=%d, target_sr=%d)", self._capture_sr, self.target_sr)

    def stop(self) -> None:
        """Stop the audio stream."""
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("Error closing stream: %s", e)
            self._stream = None
        log.info("Microphone stream stopped.")

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def capture_samplerate(self) -> int:
        return self._capture_sr

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, float]]:
        """
        Retrieve the next available 1.0-second chunk (16000 samples) from the queue.
        Returns (chunk_array, timestamp) or None if timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Standalone CLI Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = argparse.ArgumentParser(description="VAANI-SHIELD Live Microphone Test")
    parser.add_argument("--seconds", type=int, default=3, help="Seconds to capture")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices")
    parser.add_argument("--device", type=int, default=None, help="Device index")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable Audio Input Devices:")
        for dev in list_input_devices():
            print(f"  [{dev['index']}] {dev['name']} (channels={dev['channels']}, default_sr={dev['default_samplerate']})")
        exit(0)

    print(f"\nTesting live microphone capture for {args.seconds} seconds...")
    mic = LiveMicrophoneStream(device_index=args.device)
    mic.start()

    t_end = time.time() + args.seconds
    count = 0
    try:
        while time.time() < t_end:
            item = mic.get_chunk(timeout=1.2)
            if item is not None:
                chunk, ts = item
                count += 1
                rms = float(np.sqrt(np.mean(np.square(chunk))))
                max_amp = float(np.max(np.abs(chunk)))
                silent = is_silent(chunk)
                print(f"  Received chunk {count}: len={len(chunk)} | max_amp={max_amp:.4f} | RMS={rms:.4f} | Silent={silent}")
    finally:
        mic.stop()

    print(f"Capture complete. Received {count} complete 1.0-second chunks.\n")
