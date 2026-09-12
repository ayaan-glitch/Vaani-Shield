"""
Tests for src.live_mic — LiveMicrophoneStream, audio accumulation, resampling, and device discovery.
"""

from __future__ import annotations

import math
import queue
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import VaaniConfig
from src.live_mic import LiveMicrophoneStream, list_input_devices


@pytest.fixture
def config() -> VaaniConfig:
    return VaaniConfig(sample_rate=16000, chunk_seconds=1.0)


class TestDeviceListing:
    def test_list_input_devices_returns_list(self):
        devices = list_input_devices()
        assert isinstance(devices, list)
        for dev in devices:
            assert "index" in dev
            assert "name" in dev
            assert "channels" in dev
            assert "default_samplerate" in dev
            assert dev["channels"] > 0


class TestLiveMicrophoneStream:
    def test_init_defaults(self, config):
        stream = LiveMicrophoneStream(config=config)
        assert stream.target_sr == 16000
        assert stream.chunk_samples == 16000
        assert stream.is_running is False
        assert stream.get_chunk(timeout=0.01) is None

    def test_audio_callback_direct_16k_mono(self, config):
        stream = LiveMicrophoneStream(config=config)
        stream._is_running = True
        stream._capture_sr = 16000
        stream._channels = 1

        # Feed 16,000 samples in two blocks of 8,000
        block1 = np.ones((8000, 1), dtype=np.float32) * 0.1
        block2 = np.ones((8000, 1), dtype=np.float32) * 0.2

        stream._audio_callback(block1, 8000, {}, 0)
        assert stream.get_chunk(timeout=0.01) is None  # Only 0.5s, no chunk yet

        stream._audio_callback(block2, 8000, {}, 0)
        item = stream.get_chunk(timeout=0.1)
        assert item is not None
        chunk, ts = item
        assert isinstance(chunk, np.ndarray)
        assert chunk.shape == (16000,)
        assert chunk.dtype == np.float32
        assert isinstance(ts, float)

    def test_audio_callback_resampling_from_44100_stereo(self, config):
        callback_mock = MagicMock()
        stream = LiveMicrophoneStream(config=config, chunk_callback=callback_mock)
        stream._is_running = True
        stream._capture_sr = 44100
        stream._channels = 2

        # Send 1.0 second of 44.1kHz stereo audio
        stereo_44k = np.ones((44100, 2), dtype=np.float32) * 0.25
        stream._audio_callback(stereo_44k, 44100, {}, 0)

        item = stream.get_chunk(timeout=0.5)
        assert item is not None
        chunk, ts = item
        assert chunk.shape == (16000,)
        assert chunk.dtype == np.float32
        assert callback_mock.called

    def test_start_and_stop_lifecycle(self, config):
        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_inst = MagicMock()
            mock_stream_cls.return_value = mock_inst

            stream = LiveMicrophoneStream(config=config)
            stream.start()
            assert stream.is_running is True
            mock_inst.start.assert_called_once()

            stream.stop()
            assert stream.is_running is False
            mock_inst.stop.assert_called_once()
            mock_inst.close.assert_called_once()
