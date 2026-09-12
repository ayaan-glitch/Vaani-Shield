"""
Tests for src.audio — load, chunk, silence detection, file discovery.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.audio import (
    load_audio,
    chunk_audio,
    is_silent,
    find_audio_files,
)
from src.config import VaaniConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config() -> VaaniConfig:
    return VaaniConfig(sample_rate=16000, chunk_seconds=1.0)


@pytest.fixture
def tmp_wav(tmp_path: Path, config: VaaniConfig) -> Path:
    """Write a 3-second 16kHz sine-wave WAV and return its path."""
    sr = config.sample_rate
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "test_tone.wav"
    sf.write(str(wav_path), audio, sr)
    return wav_path


@pytest.fixture
def tmp_silent_wav(tmp_path: Path, config: VaaniConfig) -> Path:
    sr = config.sample_rate
    audio = np.zeros(sr, dtype=np.float32)
    wav_path = tmp_path / "silent.wav"
    sf.write(str(wav_path), audio, sr)
    return wav_path


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------
class TestLoadAudio:
    def test_loads_and_returns_mono_float32(self, tmp_wav, config):
        samples, sr = load_audio(tmp_wav, config)
        assert samples.ndim == 1
        assert samples.dtype == np.float32
        assert sr == config.sample_rate

    def test_resamples_correctly(self, tmp_path, config):
        """Write at 8kHz, expect output at 16kHz."""
        sr_orig = 8000
        t = np.linspace(0, 1.0, sr_orig, endpoint=False, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        wav = tmp_path / "low_sr.wav"
        sf.write(str(wav), audio, sr_orig)

        samples, sr = load_audio(wav, config)
        assert sr == config.sample_rate
        # Duration should be ~1 second → ~16000 samples (±5%)
        assert abs(len(samples) - config.sample_rate) < 0.05 * config.sample_rate

    def test_converts_stereo_to_mono(self, tmp_path, config):
        sr = config.sample_rate
        stereo = np.random.rand(sr * 2, 2).astype(np.float32)
        wav = tmp_path / "stereo.wav"
        sf.write(str(wav), stereo, sr)

        samples, _ = load_audio(wav, config)
        assert samples.ndim == 1

    def test_raises_file_not_found(self, config):
        with pytest.raises(FileNotFoundError):
            load_audio(Path("/nonexistent/path/audio.wav"), config)


# ---------------------------------------------------------------------------
# chunk_audio
# ---------------------------------------------------------------------------
class TestChunkAudio:
    def test_correct_number_of_chunks_exact(self, config):
        sr = config.sample_rate
        # exactly 3 seconds → 3 chunks
        samples = np.random.rand(sr * 3).astype(np.float32)
        chunks = chunk_audio(samples, sr, config, pad_last=True)
        assert len(chunks) == 3

    def test_correct_number_of_chunks_with_remainder(self, config):
        sr = config.sample_rate
        # 3.5 seconds → 4 chunks (last padded)
        samples = np.random.rand(int(sr * 3.5)).astype(np.float32)
        chunks = chunk_audio(samples, sr, config, pad_last=True)
        assert len(chunks) == 4

    def test_all_chunks_same_length_when_padded(self, config):
        sr = config.sample_rate
        samples = np.random.rand(int(sr * 2.7)).astype(np.float32)
        chunks = chunk_audio(samples, sr, config, pad_last=True)
        expected_len = int(sr * config.chunk_seconds)
        for c in chunks:
            assert len(c) == expected_len

    def test_last_chunk_shorter_without_padding(self, config):
        sr = config.sample_rate
        samples = np.random.rand(int(sr * 1.7)).astype(np.float32)
        chunks = chunk_audio(samples, sr, config, pad_last=False)
        assert len(chunks) == 2
        assert len(chunks[-1]) < sr  # last chunk is short

    def test_empty_input_returns_empty_list(self, config):
        sr = config.sample_rate
        chunks = chunk_audio(np.array([], dtype=np.float32), sr, config)
        assert chunks == []


# ---------------------------------------------------------------------------
# is_silent
# ---------------------------------------------------------------------------
class TestIsSilent:
    def test_detects_silence(self, tmp_silent_wav, config):
        sr = config.sample_rate
        silence = np.zeros(sr, dtype=np.float32)
        assert is_silent(silence, config) is True

    def test_not_silent_for_tone(self, config):
        sr = config.sample_rate
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        tone = 0.3 * np.sin(2 * np.pi * 440 * t)
        assert is_silent(tone, config) is False


# ---------------------------------------------------------------------------
# find_audio_files
# ---------------------------------------------------------------------------
class TestFindAudioFiles:
    def test_finds_wav_files(self, tmp_path, config):
        sr = config.sample_rate
        audio = np.zeros(sr, dtype=np.float32)
        for name in ["a.wav", "b.wav"]:
            sf.write(str(tmp_path / name), audio, sr)

        found = find_audio_files(tmp_path)
        assert len(found) == 2

    def test_recursive_search(self, tmp_path, config):
        sub = tmp_path / "sub"
        sub.mkdir()
        sr = config.sample_rate
        audio = np.zeros(sr, dtype=np.float32)
        sf.write(str(sub / "nested.wav"), audio, sr)

        found = find_audio_files(tmp_path)
        assert len(found) == 1
        assert found[0].name == "nested.wav"

    def test_empty_directory(self, tmp_path):
        found = find_audio_files(tmp_path)
        assert found == []

    def test_nonexistent_directory(self):
        found = find_audio_files(Path("/nonexistent"))
        assert found == []

    def test_sorted_order(self, tmp_path, config):
        sr = config.sample_rate
        audio = np.zeros(sr, dtype=np.float32)
        for name in ["c.wav", "a.wav", "b.wav"]:
            sf.write(str(tmp_path / name), audio, sr)

        found = find_audio_files(tmp_path)
        names = [f.name for f in found]
        assert names == sorted(names)
