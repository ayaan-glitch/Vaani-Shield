"""
Tests for src.features — log-Mel spectrogram extraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import VaaniConfig
from src.features import extract_log_mel, batch_extract_log_mel


@pytest.fixture
def config() -> VaaniConfig:
    return VaaniConfig(sample_rate=16000, chunk_seconds=1.0, n_mels=64)


def make_tone(config: VaaniConfig, freq: float = 440.0) -> np.ndarray:
    sr = config.sample_rate
    n = int(sr * config.chunk_seconds)
    t = np.linspace(0, config.chunk_seconds, n, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * freq * t)


class TestExtractLogMel:
    def test_output_shape(self, config):
        chunk = make_tone(config)
        feat = extract_log_mel(chunk, config.sample_rate, config)
        assert feat.shape == (config.n_mels, config.n_frames), (
            f"Expected ({config.n_mels}, {config.n_frames}), got {feat.shape}"
        )

    def test_output_dtype(self, config):
        chunk = make_tone(config)
        feat = extract_log_mel(chunk, config.sample_rate, config)
        assert feat.dtype == np.float32

    def test_output_range(self, config):
        """All values must be in [0, 1]."""
        chunk = make_tone(config)
        feat = extract_log_mel(chunk, config.sample_rate, config)
        assert float(feat.min()) >= 0.0 - 1e-6
        assert float(feat.max()) <= 1.0 + 1e-6

    def test_no_nan_or_inf(self, config):
        chunk = make_tone(config)
        feat = extract_log_mel(chunk, config.sample_rate, config)
        assert not np.any(np.isnan(feat)), "NaN values in feature"
        assert not np.any(np.isinf(feat)), "Inf values in feature"

    def test_deterministic(self, config):
        """Same input must produce identical output (no random state)."""
        chunk = make_tone(config)
        feat1 = extract_log_mel(chunk, config.sample_rate, config)
        feat2 = extract_log_mel(chunk, config.sample_rate, config)
        np.testing.assert_array_equal(feat1, feat2)

    def test_silence_gives_low_values(self, config):
        """Silent chunk should produce near-zero features after normalisation."""
        silence = np.zeros(int(config.sample_rate * config.chunk_seconds), dtype=np.float32)
        feat = extract_log_mel(silence, config.sample_rate, config)
        # Silence → everything at -top_db dB → normalised to 0.0
        assert float(feat.mean()) < 0.1, "Silence should give near-zero features"

    def test_short_chunk_padded(self, config):
        """Chunk shorter than one second should be padded, not crash."""
        short = make_tone(config)[:1000]  # very short
        feat = extract_log_mel(short, config.sample_rate, config)
        assert feat.shape == (config.n_mels, config.n_frames)

    def test_long_chunk_trimmed(self, config):
        """Chunk longer than one second should be trimmed, not crash."""
        long_chunk = np.tile(make_tone(config), 3)
        feat = extract_log_mel(long_chunk, config.sample_rate, config)
        assert feat.shape == (config.n_mels, config.n_frames)

    def test_different_inputs_differ(self, config):
        """440Hz and 1000Hz tones should produce different features."""
        feat_440 = extract_log_mel(make_tone(config, 440), config.sample_rate, config)
        feat_1k = extract_log_mel(make_tone(config, 1000), config.sample_rate, config)
        assert not np.allclose(feat_440, feat_1k), "Different tones should give different features"


class TestBatchExtractLogMel:
    def test_batch_shape(self, config):
        chunks = [make_tone(config) for _ in range(5)]
        batch = batch_extract_log_mel(chunks, config.sample_rate, config)
        assert batch.shape == (5, config.n_mels, config.n_frames)

    def test_batch_consistent_with_single(self, config):
        """Batch extraction must give same result as single extraction."""
        chunks = [make_tone(config, f) for f in [220, 440, 880]]
        batch = batch_extract_log_mel(chunks, config.sample_rate, config)
        for i, chunk in enumerate(chunks):
            single = extract_log_mel(chunk, config.sample_rate, config)
            np.testing.assert_array_equal(batch[i], single)
