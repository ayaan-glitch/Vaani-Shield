"""
Tests for src.inference — VaaniInferenceEngine, aggregation, evaluate helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.config import VaaniConfig
from src.inference import VaaniInferenceEngine
from src.aggregation import ChunkAggregator
from src.evaluate import compute_eer, compute_auc, compute_confusion_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config() -> VaaniConfig:
    return VaaniConfig(sample_rate=16000, chunk_seconds=1.0, n_mels=64)


@pytest.fixture
def tmp_wav(tmp_path: Path, config: VaaniConfig) -> Path:
    sr = config.sample_rate
    t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def untrained_engine(config) -> VaaniInferenceEngine:
    """Engine with no checkpoint → UNTRAINED mode."""
    return VaaniInferenceEngine(checkpoint_path=None, config=config)


# ---------------------------------------------------------------------------
# VaaniInferenceEngine — UNTRAINED mode
# ---------------------------------------------------------------------------
class TestInferenceEngineUntrained:
    def test_is_not_trained(self, untrained_engine):
        assert untrained_engine.is_trained is False

    def test_predict_chunk_returns_05(self, untrained_engine, config):
        sr = config.sample_rate
        chunk = 0.3 * np.sin(np.linspace(0, 1, sr, dtype=np.float32))
        prob = untrained_engine.predict_chunk(chunk, sr)
        assert prob == 0.5

    def test_predict_file_returns_two_tuple(self, untrained_engine, tmp_wav):
        final, chunks = untrained_engine.predict(tmp_wav)
        assert isinstance(final, float)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_predict_file_all_chunks_05(self, untrained_engine, tmp_wav):
        _, chunk_scores = untrained_engine.predict(tmp_wav)
        for score in chunk_scores:
            assert score == 0.5, f"Expected 0.5 in untrained mode, got {score}"

    def test_missing_checkpoint_gives_untrained(self, config):
        engine = VaaniInferenceEngine(
            checkpoint_path=Path("/nonexistent/checkpoint.pt"), config=config
        )
        assert not engine.is_trained


# ---------------------------------------------------------------------------
# VaaniInferenceEngine — TRAINED mode (fake checkpoint)
# ---------------------------------------------------------------------------
class TestInferenceEngineTrained:
    @pytest.fixture
    def trained_engine(self, tmp_path, config):
        """Create a real (random-weight) checkpoint and load it."""
        from src.model import build_model
        model = build_model(config)
        ckpt_path = tmp_path / "fake_best.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "val_eer": 0.5,
                "config": config,
            },
            ckpt_path,
        )
        return VaaniInferenceEngine(checkpoint_path=ckpt_path, config=config)

    def test_is_trained(self, trained_engine):
        assert trained_engine.is_trained is True

    def test_predict_chunk_returns_float_in_range(self, trained_engine, config):
        sr = config.sample_rate
        chunk = np.random.randn(sr).astype(np.float32) * 0.1
        prob = trained_engine.predict_chunk(chunk, sr)
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range"

    def test_predict_file_final_in_range(self, trained_engine, tmp_wav):
        final, _ = trained_engine.predict(tmp_wav)
        assert 0.0 <= final <= 1.0

    def test_silent_chunk_returns_zero(self, trained_engine, config):
        sr = config.sample_rate
        silence = np.zeros(sr, dtype=np.float32)
        prob = trained_engine.predict_chunk(silence, sr)
        assert prob == 0.0

    def test_batch_predict_length(self, trained_engine, config):
        sr = config.sample_rate
        chunks = [np.random.randn(sr).astype(np.float32) * 0.1 for _ in range(5)]
        probs = trained_engine.predict_batch(chunks, sr)
        assert len(probs) == 5

    def test_batch_predict_range(self, trained_engine, config):
        sr = config.sample_rate
        chunks = [np.random.randn(sr).astype(np.float32) * 0.1 for _ in range(5)]
        probs = trained_engine.predict_batch(chunks, sr)
        for p in probs:
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# ChunkAggregator
# ---------------------------------------------------------------------------
class TestChunkAggregator:
    def test_empty_score_is_zero(self, config):
        agg = ChunkAggregator(config)
        assert agg.current_score() == 0.0
        assert agg.final_score() == 0.0

    def test_update_returns_score(self, config):
        agg = ChunkAggregator(config)
        score = agg.update(0.8)
        assert isinstance(score, float)

    def test_single_chunk_score(self, config):
        agg = ChunkAggregator(config)
        agg.update(0.6)
        # avg=0.6, peak=0.6, final = 0.5*0.6 + 0.5*0.6 = 0.6
        assert abs(agg.final_score() - 0.6) < 1e-6

    def test_peak_dominates_on_high_variance(self, config):
        agg = ChunkAggregator(config)
        for p in [0.1, 0.1, 0.1, 0.9]:
            agg.update(p)
        final = agg.final_score()
        avg = 0.3
        peak = 0.9
        expected = 0.5 * avg + 0.5 * peak
        assert abs(final - expected) < 1e-5

    def test_classify_thresholds(self, config):
        agg = ChunkAggregator(config)
        assert agg.classify(0.1) == "LOW"
        assert agg.classify(config.low_threshold) == "MEDIUM"
        assert agg.classify(config.medium_threshold) == "HIGH"

    def test_reset_clears_state(self, config):
        agg = ChunkAggregator(config)
        agg.update(0.9)
        agg.reset()
        assert agg.num_chunks == 0
        assert agg.final_score() == 0.0

    def test_window_limits_rolling(self, config):
        agg = ChunkAggregator(config, window=3)
        for p in [0.9, 0.9, 0.9, 0.0, 0.0, 0.0]:
            agg.update(p)
        # Window of 3 → only last 3 zeros are active
        assert agg.current_score() == 0.0
        # But final_score uses ALL chunks
        assert agg.final_score() > 0.0


# ---------------------------------------------------------------------------
# Evaluate helpers
# ---------------------------------------------------------------------------
class TestComputeEER:
    def test_perfect_classifier(self):
        labels = [0, 0, 1, 1]
        scores = [0.0, 0.0, 1.0, 1.0]
        eer = compute_eer(labels, scores)
        assert eer < 0.05, f"Perfect classifier EER should be ~0, got {eer}"

    def test_random_classifier(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(200) > 0.5).astype(int).tolist()
        scores = rng.random(200).tolist()
        eer = compute_eer(labels, scores)
        # Random should be near 0.5
        assert 0.3 < eer < 0.7, f"Random EER should be ~0.5, got {eer}"

    def test_empty_inputs(self):
        eer = compute_eer([], [])
        assert np.isnan(eer)

    def test_only_one_class(self):
        eer = compute_eer([1, 1, 1], [0.8, 0.9, 0.7])
        assert np.isnan(eer)


class TestComputeAUC:
    def test_perfect_auc(self):
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        auc = compute_auc(labels, scores)
        assert abs(auc - 1.0) < 0.01

    def test_random_auc_near_05(self):
        rng = np.random.default_rng(42)
        labels = (rng.random(500) > 0.5).astype(int).tolist()
        scores = rng.random(500).tolist()
        auc = compute_auc(labels, scores)
        assert 0.4 < auc < 0.6

    def test_worst_auc(self):
        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.1, 0.2]
        auc = compute_auc(labels, scores)
        assert auc < 0.1


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        labels = [0, 0, 1, 1]
        scores = [0.0, 0.0, 1.0, 1.0]
        tp, fp, tn, fn = compute_confusion_matrix(labels, scores, threshold=0.5)
        assert tp == 2
        assert tn == 2
        assert fp == 0
        assert fn == 0

    def test_all_wrong(self):
        labels = [0, 0, 1, 1]
        scores = [1.0, 1.0, 0.0, 0.0]
        tp, fp, tn, fn = compute_confusion_matrix(labels, scores, threshold=0.5)
        assert tp == 0
        assert tn == 0
        assert fp == 2
        assert fn == 2
