"""
Tests for src.dataset — VaaniDataset with dummy audio files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.config import VaaniConfig
from src.dataset import VaaniDataset, LABEL_REAL, LABEL_SYNTHETIC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config() -> VaaniConfig:
    return VaaniConfig(
        sample_rate=16000,
        chunk_seconds=1.0,
        n_mels=64,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42,
    )


def _write_dummy_wav(path: Path, sr: int = 16000, duration: float = 1.5) -> None:
    """Write a short sine-wave WAV file."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr)


@pytest.fixture
def tiny_dataset(tmp_path: Path, config: VaaniConfig):
    """
    Create a tiny dataset:
      dataset/real/      — 6 files
      dataset/synthetic/ — 6 files
    Returns (real_dir, synthetic_dir, tmp_path).
    """
    real_dir = tmp_path / "real"
    syn_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    syn_dir.mkdir()

    sr = config.sample_rate
    for i in range(6):
        _write_dummy_wav(real_dir / f"real_{i:03d}.wav", sr)
        _write_dummy_wav(syn_dir / f"syn_{i:03d}.wav", sr)

    return real_dir, syn_dir, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestVaaniDataset:
    def test_train_split_not_empty(self, tiny_dataset, config):
        real_dir, syn_dir, _ = tiny_dataset
        ds = VaaniDataset("train", config, real_dir, syn_dir)
        assert len(ds) > 0

    def test_val_split_not_empty(self, tiny_dataset, config):
        real_dir, syn_dir, _ = tiny_dataset
        ds = VaaniDataset("val", config, real_dir, syn_dir)
        assert len(ds) > 0

    def test_total_samples_accounted_for(self, tiny_dataset, config):
        """Train + val + test must cover all 12 files (no silent wasted chunks)."""
        real_dir, syn_dir, _ = tiny_dataset
        n_total = 12  # 6 real + 6 synthetic
        n_train = len(VaaniDataset("train", config, real_dir, syn_dir))
        n_val = len(VaaniDataset("val", config, real_dir, syn_dir))
        n_test = len(VaaniDataset("test", config, real_dir, syn_dir))
        assert n_train + n_val + n_test == n_total

    def test_item_shapes(self, tiny_dataset, config):
        """Each item must be (Tensor(1, n_mels, n_frames), Tensor(scalar))."""
        real_dir, syn_dir, _ = tiny_dataset
        ds = VaaniDataset("train", config, real_dir, syn_dir)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (1, config.n_mels, config.n_frames), f"Bad shape: {x.shape}"
        assert y.ndim == 0  # scalar

    def test_labels_binary(self, tiny_dataset, config):
        """Labels must be 0.0 or 1.0 (float32 binary)."""
        real_dir, syn_dir, _ = tiny_dataset
        ds = VaaniDataset("train", config, real_dir, syn_dir)
        for _, y in ds:
            assert float(y) in (0.0, 1.0), f"Label {float(y)} is not 0 or 1"

    def test_both_classes_present(self, tiny_dataset, config):
        real_dir, syn_dir, _ = tiny_dataset
        ds = VaaniDataset("train", config, real_dir, syn_dir)
        labels = [float(ds[i][1]) for i in range(len(ds))]
        assert 0.0 in labels, "Real class missing from train split"
        assert 1.0 in labels, "Synthetic class missing from train split"

    def test_class_weights_shape(self, tiny_dataset, config):
        real_dir, syn_dir, _ = tiny_dataset
        ds = VaaniDataset("train", config, real_dir, syn_dir)
        w = ds.class_weights()
        assert w.shape == (2,)
        assert torch.all(w > 0)

    def test_invalid_split_raises(self, tiny_dataset, config):
        real_dir, syn_dir, _ = tiny_dataset
        with pytest.raises(AssertionError):
            VaaniDataset("invalid_split", config, real_dir, syn_dir)

    def test_empty_dataset_warns(self, tmp_path, config):
        """Empty directories must warn (not crash)."""
        real_dir = tmp_path / "real"
        syn_dir = tmp_path / "synthetic"
        real_dir.mkdir()
        syn_dir.mkdir()
        with pytest.warns(UserWarning, match="No audio files found"):
            ds = VaaniDataset("train", config, real_dir, syn_dir)
        assert len(ds) == 0

    def test_feature_cache_created(self, tiny_dataset, config):
        """After first access, .vaani.npy cache files should appear."""
        real_dir, syn_dir, _ = tiny_dataset
        ds = VaaniDataset("train", config, real_dir, syn_dir)
        _ = ds[0]  # trigger cache creation
        cache_files = list(real_dir.glob("*.vaani.npy")) + list(syn_dir.glob("*.vaani.npy"))
        assert len(cache_files) > 0, "No cache files created"
