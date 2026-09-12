"""
Tests for src.model — architecture, forward pass, parameter count.
"""

from __future__ import annotations

import pytest
import torch

from src.config import VaaniConfig
from src.model import VaaniLCNN, build_model, count_parameters, model_summary


@pytest.fixture
def config() -> VaaniConfig:
    return VaaniConfig(sample_rate=16000, chunk_seconds=1.0, n_mels=64)


@pytest.fixture
def model(config) -> VaaniLCNN:
    return build_model(config)


class TestModelArchitecture:
    def test_builds_without_error(self, config):
        m = build_model(config)
        assert m is not None

    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_parameter_count_reasonable(self, model):
        """Model should be lightweight: < 5M parameters."""
        n = count_parameters(model)
        assert n > 0, "Model has no parameters"
        assert n < 5_000_000, f"Model too large: {n:,} params (expected < 5M)"
        print(f"\n  Parameters: {n:,}")

    def test_model_summary_string(self, model, config):
        s = model_summary(model, config)
        assert "VaaniLCNN" in s
        assert "params" in s


class TestForwardPass:
    def test_output_shape(self, model, config):
        """Single sample: output must be (1, 1)."""
        x = torch.zeros(1, 1, config.n_mels, config.n_frames)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"

    def test_batch_output_shape(self, model, config):
        """Batch of 8: output must be (8, 1)."""
        x = torch.zeros(8, 1, config.n_mels, config.n_frames)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 1), f"Expected (8, 1), got {out.shape}"

    def test_output_is_logit_not_probability(self, model, config):
        """
        The model returns raw logits. We check that sigmoid(logit) ∈ (0, 1)
        but the logit itself can be any real number.
        """
        x = torch.randn(4, 1, config.n_mels, config.n_frames)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
        assert torch.all(probs > 0.0) and torch.all(probs < 1.0)

    def test_no_nan_in_output(self, model, config):
        x = torch.randn(4, 1, config.n_mels, config.n_frames)
        with torch.no_grad():
            out = model(x)
        assert not torch.any(torch.isnan(out)), "NaN in model output"

    def test_no_inf_in_output(self, model, config):
        x = torch.randn(4, 1, config.n_mels, config.n_frames)
        with torch.no_grad():
            out = model(x)
        assert not torch.any(torch.isinf(out)), "Inf in model output"

    def test_gradient_flows(self, model, config):
        """Backward pass must not raise and gradients must be non-None."""
        x = torch.randn(2, 1, config.n_mels, config.n_frames)
        labels = torch.zeros(2, 1)
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_eval_mode_no_grad(self, model, config):
        """eval() mode + no_grad should not crash."""
        model.eval()
        x = torch.randn(1, 1, config.n_mels, config.n_frames)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_deterministic_in_eval(self, model, config):
        """eval() mode must produce identical outputs for same input."""
        model.eval()
        x = torch.randn(1, 1, config.n_mels, config.n_frames)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)
