"""
VAANI-SHIELD — LCNN model for synthetic-speech detection.

Architecture: Light CNN (LCNN) with Max-Feature-Map (MFM) activations
              followed by a single GRU layer and fully-connected head.

Why LCNN?
---------
- State-of-the-art on ASVspoof 2015/2019 LA with lightweight budgets.
- MFM activation suppresses noisy activations, acting as a built-in feature
  selector — better than ReLU for audio discrimination tasks.
- ~300–600K parameters, ~2–4 MB on disk (FP32).
- Exports cleanly to ONNX (no dynamic shapes, no custom ops).
- Quantises well to INT8 for Android deployment.

Input shape:  (batch, 1, n_mels, n_frames)  — channel-first, single-channel
Output shape: (batch, 1)                    — raw logit (apply sigmoid for prob)

For inference, pass the logit through torch.sigmoid() to get a probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VaaniConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Max-Feature-Map (MFM) activation
# ---------------------------------------------------------------------------
class MaxFeatureMap(nn.Module):
    """
    MFM activation: split channel dim in half, take element-wise max.
    Input channels C → output channels C//2.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) % 2 == 0, "Channel dim must be even for MFM."
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


# ---------------------------------------------------------------------------
# LCNN block: Conv → BN → MFM → optional pool
# ---------------------------------------------------------------------------
class LCNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # actual output = out_channels // 2 after MFM
        kernel_size: int,
        padding: int = 0,
        pool: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.mfm = MaxFeatureMap()
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mfm(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


# ---------------------------------------------------------------------------
# LCNN backbone
# ---------------------------------------------------------------------------
class LCNNBackbone(nn.Module):
    """
    5-block LCNN feature extractor.

    Input : (B, 1, n_mels, n_frames)
    Output: (B, T', C') — sequence for GRU, where T' is along time axis
    """

    def __init__(self) -> None:
        super().__init__()
        # Each conv is specified as (in_ch, out_ch_before_mfm, kernel, pad, pool)
        self.block1 = LCNNBlock(1, 64, kernel_size=5, padding=2, pool=True)   # → 32 ch
        self.block2 = LCNNBlock(32, 64, kernel_size=1, padding=0, pool=False)  # → 32 ch
        self.block3 = LCNNBlock(32, 96, kernel_size=3, padding=1, pool=True)   # → 48 ch
        self.block4 = LCNNBlock(48, 96, kernel_size=1, padding=0, pool=False)  # → 48 ch
        self.block5 = LCNNBlock(48, 128, kernel_size=3, padding=1, pool=True)  # → 64 ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, n_frames)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # x: (B, 64, freq', time')
        # Collapse frequency axis → treat time axis as sequence for GRU
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)       # (B, T, C, F)
        x = x.reshape(B, T, C * F)       # (B, T, C*F)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class VaaniLCNN(nn.Module):
    """
    VAANI-SHIELD LCNN + GRU synthetic-speech detector.

    Returns raw logits (NOT probabilities). Call torch.sigmoid() on the
    output for inference, or use BCEWithLogitsLoss during training.
    """

    def __init__(self, config: VaaniConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config

        self.backbone = LCNNBackbone()

        # Compute GRU input size by doing a forward pass on a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(
                1, 1, config.n_mels, config.n_frames
            )
            gru_in = self.backbone(dummy).shape[-1]

        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.Linear(config.gru_hidden, config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, n_mels, n_frames)

        Returns
        -------
        logits : (B, 1)  — raw logit; apply sigmoid for probability
        """
        feat = self.backbone(x)              # (B, T', gru_in)
        _, h_n = self.gru(feat)              # h_n: (layers, B, hidden)
        h = h_n[-1]                          # last layer: (B, hidden)
        logits = self.head(h)                # (B, 1)
        return logits


# ---------------------------------------------------------------------------
# Factory + parameter counting
# ---------------------------------------------------------------------------
def build_model(config: VaaniConfig = DEFAULT_CONFIG) -> VaaniLCNN:
    """Construct and return a VaaniLCNN instance."""
    return VaaniLCNN(config)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, config: VaaniConfig = DEFAULT_CONFIG) -> str:
    """Return a brief human-readable summary string."""
    n_params = count_parameters(model)
    size_mb = n_params * 4 / 1e6  # float32
    return (
        f"VaaniLCNN | params: {n_params:,} | "
        f"FP32 size: ~{size_mb:.1f} MB | "
        f"Input: (B, 1, {config.n_mels}, {config.n_frames})"
    )
