"""
VAANI-SHIELD — Central configuration.

All hyperparameters, paths, and thresholds live here.
Change values here; nothing else should contain magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, not this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_DIR: Path = PROJECT_ROOT / "dataset"
REAL_DIR: Path = DATASET_DIR / "real"
SYNTHETIC_DIR: Path = DATASET_DIR / "synthetic"

CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

AUDIO_DIR: Path = PROJECT_ROOT / "audio"

# ---------------------------------------------------------------------------
# ASVspoof 2019 LA paths
# ---------------------------------------------------------------------------
# Override with env var VAANI_ASVSPOOF_ROOT to point at a different location.
import os as _os
_asvspoof_default = Path("C:/Users/ayaan/Downloads/ASVspoof2019_LA/LA")
ASVSPOOF_ROOT: Path = Path(_os.environ.get("VAANI_ASVSPOOF_ROOT", str(_asvspoof_default)))

# Sub-directories (relative to ASVSPOOF_ROOT)
ASVSPOOF_TRAIN_DIR:    Path = ASVSPOOF_ROOT / "ASVspoof2019_LA_train" / "flac"
ASVSPOOF_DEV_DIR:      Path = ASVSPOOF_ROOT / "ASVspoof2019_LA_dev"   / "flac"
ASVSPOOF_EVAL_DIR:     Path = ASVSPOOF_ROOT / "ASVspoof2019_LA_eval"  / "flac"
ASVSPOOF_PROTOCOL_DIR: Path = ASVSPOOF_ROOT / "ASVspoof2019_LA_cm_protocols"

ASVSPOOF_TRAIN_PROTOCOL: Path = ASVSPOOF_PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
ASVSPOOF_DEV_PROTOCOL:   Path = ASVSPOOF_PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
# NOTE: eval protocol is intentionally NOT aliased to a training-related name.
ASVSPOOF_EVAL_PROTOCOL:  Path = ASVSPOOF_PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"



# ---------------------------------------------------------------------------
# Config dataclass — one instance shared everywhere
# ---------------------------------------------------------------------------
@dataclass
class VaaniConfig:
    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------
    sample_rate: int = 16_000          # Hz — must match dataset recording SR
    chunk_seconds: float = 1.0         # seconds per inference window
    silence_rms_threshold: float = 1e-4

    # ------------------------------------------------------------------
    # Log-Mel spectrogram
    # ------------------------------------------------------------------
    n_mels: int = 64                   # mel filter banks
    n_fft: int = 512                   # FFT window size (32 ms @ 16kHz)
    hop_length: int = 160              # hop size (10 ms @ 16kHz)
    win_length: int = 400              # analysis window (25 ms @ 16kHz)
    f_min: float = 80.0               # lowest mel frequency (Hz)
    f_max: float = 7600.0             # highest mel frequency (Hz)
    top_db: float = 80.0              # dynamic range for power_to_db

    # Derived: number of time frames for exactly chunk_seconds of audio
    @property
    def n_frames(self) -> int:
        chunk_samples = int(self.sample_rate * self.chunk_seconds)
        return (chunk_samples - self.n_fft) // self.hop_length + 1

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    lcnn_channels: list[int] = field(
        default_factory=lambda: [32, 32, 64, 64, 64]
    )
    gru_hidden: int = 128
    gru_layers: int = 1
    fc_hidden: int = 64
    dropout: float = 0.3

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stopping_patience: int = 5    # epochs without val EER improvement
    warmup_epochs: int = 2
    num_workers: int = 0                # DataLoader workers (0 = main process)
    pin_memory: bool = False            # set True only with GPU

    # ------------------------------------------------------------------
    # Dataset split
    # ------------------------------------------------------------------
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    speaker_id_regex: str = r"^(p\d+|s\d+|[A-Za-z]+\d+)"  # VCTK / LibriSpeech
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Inference thresholds
    # ------------------------------------------------------------------
    low_threshold: float = 0.35
    medium_threshold: float = 0.65

    # ------------------------------------------------------------------
    # Aggregation weights
    # ------------------------------------------------------------------
    agg_weight_avg: float = 0.50
    agg_weight_peak: float = 0.50

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    onnx_opset: int = 17


# Singleton — import this everywhere instead of constructing a new one.
DEFAULT_CONFIG = VaaniConfig()
