"""
VAANI-SHIELD — Training loop.

Usage
-----
    python -m src.train [--epochs N] [--batch-size N] [--lr LR] [--device cpu|cuda]

Features
--------
- BCE loss with logits + optional label smoothing
- AdamW optimiser + CosineAnnealingLR with linear warmup
- Early stopping on validation EER
- Saves best checkpoint and last checkpoint separately
- Logs per-epoch metrics to outputs/metrics/training_log.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import VaaniConfig, DEFAULT_CONFIG, CHECKPOINTS_DIR, METRICS_DIR, ASVSPOOF_ROOT
from .dataset import VaaniDataset, ASVspoofDataset
from .model import build_model, model_summary
from .evaluate import compute_eer

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stop training when validation EER stops improving."""

    def __init__(self, patience: int, delta: float = 1e-4) -> None:
        self.patience = patience
        self.delta = delta
        self._best_eer = float("inf")
        self._counter = 0
        self.should_stop = False

    def step(self, eer: float) -> bool:
        """Returns True if improvement was seen."""
        if eer < self._best_eer - self.delta:
            self._best_eer = eer
            self._counter = 0
            return True
        self._counter += 1
        if self._counter >= self.patience:
            self.should_stop = True
        return False


def _smooth_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing: 0 → smoothing/2, 1 → 1 - smoothing/2."""
    return labels * (1.0 - smoothing) + smoothing * 0.5


def _warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
):
    """Linear warmup then cosine annealing."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    label_smoothing: float,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = _smooth_labels(y, label_smoothing).unsqueeze(1).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (val_loss, val_eer)."""
    model.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for x, y in loader:
        x = x.to(device)
        y_dev = y.unsqueeze(1).to(device)
        logits = model(x)
        loss = criterion(logits, y_dev)
        total_loss += loss.item() * x.size(0)
        probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(y.tolist())

    val_loss = total_loss / max(1, len(loader.dataset))
    eer = compute_eer(all_labels, all_probs)
    return val_loss, eer


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    config: VaaniConfig = DEFAULT_CONFIG,
    device_str: str = "cpu",
    dataset_type: str = "auto",
    asvspoof_root: Path | None = None,
    real_dir: Path | None = None,
    synthetic_dir: Path | None = None,
) -> Path:
    """
    Full training run.

    Parameters
    ----------
    config         : VaaniConfig
    device_str     : 'cpu' | 'cuda'
    dataset_type   : 'auto' | 'asvspoof' | 'generic'
    asvspoof_root  : Path to ASVspoof root (if using ASVspoof)
    real_dir       : Path to real audio (if using generic VaaniDataset)
    synthetic_dir  : Path to synthetic audio (if using generic VaaniDataset)

    Returns the path to the best checkpoint saved.
    """
    # Reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    log.info("Training device: %s", device)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    _root = Path(asvspoof_root) if asvspoof_root is not None else ASVSPOOF_ROOT
    if dataset_type == "auto":
        proto_dir = _root / "ASVspoof2019_LA_cm_protocols"
        dataset_type = "asvspoof" if proto_dir.exists() else "generic"

    if dataset_type == "asvspoof":
        log.info("Loading ASVspoof 2019 LA dataset from: %s", _root)
        train_ds = ASVspoofDataset("train", config, asvspoof_root=_root)
        val_ds = ASVspoofDataset("dev", config, asvspoof_root=_root)
    else:
        log.info("Loading generic dataset from %s / %s", real_dir, synthetic_dir)
        train_ds = VaaniDataset("train", config, real_dir, synthetic_dir)
        val_ds = VaaniDataset("val", config, real_dir, synthetic_dir)

    if len(train_ds) == 0:
        log.error(
            "Training dataset is empty!\n"
            "For ASVspoof: ensure ASVSPOOF_ROOT is valid.\n"
            "For generic: populate dataset/real/ and dataset/synthetic/ with WAV files.\n"
            "See README.md for dataset acquisition instructions."
        )
        raise RuntimeError("No training data found.")

    log.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimiser
    # ------------------------------------------------------------------
    model = build_model(config).to(device)
    log.info(model_summary(model, config))

    # Imbalanced dataset → use positive class weight
    weights = train_ds.class_weights()
    pos_weight = (weights[1] / weights[0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup = getattr(config, "warmup_epochs", 2)
    scheduler = _warmup_cosine_scheduler(optimizer, warmup, config.epochs)
    stopper = EarlyStopping(config.early_stopping_patience)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_ckpt_path = CHECKPOINTS_DIR / "best.pt"
    last_ckpt_path = CHECKPOINTS_DIR / "last.pt"
    log_path = METRICS_DIR / "training_log.csv"

    n_train_samples = len(train_ds)
    n_val_samples = len(val_ds)
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_eer", "lr", "elapsed_s",
            "train_samples", "val_samples", "train_batches", "val_batches",
        ])

    log.info("Starting training for %d epochs (warmup=%d)...", config.epochs, warmup)
    log.info(
        "Train: %d samples (%d batches) | Val: %d samples (%d batches)",
        n_train_samples, n_train_batches, n_val_samples, n_val_batches,
    )
    best_eer = float("inf")

    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device, config.label_smoothing
        )

        if len(val_ds) > 0:
            val_loss, val_eer = _val_epoch(model, val_loader, criterion, device)
        else:
            val_loss, val_eer = float("nan"), float("nan")
            log.warning("Validation set is empty; EER cannot be computed.")

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_EER=%.4f | "
            "lr=%.2e | %.1fs | train_b=%d val_b=%d",
            epoch, config.epochs, train_loss, val_loss, val_eer, lr, elapsed,
            n_train_batches, n_val_batches,
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_loss, val_eer, lr, elapsed,
                n_train_samples, n_val_samples, n_train_batches, n_val_batches,
            ])

        # Save last checkpoint always
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_eer": val_eer,
                "config": config,
            },
            last_ckpt_path,
        )

        # Save best checkpoint on improvement
        improved = stopper.step(val_eer)
        if improved:
            best_eer = val_eer
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_eer": val_eer,
                    "config": config,
                },
                best_ckpt_path,
            )
            log.info("  ↑ New best EER: %.4f — checkpoint saved.", best_eer)

        if stopper.should_stop:
            log.info("Early stopping triggered at epoch %d.", epoch)
            break

    log.info("Training complete. Best val EER: %.4f", best_eer)
    log.info("Best checkpoint: %s", best_ckpt_path)
    return best_ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAANI-SHIELD training script")
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG.learning_rate)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--dataset",
        type=str,
        choices=["auto", "asvspoof", "generic"],
        default="auto",
        help="Dataset type ('asvspoof', 'generic', or 'auto')",
    )
    p.add_argument("--asvspoof-root", type=Path, default=None)
    p.add_argument("--real-dir", type=Path, default=None)
    p.add_argument("--synthetic-dir", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = VaaniConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    train(
        cfg,
        device_str=args.device,
        dataset_type=args.dataset,
        asvspoof_root=args.asvspoof_root,
        real_dir=args.real_dir,
        synthetic_dir=args.synthetic_dir,
    )
