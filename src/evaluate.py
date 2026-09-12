"""
VAANI-SHIELD — Evaluation metrics.

Computes:
  - Equal Error Rate (EER)     — primary anti-spoofing metric
  - AUC-ROC
  - Confusion matrix
  - Classification report

Saves plots to outputs/plots/ and scores to outputs/metrics/.

Usage
-----
    python -m src.evaluate --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EER (Equal Error Rate)
# ---------------------------------------------------------------------------
def compute_eer(
    labels: List[int],
    scores: List[float],
) -> float:
    """
    Compute the Equal Error Rate (EER) using threshold-sweep with interpolation.

    Parameters
    ----------
    labels : list of int   -- 0 = real, 1 = synthetic
    scores : list of float -- predicted synthetic probability in [0, 1]

    Returns
    -------
    eer : float in [0, 1]  -- lower is better; 0.5 = random chance
    """
    if not labels or not scores:
        return float("nan")

    labels_arr = np.array(labels, dtype=int)
    scores_arr = np.array(scores, dtype=float)

    n_pos = int(labels_arr.sum())       # synthetic (positive)
    n_neg = int((1 - labels_arr).sum()) # real      (negative)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sweep thresholds in descending order (high threshold → low FAR, high FRR)
    sorted_idx = np.argsort(-scores_arr)
    thresholds = scores_arr[sorted_idx]

    far_vals: List[float] = []
    frr_vals: List[float] = []
    fp = 0  # false positives (real predicted synthetic)
    fn = n_pos  # false negatives (synthetic predicted real)

    for i, idx in enumerate(sorted_idx):
        if labels_arr[idx] == 1:
            fn -= 1  # this synthetic sample is now correctly accepted
        else:
            fp += 1  # this real sample is incorrectly rejected

        far_vals.append(fp / n_neg)
        frr_vals.append(fn / n_pos)

    far_arr = np.array(far_vals)
    frr_arr = np.array(frr_vals)

    # Find where FAR and FRR cross (FAR increases, FRR decreases)
    # EER is the value at or near the crossing point
    diff = far_arr - frr_arr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # No crossing found: pick the index with minimal |FAR - FRR|
        best = int(np.argmin(np.abs(diff)))
        return float((far_arr[best] + frr_arr[best]) / 2.0)

    # Interpolate at each sign change and take the minimum
    eer_candidates: List[float] = []
    for i in sign_changes:
        # Linear interpolation between point i and i+1
        far1, far2 = far_arr[i], far_arr[i + 1]
        frr1, frr2 = frr_arr[i], frr_arr[i + 1]
        # Solve: far1 + t*(far2-far1) == frr1 + t*(frr2-frr1)
        denom = (far2 - far1) - (frr2 - frr1)
        if abs(denom) < 1e-12:
            eer_candidates.append((far1 + frr1) / 2.0)
        else:
            t = (frr1 - far1) / denom
            t = max(0.0, min(1.0, t))
            eer_val = far1 + t * (far2 - far1)
            eer_candidates.append(float(eer_val))

    return min(eer_candidates)


# ---------------------------------------------------------------------------
# AUC-ROC
# ---------------------------------------------------------------------------
def compute_auc(labels: List[int], scores: List[float]) -> float:
    """Compute AUC-ROC using the trapezoidal rule (no sklearn dependency)."""
    if not labels or not scores:
        return float("nan")

    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    auc = 0.0
    tp = 0
    fp = 0
    prev_tp = 0
    prev_fp = 0
    prev_score = None

    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def compute_confusion_matrix(
    labels: List[int],
    scores: List[float],
    threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Returns (TP, FP, TN, FN)."""
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    return tp, fp, tn, fn


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------
def evaluate_checkpoint(
    checkpoint_path: Path,
    config=None,
    device_str: str = "cpu",
    dataset_type: str = "auto",
    split: str = "dev",
    asvspoof_root: Optional[Path] = None,
    real_dir: Optional[Path] = None,
    synthetic_dir: Optional[Path] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Load a checkpoint, run inference on the dev/test split, compute all metrics.

    Returns a dict with keys:
        eer, auc, accuracy, precision, recall, f1, far, frr,
        tp, fp, tn, fn, threshold, n_samples.
    """
    import torch
    from .config import DEFAULT_CONFIG, METRICS_DIR, PLOTS_DIR, ASVSPOOF_ROOT
    from .model import build_model
    from .dataset import VaaniDataset, ASVspoofDataset
    from torch.utils.data import DataLoader

    if config is None:
        config = DEFAULT_CONFIG

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", config)

    model = build_model(ckpt_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint from epoch %s", ckpt.get("epoch", "?"))

    # Determine dataset
    _root = Path(asvspoof_root) if asvspoof_root is not None else ASVSPOOF_ROOT
    if dataset_type == "auto":
        proto_dir = _root / "ASVspoof2019_LA_cm_protocols"
        dataset_type = "asvspoof" if proto_dir.exists() else "generic"

    if dataset_type == "asvspoof":
        eval_split = "dev" if split in ("dev", "val") else split
        log.info("Evaluating on ASVspoof 2019 LA '%s' split from %s", eval_split, _root)
        eval_ds = ASVspoofDataset(eval_split, ckpt_config, asvspoof_root=_root)
    else:
        eval_split = "test" if split in ("test", "dev", "val") else split
        log.info("Evaluating on generic '%s' split from %s / %s", eval_split, real_dir, synthetic_dir)
        eval_ds = VaaniDataset(eval_split, ckpt_config, real_dir, synthetic_dir)

    if len(eval_ds) == 0:
        log.error("Evaluation dataset is empty. Cannot evaluate.")
        return {}

    loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=0)
    all_probs: List[float] = []
    all_labels: List[int] = []

    log.info("Running evaluation on %d samples...", len(eval_ds))
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(y.int().tolist())

    # Continuous score metrics
    eer = compute_eer(all_labels, all_probs)
    auc = compute_auc(all_labels, all_probs)

    # Threshold-based metrics
    tp, fp, tn, fn = compute_confusion_matrix(all_labels, all_probs, threshold=threshold)
    n = len(all_labels)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Alarm Rate on negative (bona-fide)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # False Rejection Rate on positive (spoof)

    metrics = {
        "dataset_type": dataset_type,
        "split": eval_split,
        "threshold": threshold,
        "n_samples": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "eer": eer,
        "far": far,
        "frr": frr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    log.info("Split:         %s (%d samples)", eval_split, n)
    log.info("Threshold:     %.2f", threshold)
    log.info("Accuracy:      %.4f", accuracy)
    log.info("Precision:     %.4f", precision)
    log.info("Recall:        %.4f", recall)
    log.info("F1 Score:      %.4f", f1)
    log.info("ROC-AUC:       %.4f", auc)
    log.info("EER:           %.4f", eer)
    log.info("FAR (FPR):     %.4f", far)
    log.info("FRR (FNR):     %.4f", frr)
    log.info("TP=%d, FP=%d, TN=%d, FN=%d", tp, fp, tn, fn)

    # Save metrics JSON
    metrics_file = METRICS_DIR / f"{eval_split}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", metrics_file)

    # Save per-sample predictions
    from .config import PREDICTIONS_DIR
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_file = PREDICTIONS_DIR / f"{eval_split}_predictions.csv"
    with open(pred_file, "w") as f:
        f.write("label,score\n")
        for lbl, sc in zip(all_labels, all_probs):
            f.write(f"{lbl},{sc:.6f}\n")
    log.info("Predictions saved to %s", pred_file)

    # Plot ROC curve & score distribution
    try:
        _plot_roc(all_labels, all_probs, PLOTS_DIR / f"{eval_split}_roc_curve.png")
        _plot_score_distribution(all_labels, all_probs, PLOTS_DIR / f"{eval_split}_score_dist.png")
    except Exception as e:
        log.warning("Could not generate plots: %s", e)

    return metrics


def _plot_roc(labels, scores, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    tprs, fprs = [0.0], [0.0]
    tp = fp = 0
    for s, l in pairs:
        if l == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)
    tprs.append(1.0)
    fprs.append(1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs, tprs, lw=2, color="#e63946", label="ROC")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — VAANI-SHIELD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("ROC curve saved to %s", save_path)


def _plot_score_distribution(labels, scores, save_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    real_scores = [s for s, l in zip(scores, labels) if l == 0]
    syn_scores = [s for s, l in zip(scores, labels) if l == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(real_scores, bins=bins, alpha=0.6, label="Real", color="#457b9d")
    ax.hist(syn_scores, bins=bins, alpha=0.6, label="Synthetic", color="#e63946")
    ax.set_xlabel("Synthetic Probability")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — VAANI-SHIELD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Score distribution saved to %s", save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAANI-SHIELD evaluation script")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--dataset",
        type=str,
        choices=["auto", "asvspoof", "generic"],
        default="auto",
        help="Dataset type ('asvspoof', 'generic', or 'auto')",
    )
    p.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split to evaluate ('dev' for ASVspoof, 'test' for generic)",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--asvspoof-root", type=Path, default=None)
    p.add_argument("--real-dir", type=Path, default=None)
    p.add_argument("--synthetic-dir", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = _parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        device_str=args.device,
        dataset_type=args.dataset,
        split=args.split,
        asvspoof_root=args.asvspoof_root,
        real_dir=args.real_dir,
        synthetic_dir=args.synthetic_dir,
        threshold=args.threshold,
    )
