"""
VAANI-SHIELD — ONNX export.

Exports a trained PyTorch checkpoint to ONNX, then sanity-checks the
exported model's output against PyTorch's output (must agree to 1e-4).

Usage
-----
    python -m src.export --checkpoint checkpoints/best.pt
    python -m src.export --checkpoint checkpoints/best.pt --output models/vaani_shield.onnx

Android path (after this script)
---------------------------------
    1. Convert ONNX → TFLite:
       pip install onnx-tf tensorflow
       python -m onnx_tf.backend.prepare --input models/vaani_shield.onnx --output models/vaani_tf/
       tflite_convert --saved_model_dir=models/vaani_tf --output_file=models/vaani_shield.tflite

    2. Or use ONNX Runtime for Android directly (onnxruntime-android AAR).
       The ONNX model produced here works with ORT Mobile out of the box.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .config import VaaniConfig, DEFAULT_CONFIG, MODELS_DIR, CHECKPOINTS_DIR
from .model import build_model

log = logging.getLogger(__name__)


def export_onnx(
    checkpoint_path: Path,
    output_path: Path | None = None,
    config: VaaniConfig = DEFAULT_CONFIG,
    opset: int | None = None,
) -> Path:
    """
    Export a trained VaaniLCNN checkpoint to ONNX.

    Parameters
    ----------
    checkpoint_path : path to .pt checkpoint
    output_path     : where to save the .onnx file (default: models/vaani_shield.onnx)
    config          : VaaniConfig (overridden by checkpoint's config if present)
    opset           : ONNX opset version (default: config.onnx_opset)

    Returns
    -------
    Path to the exported .onnx file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = MODELS_DIR / "vaani_shield.onnx"

    opset = opset or config.onnx_opset

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    log.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", config)

    model = build_model(ckpt_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log.info(
        "Checkpoint: epoch=%s | val_EER=%s",
        ckpt.get("epoch", "?"),
        ckpt.get("val_eer", "?"),
    )

    # ------------------------------------------------------------------
    # Dummy input
    # ------------------------------------------------------------------
    dummy_input = torch.zeros(
        1, 1, ckpt_config.n_mels, ckpt_config.n_frames
    )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    log.info("Exporting to ONNX (opset %d): %s", opset, output_path)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["log_mel"],
        output_names=["logit"],
        dynamic_axes={
            "log_mel": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
    )

    size_mb = output_path.stat().st_size / 1e6
    log.info("Exported ONNX model: %.2f MB", size_mb)

    # ------------------------------------------------------------------
    # Sanity check: ONNX output vs PyTorch output
    # ------------------------------------------------------------------
    _sanity_check(model, dummy_input, output_path)

    print(f"\n[OK] ONNX model exported: {output_path}  ({size_mb:.2f} MB)")
    print("\nNext steps for Android deployment:")
    print("  Option A - ONNX Runtime Mobile:")
    print("    Add onnxruntime-android AAR to your Android project.")
    print("    Load vaani_shield.onnx directly - no conversion needed.")
    print()
    print("  Option B - TFLite (via onnx-tf):")
    print("    pip install onnx-tf tensorflow")
    print("    python -m onnx_tf.backend.prepare \\")
    print(f"        --input {output_path} \\")
    print("        --output models/vaani_tf/")
    print("    tflite_convert \\")
    print("        --saved_model_dir=models/vaani_tf \\")
    print("        --output_file=models/vaani_shield.tflite")

    return output_path


def _sanity_check(model: torch.nn.Module, dummy: torch.Tensor, onnx_path: Path) -> None:
    """Verify ONNX output matches PyTorch output within tolerance."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ONNX sanity check.")
        return

    try:
        import onnx
        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        log.info("ONNX graph structure: ✅ valid")
    except Exception as exc:
        log.error("ONNX validation failed: %s", exc)
        return

    # Run PyTorch
    with torch.no_grad():
        pt_out = torch.sigmoid(model(dummy)).numpy()

    # Run ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logit"], {"log_mel": dummy.numpy()})
    ort_prob = 1 / (1 + np.exp(-ort_out[0]))  # sigmoid

    max_diff = float(np.max(np.abs(pt_out - ort_prob)))
    if max_diff < 1e-4:
        log.info("ONNX sanity check: ✅ max_diff=%.2e (< 1e-4)", max_diff)
        print(f"  Sanity check: [OK] PyTorch vs ONNX max diff = {max_diff:.2e}")
    else:
        log.error(
            "ONNX sanity check FAILED: max_diff=%.4f > 1e-4. "
            "Check for unsupported ops or precision issues.",
            max_diff,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    default_ckpt = CHECKPOINTS_DIR / "best.pt"
    p = argparse.ArgumentParser(description="VAANI-SHIELD ONNX export")
    p.add_argument("--checkpoint", type=Path, default=default_ckpt)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--opset", type=int, default=DEFAULT_CONFIG.onnx_opset)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = _parse_args()
    export_onnx(args.checkpoint, args.output, opset=args.opset)
