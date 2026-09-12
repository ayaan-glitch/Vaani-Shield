"""
VAANI-SHIELD — Inference engine.

Provides a clean API for loading a trained checkpoint and running
per-chunk or per-file predictions.

Usage (CLI)
-----------
    python -m src.inference --audio audio/test.wav
    python -m src.inference --audio audio/test.wav --checkpoint checkpoints/best.pt

Usage (Python)
--------------
    from src.inference import VaaniInferenceEngine
    engine = VaaniInferenceEngine("checkpoints/best.pt")
    prob = engine.predict("audio/test.wav")
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .audio import load_audio, chunk_audio, is_silent
from .config import VaaniConfig, DEFAULT_CONFIG, CHECKPOINTS_DIR
from .features import extract_log_mel

log = logging.getLogger(__name__)


class VaaniInferenceEngine:
    """
    Load a trained VAANI-SHIELD checkpoint and run inference.

    If no checkpoint is provided (or the file doesn't exist) the engine
    operates in UNTRAINED mode — it still runs the full pipeline but
    outputs a constant 0.5 (maximum uncertainty) with a clear warning.
    This allows the pipeline to be smoke-tested before training.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        config: VaaniConfig = DEFAULT_CONFIG,
        device_str: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device_str)
        self._trained = False

        self._model: Optional[torch.nn.Module] = None
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[Path | str]) -> None:
        from .model import build_model

        if checkpoint_path is None or str(checkpoint_path).lower() in ("none", ""):
            log.warning(
                "No checkpoint path given. Inference engine running in UNTRAINED mode."
                " All predictions will return 0.5 (maximum uncertainty)."
            )
            return

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            log.warning(
                "Checkpoint not found: %s. Running in UNTRAINED mode.", ckpt_path
            )
            return

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            ckpt_config = ckpt.get("config", self.config)
            self.config = ckpt_config  # use the config the model was trained with
            model = build_model(ckpt_config).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._model = model
            self._trained = True
            epoch = ckpt.get("epoch", "?")
            eer = ckpt.get("val_eer", "?")
            log.info(
                "Loaded checkpoint (epoch=%s, val_EER=%s) from %s", epoch, eer, ckpt_path
            )
        except Exception as exc:
            log.error("Failed to load checkpoint: %s — running in UNTRAINED mode.", exc)

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Core prediction methods
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_chunk(
        self,
        samples: np.ndarray,
        sr: int,
    ) -> float:
        """
        Predict synthetic probability for a single audio chunk.

        Parameters
        ----------
        samples : 1-D float32 numpy array (one chunk, any length — will be padded/trimmed)
        sr      : sample rate of the samples array

        Returns
        -------
        synthetic_probability : float ∈ [0, 1]
        """
        if not self._trained:
            return 0.5  # maximum uncertainty sentinel

        if is_silent(samples, self.config):
            return 0.0

        feat = extract_log_mel(samples, sr, self.config)            # (n_mels, n_frames)
        x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)       # (1, 1, n_mels, n_frames)
        x = x.to(self.device)
        logit = self._model(x)                                       # (1, 1)
        prob = float(torch.sigmoid(logit).squeeze().cpu())
        return prob

    def predict(
        self,
        audio_path: Path | str,
    ) -> Tuple[float, List[float]]:
        """
        Predict synthetic probability for an audio file.

        Returns
        -------
        final_score  : float ∈ [0, 1]  — aggregated over all chunks
        chunk_scores : list of float   — per-chunk probabilities
        """
        from .aggregation import ChunkAggregator

        samples, sr = load_audio(Path(audio_path), self.config)
        chunks = chunk_audio(samples, sr, self.config, pad_last=True)

        aggregator = ChunkAggregator(self.config)
        chunk_scores: List[float] = []

        for chunk in chunks:
            prob = self.predict_chunk(chunk, sr)
            aggregator.update(prob)
            chunk_scores.append(prob)

        return aggregator.final_score(), chunk_scores

    def predict_batch(
        self,
        chunks: List[np.ndarray],
        sr: int,
    ) -> List[float]:
        """
        Predict synthetic probabilities for a batch of chunks (more efficient).

        Parameters
        ----------
        chunks : list of 1-D float32 arrays
        sr     : sample rate

        Returns
        -------
        list of float probabilities, one per chunk
        """
        if not self._trained:
            return [0.5] * len(chunks)

        feats = []
        silent_mask = []
        for chunk in chunks:
            if is_silent(chunk, self.config):
                silent_mask.append(True)
                feats.append(np.zeros((self.config.n_mels, self.config.n_frames), dtype=np.float32))
            else:
                silent_mask.append(False)
                feats.append(extract_log_mel(chunk, sr, self.config))

        x = torch.from_numpy(np.stack(feats)).unsqueeze(1).to(self.device)  # (B, 1, n_mels, n_frames)

        with torch.no_grad():
            logits = self._model(x)  # (B, 1)
            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()

        # Zero out silent chunks
        for i, silent in enumerate(silent_mask):
            if silent:
                probs[i] = 0.0

        return probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAANI-SHIELD inference")
    p.add_argument("--audio", type=Path, required=True, help="Path to audio file")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (omit for untrained mode)",
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = _parse_args()

    engine = VaaniInferenceEngine(
        checkpoint_path=args.checkpoint,
        device_str=args.device,
    )

    mode = "TRAINED" if engine.is_trained else "UNTRAINED (no checkpoint)"
    print(f"\n{'='*55}")
    print(f"  VAANI-SHIELD Inference - {mode}")
    print(f"{'='*55}")

    t0 = time.perf_counter()
    final_score, chunk_scores = engine.predict(args.audio)
    elapsed = time.perf_counter() - t0

    from .config import DEFAULT_CONFIG

    cfg = engine.config

    print(f"\nFile   : {args.audio}")
    print(f"Chunks : {len(chunk_scores)}")
    print(f"\nPer-chunk probabilities:")
    for i, prob in enumerate(chunk_scores, 1):
        bar = "#" * int(prob * 20) + "." * (20 - int(prob * 20))
        risk = "HIGH" if prob >= cfg.medium_threshold else "MED" if prob >= cfg.low_threshold else "LOW"
        print(f"  Chunk {i:2d}: [{bar}] {prob:.3f}  {risk}")

    print(f"\nFinal risk score : {final_score:.3f}")
    risk = "HIGH" if final_score >= cfg.medium_threshold else "MEDIUM" if final_score >= cfg.low_threshold else "LOW"
    print(f"Risk level       : {risk}")
    print(f"Processing time  : {elapsed*1000:.1f} ms")

    if not engine.is_trained:
        print(
            "\n[!] NOTE: Scores above are 0.5 (maximum uncertainty) because no "
            "trained checkpoint was loaded. Train the model first:\n"
            "    python -m src.train\n"
            "Then re-run with: --checkpoint checkpoints/best.pt"
        )
