"""
VAANI-SHIELD — Real-time simulation using the trained model.

This is the upgraded successor to the root realtime.py (Phase 1).
- If a trained checkpoint exists at checkpoints/best.pt, it is loaded and
  used for REAL inference.
- If no checkpoint is found, the engine falls back to UNTRAINED mode
  (0.5 for every non-silent chunk) with a clear warning.

The Phase 1 root realtime.py is preserved untouched as a reference.

Usage
-----
    python -m src.realtime                              # uses default test.wav
    python -m src.realtime --audio path/to/audio.wav
    python -m src.realtime --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .audio import load_audio, chunk_audio
from .aggregation import ChunkAggregator
from .config import VaaniConfig, DEFAULT_CONFIG, CHECKPOINTS_DIR, AUDIO_DIR
from .inference import VaaniInferenceEngine


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_simulation(
    audio_path: Path,
    checkpoint_path: Path | None = None,
    config: VaaniConfig = DEFAULT_CONFIG,
    device_str: str = "cpu",
) -> None:
    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VAANI-SHIELD - Real-Time Simulation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load engine
    # ------------------------------------------------------------------
    engine = VaaniInferenceEngine(
        checkpoint_path=checkpoint_path,
        config=config,
        device_str=device_str,
    )

    if engine.is_trained:
        print("  Model   : TRAINED checkpoint loaded [OK]")
    else:
        print("  Model   : UNTRAINED mode [!] (scores will be 0.5 for all chunks)")
        print("            -> Train first: python -m src.train")

    print(f"  Audio   : {audio_path}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Load audio
    # ------------------------------------------------------------------
    try:
        samples, sr = load_audio(audio_path, config)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    total_samples = len(samples)
    audio_duration = total_samples / sr

    if total_samples == 0:
        print("ERROR: Audio file contains no samples.")
        sys.exit(1)

    chunks = chunk_audio(samples, sr, config, pad_last=True)
    aggregator = ChunkAggregator(config)
    total_proc_time = 0.0

    # ------------------------------------------------------------------
    # Process chunks
    # ------------------------------------------------------------------
    for i, chunk in enumerate(chunks, 1):
        chunk_start = (i - 1) * config.chunk_seconds
        chunk_end = min(i * config.chunk_seconds, audio_duration)

        t0 = time.perf_counter()
        prob = engine.predict_chunk(chunk, sr)
        elapsed = time.perf_counter() - t0
        total_proc_time += elapsed

        current_risk = aggregator.update(prob)
        risk_label = aggregator.classify(prob)

        bar = "#" * int(prob * 20) + "." * (20 - int(prob * 20))
        print(
            f"Chunk {i:3d} | {chunk_start:5.1f}-{chunk_end:5.1f}s | "
            f"[{bar}] {prob:.3f} | {risk_label:<6} | "
            f"{elapsed*1000:.0f}ms"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    final_score = aggregator.final_score()
    final_risk = aggregator.classify(final_score)
    rtf = audio_duration / total_proc_time if total_proc_time > 0 else float("inf")
    avg_chunk_ms = (total_proc_time / len(chunks) * 1000) if chunks else 0

    print("\n" + "-" * 60)
    print("  SUMMARY")
    print("-" * 60)
    print(f"  Audio duration      : {audio_duration:.2f} s")
    print(f"  Chunks processed    : {len(chunks)}")
    print(f"  Average probability : {sum(aggregator.chunk_probabilities)/len(aggregator.chunk_probabilities):.3f}")
    print(f"  Peak probability    : {max(aggregator.chunk_probabilities):.3f}")
    print(f"  Final risk score    : {final_score:.3f}  [{final_risk}]")
    print(f"  Total proc time     : {total_proc_time*1000:.1f} ms")
    print(f"  Avg per-chunk       : {avg_chunk_ms:.1f} ms")
    print(f"  Real-time factor    : {rtf:.1f}x")
    print("-" * 60)

    if not engine.is_trained:
        print(
            "\n[!] Reminder: All scores above are 0.5 (UNTRAINED mode).\n"
            "   This output only validates the streaming pipeline works.\n"
            "   Populate dataset/ and run: python -m src.train"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    default_audio = AUDIO_DIR / "test.wav"
    default_ckpt = CHECKPOINTS_DIR / "best.pt"
    p = argparse.ArgumentParser(description="VAANI-SHIELD real-time simulation")
    p.add_argument("--audio", type=Path, default=default_audio)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt if default_ckpt.exists() else None,
    )
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_simulation(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        device_str=args.device,
    )
