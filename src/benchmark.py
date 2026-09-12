"""
VAANI-SHIELD — Latency and real-time factor benchmark.

Measures:
  - Feature extraction latency per chunk (p50, p95, p99)
  - Model inference latency per chunk (p50, p95, p99)
  - End-to-end pipeline latency
  - Real-time factor (RTF): audio_duration / processing_time

Usage
-----
    python -m src.benchmark
    python -m src.benchmark --audio audio/test.wav --n-repeats 100
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np

from .audio import load_audio, chunk_audio
from .config import VaaniConfig, DEFAULT_CONFIG, AUDIO_DIR, CHECKPOINTS_DIR
from .features import extract_log_mel
from .inference import VaaniInferenceEngine

log = logging.getLogger(__name__)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    arr = sorted(values)
    idx = (len(arr) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(arr) - 1)
    return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)


def benchmark(
    audio_path: Path,
    checkpoint_path: Path | None = None,
    config: VaaniConfig = DEFAULT_CONFIG,
    n_repeats: int = 50,
    device_str: str = "cpu",
) -> dict:
    """
    Run a latency benchmark on the given audio file.

    Parameters
    ----------
    audio_path      : path to .wav file
    checkpoint_path : trained checkpoint (None → untrained mode)
    config          : VaaniConfig
    n_repeats       : number of times to repeat chunk processing for stats
    device_str      : 'cpu' or 'cuda'

    Returns
    -------
    dict with keys: feat_p50_ms, feat_p95_ms, feat_p99_ms,
                    infer_p50_ms, infer_p95_ms, infer_p99_ms,
                    e2e_ms, rtf, n_chunks
    """
    print("\n" + "=" * 55)
    print("  VAANI-SHIELD Latency Benchmark")
    print("=" * 55)

    engine = VaaniInferenceEngine(checkpoint_path, config, device_str)
    mode = "TRAINED" if engine.is_trained else "UNTRAINED"
    print(f"  Mode     : {mode}")
    print(f"  Audio    : {audio_path}")
    print(f"  Repeats  : {n_repeats}")
    print("=" * 55 + "\n")

    samples, sr = load_audio(audio_path, config)
    chunks = chunk_audio(samples, sr, config, pad_last=True)
    audio_duration = len(samples) / sr
    n_chunks = len(chunks)

    feat_times: List[float] = []
    infer_times: List[float] = []

    # Warm-up (1 pass, not counted)
    for chunk in chunks:
        extract_log_mel(chunk, sr, config)
        engine.predict_chunk(chunk, sr)

    # Benchmarked passes
    for _ in range(n_repeats):
        for chunk in chunks:
            t0 = time.perf_counter()
            feat = extract_log_mel(chunk, sr, config)
            feat_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            engine.predict_chunk(chunk, sr)
            infer_times.append((time.perf_counter() - t0) * 1000)

    # E2E timing (single pass including load)
    t_start = time.perf_counter()
    for chunk in chunks:
        engine.predict_chunk(chunk, sr)
    e2e_ms = (time.perf_counter() - t_start) * 1000
    rtf = (audio_duration * 1000) / e2e_ms if e2e_ms > 0 else float("inf")

    results = {
        "feat_p50_ms": _percentile(feat_times, 50),
        "feat_p95_ms": _percentile(feat_times, 95),
        "feat_p99_ms": _percentile(feat_times, 99),
        "infer_p50_ms": _percentile(infer_times, 50),
        "infer_p95_ms": _percentile(infer_times, 95),
        "infer_p99_ms": _percentile(infer_times, 99),
        "e2e_ms": e2e_ms,
        "rtf": rtf,
        "n_chunks": n_chunks,
        "audio_duration_s": audio_duration,
    }

    _print_results(results, n_repeats)
    return results


def _print_results(r: dict, n_repeats: int) -> None:
    print("Feature Extraction (per chunk):")
    print(f"  p50 : {r['feat_p50_ms']:.2f} ms")
    print(f"  p95 : {r['feat_p95_ms']:.2f} ms")
    print(f"  p99 : {r['feat_p99_ms']:.2f} ms")
    print()
    print("Model Inference (per chunk):")
    print(f"  p50 : {r['infer_p50_ms']:.2f} ms")
    print(f"  p95 : {r['infer_p95_ms']:.2f} ms")
    print(f"  p99 : {r['infer_p99_ms']:.2f} ms")
    print()
    print(f"End-to-end ({r['n_chunks']} chunks, {r['audio_duration_s']:.1f}s audio):")
    print(f"  Total time : {r['e2e_ms']:.1f} ms")
    print(f"  RTF        : {r['rtf']:.1f}x  (audio_duration / processing_time)")
    print(
        f"               {'[OK] Faster than real-time' if r['rtf'] >= 1.0 else '[!!] Slower than real-time'}"
    )
    print()
    print(f"  (Statistics over {n_repeats} x {r['n_chunks']} chunk runs)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    default_audio = AUDIO_DIR / "test.wav"
    default_ckpt = CHECKPOINTS_DIR / "best.pt"
    p = argparse.ArgumentParser(description="VAANI-SHIELD latency benchmark")
    p.add_argument("--audio", type=Path, default=default_audio)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt if default_ckpt.exists() else None,
    )
    p.add_argument("--n-repeats", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()
    benchmark(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        n_repeats=args.n_repeats,
        device_str=args.device,
    )
