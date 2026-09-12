"""
VAANI-SHIELD — 30-Second Real Microphone Diagnostic Test.

Runs continuous live audio capture from the default microphone, performs
1-second chunk inference with VaaniLCNN (checkpoints/best.pt), aggregates
into 5-second decision windows, and outputs a diagnostic report.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np

from src.config import CHECKPOINTS_DIR, DEFAULT_CONFIG
from src.inference import VaaniInferenceEngine
from src.aggregation import DecisionWindowAggregator
from src.live_mic import LiveMicrophoneStream
from src.audio import is_silent


def run_diagnostic(duration_sec: int = 30, threshold: float = 0.50):
    print("=" * 70)
    print("VAANI-SHIELD: 30-SECOND REAL MICROPHONE DIAGNOSTIC TEST")
    print("=" * 70)
    ckpt_path = CHECKPOINTS_DIR / "best.pt"
    print(f"Loading model: {ckpt_path.name}")
    engine = VaaniInferenceEngine(checkpoint_path=ckpt_path, device_str="cpu")
    aggregator = DecisionWindowAggregator(window_size=5, config=DEFAULT_CONFIG)

    print("Initializing real microphone stream...")
    mic = LiveMicrophoneStream(config=DEFAULT_CONFIG)
    mic.start()
    print(f"Microphone capturing at {mic.capture_samplerate} Hz -> 16,000 Hz Mono")
    print(f"Test duration: {duration_sec} seconds. Please speak normally...")
    print("-" * 70)

    start_time = time.time()
    end_time = start_time + duration_sec
    chunk_idx = 0
    latencies = []
    chunk_records = []
    window_records = []

    try:
        while time.time() < end_time:
            item = mic.get_chunk(timeout=1.5)
            if item is None:
                continue

            chunk, ts = item
            chunk_idx += 1

            t0 = time.perf_counter()
            prob = engine.predict_chunk(chunk, 16_000)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            silent = is_silent(chunk, DEFAULT_CONFIG)
            curr_p, max_p = aggregator.current_window_progress
            chunk_records.append({
                "chunk": chunk_idx,
                "prob": prob,
                "silent": silent,
                "latency_ms": latency_ms,
            })

            silent_tag = " (SILENCE)" if silent else ""
            print(f"  [Sec {chunk_idx:2d}] 1s P(Spoof) = {prob*100:5.1f}%{silent_tag:10s} | Latency = {latency_ms:4.1f}ms | Window: {curr_p + 1}/5")

            res = aggregator.add_chunk(prob, is_silent=silent)
            if res is not None:
                w_idx = res["window_idx"]
                w_score = res["score"]
                is_susp = w_score >= threshold
                susp_tag = "SUSPICIOUS (>= Thresh)" if is_susp else "BENIGN / SAFE"
                window_records.append({
                    "window_idx": w_idx,
                    "score": w_score,
                    "avg": res["avg"],
                    "peak": res["peak"],
                    "probs": res["probs"],
                    "suspicious": is_susp,
                })
                p_strs = ", ".join(f"{p*100:.0f}%" for p in res["probs"])
                print("  " + "=" * 66)
                print(f"  * 5-SECOND DECISION WINDOW #{w_idx}:")
                print(f"     Aggregated Risk Score: {w_score*100:.1f}%")
                print(f"     Individual Chunks:     [{p_strs}]")
                print(f"     Mean: {res['avg']*100:.1f}% | Peak: {res['peak']*100:.1f}%")
                print(f"     Decision Status:       {susp_tag}")
                print("  " + "=" * 66)

    finally:
        mic.stop()
        print("\nMicrophone stream stopped.")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST SUMMARY REPORT")
    print("=" * 70)
    total_chunks = len(chunk_records)
    total_windows = len(window_records)
    susp_windows = sum(1 for w in window_records if w["suspicious"])
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    rtf = (avg_latency / 1000.0) / 1.0

    print(f"Total 1-Second Chunks Captured:      {total_chunks}")
    print(f"Total 5-Second Windows Completed:     {total_windows}")
    print(f"Windows Classified as Suspicious:    {susp_windows} / {total_windows} (Threshold = {threshold:.2f})")
    print(f"Mean Chunk Inference Latency:        {avg_latency:.2f} ms")
    print(f"95th Percentile Latency:             {p95_latency:.2f} ms")
    print(f"Real-Time Factor (RTF):              {rtf:.4f}x (~{1.0/max(1e-5, rtf):.0f}x faster than real-time)")
    print("-" * 70)
    print("Window-by-Window Breakdown:")
    for w in window_records:
        tag = "FLAGGED" if w["suspicious"] else "CLEARED"
        print(f"  * Window #{w['window_idx']}: Risk = {w['score']*100:5.1f}% | [{tag}] | Chunks: {[f'{p*100:.1f}%' for p in w['probs']]}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    run_diagnostic(duration_sec=args.seconds, threshold=args.threshold)
