"""
VAANI-SHIELD - Phase 1 Prototype
Real-time microphone simulation using a pre-recorded WAV file.

NO TRAINED MODEL EXISTS YET.
predict_synthetic_probability() below is a DEMO PLACEHOLDER only.
It does NOT perform real synthetic-voice detection. Its only job
right now is to prove the chunked streaming pipeline works end to end
(load -> resample -> chunk -> feature-extract -> "infer" -> report).
"""

import sys
import time
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
AUDIO_PATH = BASE_DIR / "audio" / "test.wav"

TARGET_SR = 16000          # simulate a 16kHz mono mic stream
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)

LOW_THRESHOLD = 0.35
MEDIUM_THRESHOLD = 0.65

SILENCE_RMS_THRESHOLD = 1e-4  # below this, treat chunk as silence


# ----------------------------------------------------------------------
# STEP 1-2: LOAD + RESAMPLE/MONO
# ----------------------------------------------------------------------
def load_and_prepare_audio(path: Path):
    """Load an audio file and convert it to mono float32 at TARGET_SR."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, sr = sf.read(str(path), always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read audio file '{path}': {exc}") from exc

    data = np.asarray(data, dtype=np.float32)

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to TARGET_SR if needed
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return data, sr


# ----------------------------------------------------------------------
# STEP 3: LIGHTWEIGHT FEATURE EXTRACTION
# ----------------------------------------------------------------------
def extract_features(chunk: np.ndarray, sr: int) -> dict:
    """
    Extract a small, cheap set of audio features from one chunk.
    This is intentionally lightweight (no deep model) - just enough
    to feed the placeholder inference function and prove data is
    flowing correctly through the pipeline.
    """
    if chunk.size == 0:
        return {
            "rms": 0.0,
            "zcr": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "mfcc_mean": 0.0,
            "is_silent": True,
        }

    rms = float(np.sqrt(np.mean(np.square(chunk))))
    is_silent = rms < SILENCE_RMS_THRESHOLD

    if is_silent:
        # Skip expensive spectral math on silence; return zeros safely.
        return {
            "rms": rms,
            "zcr": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "mfcc_mean": 0.0,
            "is_silent": True,
        }

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(chunk)))

    centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
    centroid = float(np.nan_to_num(np.mean(centroid)))

    bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
    bandwidth = float(np.nan_to_num(np.mean(bandwidth)))

    n_fft = min(2048, max(256, chunk.size))
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13, n_fft=n_fft)
    mfcc_mean = float(np.nan_to_num(np.mean(mfcc)))

    return {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "mfcc_mean": mfcc_mean,
        "is_silent": False,
    }


# ----------------------------------------------------------------------
# STEP 5: PLACEHOLDER "INFERENCE" - NOT A REAL DETECTOR
# ----------------------------------------------------------------------
def predict_synthetic_probability(features: dict) -> float:
    """
    *** DEMO PLACEHOLDER ONLY - NOT REAL AI DETECTION ***

    There is no trained model behind this yet. This function exists
    purely to give the streaming pipeline something deterministic to
    call at the spot where real inference will eventually go.

    It derives a pseudo-probability from simple feature values so the
    demo output looks plausible and varies chunk-to-chunk, but it has
    NO relationship to actual synthetic-voice likelihood.
    """
    if features.get("is_silent"):
        return 0.0

    # Purely arbitrary deterministic combination - replace with a real
    # model's output in a later phase.
    score = (
        0.4 * min(features["zcr"] * 10, 1.0)
        + 0.3 * min(features["spectral_centroid"] / 4000.0, 1.0)
        + 0.3 * min(abs(features["mfcc_mean"]) / 50.0, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def classify_risk(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "LOW"
    elif prob < MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


# ----------------------------------------------------------------------
# MAIN SIMULATION
# ----------------------------------------------------------------------
def run_simulation():
    print("VAANI-SHIELD REAL-TIME SIMULATION")
    print("(Phase 1 prototype - placeholder inference, no trained model)\n")

    try:
        audio, sr = load_and_prepare_audio(AUDIO_PATH)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    total_samples = len(audio)
    audio_duration = total_samples / sr

    if total_samples == 0:
        print("ERROR: Audio file contains no samples.")
        sys.exit(1)

    if audio_duration < CHUNK_SECONDS:
        print(
            f"NOTE: Audio ({audio_duration:.2f}s) is shorter than one "
            f"{CHUNK_SECONDS:.0f}s chunk. Processing it as a single chunk.\n"
        )

    results = []
    total_processing_time = 0.0
    chunk_index = 0
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + CHUNK_SAMPLES, total_samples)
        chunk = audio[start_sample:end_sample]

        chunk_start_sec = start_sample / sr
        chunk_end_sec = end_sample / sr
        is_partial = (end_sample - start_sample) < CHUNK_SAMPLES

        t0 = time.perf_counter()
        features = extract_features(chunk, sr)
        prob = predict_synthetic_probability(features)
        elapsed = time.perf_counter() - t0
        total_processing_time += elapsed

        chunk_index += 1
        risk = classify_risk(prob)
        tag = " (partial chunk)" if is_partial else ""
        silent_tag = " [SILENT]" if features.get("is_silent") else ""

        print(
            f"Chunk {chunk_index} | {chunk_start_sec:.2f}-{chunk_end_sec:.2f} sec | "
            f"{elapsed * 1000:.0f} ms | Synthetic: {prob * 100:.0f}% | "
            f"{risk}{tag}{silent_tag}"
        )

        results.append(
            {
                "index": chunk_index,
                "start": chunk_start_sec,
                "end": chunk_end_sec,
                "probability": prob,
                "risk": risk,
                "silent": features.get("is_silent", False),
            }
        )

        start_sample = end_sample

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------
    num_chunks = len(results)
    probs = [r["probability"] for r in results]
    avg_prob = float(np.mean(probs)) if probs else 0.0
    peak_prob = float(np.max(probs)) if probs else 0.0
    suspicious_chunks = [r for r in results if r["risk"] in ("MEDIUM", "HIGH")]
    final_risk_score = float(0.5 * avg_prob + 0.5 * peak_prob)
    real_time_factor = (
        audio_duration / total_processing_time
        if total_processing_time > 0
        else float("inf")
    )

    print("\n--- SUMMARY ---")
    print(f"Audio duration:            {audio_duration:.2f} sec")
    print(f"Number of chunks:          {num_chunks}")
    print(f"Average synthetic prob:    {avg_prob * 100:.1f}%")
    print(f"Suspicious chunks:         {len(suspicious_chunks)} / {num_chunks}")
    print(f"Peak probability:          {peak_prob * 100:.1f}%")
    print(f"Final demo risk score:     {final_risk_score * 100:.1f}% "
          f"({classify_risk(final_risk_score)})")
    print(f"Total processing time:     {total_processing_time * 1000:.1f} ms")
    print(f"Real-time factor:          {real_time_factor:.1f}x "
          f"(audio_duration / processing_time)")
    print(
        "\nReminder: 'Synthetic %' values above are DEMO placeholders "
        "from predict_synthetic_probability(). No trained model is in "
        "the loop yet - this run only validates the streaming pipeline."
    )


if __name__ == "__main__":
    run_simulation()
