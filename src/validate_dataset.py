"""
VAANI-SHIELD — Dataset Validation Script for ASVspoof 2019 LA.

Run from the project root:
    python -m src.validate_dataset

This script validates the dataset WITHOUT training the model.
It reports statistics, checks speaker overlap, verifies audio files,
and runs a mini smoke-test on 15 randomly-chosen utterances.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("validate_dataset")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from src.config import (
    VaaniConfig,
    ASVSPOOF_ROOT,
    ASVSPOOF_TRAIN_PROTOCOL,
    ASVSPOOF_DEV_PROTOCOL,
)
from src.dataset import (
    ASVspoofDataset,
    ASVspoofRecord,
    parse_asvspoof_protocol,
    check_speaker_overlap,
    LABEL_REAL,
    LABEL_SYNTHETIC,
)
from src.audio import load_audio, chunk_audio, is_silent
from src.features import extract_log_mel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sep(char: str = "=", width: int = 60) -> str:
    return char * width


def _print_section(title: str) -> None:
    print(f"\n{_sep()}")
    print(f"  {title}")
    print(_sep())


def _smoke_test_record(record: ASVspoofRecord, config: VaaniConfig) -> dict:
    """
    Run a single utterance through the full pipeline.
    Returns a dict with keys: ok, error, label, shape.
    """
    result: dict = {
        "utterance_id": record.utterance_id,
        "label": record.label,
        "ok": False,
        "error": None,
        "shape": None,
    }
    try:
        # 1. Audio loads
        if not record.audio_path.exists():
            raise FileNotFoundError(f"Missing: {record.audio_path}")

        samples, sr = load_audio(record.audio_path, config)

        # 2. Chunking works
        chunks = chunk_audio(samples, sr, config, pad_last=True)
        if not chunks:
            raise RuntimeError("chunk_audio returned no chunks")

        # 3. Pick first non-silent chunk (or fall back to first)
        chunk = chunks[0]
        for c in chunks:
            if not is_silent(c, config):
                chunk = c
                break

        # 4. Feature extraction works
        feat = extract_log_mel(chunk, sr, config)

        # 5. Shape matches model expectation
        expected = (config.n_mels, config.n_frames)
        if feat.shape != expected:
            raise ValueError(f"Feature shape {feat.shape} != expected {expected}")

        result["ok"] = True
        result["shape"] = feat.shape

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def validate(config: VaaniConfig = VaaniConfig(), n_smoke: int = 15) -> bool:
    """
    Run full dataset validation.

    Returns True if dataset is READY FOR TRAINING, False otherwise.
    """
    errors: List[str] = []

    print(_sep())
    print("  VAANI-SHIELD DATASET VALIDATION")
    print("  ASVspoof 2019 LA")
    print(_sep())

    # ------------------------------------------------------------------
    # 1. Check paths
    # ------------------------------------------------------------------
    print(f"\nDataset root   : {ASVSPOOF_ROOT}")
    print(f"Train protocol : {ASVSPOOF_TRAIN_PROTOCOL}")
    print(f"Dev protocol   : {ASVSPOOF_DEV_PROTOCOL}")

    train_dir = ASVSPOOF_ROOT / "ASVspoof2019_LA_train" / "flac"
    dev_dir   = ASVSPOOF_ROOT / "ASVspoof2019_LA_dev"   / "flac"

    for path, label in [
        (ASVSPOOF_ROOT, "ASVSPOOF_ROOT"),
        (ASVSPOOF_TRAIN_PROTOCOL, "Train protocol"),
        (ASVSPOOF_DEV_PROTOCOL,   "Dev protocol"),
        (train_dir, "Train audio dir"),
        (dev_dir,   "Dev audio dir"),
    ]:
        if not path.exists():
            errors.append(f"NOT FOUND: {label} = {path}")
            print(f"  [MISSING] {label}: {path}")
        else:
            print(f"  [OK]      {label}: {path}")

    if any("MISSING" in e for e in errors):
        _print_section("STATUS: DATASET PATHS MISSING")
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    # ------------------------------------------------------------------
    # 2. Parse protocols
    # ------------------------------------------------------------------
    print("\nParsing train protocol...")
    try:
        train_records = parse_asvspoof_protocol(ASVSPOOF_TRAIN_PROTOCOL, train_dir)
    except Exception as exc:
        errors.append(f"Train protocol parse error: {exc}")
        print(f"  ERROR: {exc}")
        return False

    print("Parsing dev protocol...")
    try:
        dev_records = parse_asvspoof_protocol(ASVSPOOF_DEV_PROTOCOL, dev_dir)
    except Exception as exc:
        errors.append(f"Dev protocol parse error: {exc}")
        print(f"  ERROR: {exc}")
        return False

    # ------------------------------------------------------------------
    # 3. Statistics
    # ------------------------------------------------------------------
    train_bon   = sum(1 for r in train_records if r.label == LABEL_REAL)
    train_spoof = sum(1 for r in train_records if r.label == LABEL_SYNTHETIC)
    dev_bon     = sum(1 for r in dev_records   if r.label == LABEL_REAL)
    dev_spoof   = sum(1 for r in dev_records   if r.label == LABEL_SYNTHETIC)

    train_speakers = sorted({r.speaker_id for r in train_records})
    dev_speakers   = sorted({r.speaker_id for r in dev_records})
    overlap        = sorted(set(train_speakers) & set(dev_speakers))

    train_systems = sorted({r.system_id for r in train_records if r.system_id != "-"})
    dev_systems   = sorted({r.system_id for r in dev_records   if r.system_id != "-"})
    all_systems   = sorted(set(train_systems) | set(dev_systems))

    _print_section("TRAIN SPLIT")
    print(f"  Protocol  : {ASVSPOOF_TRAIN_PROTOCOL.name}")
    print(f"  Files     : {len(train_records)}")
    print(f"  Speakers  : {len(train_speakers)}")
    print(f"  Bona-fide : {train_bon}")
    print(f"  Spoof     : {train_spoof}")
    print(f"  Systems   : {', '.join(train_systems)}")

    _print_section("DEVELOPMENT SPLIT")
    print(f"  Protocol  : {ASVSPOOF_DEV_PROTOCOL.name}")
    print(f"  Files     : {len(dev_records)}")
    print(f"  Speakers  : {len(dev_speakers)}")
    print(f"  Bona-fide : {dev_bon}")
    print(f"  Spoof     : {dev_spoof}")
    print(f"  Systems   : {', '.join(dev_systems)}")

    # ------------------------------------------------------------------
    # 4. Speaker leakage check
    # ------------------------------------------------------------------
    _print_section("SPEAKER LEAKAGE CHECK")
    print(f"  Train speakers : {len(train_speakers)}")
    print(f"  Dev speakers   : {len(dev_speakers)}")
    print(f"  Overlap        : {len(overlap)}")

    if overlap:
        errors.append(f"SPEAKER LEAKAGE DETECTED: {overlap}")
        print(f"\n  *** ERROR: Speakers appear in BOTH train and dev: {overlap} ***")
        print("  This is a dataset integrity problem. DO NOT train with this data.")
    else:
        print("  TRAIN speakers and DEV speakers are DISJOINT [OK]")

    # ------------------------------------------------------------------
    # 5. Audio file spot-check
    # ------------------------------------------------------------------
    _print_section("AUDIO FILE CHECK")
    train_missing = [r for r in train_records if not r.audio_path.exists()]
    dev_missing   = [r for r in dev_records   if not r.audio_path.exists()]

    print(f"  Train missing : {len(train_missing)} / {len(train_records)}")
    print(f"  Dev missing   : {len(dev_missing)} / {len(dev_records)}")

    if train_missing:
        errors.append(f"{len(train_missing)} train audio files missing")
        for r in train_missing[:5]:
            print(f"    MISSING: {r.audio_path}")

    if dev_missing:
        errors.append(f"{len(dev_missing)} dev audio files missing")
        for r in dev_missing[:5]:
            print(f"    MISSING: {r.audio_path}")

    # ------------------------------------------------------------------
    # 6. Sample rate verification (check first available file)
    # ------------------------------------------------------------------
    _print_section("SAMPLE RATE CHECK")
    sr_verified = False
    for r in train_records:
        if r.audio_path.exists():
            try:
                import soundfile as sf
                info = sf.info(str(r.audio_path))
                print(f"  File        : {r.audio_path.name}")
                print(f"  Sample rate : {info.samplerate} Hz")
                print(f"  Channels    : {info.channels}")
                print(f"  Format      : {info.format} / {info.subtype}")
                if info.samplerate != config.sample_rate:
                    errors.append(
                        f"Sample rate mismatch: file has {info.samplerate} Hz, "
                        f"config expects {config.sample_rate} Hz"
                    )
                else:
                    print(f"  [OK] Sample rate matches config ({config.sample_rate} Hz)")
                sr_verified = True
                break
            except Exception as exc:
                print(f"  WARNING: Could not read {r.audio_path}: {exc}")
                break

    if not sr_verified:
        errors.append("Could not verify sample rate (no accessible audio file)")

    # ------------------------------------------------------------------
    # 7. Smoke test — full pipeline on N random utterances
    # ------------------------------------------------------------------
    _print_section(f"SMOKE TEST ({n_smoke} random utterances)")
    all_records = list(train_records) + list(dev_records)
    available   = [r for r in all_records if r.audio_path.exists()]

    if not available:
        errors.append("No audio files available for smoke test")
        print("  ERROR: No accessible audio files found")
    else:
        sample = random.Random(42).sample(available, min(n_smoke, len(available)))
        passed = 0
        failed = 0
        for record in sample:
            res = _smoke_test_record(record, config)
            tag = "[OK]   " if res["ok"] else "[FAIL] "
            label_str = "bonafide" if res["label"] == LABEL_REAL else "spoof   "
            if res["ok"]:
                print(f"  {tag} {record.utterance_id}  {label_str}  shape={res['shape']}")
                passed += 1
            else:
                print(f"  {tag} {record.utterance_id}  {label_str}  ERROR: {res['error']}")
                failed += 1

        print(f"\n  Smoke test: {passed} passed, {failed} failed")
        if failed > 0:
            errors.append(f"Smoke test: {failed} failures")

        # Verify feature shape
        if passed > 0:
            print(f"  Feature shape  : ({config.n_mels}, {config.n_frames})")
            print(f"  Tensor shape   : (1, {config.n_mels}, {config.n_frames})")

    # ------------------------------------------------------------------
    # 8. Final report
    # ------------------------------------------------------------------
    _print_section("FINAL REPORT")
    print(f"  Dataset root        : {ASVSPOOF_ROOT}")
    print(f"  Train utterances    : {len(train_records)}")
    print(f"  Dev utterances      : {len(dev_records)}")
    print(f"  Train speakers      : {len(train_speakers)}")
    print(f"  Dev speakers        : {len(dev_speakers)}")
    print(f"  Speaker overlap     : {len(overlap)} (must be 0)")
    print(f"  Train bona-fide     : {train_bon}")
    print(f"  Train spoof         : {train_spoof}")
    print(f"  Dev bona-fide       : {dev_bon}")
    print(f"  Dev spoof           : {dev_spoof}")
    print(f"  Attack systems      : {', '.join(all_systems)}")
    print(f"  Sample rate         : 16000 Hz (FLAC, 16-bit)")
    print(f"  Feature shape       : (1, {config.n_mels}, {config.n_frames})")

    ready = len(errors) == 0

    print(f"\n{'=' * 60}")
    if ready:
        print("  Dataset status: READY FOR TRAINING")
    else:
        print("  Dataset status: NOT READY")
        for e in errors:
            print(f"    ERROR: {e}")
    print(_sep())
    print()

    return ready


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate ASVspoof 2019 LA dataset integration")
    parser.add_argument(
        "--smoke", type=int, default=15,
        help="Number of utterances to smoke-test (default: 15)"
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Override ASVSPOOF_ROOT (default: from config.py / env var)"
    )
    args = parser.parse_args()

    if args.root:
        import os
        os.environ["VAANI_ASVSPOOF_ROOT"] = str(args.root)
        # Re-import to pick up env var
        from importlib import reload
        import src.config as _cfg
        reload(_cfg)

    cfg = VaaniConfig()
    ready = validate(cfg, n_smoke=args.smoke)
    sys.exit(0 if ready else 1)
