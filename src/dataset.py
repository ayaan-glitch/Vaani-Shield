"""
VAANI-SHIELD — PyTorch Dataset for binary synthetic-speech classification.

Directory layout expected:
    dataset/
        real/        *.wav  (and sub-folders — scanned recursively)
        synthetic/   *.wav

Speaker-aware splitting
-----------------------
If file names contain a recognisable speaker-ID prefix (configurable via
config.speaker_id_regex), speakers are split at the speaker level so that
no speaker appears in more than one partition.  This prevents the model from
learning speaker identity instead of synthetic vs real.

If no speaker IDs are found, the split falls back to file-level random split
with a warning.

Feature caching
---------------
Extracted log-Mel features are saved alongside the audio as
    <audio_stem>.vaani.npy
so they are computed only once per file. Delete these files to force
re-extraction.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import find_audio_files, load_audio, chunk_audio, is_silent
from .config import VaaniConfig, DEFAULT_CONFIG
from .features import extract_log_mel

log = logging.getLogger(__name__)

# Labels
LABEL_REAL = 0
LABEL_SYNTHETIC = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _speaker_id(path: Path, pattern: str) -> Optional[str]:
    """Extract speaker ID from filename stem using regex. Returns None if no match."""
    m = re.match(pattern, path.stem)
    return m.group(0) if m else None


def _speaker_aware_split(
    items: List[Tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    config: VaaniConfig,
    rng: np.random.Generator,
) -> Tuple[List, List, List]:
    """
    Split (path, label) pairs at the speaker level.
    Falls back to file-level split if no speaker IDs found.
    """
    # Group by speaker
    speaker_to_items: Dict[str, List] = {}
    no_id_items: List = []
    for item in items:
        sid = _speaker_id(item[0], config.speaker_id_regex)
        if sid:
            speaker_to_items.setdefault(sid, []).append(item)
        else:
            no_id_items.append(item)

    if speaker_to_items:
        speakers = sorted(speaker_to_items.keys())
        rng.shuffle(speakers)  # type: ignore[arg-type]
        n = len(speakers)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_sp = speakers[:n_train]
        val_sp = speakers[n_train : n_train + n_val]
        test_sp = speakers[n_train + n_val :]

        train = [i for s in train_sp for i in speaker_to_items[s]]
        val = [i for s in val_sp for i in speaker_to_items[s]]
        test = [i for s in test_sp for i in speaker_to_items[s]]

        # Files without IDs go into train to avoid wasting them
        if no_id_items:
            warnings.warn(
                f"{len(no_id_items)} files had no speaker ID and were assigned to train.",
                stacklevel=3,
            )
            train.extend(no_id_items)

        log.info(
            "Speaker-aware split: %d train speakers, %d val speakers, %d test speakers",
            len(train_sp),
            len(val_sp),
            len(test_sp),
        )
    else:
        warnings.warn(
            "No speaker IDs found; falling back to file-level random split.",
            stacklevel=3,
        )
        all_items = list(items)
        rng.shuffle(all_items)
        n = len(all_items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = all_items[:n_train]
        val = all_items[n_train : n_train + n_val]
        test = all_items[n_train + n_val :]

    return train, val, test


def _load_or_compute_feature(
    path: Path, config: VaaniConfig
) -> np.ndarray:
    """
    Load cached log-Mel feature array, or compute and cache it.
    Cache file: <path.stem>.vaani.npy next to the audio file.
    """
    cache_path = path.parent / (path.stem + ".vaani.npy")

    if cache_path.exists():
        try:
            feat = np.load(str(cache_path))
            # Validate shape matches current config
            if feat.shape == (config.n_mels, config.n_frames):
                return feat
            else:
                log.debug("Cache shape mismatch for %s, recomputing.", path.name)
        except Exception:
            log.debug("Cache read failed for %s, recomputing.", path.name)

    # Compute
    samples, sr = load_audio(path, config)
    chunks = chunk_audio(samples, sr, config, pad_last=True)
    if not chunks:
        raise RuntimeError(f"No audio chunks produced from {path}")

    # Use first non-silent chunk; fall back to first chunk
    chunk = chunks[0]
    for c in chunks:
        if not is_silent(c, config):
            chunk = c
            break

    feat = extract_log_mel(chunk, sr, config)

    try:
        np.save(str(cache_path), feat)
    except Exception:
        pass  # non-fatal; we'll just recompute next time

    return feat


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class VaaniDataset(Dataset):
    """
    Binary synthetic-speech classification dataset.

    Parameters
    ----------
    split  : 'train' | 'val' | 'test'
    config : VaaniConfig
    real_dir     : override for dataset/real/
    synthetic_dir: override for dataset/synthetic/
    """

    def __init__(
        self,
        split: str = "train",
        config: VaaniConfig = DEFAULT_CONFIG,
        real_dir: Optional[Path] = None,
        synthetic_dir: Optional[Path] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.config = config

        # Resolve paths — use caller-provided dirs if they exist, else defaults
        from .config import REAL_DIR, SYNTHETIC_DIR
        _real_dir: Path = Path(real_dir) if real_dir is not None and Path(real_dir).is_dir() else REAL_DIR
        _syn_dir: Path = Path(synthetic_dir) if synthetic_dir is not None and Path(synthetic_dir).is_dir() else SYNTHETIC_DIR

        # Collect all files with labels
        real_files = [(p, LABEL_REAL) for p in find_audio_files(_real_dir)]
        syn_files = [(p, LABEL_SYNTHETIC) for p in find_audio_files(_syn_dir)]
        all_items = real_files + syn_files

        if not all_items:
            warnings.warn(
                f"No audio files found in:\n  {_real_dir}\n  {_syn_dir}\n"
                "The dataset is empty. Populate dataset/real/ and dataset/synthetic/ "
                "with WAV files to enable training. See README.md for instructions.",
                stacklevel=2,
            )
            self._items: List[Tuple[Path, int]] = []
            return

        rng = np.random.default_rng(config.random_seed)
        train_items, val_items, test_items = _speaker_aware_split(
            all_items,
            config.train_ratio,
            config.val_ratio,
            config,
            rng,
        )

        split_map = {"train": train_items, "val": val_items, "test": test_items}
        self._items = split_map[split]

        log.info(
            "VaaniDataset[%s]: %d samples (%d real, %d synthetic)",
            split,
            len(self._items),
            sum(1 for _, lbl in self._items if lbl == LABEL_REAL),
            sum(1 for _, lbl in self._items if lbl == LABEL_SYNTHETIC),
        )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self._items[idx]
        feat = _load_or_compute_feature(path, self.config)
        # Shape: (1, n_mels, n_frames) — channel-first for Conv2d
        x = torch.from_numpy(feat).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    @property
    def num_real(self) -> int:
        return sum(1 for _, lbl in self._items if lbl == LABEL_REAL)

    @property
    def num_synthetic(self) -> int:
        return sum(1 for _, lbl in self._items if lbl == LABEL_SYNTHETIC)

    def class_weights(self) -> torch.Tensor:
        """
        Returns [w_real, w_synthetic] balanced class weights.
        Use with BCEWithLogitsLoss(pos_weight=...) for imbalanced data.
        """
        n = len(self._items)
        if n == 0:
            return torch.ones(2)
        n_real = self.num_real
        n_syn = self.num_synthetic
        w_real = n / (2 * n_real) if n_real > 0 else 1.0
        w_syn = n / (2 * n_syn) if n_syn > 0 else 1.0
        return torch.tensor([w_real, w_syn], dtype=torch.float32)


# ---------------------------------------------------------------------------
# ASVspoof 2019 LA Dataset
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import NamedTuple as _NamedTuple


class ASVspoofRecord(_NamedTuple):
    """One row from an ASVspoof 2019 LA protocol file."""
    speaker_id:  str    # e.g. "LA_0079"
    utterance_id: str   # e.g. "LA_T_1138215"
    system_id:   str    # e.g. "A01" or "-" for bonafide
    label:       int    # 0 = bonafide, 1 = spoof
    audio_path:  Path   # absolute path to .flac file


def parse_asvspoof_protocol(protocol_path: Path, audio_dir: Path) -> List[ASVspoofRecord]:
    """
    Parse an ASVspoof 2019 LA CM protocol file and resolve audio paths.

    Protocol format (5 whitespace-separated columns):
        SPEAKER_ID  UTTERANCE_ID  SYSTEM_ID  -  KEY

    Parameters
    ----------
    protocol_path : Path to the .txt protocol file
    audio_dir     : Directory containing the .flac audio files

    Returns
    -------
    List of ASVspoofRecord (one per line)

    Raises
    ------
    FileNotFoundError if protocol_path does not exist
    ValueError        if any line has wrong column count
    """
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"ASVspoof protocol not found: {protocol_path}\n"
            "Set env var VAANI_ASVSPOOF_ROOT or extract the dataset."
        )

    records: List[ASVspoofRecord] = []
    with open(protocol_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Protocol line {lineno} has {len(parts)} columns, expected 5:\n  {line}"
                )
            # Actual column order in ASVspoof 2019 LA protocol files:
            #   col[0] = SPEAKER_ID
            #   col[1] = UTTERANCE_ID (audio filename without extension)
            #   col[2] = '-'  (always unused placeholder for LA)
            #   col[3] = SYSTEM_ID  ('A01'-'A19' for spoof, '-' for bonafide)
            #   col[4] = KEY  ('bonafide' or 'spoof')
            # NOTE: The README description of columns 3 and 4 is swapped vs actual.
            speaker_id    = parts[0]
            utterance_id  = parts[1]
            _unused       = parts[2]   # always '-'
            system_id     = parts[3]   # 'A01'-'A19' or '-' for bonafide
            key           = parts[4]

            # Validate key
            if key not in ("bonafide", "spoof"):
                raise ValueError(
                    f"Unknown key '{key}' on line {lineno} — expected 'bonafide' or 'spoof'"
                )

            label = LABEL_REAL if key == "bonafide" else LABEL_SYNTHETIC

            audio_path = audio_dir / f"{utterance_id}.flac"

            records.append(ASVspoofRecord(
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                system_id=system_id,
                label=label,
                audio_path=audio_path,
            ))

    log.info("Parsed %d records from %s", len(records), protocol_path.name)
    return records


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 Logical Access (LA).

    This dataset uses the OFFICIAL protocol files to assign labels.
    Labels are NEVER inferred from filenames or directory structure.

        bonafide -> label 0 (real)
        spoof    -> label 1 (synthetic)

    Parameters
    ----------
    split : 'train' | 'dev'
        'train' loads ASVspoof2019.LA.cm.train.trn.txt
        'dev'   loads ASVspoof2019.LA.cm.dev.trl.txt
        'eval'  is explicitly forbidden here (never used for training/validation)
    config : VaaniConfig
    asvspoof_root : Path to extracted LA/ directory.
        Defaults to ASVSPOOF_ROOT from config.py.
        Override with env var VAANI_ASVSPOOF_ROOT.
    verify_files : bool
        If True, warn for each audio file that does not exist on disk.
        Default True. Set False to skip I/O during unit tests with mocked paths.

    Notes
    -----
    - Train and dev splits are NEVER mixed.
    - Eval split is NEVER loaded here.
    - Speaker sets: train speakers != dev speakers (no overlap in ASVspoof 2019 LA).
    - Each item returned is (Tensor(1, n_mels, n_frames), Tensor(scalar float32 label)).
    - Feature caching uses the existing .vaani.npy mechanism next to each audio file.
    """

    def __init__(
        self,
        split: str,
        config: VaaniConfig = DEFAULT_CONFIG,
        asvspoof_root: Optional[Path] = None,
        verify_files: bool = True,
    ) -> None:
        if split == "eval":
            raise ValueError(
                "split='eval' is not allowed in ASVspoofDataset. "
                "The eval set must never be used during training."
            )
        if split not in ("train", "dev"):
            raise ValueError(f"split must be 'train' or 'dev', got '{split}'")

        self.split = split
        self.config = config

        # Resolve root
        from .config import (
            ASVSPOOF_ROOT,
            ASVSPOOF_TRAIN_DIR, ASVSPOOF_DEV_DIR,
            ASVSPOOF_TRAIN_PROTOCOL, ASVSPOOF_DEV_PROTOCOL,
        )
        _root = Path(asvspoof_root) if asvspoof_root is not None else ASVSPOOF_ROOT

        if split == "train":
            protocol_path = _root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
            audio_dir = _root / "ASVspoof2019_LA_train" / "flac"
        else:  # dev
            protocol_path = _root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
            audio_dir = _root / "ASVspoof2019_LA_dev" / "flac"

        self._records: List[ASVspoofRecord] = parse_asvspoof_protocol(protocol_path, audio_dir)

        # Optional: warn about missing audio files (non-fatal so tests can use mock paths)
        if verify_files:
            missing = [r for r in self._records if not r.audio_path.exists()]
            if missing:
                warnings.warn(
                    f"ASVspoofDataset[{split}]: {len(missing)} audio files not found on disk. "
                    f"First missing: {missing[0].audio_path}",
                    stacklevel=2,
                )

        n_bon = sum(1 for r in self._records if r.label == LABEL_REAL)
        n_spoof = sum(1 for r in self._records if r.label == LABEL_SYNTHETIC)
        log.info(
            "ASVspoofDataset[%s]: %d utterances (%d bonafide, %d spoof)",
            split, len(self._records), n_bon, n_spoof,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self._records[idx]
        feat = _load_or_compute_feature(record.audio_path, self.config)
        x = torch.from_numpy(feat).unsqueeze(0)   # (1, n_mels, n_frames)
        y = torch.tensor(record.label, dtype=torch.float32)
        return x, y

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def records(self) -> List[ASVspoofRecord]:
        """Read-only list of all ASVspoofRecord instances."""
        return list(self._records)

    @property
    def speaker_ids(self) -> List[str]:
        return sorted({r.speaker_id for r in self._records})

    @property
    def system_ids(self) -> List[str]:
        """Unique attack/TTS system IDs (excludes '-' bonafide placeholder)."""
        return sorted({r.system_id for r in self._records if r.system_id != "-"})

    @property
    def num_bonafide(self) -> int:
        return sum(1 for r in self._records if r.label == LABEL_REAL)

    @property
    def num_spoof(self) -> int:
        return sum(1 for r in self._records if r.label == LABEL_SYNTHETIC)

    def class_weights(self) -> torch.Tensor:
        """Returns [w_bonafide, w_spoof] balanced class weights."""
        n = len(self._records)
        if n == 0:
            return torch.ones(2)
        n_bon = self.num_bonafide
        n_spoof = self.num_spoof
        w_bon   = n / (2 * n_bon)   if n_bon   > 0 else 1.0
        w_spoof = n / (2 * n_spoof) if n_spoof > 0 else 1.0
        return torch.tensor([w_bon, w_spoof], dtype=torch.float32)


def check_speaker_overlap(
    train_dataset: ASVspoofDataset,
    dev_dataset: ASVspoofDataset,
) -> Tuple[bool, List[str]]:
    """
    Check whether any speaker appears in both train and dev splits.

    Returns
    -------
    (has_overlap, overlap_list)
        has_overlap  : True if TRAIN and DEV share at least one speaker
        overlap_list : list of overlapping speaker IDs (empty if none)
    """
    train_speakers = set(train_dataset.speaker_ids)
    dev_speakers   = set(dev_dataset.speaker_ids)
    overlap = sorted(train_speakers & dev_speakers)
    return (len(overlap) > 0, overlap)

