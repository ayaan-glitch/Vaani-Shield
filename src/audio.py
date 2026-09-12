"""
VAANI-SHIELD — Audio I/O utilities.

Wraps Phase 1 load/resample/chunk logic into clean, reusable functions.
All functions are stateless and side-effect-free (no globals modified).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import librosa

from .config import VaaniConfig, DEFAULT_CONFIG

# Supported audio extensions
_AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_audio(
    path: Path | str,
    config: VaaniConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to float32 mono, resample to config.sample_rate.

    Returns
    -------
    samples : np.ndarray  shape (N,), dtype float32
    sr      : int         always == config.sample_rate
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, sr = sf.read(str(path), always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read audio file '{path}': {exc}") from exc

    data = np.asarray(data, dtype=np.float32)

    # Stereo / multi-channel → mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != config.sample_rate:
        data = librosa.resample(data, orig_sr=sr, target_sr=config.sample_rate)

    return data, config.sample_rate


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_audio(
    samples: np.ndarray,
    sr: int,
    config: VaaniConfig = DEFAULT_CONFIG,
    pad_last: bool = True,
) -> List[np.ndarray]:
    """
    Split a 1-D float32 audio array into fixed-length chunks.

    Parameters
    ----------
    samples  : 1-D float32 array of audio
    sr       : sample rate (should match config.sample_rate)
    config   : VaaniConfig instance
    pad_last : if True, zero-pad the final short chunk to full length;
               if False, the final chunk may be shorter than chunk_samples.

    Returns
    -------
    List of np.ndarray chunks, each of shape (chunk_samples,) or shorter if
    pad_last=False.
    """
    chunk_samples = int(sr * config.chunk_seconds)
    if chunk_samples <= 0:
        raise ValueError(f"chunk_samples must be positive, got {chunk_samples}")

    chunks: List[np.ndarray] = []
    start = 0
    total = len(samples)

    while start < total:
        end = start + chunk_samples
        chunk = samples[start:end]

        if len(chunk) < chunk_samples:
            if pad_last:
                # Zero-pad to full chunk length
                pad = np.zeros(chunk_samples - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])
            # else: keep the short chunk as-is

        chunks.append(chunk)
        start = end

    return chunks


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------
def is_silent(chunk: np.ndarray, config: VaaniConfig = DEFAULT_CONFIG) -> bool:
    """Return True if the chunk's RMS is below the silence threshold."""
    rms = float(np.sqrt(np.mean(np.square(chunk))))
    return rms < config.silence_rms_threshold


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------
def find_audio_files(directory: Path | str) -> List[Path]:
    """
    Recursively find all audio files under *directory*.

    Returns a sorted list of Path objects (deterministic ordering).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    files: List[Path] = []
    for ext in _AUDIO_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))

    return sorted(files)
