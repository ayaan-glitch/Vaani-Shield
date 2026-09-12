"""
VAANI-SHIELD — Log-Mel spectrogram feature extraction.

Design constraints:
  - Stateless: same input → same output, no random state, no global mutation.
  - Fixed output shape: always (n_mels, n_frames) regardless of input length.
  - Uses librosa so it works without PyTorch (needed for dataset pre-caching).
  - The same mel filterbank parameters are used during training AND inference
    (enforced through VaaniConfig).
"""

from __future__ import annotations

import numpy as np
import librosa

from .config import VaaniConfig, DEFAULT_CONFIG


def extract_log_mel(
    chunk: np.ndarray,
    sr: int,
    config: VaaniConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Compute a log-power Mel spectrogram for a single audio chunk.

    Parameters
    ----------
    chunk  : 1-D float32 array, expected length = sr * chunk_seconds
    sr     : sample rate (must equal config.sample_rate for correct mel bins)
    config : VaaniConfig instance

    Returns
    -------
    log_mel : np.ndarray, shape (n_mels, n_frames), dtype float32
              Values are in dB, normalised to [0, 1].
    """
    # Guard: ensure chunk is exactly the right length (pad or trim)
    expected_len = int(sr * config.chunk_seconds)
    if len(chunk) < expected_len:
        pad = np.zeros(expected_len - len(chunk), dtype=np.float32)
        chunk = np.concatenate([chunk, pad])
    elif len(chunk) > expected_len:
        chunk = chunk[:expected_len]

    # Compute mel-power spectrogram
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.f_min,
        fmax=config.f_max,
        power=2.0,
    )

    # Convert to dB (log-power)
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=config.top_db)

    # Normalise to [0, 1] (shift from [-top_db, 0] to [0, 1])
    log_mel = (log_mel + config.top_db) / config.top_db
    log_mel = np.clip(log_mel, 0.0, 1.0).astype(np.float32)

    # Ensure exact frame count (librosa may vary by ±1 frame)
    target_frames = config.n_frames
    if log_mel.shape[1] < target_frames:
        pad_cols = target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_cols)), mode="constant")
    elif log_mel.shape[1] > target_frames:
        log_mel = log_mel[:, :target_frames]

    return log_mel  # shape: (n_mels, n_frames)


def batch_extract_log_mel(
    chunks: list[np.ndarray],
    sr: int,
    config: VaaniConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Extract log-Mel features for a list of audio chunks.

    Returns
    -------
    np.ndarray, shape (N, n_mels, n_frames), dtype float32
    """
    return np.stack(
        [extract_log_mel(c, sr, config) for c in chunks], axis=0
    )
