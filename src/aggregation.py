"""
VAANI-SHIELD — Temporal chunk aggregation.

Converts a stream of per-chunk synthetic probabilities into:
  - A rolling current risk (updated after each chunk)
  - A final call-level risk score when the stream ends

The aggregation formula intentionally matches Phase 1:
    final = 0.5 * mean_prob + 0.5 * peak_prob

Both weights are configurable via VaaniConfig.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

from .config import VaaniConfig, DEFAULT_CONFIG


class ChunkAggregator:
    """
    Accumulates per-chunk synthetic probabilities and computes risk scores.

    Parameters
    ----------
    config   : VaaniConfig
    window   : optional rolling window size (in chunks).
               If None, uses all chunks seen so far.
    """

    def __init__(
        self,
        config: VaaniConfig = DEFAULT_CONFIG,
        window: Optional[int] = None,
    ) -> None:
        self.config = config
        self._all: List[float] = []
        self._window: Optional[Deque[float]] = (
            deque(maxlen=window) if window is not None else None
        )

    def update(self, prob: float) -> float:
        """
        Add a new per-chunk probability and return the current rolling risk score.

        Parameters
        ----------
        prob : float ∈ [0, 1]

        Returns
        -------
        current_score : float ∈ [0, 1]
        """
        self._all.append(prob)
        if self._window is not None:
            self._window.append(prob)
        return self.current_score()

    def current_score(self) -> float:
        """Current rolling risk score over the active window (or all chunks)."""
        probs = list(self._window) if self._window is not None else self._all
        if not probs:
            return 0.0
        avg = sum(probs) / len(probs)
        peak = max(probs)
        return (
            self.config.agg_weight_avg * avg
            + self.config.agg_weight_peak * peak
        )

    def final_score(self) -> float:
        """Final risk score computed over ALL chunks (not just the window)."""
        if not self._all:
            return 0.0
        avg = sum(self._all) / len(self._all)
        peak = max(self._all)
        return (
            self.config.agg_weight_avg * avg
            + self.config.agg_weight_peak * peak
        )

    def classify(self, score: Optional[float] = None) -> str:
        """Return 'LOW', 'MEDIUM', or 'HIGH' for a given score (or current score)."""
        if score is None:
            score = self.current_score()
        if score >= self.config.medium_threshold:
            return "HIGH"
        if score >= self.config.low_threshold:
            return "MEDIUM"
        return "LOW"

    def reset(self) -> None:
        """Clear all accumulated state (useful for re-use across calls)."""
        self._all.clear()
        if self._window is not None:
            self._window.clear()

    @property
    def num_chunks(self) -> int:
        return len(self._all)

    @property
    def chunk_probabilities(self) -> List[float]:
        return list(self._all)


class DecisionWindowAggregator:
    """
    Accumulates 1-second chunk probabilities into fixed-size decision windows
    (default: 5 seconds / 5 chunks) before computing a security risk score.

    Aggregation formula matches ChunkAggregator:
        risk_score = agg_weight_avg * mean(probs) + agg_weight_peak * max(probs)
    where default weights are 0.50 average + 0.50 peak.

    Parameters
    ----------
    window_size : int, default 5 (number of 1-second chunks per decision window)
    config      : VaaniConfig instance
    """

    def __init__(
        self,
        window_size: int = 5,
        config: VaaniConfig = DEFAULT_CONFIG,
    ) -> None:
        self.window_size = window_size
        self.config = config
        self._current_window: List[float] = []
        self._current_silent: List[bool] = []
        self._total_chunks: int = 0
        self._completed_windows: List[dict] = []

    def add_chunk(self, prob: float, is_silent: bool = False) -> Optional[dict]:
        """
        Add a 1-second chunk probability.

        Returns
        -------
        window_result : dict or None
            If the window reached window_size, returns a dict with:
                - 'window_idx': int (1-based index of completed window)
                - 'score': float (aggregated risk score)
                - 'avg': float (mean probability)
                - 'peak': float (maximum probability)
                - 'probs': List[float] (the raw model probabilities)
                - 'all_silent': bool (True if all chunks in window were silent)
            If the window has not yet filled (e.g. chunks 1-4), returns None.
        """
        self._total_chunks += 1
        self._current_window.append(prob)
        self._current_silent.append(is_silent)

        if len(self._current_window) >= self.window_size:
            probs = list(self._current_window)
            silent_flags = list(self._current_silent)

            avg = sum(probs) / len(probs)
            peak = max(probs)
            score = (
                self.config.agg_weight_avg * avg
                + self.config.agg_weight_peak * peak
            )
            all_silent = all(silent_flags)

            window_result = {
                "window_idx": len(self._completed_windows) + 1,
                "score": float(score),
                "avg": float(avg),
                "peak": float(peak),
                "probs": probs,
                "all_silent": all_silent,
            }
            self._completed_windows.append(window_result)

            # Reset window buffer for next decision window
            self._current_window.clear()
            self._current_silent.clear()

            return window_result

        return None

    @property
    def current_window_progress(self) -> Tuple[int, int]:
        """Returns (current_chunks_in_window, window_size), e.g. (3, 5)."""
        return len(self._current_window), self.window_size

    @property
    def current_window_probs(self) -> List[float]:
        """Returns the probabilities accumulated in the active, uncompleted window."""
        return list(self._current_window)

    @property
    def total_chunks_analyzed(self) -> int:
        return self._total_chunks

    @property
    def num_completed_windows(self) -> int:
        return len(self._completed_windows)

    @property
    def last_completed_window(self) -> Optional[dict]:
        return self._completed_windows[-1] if self._completed_windows else None

    def reset(self) -> None:
        """Reset all state across sessions."""
        self._current_window.clear()
        self._current_silent.clear()
        self._total_chunks = 0
        self._completed_windows.clear()

