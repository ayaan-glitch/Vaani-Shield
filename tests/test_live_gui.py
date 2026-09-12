"""
Tests for src.live_gui — Desktop Tkinter application with 5-second decision window.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from src.config import CHECKPOINTS_DIR, VaaniConfig
from src.aggregation import DecisionWindowAggregator
from src.live_gui import LiveDetectionApp


@pytest.fixture(scope="module")
def tk_root():
    """Create a single hidden Tk root for GUI unit testing."""
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture(scope="module")
def app(tk_root):
    checkpoint = CHECKPOINTS_DIR / "best.pt"
    cp_path = checkpoint if checkpoint.exists() else Path("non_existent.pt")
    app_inst = LiveDetectionApp(root=tk_root, checkpoint_path=cp_path, auto_test_mode=False)
    yield app_inst
    if app_inst.is_running:
        app_inst.stop_monitoring()


@pytest.fixture(autouse=True)
def reset_app(app):
    """Automatically reset session state before each test."""
    app.reset_session()
    yield


class TestDecisionWindowAggregator:
    def test_window_requires_exactly_five_chunks(self):
        agg = DecisionWindowAggregator(window_size=5)
        # Chunks 1 to 4 should return None
        for i in range(1, 5):
            res = agg.add_chunk(0.20)
            assert res is None
            assert agg.current_window_progress == (i, 5)
            assert agg.total_chunks_analyzed == i

        # Chunk 5 triggers window calculation
        res = agg.add_chunk(0.40)
        assert res is not None
        assert res["window_idx"] == 1
        # Probs: [0.20, 0.20, 0.20, 0.20, 0.40]
        # avg = 1.20 / 5 = 0.24, peak = 0.40
        # score = 0.5 * 0.24 + 0.5 * 0.40 = 0.32
        assert pytest.approx(res["avg"], abs=1e-4) == 0.24
        assert pytest.approx(res["peak"], abs=1e-4) == 0.40
        assert pytest.approx(res["score"], abs=1e-4) == 0.32
        assert agg.num_completed_windows == 1
        assert agg.current_window_progress == (0, 5)

    def test_second_window_completes_at_chunk_10(self):
        agg = DecisionWindowAggregator(window_size=5)
        for _ in range(5):
            agg.add_chunk(0.10)
        assert agg.num_completed_windows == 1

        for _ in range(4):
            assert agg.add_chunk(0.90) is None
        res2 = agg.add_chunk(0.90)
        assert res2 is not None
        assert res2["window_idx"] == 2
        assert pytest.approx(res2["score"], abs=1e-4) == 0.90
        assert agg.num_completed_windows == 2


class TestLiveGUIApp:
    def test_initial_state(self, app):
        assert app.status_var.get() == "STOPPED"
        assert app.threshold_var.get() == 0.50
        assert app.threshold_display_var.get() == "0.50"
        assert app.main_verdict_var.get() == "IDLE"
        assert app.five_sec_risk_str_var.get() == "--.-%"
        assert app.total_chunks_var.get() == "0"
        assert app.total_windows_var.get() == "0"
        assert app.suspicious_windows_var.get() == "0"
        assert app.is_running is False

    def test_threshold_presets(self, app):
        app.set_threshold_preset(0.25)
        assert app.threshold_var.get() == 0.25
        assert app.threshold_display_var.get() == "0.25"

        app.set_threshold_preset(0.16)
        assert app.threshold_var.get() == 0.16
        assert app.threshold_display_var.get() == "0.16"

        app.set_threshold_preset(0.50)
        assert app.threshold_var.get() == 0.50
        assert app.threshold_display_var.get() == "0.50"

    def test_chunks_1_to_4_do_not_update_main_verdict(self, app):
        # Feed 4 individual chunks
        sample_probs = [0.85, 0.70, 0.90, 0.65]
        for idx, p in enumerate(sample_probs, 1):
            msg = {
                "chunk_idx": idx,
                "timestamp": time.time(),
                "prob": p,
                "silent": False,
                "latency_ms": 15.0,
            }
            app._update_ui_with_chunk(msg)

            # Main verdict MUST NOT change prematurely
            assert app.main_verdict_var.get() == "IDLE"
            assert app.total_windows_var.get() == "0"
            assert f"{idx} / 5" in app.window_progress_str_var.get()
            assert f"{p*100:.1f}%" in app.latest_chunk_prob_var.get()

        assert app.total_chunks_var.get() == "4"
        assert len(app.prob_history) == 4

    def test_chunk_5_triggers_main_decision(self, app):
        # Chunks 1 to 4
        for idx in range(1, 5):
            app._update_ui_with_chunk({
                "chunk_idx": idx,
                "timestamp": time.time(),
                "prob": 0.80,
                "silent": False,
                "latency_ms": 15.0,
            })
            assert app.main_verdict_var.get() == "IDLE"

        # Chunk 5 completes the 5-second window
        app._update_ui_with_chunk({
            "chunk_idx": 5,
            "timestamp": time.time(),
            "prob": 0.90,
            "silent": False,
            "latency_ms": 16.0,
        })

        # Window completed!
        assert app.total_windows_var.get() == "1"
        assert app.total_chunks_var.get() == "5"
        # 4 chunks of 0.80, 1 chunk of 0.90 -> avg = 0.82, peak = 0.90 -> score = 0.86 (Critical)
        assert app.main_verdict_var.get() == "CRITICAL RISK"
        assert "86.0%" in app.five_sec_risk_str_var.get()
        assert app.suspicious_windows_var.get() == "1"
        assert "0 / 5" in app.window_progress_str_var.get()

    def test_second_window_updates_decision_on_chunk_10(self, app):
        # Window 1: 5 chunks with high risk
        for idx in range(1, 6):
            app._update_ui_with_chunk({
                "chunk_idx": idx,
                "timestamp": time.time(),
                "prob": 0.80,
                "silent": False,
                "latency_ms": 15.0,
            })
        assert app.main_verdict_var.get() == "HIGH RISK"
        assert app.total_windows_var.get() == "1"

        # Chunks 6 to 9 with low risk: verdict stays at Window 1's verdict
        for idx in range(6, 10):
            app._update_ui_with_chunk({
                "chunk_idx": idx,
                "timestamp": time.time(),
                "prob": 0.05,
                "silent": False,
                "latency_ms": 15.0,
            })
            assert app.main_verdict_var.get() == "HIGH RISK"  # Unchanged during accumulation

        # Chunk 10 completes Window 2
        app._update_ui_with_chunk({
            "chunk_idx": 10,
            "timestamp": time.time(),
            "prob": 0.05,
            "silent": False,
            "latency_ms": 15.0,
        })
        assert app.total_windows_var.get() == "2"
        assert app.main_verdict_var.get() == "LOW RISK"  # Updated to LOW RISK

    def test_all_silent_window(self, app):
        for idx in range(1, 6):
            app._update_ui_with_chunk({
                "chunk_idx": idx,
                "timestamp": time.time(),
                "prob": 0.0,
                "silent": True,
                "latency_ms": 14.0,
            })
        assert app.total_windows_var.get() == "1"
        assert app.main_verdict_var.get() == "SILENCE / AMBIENT"

    def test_reset_session(self, app):
        for idx in range(1, 6):
            app._update_ui_with_chunk({
                "chunk_idx": idx,
                "timestamp": time.time(),
                "prob": 0.75,
                "silent": False,
                "latency_ms": 15.0,
            })
        assert app.total_windows_var.get() == "1"

        app.reset_session()
        assert app.total_windows_var.get() == "0"
        assert app.total_chunks_var.get() == "0"
        assert app.suspicious_windows_var.get() == "0"
        assert app.main_verdict_var.get() == "IDLE"
        assert app.five_sec_risk_str_var.get() == "--.-%"
        assert len(app.prob_history) == 0
