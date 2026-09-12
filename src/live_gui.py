"""
VAANI-SHIELD — Live Microphone Real-Time Test GUI (5-Second Decision Window).

Desktop Tkinter GUI for continuous real-time voice cloning detection from
a local microphone using the trained VaaniLCNN model checkpoint.

Architecture
------------
- 1-Second Audio Input: Captures 16,000 samples (1.0s at 16 kHz) and runs
  PyTorch inference on each chunk in real time (<20 ms per chunk on CPU).
- 5-Second Security Decision Window: Accumulates five consecutive 1-second
  probabilities before computing the aggregated risk score and updating
  the main security verdict.
- Non-blocking multithreaded architecture (sounddevice audio worker + Tkinter UI).
- Real-time graph of individual 1-second probabilities with 5-second boundary markers.
- Decision threshold slider with presets (0.50 default, 0.25 best F1, 0.16 high-security).
- Real-time factor (RTF) and inference latency telemetry.
- Standalone smoke-test mode (--test) for automated verification.

Usage
-----
    python -m src.live_gui
    python -m src.live_gui --checkpoint checkpoints/best.pt
    python -m src.live_gui --test
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import DEFAULT_CONFIG, CHECKPOINTS_DIR, VaaniConfig
from .inference import VaaniInferenceEngine
from .aggregation import ChunkAggregator, DecisionWindowAggregator
from .live_mic import LiveMicrophoneStream, list_input_devices
from .audio import is_silent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("live_gui")


# ---------------------------------------------------------------------------
# Visual Theme & Colors
# ---------------------------------------------------------------------------
THEME_BG = "#1e1e24"          # Dark Charcoal
THEME_CARD_BG = "#2b2d42"     # Midnight Navy
THEME_TEXT = "#edf2f4"        # Off-white
THEME_MUTED = "#8d99ae"       # Muted Gray
COLOR_SAFE = "#2a9d8f"        # Teal / Safe
COLOR_WARN = "#e9c46a"        # Amber / Warning
COLOR_DANGER = "#f4a261"      # Orange / High Risk
COLOR_CRITICAL = "#e63946"    # Crimson / Critical Threat
COLOR_ACCENT = "#457b9d"      # Slate Blue


class LiveDetectionApp:
    """Main desktop application window for live microphone inference with 5s decision window."""

    def __init__(
        self,
        root: tk.Tk,
        checkpoint_path: Path,
        auto_test_mode: bool = False,
        test_duration_sec: int = 6,
    ) -> None:
        self.root = root
        self.checkpoint_path = checkpoint_path
        self.auto_test_mode = auto_test_mode
        self.test_duration_sec = test_duration_sec

        self.root.title("VAANI-SHIELD — Live Voice Cloning Detection (5-Sec Decision Window)")
        self.root.geometry("980x790")
        self.root.minsize(860, 690)
        self.root.configure(bg=THEME_BG)

        # 1. Load Inference Engine
        self.engine = VaaniInferenceEngine(checkpoint_path=self.checkpoint_path, device_str="cpu")
        if not self.engine.is_trained:
            log.warning("Warning: Model checkpoint could not be verified as trained.")

        self.config: VaaniConfig = self.engine.config
        self.session_aggregator = ChunkAggregator(config=self.config)
        self.decision_aggregator = DecisionWindowAggregator(window_size=5, config=self.config)

        # 2. State variables
        self.threshold_var = tk.DoubleVar(value=0.50)
        self.threshold_display_var = tk.StringVar(value="0.50")
        self.status_var = tk.StringVar(value="STOPPED")

        # 5-second main security decision states
        self.main_verdict_var = tk.StringVar(value="IDLE")
        self.five_sec_risk_str_var = tk.StringVar(value="--.-%")
        self.window_progress_str_var = tk.StringVar(value="[░░░░░░░░░░] 0 / 5 sec")

        # Telemetry & session counters
        self.latest_chunk_prob_var = tk.StringVar(value="--.-%")
        self.session_risk_var = tk.StringVar(value="0.0%")
        self.total_chunks_var = tk.StringVar(value="0")
        self.total_windows_var = tk.StringVar(value="0")
        self.suspicious_windows_var = tk.StringVar(value="0")
        self.suspicious_windows_count = 0
        self.latency_var = tk.StringVar(value="0.0 ms")
        self.rtf_var = tk.StringVar(value="0.00x")
        self.device_info_var = tk.StringVar(value="Initializing audio...")

        self.prob_history: List[float] = []
        self.window_boundary_indices: List[int] = []
        self.max_history_len = 35

        # 3. Threading communication
        self.ui_queue: queue.Queue[dict] = queue.Queue()
        self.mic_stream: Optional[LiveMicrophoneStream] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False

        # 4. Build UI
        self._build_ui()

        # 5. Start poll timer
        self.root.after(40, self._poll_queue)

        # 6. Window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 7. If auto test mode requested
        if self.auto_test_mode:
            log.info("Auto-test mode enabled: running for %d seconds...", self.test_duration_sec)
            self.root.after(500, self.start_monitoring)
            self.root.after(int(self.test_duration_sec * 1000), self._auto_test_finish)

    # -----------------------------------------------------------------------
    # UI Building
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Header Banner
        header_frame = tk.Frame(self.root, bg=THEME_CARD_BG, pady=12, padx=16)
        header_frame.pack(fill=tk.X, padx=12, pady=(10, 6))

        title_lbl = tk.Label(
            header_frame,
            text="VAANI-SHIELD",
            font=("Helvetica", 18, "bold"),
            fg=THEME_TEXT,
            bg=THEME_CARD_BG,
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header_frame,
            text=" | Real-Time Voice Cloning Detection (5s Decision Window)",
            font=("Helvetica", 12),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        )
        subtitle_lbl.pack(side=tk.LEFT, pady=(3, 0))

        self.badge_lbl = tk.Label(
            header_frame,
            textvariable=self.status_var,
            font=("Helvetica", 11, "bold"),
            fg="#ffffff",
            bg="#6c757d",
            padx=12,
            pady=4,
            relief=tk.FLAT,
        )
        self.badge_lbl.pack(side=tk.RIGHT)

        # Control Bar
        control_frame = tk.Frame(self.root, bg=THEME_BG, pady=6, padx=12)
        control_frame.pack(fill=tk.X)

        self.start_btn = tk.Button(
            control_frame,
            text="▶  START MONITORING",
            font=("Helvetica", 11, "bold"),
            bg=COLOR_SAFE,
            fg="#ffffff",
            activebackground="#21867a",
            activeforeground="#ffffff",
            padx=14,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_monitoring,
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = tk.Button(
            control_frame,
            text="⏹  STOP",
            font=("Helvetica", 11, "bold"),
            bg=COLOR_CRITICAL,
            fg="#ffffff",
            activebackground="#c12d38",
            activeforeground="#ffffff",
            padx=14,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED,
            command=self.stop_monitoring,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.reset_btn = tk.Button(
            control_frame,
            text="⟲  CLEAR SESSION",
            font=("Helvetica", 10),
            bg=THEME_CARD_BG,
            fg=THEME_TEXT,
            activebackground="#3a3d59",
            activeforeground=THEME_TEXT,
            padx=10,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.reset_session,
        )
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 20))

        # Threshold controls
        th_frame = tk.Frame(control_frame, bg=THEME_BG)
        th_frame.pack(side=tk.RIGHT)

        tk.Label(
            th_frame,
            text="Decision Threshold:",
            font=("Helvetica", 10),
            fg=THEME_TEXT,
            bg=THEME_BG,
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.th_scale = tk.Scale(
            th_frame,
            from_=0.05,
            to=0.95,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            bg=THEME_BG,
            fg=THEME_TEXT,
            highlightthickness=0,
            troughcolor=THEME_CARD_BG,
            activebackground=COLOR_ACCENT,
            length=120,
            command=self._on_threshold_changed,
        )
        self.th_scale.pack(side=tk.LEFT, padx=(0, 8))

        # Preset buttons
        tk.Button(
            th_frame,
            text="0.50 (Default)",
            font=("Helvetica", 8),
            bg=THEME_CARD_BG,
            fg=THEME_TEXT,
            relief=tk.FLAT,
            command=lambda: self.set_threshold_preset(0.50),
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            th_frame,
            text="0.25 (Best F1)",
            font=("Helvetica", 8),
            bg=THEME_CARD_BG,
            fg=THEME_TEXT,
            relief=tk.FLAT,
            command=lambda: self.set_threshold_preset(0.25),
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            th_frame,
            text="0.16 (High Sec)",
            font=("Helvetica", 8),
            bg=THEME_CARD_BG,
            fg=THEME_TEXT,
            relief=tk.FLAT,
            command=lambda: self.set_threshold_preset(0.16),
        ).pack(side=tk.LEFT, padx=2)

        # Dashboard Grid (Middle section)
        mid_frame = tk.Frame(self.root, bg=THEME_BG)
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Left Column: 5-Second Decision & Telemetry
        left_col = tk.Frame(mid_frame, bg=THEME_BG, width=430)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8))

        # Main 5-Second Decision Card
        decision_card = tk.Frame(left_col, bg=THEME_CARD_BG, padx=16, pady=14)
        decision_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            decision_card,
            text="MAIN SECURITY DECISION (5-SECOND WINDOW)",
            font=("Helvetica", 10, "bold"),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        ).pack(anchor=tk.W)

        self.main_verdict_lbl = tk.Label(
            decision_card,
            textvariable=self.main_verdict_var,
            font=("Helvetica", 19, "bold"),
            fg=COLOR_SAFE,
            bg=THEME_CARD_BG,
        )
        self.main_verdict_lbl.pack(anchor=tk.W, pady=(5, 2))

        self.five_sec_risk_lbl = tk.Label(
            decision_card,
            textvariable=self.five_sec_risk_str_var,
            font=("Helvetica", 34, "bold"),
            fg=THEME_TEXT,
            bg=THEME_CARD_BG,
        )
        self.five_sec_risk_lbl.pack(anchor=tk.W)

        tk.Label(
            decision_card,
            text="5-Second Aggregated Synthetic Risk Score",
            font=("Helvetica", 9),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        ).pack(anchor=tk.W)

        # 5-Second Risk level bar canvas
        self.meter_canvas = tk.Canvas(
            decision_card,
            height=14,
            bg="#1b1c2b",
            highlightthickness=0,
        )
        self.meter_canvas.pack(fill=tk.X, pady=(8, 8))

        # Decision Window Progress Indicator
        prog_frame = tk.Frame(decision_card, bg=THEME_CARD_BG)
        prog_frame.pack(fill=tk.X, pady=(2, 0))

        tk.Label(
            prog_frame,
            text="Decision Window Progress:",
            font=("Helvetica", 9),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        ).pack(side=tk.LEFT)

        self.progress_lbl = tk.Label(
            prog_frame,
            textvariable=self.window_progress_str_var,
            font=("Consolas", 10, "bold"),
            fg=COLOR_ACCENT,
            bg=THEME_CARD_BG,
        )
        self.progress_lbl.pack(side=tk.RIGHT)

        # Telemetry Card (Session Stats)
        telemetry_card = tk.Frame(left_col, bg=THEME_CARD_BG, padx=16, pady=12)
        telemetry_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            telemetry_card,
            text="LIVE TELEMETRY & AUDIT METRICS",
            font=("Helvetica", 10, "bold"),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        ).pack(anchor=tk.W, pady=(0, 8))

        # Metric grid
        grid = tk.Frame(telemetry_card, bg=THEME_CARD_BG)
        grid.pack(fill=tk.X)

        self._add_stat_row(grid, 0, "Latest 1-Sec Chunk P(Spoof):", self.latest_chunk_prob_var, is_bold=True)
        self._add_stat_row(grid, 1, "Session-Wide Risk (Rolling):", self.session_risk_var)
        self._add_stat_row(grid, 2, "Total 1-Sec Chunks Analyzed:", self.total_chunks_var)
        self._add_stat_row(grid, 3, "Completed 5-Sec Windows:", self.total_windows_var, is_bold=True)
        self._add_stat_row(grid, 4, "Suspicious 5-Sec Windows:", self.suspicious_windows_var)
        self._add_stat_row(grid, 5, "Latest Chunk Latency (1s):", self.latency_var)
        self._add_stat_row(grid, 6, "Real-Time Factor (RTF):", self.rtf_var)
        self._add_stat_row(grid, 7, "Decision Threshold Active:", self.threshold_display_var)

        # Right Column: Visualizer & Log
        right_col = tk.Frame(mid_frame, bg=THEME_BG)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Scrolling Probability Graph Card
        graph_card = tk.Frame(right_col, bg=THEME_CARD_BG, padx=12, pady=10)
        graph_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        tk.Label(
            graph_card,
            text="LIVE 1-SECOND PROBABILITIES (WITH 5-SEC WINDOW BORDERS)",
            font=("Helvetica", 10, "bold"),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        ).pack(anchor=tk.W, pady=(0, 4))

        self.graph_canvas = tk.Canvas(
            graph_card,
            bg="#1a1b26",
            highlightthickness=0,
            height=160,
        )
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)
        self.graph_canvas.bind("<Configure>", lambda _: self._draw_graph())

        # History log Card
        log_card = tk.Frame(right_col, bg=THEME_CARD_BG, padx=12, pady=10)
        log_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            log_card,
            text="CHUNK STREAM & DECISION AUDIT LOG",
            font=("Helvetica", 10, "bold"),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        ).pack(anchor=tk.W, pady=(0, 4))

        # Log text area
        log_scroll = tk.Scrollbar(log_card)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_card,
            height=6,
            bg="#1a1b26",
            fg=THEME_TEXT,
            font=("Consolas", 9),
            relief=tk.FLAT,
            yscrollcommand=log_scroll.set,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)

        # Footer Status Bar
        footer_frame = tk.Frame(self.root, bg=THEME_CARD_BG, pady=6, padx=16)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=(0, 10))

        self.device_lbl = tk.Label(
            footer_frame,
            textvariable=self.device_info_var,
            font=("Helvetica", 9),
            fg=THEME_MUTED,
            bg=THEME_CARD_BG,
        )
        self.device_lbl.pack(side=tk.LEFT)

        model_status_text = (
            f"Model: VaaniLCNN (~347K params) | Decision Window: 5 Chunks | Checkpoint: {self.checkpoint_path.name}"
        )
        tk.Label(
            footer_frame,
            text=model_status_text,
            font=("Helvetica", 9),
            fg=COLOR_SAFE if self.engine.is_trained else COLOR_WARN,
            bg=THEME_CARD_BG,
        ).pack(side=tk.RIGHT)

        # Initial meter and graph render
        self._update_meter(0.0)
        self._draw_graph()

    def _add_stat_row(
        self,
        parent: tk.Widget,
        row_idx: int,
        label_text: str,
        text_var: tk.StringVar,
        is_bold: bool = False,
    ) -> None:
        font_style = ("Helvetica", 10, "bold") if is_bold else ("Helvetica", 10)
        color = THEME_TEXT if is_bold else THEME_MUTED

        lbl = tk.Label(parent, text=label_text, font=font_style, fg=color, bg=THEME_CARD_BG)
        lbl.grid(row=row_idx, column=0, sticky=tk.W, pady=3)

        val_lbl = tk.Label(parent, textvariable=text_var, font=("Helvetica", 10, "bold"), fg=THEME_TEXT, bg=THEME_CARD_BG)
        val_lbl.grid(row=row_idx, column=1, sticky=tk.E, padx=(12, 0), pady=3)

    # -----------------------------------------------------------------------
    # Threshold handling
    # -----------------------------------------------------------------------
    def _on_threshold_changed(self, val_str: str) -> None:
        self.threshold_display_var.set(f"{float(val_str):.2f}")
        self._draw_graph()

    def set_threshold_preset(self, val: float) -> None:
        self.threshold_var.set(val)
        self.threshold_display_var.set(f"{val:.2f}")
        self._draw_graph()

    # -----------------------------------------------------------------------
    # Audio Stream and Worker Control
    # -----------------------------------------------------------------------
    def start_monitoring(self) -> None:
        if self.is_running:
            return

        try:
            self.mic_stream = LiveMicrophoneStream(config=self.config)
            self.mic_stream.start()
        except Exception as exc:
            messagebox.showerror(
                "Microphone Initialization Error",
                f"Could not initialize audio input device:\n{exc}\n\n"
                "Please verify your microphone is plugged in and access is permitted.",
            )
            return

        self.is_running = True
        self.status_var.set("LISTENING")
        self.badge_lbl.configure(bg=COLOR_SAFE)
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.device_info_var.set(
            f"Active Mic: Capture SR={self.mic_stream.capture_samplerate} Hz -> 16,000 Hz Mono"
        )

        if self.decision_aggregator.num_completed_windows == 0:
            self.main_verdict_var.set("WAITING FOR 5s WINDOW")
            self.main_verdict_lbl.configure(fg=THEME_MUTED)
            self.five_sec_risk_str_var.set("Collecting audio...")

        # Launch background inference thread
        self.worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.worker_thread.start()
        log.info("Live microphone monitoring started.")

    def stop_monitoring(self) -> None:
        if not self.is_running:
            return

        self.is_running = False
        if self.mic_stream is not None:
            self.mic_stream.stop()
            self.mic_stream = None

        self.status_var.set("STOPPED")
        self.badge_lbl.configure(bg="#6c757d")
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.device_info_var.set("Microphone paused. Ready to start.")
        log.info("Live microphone monitoring stopped.")

    def reset_session(self) -> None:
        self.session_aggregator.reset()
        self.decision_aggregator.reset()
        self.prob_history.clear()
        self.window_boundary_indices.clear()
        self.suspicious_windows_count = 0

        self.main_verdict_var.set("IDLE")
        self.main_verdict_lbl.configure(fg=COLOR_SAFE)
        self.five_sec_risk_str_var.set("--.-%")
        self.window_progress_str_var.set("[░░░░░░░░░░] 0 / 5 sec")
        self.latest_chunk_prob_var.set("--.-%")
        self.session_risk_var.set("0.0%")
        self.total_chunks_var.set("0")
        self.total_windows_var.set("0")
        self.suspicious_windows_var.set("0")
        self.latency_var.set("0.0 ms")
        self.rtf_var.set("0.00x")
        self.log_text.delete("1.0", tk.END)
        self._update_meter(0.0)
        self._draw_graph()
        log.info("Session reset.")

    # -----------------------------------------------------------------------
    # Background Inference Worker
    # -----------------------------------------------------------------------
    def _inference_worker(self) -> None:
        """Continuously pulls 1s chunks from mic_stream, runs inference, and queues results."""
        chunk_idx = 0
        while self.is_running and self.mic_stream is not None:
            item = self.mic_stream.get_chunk(timeout=0.2)
            if item is None:
                continue

            chunk, ts = item
            chunk_idx += 1

            t0 = time.perf_counter()
            prob = self.engine.predict_chunk(chunk, 16_000)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            silent = is_silent(chunk, self.config)

            msg = {
                "chunk_idx": chunk_idx,
                "timestamp": ts,
                "prob": prob,
                "silent": silent,
                "latency_ms": latency_ms,
            }
            self.ui_queue.put(msg)

    # -----------------------------------------------------------------------
    # UI Queue Polling & Updates (Runs on Main Tkinter Thread)
    # -----------------------------------------------------------------------
    def _poll_queue(self) -> None:
        """Processes all messages posted to UI queue by the background thread."""
        while not self.ui_queue.empty():
            try:
                msg = self.ui_queue.get_nowait()
                self._update_ui_with_chunk(msg)
            except queue.Empty:
                break

        # Schedule next poll
        self.root.after(40, self._poll_queue)

    def _update_ui_with_chunk(self, msg: dict) -> None:
        prob = msg["prob"]
        latency_ms = msg["latency_ms"]
        silent = msg.get("silent", False)
        chunk_idx = msg["chunk_idx"]
        th = self.threshold_var.get()

        # 1. Update 1s probability history for graph
        self.prob_history.append(prob)
        if len(self.prob_history) > self.max_history_len:
            self.prob_history.pop(0)
            self.window_boundary_indices = [
                idx - 1 for idx in self.window_boundary_indices if idx - 1 >= 0
            ]

        # 2. Update telemetry labels
        self.latest_chunk_prob_var.set(f"{prob * 100:.1f}%")
        self.latency_var.set(f"{latency_ms:.1f} ms")
        rtf = (latency_ms / 1000.0) / 1.0  # 1.0s chunk duration
        self.rtf_var.set(f"{rtf:.3f}x ({1.0 / max(1e-5, rtf):.0f}x real-time)")

        # Update session-level rolling risk
        session_risk = self.session_aggregator.update(prob)
        self.session_risk_var.set(f"{session_risk * 100:.1f}%")

        # 3. Add to 5-second decision aggregator
        window_res = self.decision_aggregator.add_chunk(prob, is_silent=silent)
        curr_chunks, max_chunks = self.decision_aggregator.current_window_progress

        self.total_chunks_var.set(str(self.decision_aggregator.total_chunks_analyzed))

        time_str = time.strftime("%H:%M:%S", time.localtime(msg["timestamp"]))

        if window_res is None:
            # Accumulating window (chunks 1 to 4)
            # Do NOT update main security verdict yet
            filled = int((curr_chunks / max_chunks) * 10)
            bar_str = "[" + "█" * filled + "░" * (10 - filled) + "]"
            self.window_progress_str_var.set(f"{bar_str} {curr_chunks} / {max_chunks} sec")

            if self.decision_aggregator.num_completed_windows == 0:
                self.five_sec_risk_str_var.set("Calculating...")

            # Log 1-second chunk
            silent_tag = " [SILENCE]" if silent else ""
            log_line = (
                f"  [{time_str}] Chunk #{chunk_idx:3d} ({curr_chunks}/{max_chunks}s) | "
                f"1s P(Spoof): {prob*100:5.1f}%{silent_tag} | Latency: {latency_ms:4.1f}ms\n"
            )
            self.log_text.insert(tk.END, log_line)
            self.log_text.see(tk.END)
        else:
            # 5-second window completed!
            w_idx = window_res["window_idx"]
            w_score = window_res["score"]
            all_silent = window_res["all_silent"]
            probs = window_res["probs"]

            # Record boundary index for live visual divider line
            self.window_boundary_indices.append(len(self.prob_history) - 1)

            # Categorize verdict
            if all_silent:
                verdict = "SILENCE / AMBIENT"
                v_color = THEME_MUTED
            elif w_score < 0.35:
                verdict = "LOW RISK"
                v_color = COLOR_SAFE
            elif w_score < 0.65:
                verdict = "MEDIUM RISK"
                v_color = COLOR_WARN
            elif w_score < 0.85:
                verdict = "HIGH RISK"
                v_color = COLOR_DANGER
            else:
                verdict = "CRITICAL RISK"
                v_color = COLOR_CRITICAL

            self.main_verdict_var.set(verdict)
            self.main_verdict_lbl.configure(fg=v_color)
            self.five_sec_risk_str_var.set(f"{w_score * 100:.1f}%")

            # Update meter bar to show the 5-second aggregated risk
            self._update_meter(w_score)

            # Update window counters
            self.total_windows_var.set(str(self.decision_aggregator.num_completed_windows))
            if w_score >= th:
                self.suspicious_windows_count += 1
            self.suspicious_windows_var.set(str(self.suspicious_windows_count))

            # Reset progress indicator for the new window starting
            self.window_progress_str_var.set("[░░░░░░░░░░] 0 / 5 sec")

            # Log the completed 5-second decision window
            p_strs = ", ".join(f"{p*100:.0f}%" for p in probs)
            log_line = (
                f"★ [{time_str}] 5s DECISION #{w_idx:2d} | Risk: {w_score*100:5.1f}% | "
                f"{verdict:15s} | Chunks: [{p_strs}] | Peak: {window_res['peak']*100:.1f}%\n"
            )
            self.log_text.insert(tk.END, log_line)
            self.log_text.see(tk.END)

        # Redraw graph with updated points & dividers
        self._draw_graph()

    def _update_meter(self, prob: float) -> None:
        self.meter_canvas.delete("all")
        w = self.meter_canvas.winfo_width()
        if w <= 1:
            w = 360
        h = 14

        fill_w = int(w * max(0.0, min(1.0, prob)))
        if prob < 0.35:
            fill_color = COLOR_SAFE
        elif prob < 0.65:
            fill_color = COLOR_WARN
        elif prob < 0.85:
            fill_color = COLOR_DANGER
        else:
            fill_color = COLOR_CRITICAL

        self.meter_canvas.create_rectangle(0, 0, fill_w, h, fill=fill_color, outline="")

    def _draw_graph(self) -> None:
        c = self.graph_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1 or h <= 1:
            return

        pad_left = 36
        pad_right = 16
        pad_top = 18
        pad_bottom = 24

        plot_w = w - pad_left - pad_right
        plot_h = h - pad_top - pad_bottom

        # Grid lines
        for y_pct in [0.0, 0.25, 0.50, 0.75, 1.0]:
            y = pad_top + plot_h * (1.0 - y_pct)
            c.create_line(pad_left, y, w - pad_right, y, fill="#2b2d42", dash=(2, 2))
            c.create_text(pad_left - 6, y, text=f"{int(y_pct*100)}%", fill=THEME_MUTED, font=("Helvetica", 7), anchor=tk.E)

        # Decision threshold line
        th = self.threshold_var.get()
        th_y = pad_top + plot_h * (1.0 - th)
        c.create_line(pad_left, th_y, w - pad_right, th_y, fill=COLOR_CRITICAL, width=1.5, dash=(4, 2))
        c.create_text(w - pad_right, th_y - 8, text=f"Threshold ({th:.2f})", fill=COLOR_CRITICAL, font=("Helvetica", 8, "bold"), anchor=tk.E)

        if not self.prob_history:
            c.create_text(w / 2, h / 2, text="Waiting for audio stream...", fill=THEME_MUTED, font=("Helvetica", 10))
            return

        n = len(self.prob_history)
        step_x = plot_w / max(1, self.max_history_len - 1)

        # Draw 5-second decision window boundary dividers
        for b_idx in self.window_boundary_indices:
            if 0 <= b_idx < n:
                bx = pad_left + (b_idx + 0.5) * step_x
                c.create_line(bx, pad_top, bx, h - pad_bottom, fill=COLOR_ACCENT, width=1.5, dash=(3, 3))
                c.create_text(bx, pad_top - 8, text="5s Window", fill=COLOR_ACCENT, font=("Helvetica", 7, "bold"))

        points = []
        for i, p in enumerate(self.prob_history):
            x = pad_left + i * step_x
            y = pad_top + plot_h * (1.0 - p)
            points.append((x, y))

            # Dot color based on probability
            dot_color = COLOR_CRITICAL if p >= th else COLOR_SAFE
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=dot_color, outline="#ffffff")

        # Connect line
        if len(points) > 1:
            flat_pts = [coord for pt in points for coord in pt]
            c.create_line(*flat_pts, fill=THEME_TEXT, width=2)

    def _auto_test_finish(self) -> None:
        log.info("Auto-test completed successfully. Closing test GUI.")
        self.stop_monitoring()
        self.root.destroy()

    def _on_close(self) -> None:
        self.stop_monitoring()
        self.root.destroy()


# ---------------------------------------------------------------------------
# CLI Launcher
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="VAANI-SHIELD Live Microphone Real-Time GUI")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINTS_DIR / "best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in headless/automated test mode for test duration and exit cleanly",
    )
    parser.add_argument(
        "--test-duration",
        type=int,
        default=6,
        help="Duration in seconds for automated test mode (default: 6s)",
    )
    args = parser.parse_args()

    root = tk.Tk()
    app = LiveDetectionApp(
        root=root,
        checkpoint_path=args.checkpoint,
        auto_test_mode=args.test,
        test_duration_sec=args.test_duration,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
