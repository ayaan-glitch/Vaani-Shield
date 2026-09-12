# VAANI-SHIELD

**AI-powered on-device synthetic speech detection.**

VAANI-SHIELD detects suspected AI-generated / voice-cloned speech in real time. Audio is
processed locally on-device — raw call audio is never uploaded to a cloud server.

---

## Architecture

```
Incoming remote-speaker audio
         ↓
 1-second audio chunks
         ↓
 16 kHz mono preprocessing
         ↓
 64-bin log-Mel spectrogram  (64 × 101 frames)
         ↓
 LCNN + GRU  (~400K params, ~2–4 MB FP32)
         ↓
 synthetic probability ∈ [0, 1]
         ↓
 temporal aggregation (0.5 × avg + 0.5 × peak)
         ↓
 real-time risk score  →  LOW / MEDIUM / HIGH
         ↓
 final call risk score
```

**Model: Light CNN (LCNN) with Max-Feature-Map activations + GRU head.**
This architecture is the anti-spoofing community standard from ASVspoof competitions.
It is lightweight (~2–4 MB), exports cleanly to ONNX, and quantises well to INT8 for
Android deployment.

---

## Project Structure

```
vaani shield/test/
├── audio/test.wav              ← sample audio for smoke-testing
├── dataset/
│   ├── real/                   ← populate with real human speech WAVs
│   └── synthetic/              ← populate with synthetic/TTS WAVs
├── checkpoints/                ← saved model weights (auto-created)
├── models/                     ← exported ONNX / TFLite (auto-created)
├── outputs/
│   ├── metrics/                ← JSON metrics, CSV training log
│   ├── plots/                  ← ROC curve, score distribution
│   └── predictions/            ← per-sample prediction CSVs
├── src/
│   ├── config.py               ← all hyperparameters
│   ├── audio.py                ← load, resample, chunk
│   ├── features.py             ← log-Mel spectrogram extractor
│   ├── dataset.py              ← PyTorch Dataset (speaker-aware split)
│   ├── model.py                ← LCNN + GRU architecture
│   ├── train.py                ← training loop + early stopping
│   ├── evaluate.py             ← EER, AUC-ROC, confusion matrix, plots
│   ├── inference.py            ← inference engine (trained + untrained modes)
│   ├── aggregation.py          ← temporal chunk aggregation
│   ├── realtime.py             ← real-time simulation using trained model
│   ├── benchmark.py            ← RTF and latency profiler
│   └── export.py               ← ONNX export + sanity check
├── tests/                      ← pytest test suite
├── main.py                     ← Phase 1 audio load test (preserved)
├── realtime.py                 ← Phase 1 prototype simulation (preserved)
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU:** Swap the torch/torchaudio lines in `requirements.txt` for the CUDA wheel:
> ```bash
> pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Run the test suite

```bash
python -m pytest tests/ -v
```

All tests should pass without any training data.

### 3. Smoke-test the pipeline (no training data needed)

```bash
python -m src.realtime
python -m src.inference --audio audio/test.wav
python -m src.benchmark
```

The pipeline runs in **UNTRAINED mode** (outputs 0.5 for all chunks) until a model is
trained. This validates the full data-flow pipeline.

---

## Training

### Step 1 — Populate the dataset

Place WAV files in:
- `dataset/real/` — real human speech recordings
- `dataset/synthetic/` — AI-generated / TTS / voice-cloned speech

**Recommended datasets:**

| Dataset | Content | Source |
|---------|---------|--------|
| ASVspoof 2019 LA | Real + synthetic (60K utterances) | https://datashare.ed.ac.uk/handle/10283/3336 |
| LJSpeech | Real (13K utterances, single speaker) | https://keithito.com/LJ-Speech-Dataset/ |
| VCTK | Real (44 speakers) | https://datashare.ed.ac.uk/handle/10283/2651 |
| LibriTTS | Real (multiple speakers) | https://openslr.org/60/ |
| Common Voice | Real (crowd-sourced) | https://commonvoice.mozilla.org/ |

**File naming for speaker-aware splitting (important!):**

Name files with a speaker ID prefix so the dataset splitter can prevent speaker leakage:
```
dataset/real/p001_001.wav   ← VCTK speaker p001, utterance 001
dataset/real/p001_002.wav
dataset/real/p002_001.wav
dataset/synthetic/tts_p001_001.wav
```

The default regex (`^(p\d+|s\d+|[A-Za-z]+\d+)`) handles VCTK and LibriSpeech naming.
Edit `config.speaker_id_regex` in `src/config.py` for other naming conventions.

### Step 2 — Train

```bash
python -m src.train
```

Options:
```
--epochs N       (default: 50)
--batch-size N   (default: 32)
--lr LR          (default: 0.001)
--device cpu|cuda
```

Training saves:
- `checkpoints/best.pt` — best validation EER checkpoint
- `checkpoints/last.pt` — latest checkpoint (for resuming)
- `outputs/metrics/training_log.csv` — per-epoch metrics

### Step 3 — Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt
```

Outputs:
- Equal Error Rate (EER)
- AUC-ROC
- Confusion matrix
- `outputs/plots/roc_curve.png`
- `outputs/plots/score_dist.png`
- `outputs/metrics/test_metrics.json`

### Step 4 — Run real-time simulation with trained model

```bash
python -m src.realtime --checkpoint checkpoints/best.pt
```

### Step 5 — Benchmark latency

```bash
python -m src.benchmark --checkpoint checkpoints/best.pt --n-repeats 100
```

---

## Export for Android

```bash
python -m src.export --checkpoint checkpoints/best.pt
```

This produces `models/vaani_shield.onnx` and runs a PyTorch vs ONNX sanity check.

**Android deployment options:**

**Option A — ONNX Runtime Mobile (recommended):**
Add the `onnxruntime-android` AAR to your Android project and load `vaani_shield.onnx`
directly. No conversion step needed.

**Option B — TFLite:**
```bash
pip install onnx-tf tensorflow
python -m onnx_tf.backend.prepare \
    --input models/vaani_shield.onnx \
    --output models/vaani_tf/
tflite_convert \
    --saved_model_dir=models/vaani_tf \
    --output_file=models/vaani_shield.tflite
```

---

## Configuration

All hyperparameters live in [`src/config.py`](src/config.py). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Audio sample rate (Hz) |
| `chunk_seconds` | 1.0 | Inference window length |
| `n_mels` | 64 | Mel filter banks |
| `n_fft` | 512 | FFT window (32ms) |
| `hop_length` | 160 | Hop size (10ms) |
| `gru_hidden` | 128 | GRU hidden dimension |
| `dropout` | 0.3 | Dropout rate |
| `epochs` | 50 | Max training epochs |
| `batch_size` | 32 | Training batch size |
| `early_stopping_patience` | 10 | Epochs without improvement before stopping |
| `low_threshold` | 0.35 | LOW → MEDIUM risk boundary |
| `medium_threshold` | 0.65 | MEDIUM → HIGH risk boundary |

---

## Important Caveats

> ⚠️ **Model accuracy depends entirely on your training data.**
>
> A model trained on limited or non-representative data will not generalise to
> real-world calls. Synthetic speech detection is an active research problem.
> No pre-trained weights are included; you must train the model yourself on
> appropriate data.

> ⚠️ **Speaker-aware splitting is critical.**
>
> If a speaker appears in both training and test sets, measured accuracy will be
> artificially inflated. Always verify that speaker IDs are correctly extracted
> (check the logs for "Speaker-aware split: N train speakers...").

> ⚠️ **This system does NOT upload audio to any server.**
>
> All processing happens locally (Python research env) or on-device (Android inference).

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Complete | Repository inspection |
| 1 | ✅ Complete | Phase 1 prototype (root `realtime.py`) |
| 2 | ✅ Complete | Full ML pipeline (`src/`) |
| 3 | 🔲 Pending | Populate dataset + train |
| 4 | 🔲 Pending | Android TFLite/ORT-Mobile integration |

---

## License

Private / research use only. Not for production deployment without proper evaluation.
