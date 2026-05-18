# Neurosystem — Full Build Log

> Complete record of everything designed, built, and fixed across this project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware Setup](#2-hardware-setup)
3. [Folder Structure](#3-folder-structure)
4. [ESP32 Firmware](#4-esp32-firmware)
5. [Browser Dashboard](#5-browser-dashboard)
6. [Python GUI — 3-Wing Architecture](#6-python-gui--3-wing-architecture)
7. [All Python Modules](#7-all-python-modules)
8. [MongoDB Integration](#8-mongodb-integration)
9. [Incremental Model Training](#9-incremental-model-training)
10. [Bugs Fixed](#10-bugs-fixed)
11. [Installation](#11-installation)
12. [How to Run](#12-how-to-run)

---

## 1. Project Overview

**Neurosystem** is a real-time EEG decision-detection system built on:

- 2× BioAmp EXG Pill analog EEG amplifiers
- ESP32 microcontroller (WiFi + WebSocket server)
- Python tkinter GUI with 3 functional wings
- LDA (Linear Discriminant Analysis) classifier
- SSVEP (Steady-State Visual Evoked Potential) color-spectrum paradigm
- MongoDB for persistent model + calibration data storage

**Paradigm:** Three flickering colours — Red @ 8 Hz, Green @ 10 Hz, Violet @ 12 Hz. The user focuses on one; the EEG classifier detects which one from the SSVEP response in the frontal and temporal channels.

---

## 2. Hardware Setup

### EEG Pills

| Pill | GPIO | Electrode Placement |
|------|------|---------------------|
| Pill 1 | GPIO 34 (ADC1_CH6) | Frontal — Fp1 (IN+) and Fp2 (IN−) |
| Pill 2 | GPIO 35 (ADC1_CH7) | Temporal — T7 (IN+) and T8 (IN−) |
| Both | Shared REF | Mastoid (behind ear, M1 or M2) |

> **Why ADC1 only?** ADC2 on ESP32 conflicts with WiFi. GPIO 34 and 35 are input-only ADC1 pins — safe for WiFi + analog simultaneously.

### Wiring (jumper-wire only, no breadboard)

```
ESP32 3.3V ──┬──► Pill 1 VCC
             └──► Pill 2 VCC   (twist two jumpers on the 3.3V pin)

ESP32 GND  ──► Pill 1 GND
ESP32 GND  ──► Pill 2 GND   (most DevKits have 2× GND pins)

GPIO 34    ──► Pill 1 OUT
GPIO 35    ──► Pill 2 OUT
```

### Network

| Parameter | Value |
|-----------|-------|
| ESP32 Static IP | `192.168.1.33` |
| WebSocket port | `81` |
| HTTP dashboard port | `80` |
| WiFi SSID | HIVE 5TH FLOOR A |

---

## 3. Folder Structure

```
Neurosystem/
├── requirements.txt
├── NEUROSYSTEM_BUILD_LOG.md
│
├── BioAmp_EXG_EEG_ESP32/
│   ├── BioAmp_EXG_EEG_ESP32.ino   ← ESP32 firmware (dual-channel)
│   └── html_page.h                 ← Browser dashboard (embedded HTML/JS)
│
└── eeg_decision/
    ├── __init__.py
    ├── config.py                   ← ALL constants (IP, freqs, MongoDB, palette)
    ├── main_gui.py                 ← Entry point — 3-wing tkinter GUI
    │
    ├── streaming/
    │   └── ws_reader.py            ← WebSocket client daemon thread
    │
    ├── preprocessing/
    │   ├── filters.py              ← Bandpass / notch / detrend
    │   └── artifact_rejection.py  ← Epoch cleaning, EOG regression
    │
    ├── features/
    │   ├── ssvep.py                ← FFT SSVEP power extraction
    │   └── band_power.py          ← 10-feature vector builder
    │
    ├── classifier/
    │   ├── train.py                ← LDA training (epochs / CSV / feature arrays)
    │   └── predict.py             ← Predictor with exponential smoothing
    │
    └── data/
        ├── model.pkl               ← Latest trained model (auto-saved)
        └── mongo_store.py         ← MongoDB read/write helpers
```

---

## 4. ESP32 Firmware

**File:** `BioAmp_EXG_EEG_ESP32/BioAmp_EXG_EEG_ESP32.ino`

### Key design points

#### Static IP — critical ordering fix
```cpp
WiFi.persistent(false);
WiFi.mode(WIFI_STA);           // MUST come before WiFi.config()
WiFi.setAutoReconnect(true);
if (!WiFi.config(LOCAL_IP, GATEWAY, SUBNET, DNS_SRV))
    Serial.println("WARNING — static IP config rejected");
WiFi.begin(SSID, PASSWORD);
```
> `WiFi.config()` before `WiFi.mode(WIFI_STA)` silently falls back to DHCP. This was the original bug.

#### Dual-channel signal chain
```
GPIO34 ──► analogRead() ──► HPF(0.5Hz) ──► LPF(45Hz) ──► Notch(50Hz) ──► buf1[]
GPIO35 ──► analogRead() ──► HPF(0.5Hz) ──► LPF(45Hz) ──► Notch(50Hz) ──► buf2[]
```
Each channel has its own independent Biquad IIR filter state (`hpf1/hpf2`, `lpf1/lpf2`, `ntch1/ntch2`).

#### Analysis per channel
`analyseChannel(buf, bpOut, relOut, abRatio, taRatio, filtRms)` — runs 256-point radix-2 FFT, computes 5 band powers (Delta/Theta/Alpha/Beta/Gamma), relative powers, α/β and θ/α ratios, and filtered RMS. Called independently for both channels.

#### WebSocket JSON payload (sent every ~1 second)
```json
{
  "epoch": 42,
  "ch":  [/* 256 × int16 × 10 — Ch1 samples */],
  "ch2": [/* 256 × int16 × 10 — Ch2 samples */],
  "bp":  {"d":..,"t":..,"a":..,"b":..,"g":..},
  "bp2": {"d":..,"t":..,"a":..,"b":..,"g":..},
  "rel": {"d":..,"t":..,"a":..,"b":..,"g":..},
  "rel2":{"d":..,"t":..,"a":..,"b":..,"g":..},
  "quality":  "good|floating|flat",
  "quality2": "good|floating|flat",
  "filt_rms":  14.2,
  "filt_rms2": 11.7,
  "ab_ratio":  0.82,
  "ab_ratio2": 0.74,
  "ta_ratio":  1.1,
  "ta_ratio2": 0.95
}
```
> Sample values are stored as `int × 10` to avoid JSON float overhead. Python divides by 10 on receipt.

`DynamicJsonDocument` size is **6000 bytes** (two 256-sample arrays ≈ 3.5 KB; was 3000 and overflowed).

---

## 5. Browser Dashboard

**File:** `BioAmp_EXG_EEG_ESP32/html_page.h`

- Dual Chart.js waveform canvases (CH1 green, CH2 blue)
- Two band-power card grids, two quality banners
- WebSocket URL hardcoded to `ws://192.168.1.33:81`

```javascript
// Was: 'ws://' + location.hostname + ':81'  ← broke when accessed by IP
var WS_URL = 'ws://192.168.1.33:81';
```

---

## 6. Python GUI — 3-Wing Architecture

**Entry point:** `python eeg_decision/main_gui.py` (run from `Neurosystem/` folder)

### Wing 1 — Analysis & Testing

- Live dual-channel waveform (matplotlib, 4-second rolling window)
- Band power bars (Delta → Gamma) for Ch1
- Ch1 + Ch2 metrics: α/β ratio, θ/α ratio, Filt RMS, SSVEP powers at 8/10/12 Hz
- Ch1 / Ch2 quality badges (good ● / floating ◐ / flat ○ / disconnected ○)
- **Download CSV** — saves all live epochs to timestamped file
- **Train from CSV** — loads existing single-channel CSV, trains 7-feature LDA
- **Real-time labeling** — radio buttons for Red/Green/Violet/Neutral, record epochs live, train immediately

### Wing 2 — Subject Testing

- SSVEP colour-spectrum stimulus canvas — three panels flickering at 8/10/12 Hz
  - Time-based `after(14, ...)` loop ≈ 70 fps; no frame-counter drift
- Live FFT spectrum plot with coloured highlights at SSVEP bins
- Prediction label (large, colour-coded) + confidence percentage
- Confidence bars for all 4 classes

### Wing 3 — Prepare Model

- Subject ID + Notes form
- **Calibration state machine:**
  1. Press Start → 3-second countdown
  2. Big colour box shows which class to focus on
  3. Collects `CALIB_EPOCHS_PER_CLASS` (30) epochs per class
  4. Auto-advances through Red → Green → Violet → Neutral
  5. Progress bars per class
- **Train Model** button → LDA trained on calibration data
  - Merges with all prior sessions for this subject (incremental learning)
  - Auto-saves to MongoDB after training
- **MongoDB status badge** — shows Connected/Offline at startup
- **Load Saved Model from MongoDB** panel — listbox of all subjects with accuracy + epoch count; "Load Selected Model" restores any previous session instantly

### Header bar

- ESP32 connection dot (green = connected, red = reconnecting)
- Current epoch number
- Dominant band for Ch1 and Ch2
- Model ready / no model status

### Shared state — `LiveState`

Thread-safe class. `WSReader` daemon thread pushes JSON → queue. GUI poll loop (`after(100, _poll)`) drains queue → `LiveState.ingest()` → rolling waveform buffers + band power dicts updated atomically under a `threading.Lock`.

---

## 7. All Python Modules

### `eeg_decision/config.py`

All constants in one place. Import with `from eeg_decision.config import *`.

```python
ESP32_IP    = "192.168.1.33"
WS_URL      = "ws://192.168.1.33:81"
SSVEP_FREQS = [8.0, 10.0, 12.0]       # Hz
SSVEP_NAMES = ["Red", "Green", "Violet"]
CLASS_NAMES = ["Red", "Green", "Violet", "Neutral"]
CALIB_EPOCHS_PER_CLASS = 30
MONGO_URI   = "mongodb://localhost:27017"
MONGO_DB    = "neurosystem"
```

---

### `eeg_decision/streaming/ws_reader.py`

`WSReader` — daemon thread, auto-reconnects every 3 s, FIFO queue (drop-oldest on overflow).

**Critical fix applied:**
```python
# ping_interval=0 disables client-side pings.
# ESP32 ArduinoWebSockets does not respond to client pings —
# websocket-client would drop the connection after ping_timeout, causing
# a permanent reconnect loop. Browser works because it doesn't send pings.
self._ws.run_forever(ping_interval=0)
```

---

### `eeg_decision/preprocessing/filters.py`

```python
bandpass(sig, lo, hi, fs, order)   # Butterworth IIR
notch(sig, freq, fs, Q)            # IIR notch
detrend(sig)                       # linear detrend
preprocess(sig, fs)                # full pipeline: detrend → bandpass → notch
```
Accepts `[samples]` or `[channels, samples]` numpy arrays.

---

### `eeg_decision/preprocessing/artifact_rejection.py`

```python
is_clean(epoch, threshold=150.0)          # True if max amplitude < threshold µV
eog_regression(epoch, frontal_ch=0)       # regress out eye-movement artifact
clean_epoch(epoch, threshold, apply_eog)  # combined
```

---

### `eeg_decision/features/ssvep.py`

```python
ssvep_powers(samples, fs, freqs)    # → {freq: power} — Hann-windowed FFT
classify_ssvep(powers, freqs)       # → (class_index, confidence)
full_spectrum(samples, fs, fmin, fmax)  # → (freqs, power) for GUI plot
```

---

### `eeg_decision/features/band_power.py`

```python
feature_vector(epoch_json)           # → 10-element numpy array (live WebSocket epoch)
feature_vector_from_csv_row(row)     # → 7-element array (existing CSV format)
```

**10 features (live):** 5 relative band powers + SSVEP power at 8/10/12 Hz + α/β ratio + θ/α ratio

**7 features (CSV):** 5 relative band powers + α/β ratio + θ/α ratio

---

### `eeg_decision/classifier/train.py`

```python
train_from_epochs(epochs, labels)
    # epochs: list of ESP32 JSON dicts → extracts features internally
    # → result dict: {ok, model, accuracy, msg, feature_dim}

train_from_feature_arrays(X, y)
    # X: np.ndarray (n_samples, n_features) — pre-built feature matrix
    # y: np.ndarray (n_samples,) — integer class labels
    # Used by incremental training (prior DB features + new session)
    # → same result dict format

train_from_csv(csv_path, label_col)
    # loads existing 7-feature CSV
    # → result dict with label_map

load_model()
    # loads model.pkl → dict with model, feature_dim, label_map
```

All training uses `Pipeline([StandardScaler, LinearDiscriminantAnalysis])` with 3-fold cross-validation.

---

### `eeg_decision/classifier/predict.py`

`Predictor` class:

```python
predict(epoch_json)                     # → (class_index, confidence, smoothed_probs)
set_model(model, feature_dim, label_map)
class_name(idx)                         # → "Red" / "Green" / "Violet" / "Neutral"
ready                                   # True once a model is loaded
```

Exponential smoothing: `alpha = 0.35` (lower = smoother, higher = more reactive).
Confidence threshold: `0.60` — returns `-1` if no class clears it.

---

### `eeg_decision/data/mongo_store.py`

MongoDB collections (all in database `neurosystem`):

| Collection | Contents |
|------------|----------|
| `subjects` | One doc per subject — ID, notes, last_trained (upserted) |
| `models` | One doc per training session — pickled sklearn model, accuracy, epoch counts |
| `epochs` | One doc per calibration epoch — pre-computed 10-element feature vector + class label |

```python
ping()                                          # → (ok, message)
save_model(subject_id, notes, train_result, calib_epochs)
                                                # → (ok, message)
load_model_for_subject(subject_id)              # → (payload_dict, message)
load_epoch_features(subject_id)                 # → (X, y, message) — for incremental training
list_subjects()                                 # → list of dicts for GUI listbox
```

---

## 8. MongoDB Integration

### What gets saved on each training session

1. **`subjects`** — upsert: sets `last_trained`, preserves `created_at`
2. **`models`** — insert: pickled sklearn Pipeline, accuracy, feature_dim, epoch_counts, hardware tag
3. **`epochs`** — insert_many: one document per calibration epoch with pre-computed feature vector

### Schema — `models` document

```json
{
  "subject_id":   "subject_01",
  "notes":        "eyes open, good signal",
  "trained_at":   "2026-05-07T14:22:00Z",
  "accuracy":     0.87,
  "feature_dim":  10,
  "class_names":  ["Red", "Green", "Violet", "Neutral"],
  "epoch_counts": {"Red": 30, "Green": 30, "Violet": 30, "Neutral": 30},
  "total_epochs": 120,
  "hardware":     "BioAmp EXG Pill x2 + ESP32 @ 192.168.1.33",
  "model_pickle": "<Binary>"
}
```

### Schema — `epochs` document

```json
{
  "subject_id":  "subject_01",
  "model_id":    "<ObjectId of parent model>",
  "class_label": 0,
  "class_name":  "Red",
  "features":    [0.12, 0.08, 0.41, 0.27, 0.12, 44.1, 12.3, 8.7, 0.82, 1.1],
  "recorded_at": "2026-05-07T14:22:00Z"
}
```

---

## 9. Incremental Model Training

When you train for a subject who already has data in MongoDB, the system **does not start from scratch**. It layers new data on top of all previous sessions.

### Flow

```
New calibration (Wing 3)
        │
        ▼
feature_vector(ep) for each new epoch → X_new, y_new
        │
        ▼
mongo_store.load_epoch_features(subject_id)
        │ returns X_prior, y_prior (ALL epochs from ALL prior sessions)
        ▼
X_combined = vstack([X_prior, X_new])
y_combined = concat([y_prior, y_new])
        │
        ▼
train_from_feature_arrays(X_combined, y_combined)
        │  LDA + StandardScaler retrained on full history
        ▼
mongo_store.save_model(...)   ← appends only the NEW session's epochs
                                 (pool keeps growing each session)
```

Status line after training shows:
```
Trained on 210 epochs | CV accuracy: 91% | ... (+90 prior epochs — 90 total)
```

---

## 10. Bugs Fixed

### 1. ESP32 static IP silently ignored
**Symptom:** ESP32 always got a DHCP address, never 192.168.1.33.
**Cause:** `WiFi.config()` was called before `WiFi.mode(WIFI_STA)`.
**Fix:** Move `WiFi.mode(WIFI_STA)` to be the very first WiFi call.

---

### 2. Single-channel analysis despite dual ADC
**Symptom:** `.ino` sampled both GPIO pins but all band power, quality, serial output, and WebSocket JSON only used `buf1[]`.
**Fix:** Extracted `analyseChannel()` and `channelQuality()` helper functions and called them for both `buf1` and `buf2`. Added all Ch2 fields to JSON output.

---

### 3. JSON document overflow
**Symptom:** WebSocket messages silently truncated / malformed.
**Cause:** `DynamicJsonDocument(3000)` — two 256-sample int16 arrays require ~3.5 KB.
**Fix:** Increased to `DynamicJsonDocument(6000)`.

---

### 4. Browser WebSocket URL broke when accessed by IP
**Symptom:** Dashboard connected fine from browser's address bar but broke if opened via IP.
**Cause:** `var WS_URL = 'ws://' + location.hostname + ':81'` — `location.hostname` returns whatever was in the browser URL bar.
**Fix:** Hardcode `var WS_URL = 'ws://192.168.1.33:81'`.

---

### 5. Python GUI WebSocket permanent reconnect loop — live graph stuck at zero
**Symptom:** Browser dashboard showed live flowing graphs; Python GUI showed flat lines.
**Cause:** `run_forever(ping_interval=20, ping_timeout=10)` — `websocket-client` sends WebSocket ping frames every 20 s. ESP32's ArduinoWebSockets library does not respond to client-initiated pings. After 10 s with no pong, Python declares the connection dead, waits 3 s, reconnects, and the cycle repeats indefinitely. Connection never stays alive long enough to receive data.
**Fix:** `run_forever(ping_interval=0)` — disables Python-side pings entirely.

---

### 6. `mongo_store.py` ImportError on startup
**Symptom:** GUI crashed on import because `mongo_store.py` imported `MONGO_URI` and `MONGO_DB` from `config.py`, which didn't have them yet.
**Fix:** Added `MONGO_URI = "mongodb://localhost:27017"` and `MONGO_DB = "neurosystem"` to `config.py`.

---

## 11. Installation

### Python requirements

```
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
numpy>=1.24.0
scipy>=1.10.0
websocket-client>=1.6.0
scikit-learn>=1.3.0
joblib>=1.1.1
threadpoolctl>=2.0.0
matplotlib>=3.7.0
contourpy>=1.0.1
cycler>=0.10.0
fonttools>=4.22.0
kiwisolver>=1.0.1
packaging>=20.0
pillow>=9.0.0
pyparsing>=2.3.1
python-dateutil>=2.7.0
six>=1.5.0
pymongo>=4.5.0
dnspython>=2.1.0
```

> `tkinter` is bundled with Python on Windows — no pip needed.
> Do **not** install `websockets` (async library) or standalone `bson` — both will break the code.

### MongoDB (optional — only for Prepare Model wing)

1. Download MongoDB Community from [mongodb.com/try/download/community](https://www.mongodb.com/try/download/community)
2. Install and start the service — it listens on `localhost:27017` by default
3. No configuration needed — the `neurosystem` database and all collections are created automatically on first save

### Arduino / ESP32 firmware

1. Install **Arduino IDE** + **ESP32 board package** (Espressif, via Boards Manager)
2. Install libraries via Library Manager:
   - `ArduinoWebsockets` by Gil Maimon
   - `ArduinoJson` by Benoit Blanchon
3. Open `BioAmp_EXG_EEG_ESP32.ino`, select your ESP32 board and COM port, upload

---

## 12. How to Run

```bash
# from the Neurosystem/ folder
python eeg_decision/main_gui.py
```

### Startup sequence

1. GUI opens — header shows red dot (Reconnecting…)
2. If ESP32 is powered and on WiFi, dot turns green within 3 seconds
3. Wing 1 waveform starts scrolling with live EEG data
4. **To test SSVEP:** Switch to Wing 2, press Start Stimulus, focus on a colour
5. **To train a new subject:**
   - Switch to Wing 3
   - Enter Subject ID + optional notes
   - Press Start Calibration — follow the colour prompts (3 s countdown per class)
   - Press Train Model — model is trained and saved to MongoDB automatically
6. **To reload a previous subject:** Wing 3 → Load Saved Model panel → select subject → Load Selected Model

---

*Generated: 2026-05-07*
