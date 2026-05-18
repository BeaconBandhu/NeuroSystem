"""
Central configuration — all constants in one place.
"""

# ── Hardware ──────────────────────────────────────────────────────────────────
ESP32_IP        = "192.168.1.33"
WS_URL          = f"ws://{ESP32_IP}:81"
HTTP_URL        = f"http://{ESP32_IP}"
SAMPLING_RATE   = 256           # Hz (ESP32 firmware value)
N_SAMPLES       = 256           # samples per epoch = 1 second
N_CHANNELS      = 2             # Pill1 (ch) + Pill2 (ch2)

CH1_LABEL       = "Frontal (Fp1–Fp2)"
CH2_LABEL       = "Temporal (T7–T8)"

# ── SSVEP color spectrum ──────────────────────────────────────────────────────
SSVEP_FREQS     = [8.0, 10.0, 12.0]        # Hz — Red, Green, Violet
SSVEP_COLORS    = ["#FF2200", "#00CC00", "#5500FF"]
SSVEP_DARK      = ["#330800", "#003300", "#110033"]
SSVEP_NAMES     = ["Red",    "Green",   "Violet"]
SSVEP_CHANNEL   = 0                         # index into ch array (Ch1 = frontal)

# ── Signal processing ─────────────────────────────────────────────────────────
BANDPASS_LO     = 0.5           # Hz
BANDPASS_HI     = 45.0          # Hz
NOTCH_FREQ      = 50.0          # Hz (India power line)
NOTCH_Q         = 30
ARTIFACT_THRESH = 150.0         # µV — epoch rejected if exceeded
FILTER_ORDER    = 4

# ── Band definitions ──────────────────────────────────────────────────────────
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BAND_COLORS = {
    "delta": "#ff6b6b",
    "theta": "#ffd93d",
    "alpha": "#6bcb77",
    "beta":  "#4d96ff",
    "gamma": "#c77dff",
}

# ── Classifier ────────────────────────────────────────────────────────────────
CALIB_EPOCHS_PER_CLASS  = 30    # epochs per SSVEP color per subject
MIN_EPOCHS_TO_TRAIN     = 10    # minimum before training fires
CONFIDENCE_THRESHOLD    = 0.60  # minimum confidence to show prediction
SMOOTHING_ALPHA         = 0.35  # exponential smoothing (higher = more reactive)
N_CLASSES               = len(SSVEP_FREQS) + 1  # Red, Green, Violet + Neutral
CLASS_NAMES             = SSVEP_NAMES + ["Neutral"]
MODEL_PATH              = "eeg_decision/data/model.pkl"

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB  = "neurosystem"

# ── Display ───────────────────────────────────────────────────────────────────
MONITOR_HZ      = 60            # monitor refresh rate
WAVEFORM_SECS   = 4             # seconds of waveform to display
WAVEFORM_PTS    = SAMPLING_RATE * WAVEFORM_SECS  # display buffer size

# ── Palette (dark GitHub-style) ───────────────────────────────────────────────
BG  = "#0d1117"; BG2 = "#161b22"; BG3 = "#21262d"; BG4 = "#30363d"
ACC = "#58a6ff"; ACC2 = "#bc8cff"
FG  = "#e6edf3"; FG2 = "#8b949e"; FG3 = "#6e7681"
OK  = "#3fb950"; WARN = "#d29922"; ERR = "#f85149"
