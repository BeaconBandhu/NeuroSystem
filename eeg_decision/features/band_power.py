"""
Band power feature extraction from raw samples or pre-computed rel/abs dicts.
"""

import numpy as np
from eeg_decision.config import BANDS, SAMPLING_RATE, SSVEP_FREQS
from eeg_decision.features.ssvep import ssvep_powers


def band_powers_from_fft(samples: np.ndarray, fs: float = SAMPLING_RATE) -> dict:
    """Compute absolute band powers from a raw 1-D sample array."""
    n = len(samples)
    win = np.hanning(n)
    psd = np.abs(np.fft.rfft(samples * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    result = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[name] = float(psd[mask].mean()) if mask.any() else 0.0
    return result


def relative_powers(abs_powers: dict) -> dict:
    total = sum(abs_powers.values()) + 1e-10
    return {k: v / total for k, v in abs_powers.items()}


def feature_vector(epoch_json: dict) -> np.ndarray:
    """
    Build a 10-element feature vector from one ESP32 WebSocket epoch dict.

    Features:
      [0-4]  D_rel, T_rel, A_rel, B_rel, G_rel   (from 'rel' field)
      [5-7]  SSVEP power at 8, 10, 12 Hz          (from raw 'ch' samples)
      [8]    Alpha/Beta ratio
      [9]    Theta/Alpha ratio
    """
    rel = epoch_json.get("rel", {})
    d = float(rel.get("d", 0))
    t = float(rel.get("t", 0))
    a = float(rel.get("a", 0))
    b = float(rel.get("b", 0))
    g = float(rel.get("g", 0))

    samples = np.array(epoch_json.get("ch", []), dtype=np.float32) / 10.0
    sp = ssvep_powers(samples) if len(samples) > 0 else {f: 0.0 for f in SSVEP_FREQS}
    sp_vals = [sp[f] for f in SSVEP_FREQS]

    ab = a / (b + 1e-10)
    ta = t / (a + 1e-10)

    return np.array([d, t, a, b, g, *sp_vals, ab, ta], dtype=np.float32)


def feature_vector_from_csv_row(row: dict) -> np.ndarray:
    """
    Build a 7-element feature vector from a CSV row (no raw samples available).

    Features: D_rel, T_rel, A_rel, B_rel, G_rel, AB_ratio, TA_ratio
    """
    return np.array([
        float(row.get("D_rel", 0)),
        float(row.get("T_rel", 0)),
        float(row.get("A_rel", 0)),
        float(row.get("B_rel", 0)),
        float(row.get("G_rel", 0)),
        float(row.get("A_B_ratio", 0)),
        float(row.get("T_A_ratio", 0)),
    ], dtype=np.float32)
