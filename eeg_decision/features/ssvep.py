"""
SSVEP feature extraction — FFT power at 8, 10, 12 Hz.
"""

import numpy as np
from eeg_decision.config import SSVEP_FREQS, SAMPLING_RATE


def fft_power(samples: np.ndarray, fs: float = SAMPLING_RATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (freqs, power_spectrum) for a 1-D sample array.
    Uses a Hann window for spectral leakage reduction.
    """
    n = len(samples)
    win = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(samples * win)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spectrum


def power_at(freqs: np.ndarray, spectrum: np.ndarray,
             target: float, bw: float = 0.5) -> float:
    """Mean power in [target-bw, target+bw] Hz window."""
    mask = (freqs >= target - bw) & (freqs <= target + bw)
    return float(np.mean(spectrum[mask])) if mask.any() else 0.0


def ssvep_powers(samples: np.ndarray, fs: float = SAMPLING_RATE,
                 freqs: list = SSVEP_FREQS) -> dict:
    """
    Return dict {freq: power} for each SSVEP frequency.
    samples: 1-D array (one channel, one epoch).
    """
    f, psd = fft_power(samples, fs)
    return {freq: power_at(f, psd, freq) for freq in freqs}


def classify_ssvep(powers: dict, freqs: list = SSVEP_FREQS) -> tuple[int, float]:
    """
    Returns (class_index, confidence).
    class_index 0/1/2 → Red/Green/Violet; -1 if no signal.
    confidence = fraction of total power at winning freq.
    """
    vals = np.array([powers[f] for f in freqs])
    total = vals.sum()
    if total < 1e-10:
        return -1, 0.0
    probs = vals / total
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])


def full_spectrum(samples: np.ndarray, fs: float = SAMPLING_RATE,
                  fmin: float = 1.0, fmax: float = 50.0):
    """Return (freqs, power) clipped to [fmin, fmax] for GUI display."""
    f, psd = fft_power(samples, fs)
    mask = (f >= fmin) & (f <= fmax)
    return f[mask], psd[mask]
