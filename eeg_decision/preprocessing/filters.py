"""
Signal filtering — bandpass, notch, detrend.
All functions accept numpy arrays shaped [samples] or [channels, samples].
"""

import numpy as np

try:
    from scipy.signal import butter, filtfilt, iirnotch
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


def bandpass(sig: np.ndarray, lo: float, hi: float, fs: float, order: int = 4) -> np.ndarray:
    if not SCIPY_OK:
        return sig
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    axis = 1 if sig.ndim == 2 else 0
    return filtfilt(b, a, sig, axis=axis).astype(np.float32)


def notch(sig: np.ndarray, freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    if not SCIPY_OK:
        return sig
    b, a = iirnotch(freq / (fs / 2.0), Q)
    axis = 1 if sig.ndim == 2 else 0
    return filtfilt(b, a, sig, axis=axis).astype(np.float32)


def detrend(sig: np.ndarray) -> np.ndarray:
    if SCIPY_OK:
        from scipy.signal import detrend as sp_detrend
        axis = 1 if sig.ndim == 2 else 0
        return sp_detrend(sig, axis=axis).astype(np.float32)
    return (sig - np.mean(sig, axis=-1, keepdims=True)).astype(np.float32)


def preprocess(sig: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """Full pipeline: notch → bandpass → detrend."""
    from eeg_decision.config import BANDPASS_LO, BANDPASS_HI, NOTCH_FREQ, NOTCH_Q, FILTER_ORDER
    sig = notch(sig, NOTCH_FREQ, fs, NOTCH_Q)
    sig = bandpass(sig, BANDPASS_LO, BANDPASS_HI, fs, FILTER_ORDER)
    sig = detrend(sig)
    return sig
