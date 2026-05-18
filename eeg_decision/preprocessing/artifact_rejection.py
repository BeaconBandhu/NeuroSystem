"""
Artifact rejection — threshold and IQR based.
"""

import numpy as np


def is_clean(epoch: np.ndarray, threshold: float = 150.0) -> bool:
    """Return True if epoch has no sample exceeding threshold µV."""
    return float(np.max(np.abs(epoch))) < threshold


def eog_regression(epoch: np.ndarray, frontal_ch: int = 0) -> np.ndarray:
    """
    Subtract frontal channel's contribution from all other channels.
    epoch: [channels x samples]
    """
    if epoch.ndim == 1:
        return epoch
    ref = epoch[frontal_ch]
    denom = float(np.dot(ref, ref))
    if denom < 1e-10:
        return epoch
    out = epoch.copy()
    for ch in range(epoch.shape[0]):
        if ch == frontal_ch:
            continue
        beta = np.dot(ref, epoch[ch]) / denom
        out[ch] = epoch[ch] - beta * ref
    return out


def clean_epoch(epoch: np.ndarray, threshold: float = 150.0,
                apply_eog: bool = True) -> np.ndarray | None:
    """Return cleaned epoch or None if it should be discarded."""
    if not is_clean(epoch, threshold):
        return None
    if apply_eog and epoch.ndim == 2 and epoch.shape[0] > 1:
        epoch = eog_regression(epoch)
    return epoch


def iqr_mask(sig: np.ndarray, mult: float = 3.0) -> np.ndarray:
    q1, q3 = np.percentile(sig, [25, 75])
    iqr = q3 - q1
    return (sig < q1 - mult * iqr) | (sig > q3 + mult * iqr)
