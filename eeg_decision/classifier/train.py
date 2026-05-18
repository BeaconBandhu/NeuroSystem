"""
Model training — sklearn LDA (fast, works on small datasets).
Optional EEGNet (tensorflow) if available.
"""

import os
import csv
import pickle
import numpy as np
from pathlib import Path

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from eeg_decision.config import MODEL_PATH, CLASS_NAMES
from eeg_decision.features.band_power import feature_vector, feature_vector_from_csv_row


def train_from_epochs(epochs: list, labels: list) -> dict:
    """
    Train from live accumulated epochs.
    epochs: list of ESP32 JSON dicts
    labels: list of int class indices matching each epoch
    Returns result dict with model, accuracy, message.
    """
    if not SKLEARN_OK:
        return {"ok": False, "msg": "sklearn not installed. Run: pip install scikit-learn"}

    if len(epochs) < 6:
        return {"ok": False, "msg": f"Need at least 6 epochs, got {len(epochs)}"}

    X = np.array([feature_vector(e) for e in epochs])
    y = np.array(labels)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lda",    LinearDiscriminantAnalysis()),
    ])

    # Cross-validation accuracy
    acc = 0.0
    if len(set(y)) >= 2 and len(y) >= 6:
        scores = cross_val_score(model, X, y, cv=min(3, len(y) // 2))
        acc = float(scores.mean())

    model.fit(X, y)
    _save(model, X.shape[1])

    msg = f"Trained on {len(epochs)} epochs  |  CV accuracy: {acc*100:.0f}%  |  Classes: {[CLASS_NAMES[i] for i in sorted(set(y))]}"
    return {"ok": True, "model": model, "accuracy": acc, "msg": msg, "feature_dim": X.shape[1]}


def train_from_feature_arrays(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Train directly from a pre-built feature matrix and label vector.
    Used for incremental training: combine prior DB epochs with new ones,
    then call this instead of train_from_epochs.
    """
    if not SKLEARN_OK:
        return {"ok": False, "msg": "sklearn not installed. Run: pip install scikit-learn"}
    if len(X) < 6:
        return {"ok": False, "msg": f"Need at least 6 samples, got {len(X)}"}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lda",    LinearDiscriminantAnalysis()),
    ])

    acc = 0.0
    if len(set(y)) >= 2 and len(y) >= 6:
        scores = cross_val_score(model, X, y, cv=min(3, len(y) // 2))
        acc = float(scores.mean())

    model.fit(X, y)
    _save(model, X.shape[1])

    msg = (f"Trained on {len(X)} epochs  |  CV accuracy: {acc*100:.0f}%  |  "
           f"Classes: {[CLASS_NAMES[i] for i in sorted(set(y.tolist()))]}")
    return {"ok": True, "model": model, "accuracy": acc, "msg": msg,
            "feature_dim": X.shape[1]}


def train_from_csv(csv_path: str, label_col: str = "label") -> dict:
    """
    Train from an existing CSV file (single BioAmp, 7-feature vector).
    CSV must have columns: label, D_rel, T_rel, A_rel, B_rel, G_rel, A_B_ratio, T_A_ratio
    """
    if not SKLEARN_OK:
        return {"ok": False, "msg": "sklearn not installed. Run: pip install scikit-learn"}

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < 6:
        return {"ok": False, "msg": f"CSV has only {len(rows)} rows — need at least 6"}

    label_set = sorted(set(r[label_col] for r in rows if label_col in r))
    label_map  = {name: i for i, name in enumerate(label_set)}

    X, y = [], []
    for row in rows:
        if label_col not in row:
            continue
        feat = feature_vector_from_csv_row(row)
        X.append(feat); y.append(label_map[row[label_col]])

    X = np.array(X); y = np.array(y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lda",    LinearDiscriminantAnalysis()),
    ])

    acc = 0.0
    if len(set(y)) >= 2 and len(y) >= 6:
        scores = cross_val_score(model, X, y, cv=min(3, len(y) // 2))
        acc = float(scores.mean())

    model.fit(X, y)
    _save(model, X.shape[1], label_map=label_map)

    msg = (f"Trained from CSV  |  {len(X)} rows  |  "
           f"Labels: {label_set}  |  CV accuracy: {acc*100:.0f}%")
    return {"ok": True, "model": model, "accuracy": acc, "msg": msg,
            "label_map": label_map, "feature_dim": X.shape[1]}


def _save(model, feature_dim: int, label_map: dict = None):
    path = Path(MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_dim": feature_dim}
    if label_map:
        payload["label_map"] = label_map
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_model() -> dict | None:
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
