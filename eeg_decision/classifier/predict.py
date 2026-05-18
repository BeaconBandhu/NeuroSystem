"""
Real-time predictor with exponential smoothing.
"""

import numpy as np
from eeg_decision.config import (SMOOTHING_ALPHA, CONFIDENCE_THRESHOLD,
                                   N_CLASSES, CLASS_NAMES, SSVEP_FREQS)
from eeg_decision.features.band_power import feature_vector
from eeg_decision.classifier.train import load_model


class Predictor:
    def __init__(self):
        self._model       = None
        self._feat_dim    = None
        self._label_map   = None          # {name: idx} from CSV training
        self.smoothed     = np.ones(N_CLASSES) / N_CLASSES
        self.alpha        = SMOOTHING_ALPHA
        self.threshold    = CONFIDENCE_THRESHOLD
        self.last_class   = -1
        self.last_conf    = 0.0
        self._try_load()

    def _try_load(self):
        payload = load_model()
        if payload:
            self._model     = payload["model"]
            self._feat_dim  = payload.get("feature_dim", 10)
            self._label_map = payload.get("label_map")

    def reload(self):
        self._try_load()
        self.reset()

    def set_model(self, model, feature_dim: int, label_map: dict = None):
        self._model     = model
        self._feat_dim  = feature_dim
        self._label_map = label_map
        self.reset()

    def predict(self, epoch_json: dict) -> tuple[int, float, np.ndarray]:
        """
        Returns (class_index, confidence, smoothed_probs).
        class_index = -1 if no model or low confidence.
        """
        if self._model is None:
            return -1, 0.0, self.smoothed

        feat = feature_vector(epoch_json)

        # If model was trained on 7-feature CSV data, trim to 7
        if self._feat_dim == 7:
            feat = feat[:7]

        raw_probs = self._model.predict_proba(feat.reshape(1, -1))[0]

        # Pad or trim to N_CLASSES
        n = len(raw_probs)
        if n < N_CLASSES:
            raw_probs = np.pad(raw_probs, (0, N_CLASSES - n))
        elif n > N_CLASSES:
            raw_probs = raw_probs[:N_CLASSES]

        self.smoothed = (self.alpha * raw_probs +
                         (1 - self.alpha) * self.smoothed)
        self.smoothed /= self.smoothed.sum()

        idx  = int(np.argmax(self.smoothed))
        conf = float(self.smoothed[idx])

        if conf < self.threshold:
            self.last_class = -1
            self.last_conf  = conf
            return -1, conf, self.smoothed

        self.last_class = idx
        self.last_conf  = conf
        return idx, conf, self.smoothed

    def reset(self):
        self.smoothed   = np.ones(N_CLASSES) / N_CLASSES
        self.last_class = -1
        self.last_conf  = 0.0

    @property
    def ready(self) -> bool:
        return self._model is not None

    def class_name(self, idx: int) -> str:
        if self._label_map:
            inv = {v: k for k, v in self._label_map.items()}
            return inv.get(idx, f"Class {idx}")
        return CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "Unknown"
