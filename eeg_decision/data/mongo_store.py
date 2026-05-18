"""
MongoDB store for trained models and calibration data.

Collections (all in database 'neurosystem'):
  models     — one document per training session (includes pickled model binary)
  epochs     — feature vectors per calibration epoch (lightweight, no raw samples)
  subjects   — subject metadata, upserted on subject_id

Install: pip install pymongo
"""

import pickle
import threading
from datetime import datetime, timezone

import numpy as np

try:
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    from bson.binary import Binary
    MONGO_OK = True
except ImportError:
    MONGO_OK = False

from eeg_decision.config import MONGO_URI, MONGO_DB
from eeg_decision.features.band_power import feature_vector


# ── singleton client ──────────────────────────────────────────────────────────

_client = None
_lock   = threading.Lock()


def _get_db():
    global _client
    if not MONGO_OK:
        raise RuntimeError("pymongo not installed. Run: pip install pymongo")
    with _lock:
        if _client is None:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        return _client[MONGO_DB]


def ping() -> tuple[bool, str]:
    """Return (ok, message). Safe to call even if pymongo is missing."""
    if not MONGO_OK:
        return False, "pymongo not installed — run: pip install pymongo"
    try:
        db = _get_db()
        db.command("ping")
        return True, f"Connected  ·  {MONGO_URI}"
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return False, f"Cannot reach MongoDB: {e}"
    except Exception as e:
        return False, str(e)


# ── save ─────────────────────────────────────────────────────────────────────

def save_model(subject_id: str, notes: str, train_result: dict,
               calib_epochs: dict) -> tuple[bool, str]:
    """
    Persist a trained model session to MongoDB.

    Parameters
    ----------
    subject_id   : str   e.g. "subject_01"
    notes        : str   free-text session notes
    train_result : dict  result dict from classifier/train.py
    calib_epochs : dict  {class_idx: [epoch_json, ...]} from WingPrepare

    Returns (ok, message)
    """
    if not train_result.get("ok"):
        return False, "train_result is not ok — nothing to save"

    try:
        db = _get_db()
        now = datetime.now(timezone.utc)

        # ── upsert subject record ────────────────────────────────────────────
        db.subjects.update_one(
            {"subject_id": subject_id},
            {"$set":  {"subject_id": subject_id, "notes": notes,
                       "last_trained": now},
             "$setOnInsert": {"created_at": now}},
            upsert=True
        )

        # ── build epoch_counts summary ───────────────────────────────────────
        from eeg_decision.config import CLASS_NAMES
        epoch_counts = {CLASS_NAMES[i]: len(eps)
                        for i, eps in calib_epochs.items()}
        total_epochs = sum(epoch_counts.values())

        # ── pickle the sklearn model ─────────────────────────────────────────
        model_bytes = pickle.dumps(train_result["model"])

        # ── insert model document ────────────────────────────────────────────
        doc = {
            "subject_id":   subject_id,
            "notes":        notes,
            "trained_at":   now,
            "accuracy":     round(train_result.get("accuracy", 0.0), 4),
            "feature_dim":  train_result.get("feature_dim", 10),
            "class_names":  CLASS_NAMES,
            "epoch_counts": epoch_counts,
            "total_epochs": total_epochs,
            "hardware":     "BioAmp EXG Pill x2 + ESP32 @ 192.168.1.33",
            "model_pickle": Binary(model_bytes),
        }
        result = db.models.insert_one(doc)
        model_oid = str(result.inserted_id)

        # ── save feature vectors for each calibration epoch ──────────────────
        epoch_docs = []
        for cls_idx, eps in calib_epochs.items():
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            for ep in eps:
                feat = feature_vector(ep).tolist()
                epoch_docs.append({
                    "subject_id": subject_id,
                    "model_id":   model_oid,
                    "class_label": cls_idx,
                    "class_name":  cls_name,
                    "features":    feat,
                    "recorded_at": now,
                })
        if epoch_docs:
            db.epochs.insert_many(epoch_docs)

        msg = (f"Saved to MongoDB  ·  subject='{subject_id}'  "
               f"·  {total_epochs} epochs  ·  acc={train_result.get('accuracy',0)*100:.0f}%")
        return True, msg

    except Exception as e:
        return False, f"MongoDB save error: {e}"


# ── load ─────────────────────────────────────────────────────────────────────

def load_model_for_subject(subject_id: str) -> tuple[dict | None, str]:
    """
    Load the most recently trained model for subject_id.
    Returns (payload_dict, message) where payload_dict has keys:
      model, feature_dim, accuracy, trained_at, epoch_counts
    or (None, error_message) on failure.
    """
    try:
        db = _get_db()
        doc = db.models.find_one(
            {"subject_id": subject_id},
            sort=[("trained_at", DESCENDING)]
        )
        if doc is None:
            return None, f"No model found for subject '{subject_id}'"

        model = pickle.loads(doc["model_pickle"])
        payload = {
            "model":        model,
            "feature_dim":  doc.get("feature_dim", 10),
            "accuracy":     doc.get("accuracy", 0.0),
            "trained_at":   doc.get("trained_at"),
            "epoch_counts": doc.get("epoch_counts", {}),
            "class_names":  doc.get("class_names", []),
        }
        msg = (f"Loaded '{subject_id}'  ·  "
               f"trained {doc['trained_at'].strftime('%Y-%m-%d %H:%M')}  "
               f"·  acc={doc.get('accuracy',0)*100:.0f}%")
        return payload, msg

    except Exception as e:
        return None, f"MongoDB load error: {e}"


# ── load prior epoch features (for incremental training) ─────────────────────

def load_epoch_features(subject_id: str) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    Load all stored feature vectors and labels for subject_id across every
    training session.  Returns (X, y, message) or (None, None, error).
    """
    try:
        db = _get_db()
        docs = list(db.epochs.find(
            {"subject_id": subject_id},
            {"features": 1, "class_label": 1, "_id": 0}
        ))
        if not docs:
            return None, None, f"No prior epochs for '{subject_id}'"
        X = np.array([d["features"] for d in docs], dtype=np.float32)
        y = np.array([d["class_label"] for d in docs], dtype=np.int32)
        return X, y, f"{len(docs)} prior epochs loaded for '{subject_id}'"
    except Exception as e:
        return None, None, f"MongoDB load error: {e}"


# ── list ─────────────────────────────────────────────────────────────────────

def list_subjects() -> list[dict]:
    """
    Return list of subjects with their latest model metadata.
    Each dict: {subject_id, last_trained, accuracy, total_epochs, notes}
    """
    try:
        db = _get_db()
        # aggregate: latest model per subject
        pipeline = [
            {"$sort": {"trained_at": -1}},
            {"$group": {
                "_id":          "$subject_id",
                "last_trained": {"$first": "$trained_at"},
                "accuracy":     {"$first": "$accuracy"},
                "total_epochs": {"$first": "$total_epochs"},
            }},
            {"$sort": {"last_trained": -1}},
        ]
        rows = list(db.models.aggregate(pipeline))
        # merge notes from subjects collection
        subj_map = {s["subject_id"]: s.get("notes", "")
                    for s in db.subjects.find({}, {"subject_id":1,"notes":1})}
        result = []
        for r in rows:
            sid = r["_id"]
            result.append({
                "subject_id":   sid,
                "last_trained": r["last_trained"].strftime("%Y-%m-%d %H:%M")
                                if r.get("last_trained") else "—",
                "accuracy":     f"{r.get('accuracy',0)*100:.0f}%",
                "total_epochs": r.get("total_epochs", 0),
                "notes":        subj_map.get(sid, ""),
            })
        return result
    except Exception:
        return []
