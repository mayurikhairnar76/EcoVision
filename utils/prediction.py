"""Model loading and prediction. Nothing is hard-coded: probabilities come from the trained CNN."""
import json
import numpy as np
import tensorflow as tf
from .config import MODEL_PATH, CLASSES_PATH, METRICS_PATH


def load_artifacts():
    """Return (model, class_names, metrics). Raises RuntimeError if the model is missing/broken."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = json.loads(CLASSES_PATH.read_text())
    except Exception as e:
        raise RuntimeError(f"Could not load the trained model ({e}). Run notebooks/model_training.ipynb first.")
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return model, class_names, metrics


def predict(model, class_names, batch):
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], idx, float(probs[idx]), probs
