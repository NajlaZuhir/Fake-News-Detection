# model.py
"""
Model utilities for Fake News Detector.

- Loads TF-IDF + Logistic Regression pipeline
- Provides a clean prediction function
"""

from typing import Dict
import os
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress only the InconsistentVersionWarning (version mismatch for sklearn)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Base directory (same folder as this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to saved model
MODEL_PATH = os.path.join(BASE_DIR, "tfidf+lr_fake_news.joblib")

# Load TF-IDF + Logistic Regression pipeline once at import time
pipe = joblib.load(MODEL_PATH)

# Label mapping
LABELS = {0: "Real", 1: "Fake"}


def combine(title: str = "", text: str = "", use_sep: bool = True) -> str:
    """
    Combine title and text into a single string.
    Optionally insert a [SEP] token between them.
    """
    title = (title or "").strip()
    text = (text or "").strip()

    if title and text and use_sep:
        return f"{title} [SEP] {text}"

    return (title + " " + text).strip()


def predict_tfidf(
    title: str = "",
    text: str = "",
    use_sep: bool = True,
) -> Dict:
    """
    Run TF-IDF + Logistic Regression model and return:
    - pred_label: 0 or 1
    - label_text: "Real" or "Fake"
    - confidence: probability of Fake (class 1)
    - model: model name string
    """
    if not title and not text:
        raise ValueError("Provide at least a title or text")

    txt = combine(title, text, use_sep)

    probs = pipe.predict_proba([txt])[0]   # [p_real, p_fake]
    pred_id = int(probs.argmax())          # 0 or 1
    conf_fake = float(probs[1])            # probability of Fake

    return {
        "pred_label": pred_id,
        "label_text": LABELS[pred_id],
        "confidence": conf_fake,
        "model": "tfidf-logreg",
    }
