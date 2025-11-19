# models/ml_emotion_baseline.py
# ML emotion model trained on the GoEmotions dataset (Hugging Face)
import os
from typing import Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset

EMOTION_MODEL_PATH = "models/emotion_lr.joblib"
EMOTION_VECT_PATH = "models/emotion_vectorizer.joblib"

# We expose exactly these 5 emotions to the rest of the app
EMOTION_LABELS = ["anxiety", "fear", "anger", "joy", "neutral"]


def _map_goemotions_to_buckets(label_ids: List[int],
                               id2label: List[str]) -> List[str]:
    """
    Map the original GoEmotions fine-grained labels into your 5 buckets.
    This is a heuristic mapping, but it is enough for your project.
    """
    fine_labels = [id2label[i] for i in label_ids]
    buckets = set()

    # Anxiety (mix of nervous / worried negative states)
    if any(l in fine_labels for l in [
        "nervousness", "fear", "embarrassment", "sadness",
        "disappointment", "remorse", "grief"
    ]):
        buckets.add("anxiety")

    # Fear (keep a pure 'fear' bucket too)
    if "fear" in fine_labels:
        buckets.add("fear")

    # Anger / irritation
    if any(l in fine_labels for l in [
        "anger", "annoyance", "disgust"
    ]):
        buckets.add("anger")

    # Joy / positive emotions
    if any(l in fine_labels for l in [
        "joy", "amusement", "excitement", "love",
        "gratitude", "relief", "pride", "admiration",
        "approval", "optimism"
    ]):
        buckets.add("joy")

    # Neutral as fallback or explicit neutral
    if "neutral" in fine_labels or not buckets:
        buckets.add("neutral")

    return list(buckets)


def build_and_train_emotion_model():
    """
    Train a TF-IDF + One-Vs-Rest Logistic Regression classifier on the
    GoEmotions dataset, aggregated into 5 buckets.
    """
    print("[EMO] Loading GoEmotions dataset from Hugging Face…")
    ds = load_dataset("go_emotions")  # train/validation/test splits
    train = ds["train"]

    texts = train["text"]
    label_ids_list = train["labels"]          # list of list[int]
    id2label = train.features["labels"].feature.names  # index -> original label

    # Map each example's fine labels to your 5 emotion buckets
    print("[EMO] Mapping fine-grained labels to 5 emotion buckets…")
    multi_buckets = [
        _map_goemotions_to_buckets(ids, id2label)
        for ids in label_ids_list
    ]

    # Multi-label binarization into 5 target columns
    mlb = MultiLabelBinarizer(classes=EMOTION_LABELS)
    Y = mlb.fit_transform(multi_buckets)   # shape (n_samples, 5)

    # Text → TF-IDF
    print("[EMO] Building TF-IDF features…")
    vect = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X = vect.fit_transform(texts)

    # One-vs-Rest Logistic Regression (multi-label)
    print("[EMO] Training OneVsRest Logistic Regression on GoEmotions…")
    base_clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X, Y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(vect, EMOTION_VECT_PATH)
    joblib.dump(clf, EMOTION_MODEL_PATH)

    print("[EMO] Emotion model trained and saved.")
    return vect, clf, EMOTION_LABELS, False


def load_or_build_emotion_model():
    """
    Load the trained model if it exists, otherwise train it on GoEmotions.
    Returns: vectorizer, classifier, EMOTION_LABELS, loaded_flag
    """
    if os.path.exists(EMOTION_MODEL_PATH) and os.path.exists(EMOTION_VECT_PATH):
        vect = joblib.load(EMOTION_VECT_PATH)
        clf = joblib.load(EMOTION_MODEL_PATH)
        return vect, clf, EMOTION_LABELS, True

    return build_and_train_emotion_model()


def predict_emotions(vectorizer, model, emotion_labels, text: str) -> Dict[str, float]:
    """
    Predict emotion 'probabilities' for a single text.

    The underlying model is multi-label (each emotion independent),
    but we renormalize the per-class probabilities so they sum to 1.
    """
    X = vectorizer.transform([text])
    # OneVsRestClassifier -> predict_proba gives shape (n_samples, n_classes)
    probs = np.asarray(model.predict_proba(X)[0], dtype=float)

    total = probs.sum()
    if total <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total

    return {
        label: float(p)
        for label, p in zip(emotion_labels, probs)
    }
