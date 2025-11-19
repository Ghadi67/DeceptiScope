# models/ml_baseline.py

from typing import List, Tuple, Optional
import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

LR_MODEL_PATH_DEFAULT = "artifacts/deception/lr_tfidf_lr.joblib"
LR_METRICS_PATH_DEFAULT = "artifacts/deception/lr_training_metrics.json"


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a small set of validation metrics."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }


def train_lr_model(
    texts: List[str],
    labels: List[int],
    metrics_path: Optional[str] = LR_METRICS_PATH_DEFAULT,
) -> Tuple[Pipeline, dict]:
    """
    Train a simple TF-IDF + Logistic Regression model for deception detection.

    This version is SAFE for very small datasets (like your 4-row CSV):
    - min_df = 1  → keep all terms that appear at least once
    - max_df = 1.0 → don't drop frequent terms

    Everything is fit **from scratch** on your training corpus; no pre-trained
    embeddings or external classifiers are pulled in.
    """
    labels = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            max_features=8000,
            lowercase=True
        )),
        ("clf", LogisticRegression(
            max_iter=1500,
            class_weight="balanced"
        )),
    ])

    search = GridSearchCV(
        pipe,
        {"clf__C": [0.5, 1.0, 2.0], "clf__penalty": ["l2"]},
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_pipe: Pipeline = search.best_estimator_

    val_pred = best_pipe.predict(X_val)
    metrics = _evaluate_predictions(y_val, val_pred)
    metrics["best_params"] = search.best_params_

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        joblib.dump(metrics, metrics_path)

    return best_pipe, metrics


def load_or_build_lr(
    model_path: str = LR_MODEL_PATH_DEFAULT,
    metrics_path: str = LR_METRICS_PATH_DEFAULT,
) -> Tuple[Pipeline, bool, Optional[dict]]:
    """
    Load LR pipeline if it exists, otherwise train a new one using
    the training corpus from data_utils.datasets.
    """
    from data_utils.datasets import get_training_corpus

    if os.path.exists(model_path):
        pipe = joblib.load(model_path)
        return pipe, True, None

    # Get training data (your LIAR + custom_deception_dataset.csv,
    # or only custom, depending on how we set get_training_corpus)
    texts, labels = get_training_corpus()

    if len(texts) == 0:
        raise RuntimeError("No training data loaded. Check LIAR URLs and custom_deception_dataset.csv.")

    pipe, metrics = train_lr_model(texts, labels, metrics_path=metrics_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    return pipe, False, metrics

def lr_predict_proba(pipe, texts):
    """
    Wrapper so Streamlit can call predict_proba on the LR model.
    Input: list[str]
    Output: numpy array of shape (n_samples, 2)
    """
    return pipe.predict_proba(texts)
