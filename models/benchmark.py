"""Model benchmarking utilities for deception detectors."""

import json
import os
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from data_utils.datasets import get_training_corpus
from models.dl_deception_pytorch import dl_predict_proba, load_or_build_dl
from models.ml_baseline import load_or_build_lr, lr_predict_proba

DEFAULT_REPORT_PATH = "artifacts/deception/model_comparison.json"


def _summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }


def _evaluate_model_predictions(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    preds = (y_proba[:, 1] >= 0.5).astype(int)
    return _summarize_predictions(y_true, preds)


def benchmark_models(
    report_path: str = DEFAULT_REPORT_PATH,
) -> Dict[str, Dict[str, float]]:
    """
    Compare the in-house ML and DL models that are trained **from scratch** on
    the local deception corpus. No pre-trained or third-party classifier is
    involved in the benchmark to keep the evaluation focused on first-party
    models only. Results are saved to ``report_path``.
    """

    texts, labels = get_training_corpus()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=123, stratify=labels
    )

    # Ensure training artifacts exist
    lr_model, _, _ = load_or_build_lr()
    dl_model, dl_vectorizer, _ = load_or_build_dl()

    lr_probs = lr_predict_proba(lr_model, X_test)
    dl_probs = dl_predict_proba(dl_model, dl_vectorizer, X_test)

    results: Dict[str, Dict[str, float]] = {
        "logistic_regression": _evaluate_model_predictions(np.array(y_test), lr_probs),
        "deep_learning": _evaluate_model_predictions(np.array(y_test), dl_probs),
    }

    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    summary = benchmark_models()
    print(json.dumps(summary, indent=2))
