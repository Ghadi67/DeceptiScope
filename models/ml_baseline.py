# models/ml_baseline.py

from typing import List, Tuple
import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LR_MODEL_PATH_DEFAULT = "models/lr_tfidf_lr.joblib"


def train_lr_model(texts: List[str], labels: List[int]) -> Pipeline:
    """
    Train a simple TF-IDF + Logistic Regression model for deception detection.

    This version is SAFE for very small datasets (like your 4-row CSV):
    - min_df = 1  → keep all terms that appear at least once
    - max_df = 1.0 → don't drop frequent terms
    """
    labels = np.array(labels)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=1,             # <<< IMPORTANT: keep all terms
            max_df=1.0,           # <<< keep even very frequent ones
            max_features=None,    # no artificial cap
            lowercase=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )),
    ])

    pipe.fit(texts, labels)
    return pipe


def load_or_build_lr(model_path: str = LR_MODEL_PATH_DEFAULT) -> Tuple[Pipeline, bool]:
    """
    Load LR pipeline if it exists, otherwise train a new one using
    the training corpus from data_utils.datasets.
    """
    from data_utils.datasets import get_training_corpus

    if os.path.exists(model_path):
        pipe = joblib.load(model_path)
        return pipe, True

    # Get training data (your LIAR + custom_deception_dataset.csv,
    # or only custom, depending on how we set get_training_corpus)
    texts, labels = get_training_corpus()

    if len(texts) == 0:
        raise RuntimeError("No training data loaded. Check LIAR URLs and custom_deception_dataset.csv.")

    pipe = train_lr_model(texts, labels)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    return pipe, False

def lr_predict_proba(pipe, texts):
    """
    Wrapper so Streamlit can call predict_proba on the LR model.
    Input: list[str]
    Output: numpy array of shape (n_samples, 2)
    """
    return pipe.predict_proba(texts)
