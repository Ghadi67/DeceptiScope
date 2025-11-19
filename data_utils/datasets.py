# data_utils/datasets.py

import os
from typing import List, Tuple
import pandas as pd


# ---------- Paths / URLs ----------
LIAR_TRAIN_URL = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/train2.tsv"
LIAR_VAL_URL   = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/valid2.tsv"
LIAR_TEST_URL  = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/test2.tsv"

# You create this file yourself: columns: "text", "label" (0 = truthful, 1 = deceptive)
CUSTOM_DATASET_PATH = "data/custom_deception_dataset.csv"


# ---------- LIAR dataset loader ----------
def load_liar(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Load LIAR-PLUS TSV data and map labels to binary:
        deceptive = {pants-fire, false, barely-true}
        truthful  = {half-true, mostly-true, true}

    Returns:
        texts, labels (0 = truthful, 1 = deceptive)
    """
    if split == "train":
        url = LIAR_TRAIN_URL
    elif split in ("val", "valid", "dev"):
        url = LIAR_VAL_URL
    elif split == "test":
        url = LIAR_TEST_URL
    else:
        raise ValueError(f"Unknown LIAR split: {split}")

    df = pd.read_csv(url, sep="\t", header=None)

    # In LIAR-PLUS, column indices can vary between repos.
    # Common setup: label ~ col 1 or 2, text ~ col 2 or 3.
    # Adjust if needed after checking df.head().
    # Here we assume:
    #   label in column 2
    #   text  in column 3
    label_col = 2
    text_col = 3

    label_map = {
        "pants-fire": 1,
        "pants on fire": 1,
        "false": 1,
        "barely-true": 1,
        "barely true": 1,
        "half-true": 0,
        "half true": 0,
        "mostly-true": 0,
        "mostly true": 0,
        "true": 0,
    }

    raw_labels = df[label_col].astype(str).str.strip().str.lower()
    texts = df[text_col].astype(str).tolist()
    labels = [label_map.get(lbl, 0) for lbl in raw_labels]

    return texts, labels


# ---------- Custom dataset loader ----------
def load_custom_dataset(path: str = CUSTOM_DATASET_PATH) -> Tuple[List[str], List[int]]:
    """
    Load your custom deception dataset from CSV.

    Expected columns:
        text  : the statement
        label : 0 (truthful) or 1 (deceptive)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Custom dataset not found at {path}. "
            f"Create a CSV with columns 'text' and 'label'."
        )

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Custom dataset must have columns 'text' and 'label'.")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels


# ---------- Combined training corpus ----------
def get_training_corpus(
    use_liar: bool = True,
    use_custom: bool = True,
) -> Tuple[List[str], List[int]]:
    """
    Returns the combined training corpus from:
        - LIAR dataset
        - Your custom dataset

    If one of them is missing/unavailable, falls back to the other.
    """
    texts_all: List[str] = []
    labels_all: List[int] = []

    if use_liar:
        try:
            liar_texts, liar_labels = load_liar(split="train")
            texts_all.extend(liar_texts)
            labels_all.extend(liar_labels)
            print(f"[datasets] Loaded LIAR train set: {len(liar_texts)} examples")
        except Exception as e:
            print(f"[datasets] Could not load LIAR dataset: {e}")

    if use_custom:
        try:
            custom_texts, custom_labels = load_custom_dataset()
            texts_all.extend(custom_texts)
            labels_all.extend(custom_labels)
            print(f"[datasets] Loaded custom dataset: {len(custom_texts)} examples")
        except Exception as e:
            print(f"[datasets] Could not load custom dataset: {e}")

    if not texts_all:
        raise RuntimeError(
            "No training data loaded. "
            "Check LIAR URLs and custom_deception_dataset.csv."
        )

    return texts_all, labels_all
