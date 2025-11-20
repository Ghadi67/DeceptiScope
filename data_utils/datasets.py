# data_utils/datasets.py

import os
from typing import List, Tuple
import pandas as pd


# ---------- Paths / URLs ----------
DATASETS_DIR = "datasets"

LIAR_TRAIN_PATH = os.path.join(DATASETS_DIR, "train.tsv")
LIAR_VAL_PATH   = os.path.join(DATASETS_DIR, "valid.tsv")   
LIAR_TEST_PATH  = os.path.join(DATASETS_DIR, "test.tsv") 


CUSTOM_DATASET_PATH = "data/custom_deception_dataset.csv"


# ---------- LIAR dataset loader ----------
def load_liar(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Load LIAR or LIAR-PLUS TSV data from local files and map labels to binary:
        deceptive = {pants-fire, false, barely-true}
        truthful  = {half-true, mostly-true, true}
    Returns:
        texts, labels (0 = truthful, 1 = deceptive)
    """

    # Pick correct file depending on split
    if split == "train":
        path = LIAR_TRAIN_PATH
    elif split in ("val", "valid", "dev"):
        path = LIAR_VAL_PATH
    elif split == "test":
        path = LIAR_TEST_PATH
    else:
        raise ValueError(f"Unknown LIAR split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LIAR file not found at {path}. Put train/valid/test TSVs inside 'datasets/'."
        )

    # Read TSV with no header
    df = pd.read_csv(path, sep="\t", header=None)

    # Try different possible column index configurations
    # (because LIAR and LIAR-PLUS differ)
    possible_label_cols = [1, 2]  # which column label might be in
    possible_text_cols  = [2, 3]  # which column text might be in

    label_col = None
    text_col  = None

    # Try to detect proper columns automatically
    for lc in possible_label_cols:
        if lc < df.shape[1]:
            first_val = str(df.iloc[0, lc]).lower()
            if any(x in first_val for x in ["true", "false", "pants", "barely"]):
                label_col = lc
                break

    for tc in possible_text_cols:
        if tc < df.shape[1]:
            # A text column tends to be long
            if df.iloc[0, tc] and len(str(df.iloc[0, tc])) > 10:
                text_col = tc
                break

    if label_col is None or text_col is None:
        raise RuntimeError(
            f"Could not detect label/text columns in {path}. Please check the file structure."
        )

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
    texts      = df[text_col].astype(str).tolist()
    labels     = [label_map.get(lbl, 0) for lbl in raw_labels]

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
