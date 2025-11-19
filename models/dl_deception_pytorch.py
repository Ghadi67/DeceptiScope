# models/dl_deception_pytorch.py

import json
import os
from typing import Tuple, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from data_utils.datasets import get_training_corpus


DL_MODEL_PATH = "artifacts/deception/dl_deception_model.pt"
DL_VECTORIZER_PATH = "artifacts/deception/dl_vectorizer.joblib"
DL_METRICS_PATH = "artifacts/deception/dl_training_metrics.json"


# ---------- Dataset wrapper ----------
class DeceptionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- Our PyTorch model ----------
class DeceptionNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # binary output (logit)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,) logits


# ---------- Training ----------
def train_dl_model(
    texts: List[str],
    labels: List[int],
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 8,
    device: str = None,
    metrics_path: Optional[str] = DL_METRICS_PATH,
) -> Tuple[DeceptionNN, TfidfVectorizer]:
    """
    Train our own PyTorch MLP on TF-IDF features entirely from scratch.
    No pre-trained embeddings or third-party classifiers are used; both the
    vectorizer and the network weights are learned solely from the provided
    training corpus.
    """
    print("[dl_deception] Building TF-IDF features for DL model...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,    # allow words that appear only once
        max_df=1.0,  # don't drop anything based on document frequency
    )
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_train_arr = X_train_vec.toarray().astype(np.float32)
    X_val_arr = X_val_vec.toarray().astype(np.float32)
    y_train_arr = np.array(y_train, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.float32)

    input_dim = X_train_arr.shape[1]
    print(f"[dl_deception] Input dimension = {input_dim}")

    train_dataset = DeceptionDataset(X_train_arr, y_train_arr)
    val_dataset = DeceptionDataset(X_val_arr, y_val_arr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dl_deception] Training on device: {device}")

    model = DeceptionNN(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        avg_loss = epoch_loss / len(train_dataset)

        # validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds)

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"[dl_deception] Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {avg_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_f1: {val_f1:.4f}"
        )
        model.train()

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        summary = {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "val_f1": history["val_f1"],
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return model, vectorizer


# ---------- Save / load ----------
def save_dl_model(model: DeceptionNN, vectorizer: TfidfVectorizer):
    os.makedirs(os.path.dirname(DL_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), DL_MODEL_PATH)
    joblib.dump(vectorizer, DL_VECTORIZER_PATH)
    print(f"[dl_deception] Saved model to {DL_MODEL_PATH}")
    print(f"[dl_deception] Saved vectorizer to {DL_VECTORIZER_PATH}")


def load_dl_model(device: str = None) -> Tuple[DeceptionNN, TfidfVectorizer]:
    """
    Load the model + vectorizer from disk.
    Assumes they exist.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not (os.path.exists(DL_MODEL_PATH) and os.path.exists(DL_VECTORIZER_PATH)):
        raise FileNotFoundError("DL model or vectorizer not found on disk.")

    vectorizer: TfidfVectorizer = joblib.load(DL_VECTORIZER_PATH)
    input_dim = len(vectorizer.get_feature_names_out())

    model = DeceptionNN(input_dim=input_dim)
    state = torch.load(DL_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"[dl_deception] Loaded DL model from {DL_MODEL_PATH}")
    return model, vectorizer


def load_or_build_dl(
    force_retrain: bool = False,
    device: str = None
) -> Tuple[DeceptionNN, TfidfVectorizer, bool]:
    """
    Load DL model if present, otherwise train on LIAR + custom dataset.
    Returns:
        (model, vectorizer, loaded_from_disk_flag)
    """
    if not force_retrain and os.path.exists(DL_MODEL_PATH) and os.path.exists(DL_VECTORIZER_PATH):
        model, vectorizer = load_dl_model(device=device)
        return model, vectorizer, True

    print("[dl_deception] Training DL model from scratch using LIAR + custom dataset...")
    texts, labels = get_training_corpus(
        use_liar=True,
        use_custom=True,
    )
    model, vectorizer = train_dl_model(texts, labels, device=device)
    save_dl_model(model, vectorizer)
    return model, vectorizer, False


# ---------- Inference ----------
@torch.no_grad()
def dl_predict_proba(
    model: DeceptionNN,
    vectorizer: TfidfVectorizer,
    texts: List[str],
    device: str = None
) -> np.ndarray:
    """
    Returns probabilities for [truthful, deceptive] for each text.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    X = vectorizer.transform(texts)
    X = X.toarray().astype(np.float32)
    X_tensor = torch.from_numpy(X).float().to(device)

    logits = model(X_tensor)
    probs_deceptive = torch.sigmoid(logits).cpu().numpy()  # shape (N,)
    probs_truthful = 1.0 - probs_deceptive

    return np.stack([probs_truthful, probs_deceptive], axis=1)
