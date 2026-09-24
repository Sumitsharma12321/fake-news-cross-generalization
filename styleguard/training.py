"""Training and inference procedures implementing the StyleGuard method.

The framework unifies two regimes:

* SS-Data / SS-Reg  -- for neural models (BiLSTM): style-varied copies are fed
  through the model together with the originals. A consistency term drives
  predicted probabilities to agree between an article and its style variant,
  while a gradient-reversal discriminator pushes the shared encoder to discard
  style-discriminative information.
* SS-Data (classical) -- for linear/bag-of-words models, the style-augmented
  corpus is used to refit the estimator.

Every model additionally supports SS-Ensemble: at inference time the class
probability is averaged over the original and several style variants, which
reduces sensitivity to any single surface rendering.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    NUM_THREADS,
    NUM_WORKERS,
    PATIENCE,
    RANDOM_STATE,
    STYLE_ALPHA,
    STYLE_BETA,
    STYLE_GAMMA,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from .models import StyleGuardBiLSTM, device

PredictProba = Callable[[list[str]], np.ndarray]


class PairedNewsDataset(Dataset):
    """Returns (original seq, style-variant seq, label) triples."""

    def __init__(self, X_orig: np.ndarray, X_variant: np.ndarray, y: np.ndarray):
        self.X_orig = torch.tensor(X_orig, dtype=torch.long)
        self.X_var = torch.tensor(X_variant, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_orig[idx], self.X_var[idx], self.y[idx]


def _pair_arrays(texts, variants, vocabulary, max_len):
    X_orig = vocabulary.texts_to_padded(texts, max_len)
    X_var = vocabulary.texts_to_padded(variants, max_len)
    return X_orig, X_var


class TfidfBag:
    """TF-IDF vectorizer fitted on the (possibly style-augmented) corpus."""

    def __init__(self, corpus_texts, tfidf=None):
        self.tfidf = tfidf or TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            stop_words="english",
        )
        self.tfidf.fit(corpus_texts)

    def transform(self, texts):
        return self.tfidf.transform(texts)


def build_classical(name: str):
    """Instantiate one of the classical baseline estimators."""
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if name == "svm":
        return CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE), cv=5, method="sigmoid"
        )
    if name == "nb":
        return MultinomialNB()
    raise ValueError(f"Unknown classical model: {name}")


def train_classical(
    corpus_texts,
    labels,
    name: str,
) -> tuple[object, TfidfBag]:
    """Fit TF-IDF + estimator. Pass an already style-augmented corpus for SS-Data."""
    bag = TfidfBag(corpus_texts)
    X = bag.transform(corpus_texts)
    model = build_classical(name)
    model.fit(X, labels)
    return model, bag


def classical_predict_proba(model, bag: TfidfBag) -> PredictProba:
    """Build a fake-class probability callable from a trained classical model."""
    fake_index = int(np.where(model.classes_ == 1)[0][0])

    def predict(texts: list[str]) -> np.ndarray:
        X = bag.transform(texts)
        return model.predict_proba(X)[:, fake_index]

    return predict


class _MonotonicScheduler:
    """Linearly ramp the gradient-reversal strength during training."""

    def __init__(self, epochs: int):
        self.epochs = max(epochs, 1)

    def at(self, epoch: int) -> float:
        progress = (epoch + 1) / self.epochs
        return min(1.0, 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


def train_bilstm(
    texts,
    variants,
    labels,
    vocabulary,
    embedding_matrix,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    alpha: float = STYLE_ALPHA,
    beta: float = STYLE_BETA,
    gamma: float = STYLE_GAMMA,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
    num_threads: int = NUM_THREADS,
    num_workers: int = NUM_WORKERS,
    val_texts=None,
    val_variants=None,
    val_labels=None,
    ood_texts=None,
) -> StyleGuardBiLSTM:
    """Train a StyleGuard BiLSTM.

    Setting ``alpha``, ``beta`` and ``gamma`` to zero reproduces the plain
    BiLSTM baseline (original project configuration). When ``ood_texts``
    (unlabeled MegaFake samples) is supplied, a domain discriminator is
    trained to tell Human-ID from OOD representations, and its reversed
    gradient drives the encoder to be invariant to the rewrite shift.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(num_threads)

    vocab_size = len(vocabulary)
    model = StyleGuardBiLSTM(vocab_size, embedding_matrix).to(device)
    X_orig, X_var = _pair_arrays(texts, variants, vocabulary, MAX_SEQ_LEN)
    y = np.asarray(labels, dtype=np.float32)
    train_loader = DataLoader(
        PairedNewsDataset(X_orig, X_var, y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    has_val = val_texts is not None and val_labels is not None
    has_domain = ood_texts is not None and gamma > 0.0
    ood_loader = None
    if has_domain:
        X_ood = np.asarray(
            vocabulary.texts_to_padded(ood_texts, MAX_SEQ_LEN), dtype=np.int64
        )
        ood_loader = DataLoader(
            torch.tensor(X_ood),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        ood_iter = iter(ood_loader)

    criterion = nn.BCELoss()
    consistency = nn.MSELoss()
    optimism = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _MonotonicScheduler(epochs)

    # Linear warmup so the regularizers ramp in gradually and cannot
    # overpower the supervised objective during the first epochs.
    def warmup(epoch: int) -> float:
        return min(1.0, 2.0 * (epoch + 1) / max(epochs, 1))

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        model.reversal.lambd = scheduler.at(epoch)
        eff_alpha = (alpha * warmup(epoch)) if alpha > 0.0 else 0.0
        eff_beta = (beta * warmup(epoch) ** 2) if beta > 0.0 else 0.0
        eff_gamma = (gamma * warmup(epoch) ** 2) if has_domain else 0.0
        losses = []
        for X_b, Xv_b, y_b in train_loader:
            X_b = X_b.to(device)
            Xv_b = Xv_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()

            p_orig, z_orig = model(X_b)
            p_var, z_var = model(Xv_b)

            # Supervise BOTH views so the variant keeps a meaningful label.
            ce = criterion(p_orig, y_b) + criterion(p_var, y_b)
            total = ce
            show = {"ce": ce.item()}

            if eff_alpha > 0.0:
                # Symmetric consistency: neither view is detached, both are
                # supervised, so they converge toward a shared truthful
                # prediction rather than drifting apart.
                cons = consistency(p_orig, p_var)
                total = total + eff_alpha * cons
                show["cons"] = cons.item()

            if eff_beta > 0.0 and model.discriminator is not None:
                adv_orig = optimism(model.style_logits(z_orig), torch.zeros(z_orig.size(0), dtype=torch.long, device=device))
                adv_var = optimism(model.style_logits(z_var), torch.ones(z_var.size(0), dtype=torch.long, device=device))
                adv = adv_orig + adv_var
                total = total + eff_beta * adv
                show["adv"] = adv.item()

            if eff_gamma > 0.0:
                # Unlabeled-domain alignment: draw a random OOD batch and ask
                # the domain discriminator to separate ID (0) from OOD (1).
                try:
                    X_o = next(ood_iter)
                except StopIteration:
                    ood_iter = iter(ood_loader)
                    X_o = next(ood_iter)
                X_o = X_o.to(device)
                with torch.no_grad():
                    _, z_o = model.backbone(X_o)
                dom = optimism(model.domain_logits(z_orig), torch.zeros(z_orig.size(0), dtype=torch.long, device=device)) \
                    + optimism(model.domain_logits(z_o), torch.ones(z_o.size(0), dtype=torch.long, device=device))
                total = total + eff_gamma * dom
                show["dom"] = dom.item()

            total.backward()
            optimizer.step()
            losses.append(total.item())

        val_f1 = None
        if has_val:
            model.eval()
            val_loss, val_f1 = _eval_bilstm(model, criterion, val_texts, val_labels, vocabulary)
            history["epoch"].append(epoch)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif epoch + 1 > epochs // 4:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        else:
            val_loss = None

        if verbose:
            line = f"Epoch {epoch+1}/{epochs} - train_loss: {np.mean(losses):.4f}"
            for key, value in show.items():
                line += f" {key}: {value:.4f}"
            if has_val:
                line += f" val_loss: {val_loss:.4f} val_f1: {val_f1:.4f}"
            print(line)

    return model


@torch.no_grad()
def _eval_bilstm(model, criterion, texts, labels, vocabulary):
    model.eval()
    X = torch.tensor(vocabulary.texts_to_padded(texts, MAX_SEQ_LEN), dtype=torch.long).to(device)
    y = torch.tensor(np.asarray(labels, dtype=np.float32), device=device)
    probs, _ = model(X)
    loss = criterion(probs, y)
    preds = (probs.cpu().numpy() >= 0.5).astype(int)
    f1 = f1_score(np.asarray(labels), preds)
    return loss.item(), f1


def bilstm_predict_proba(model, vocabulary) -> PredictProba:
    """Build a fake-class probability callable from a trained BiLSTM."""
    model.eval()

    def predict(texts: list[str]) -> np.ndarray:
        X = torch.tensor(vocabulary.texts_to_padded(texts, MAX_SEQ_LEN), dtype=torch.long).to(device)
        with torch.no_grad():
            probs, _ = model(X)
        return probs.cpu().numpy()

    return predict


def style_ensemble_proba(
    predict: PredictProba,
    texts: list[str],
    augmentor,
    k: int = 2,
) -> np.ndarray:
    """SS-Ensemble: average the fake-class probability over style variants."""
    if k <= 0:
        return predict(texts)
    acc = [predict(texts)]
    for _ in range(k):
        variants = [augmentor.variants(t)[0] for t in texts]
        acc.append(predict(variants))
    return np.mean(acc, axis=0)


def evaluate_predictions(y_true, y_pred, y_proba, model_name: str, dataset_name: str) -> dict:
    """Project-standard metrics dict (FAKE is the positive class)."""
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 6),
        "Precision": round(precision_score(y_true, y_pred), 6),
        "Recall": round(recall_score(y_true, y_pred), 6),
        "F1": round(f1_score(y_true, y_pred), 6),
        "ROC-AUC": round(roc_auc_score(y_true, y_proba), 6),
    }


def threshold(predicted_proba: np.ndarray, cutoff: float = 0.5) -> np.ndarray:
    return (predicted_proba >= cutoff).astype(int)