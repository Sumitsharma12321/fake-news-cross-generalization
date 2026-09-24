"""Dataset loading, preprocessing, tokenization, and vocabulary management."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    DEFAULT_DATA_DIR,
    LABEL_FAKE,
    LABEL_REAL,
    MAX_SEQ_LEN,
    MAX_VOCAB_SIZE,
    RANDOM_STATE,
)


def light_clean(text: str) -> str:
    """Apply project-standard light cleaning that preserves writing style."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class NewsSplits:
    """Container for the ID train/val/test and OOD splits."""

    def __init__(self, train, val, test, ood):
        self.train = train
        self.val = val
        self.test = test
        self.ood = ood


def _text_column(frame: pd.DataFrame, preferred: list[str]) -> str:
    """Return the first available text column among the candidates."""
    for name in preferred:
        if name in frame.columns:
            return name
    raise KeyError(
        f"None of the expected text columns {preferred} were found in "
        f"{frame.columns.tolist()}. For train.csv this usually means the file "
        "still contains the Google Drive placeholder instead of real data. "
        "Download the ISOT train split and overwrite "
        "classical_ML_result/Split_data/train.csv, or rebuild it with "
        "styleguard.data.combine_human_isot()."
    )


def load_splits(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> NewsSplits:
    """Load the pre-split CSVs produced by the original project."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    ood = pd.read_csv(data_dir / "megafake_ood_test.csv")

    text_col = _text_column(train, ["text", "content_clean", "content"])
    if text_col != "text":
        train = train.rename(columns={text_col: "text"})
    train["text"] = train["text"].fillna("")

    val["text"] = val["text"].fillna("")
    test["text"] = test["text"].fillna("")
    ood_col = _text_column(ood, ["content_clean", "text"])
    if ood_col != "content_clean":
        ood = ood.rename(columns={ood_col: "content_clean"})
    ood["content_clean"] = ood["content_clean"].fillna("")
    return NewsSplits(train, val, test, ood)


def combine_human_isot(
    fake_csv: str | Path,
    true_csv: str | Path,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Rebuild the combined ISOT frame from the raw Fake.csv / True.csv files."""
    fake = pd.read_csv(fake_csv).copy()
    true = pd.read_csv(true_csv).copy()
    fake["label"] = LABEL_FAKE
    true["label"] = LABEL_REAL
    df = pd.concat([fake, true], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = df[df["content"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset="content", keep="first")
    df["content_clean"] = df["content"].apply(light_clean)
    return df


class Vocabulary:
    """Word-to-index mapping built exclusively from training data."""

    PAD = "<PAD>"
    OOV = "<OOV>"

    def __init__(self, word2idx: dict[str, int]):
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        max_size: int = MAX_VOCAB_SIZE,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))
        most_common = counter.most_common(max_size - 2)
        word2idx = {cls.PAD: 0, cls.OOV: 1}
        for word, _ in most_common:
            word2idx[word] = len(word2idx)
        return cls(word2idx)

    def encode(self, tokens: Sequence[str], max_len: int) -> list[int]:
        ids = [self.word2idx.get(tok, self.word2idx[self.OOV]) for tok in tokens[:max_len]]
        if len(ids) < max_len:
            ids = ids + [self.word2idx[self.PAD]] * (max_len - len(ids))
        return ids

    def texts_to_padded(self, texts: Sequence[str], max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        return np.array(
            [self.encode(simple_tokenize(t), max_len) for t in texts],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.word2idx)


class NewsDataset(Dataset):
    """PyTorch dataset over pre-padded sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        NewsDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
    )