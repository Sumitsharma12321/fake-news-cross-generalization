"""Neural architectures for the StyleGuard framework.

``StyleGuardBiLSTM`` extends the original project BiLSTM with (a) a
gradient-reversal module that drives the shared encoder toward style-invariant
representations and (b) a lightweight style discriminator that competes with
that encoder during training.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn as nn

from .config import (
    DOWNLOAD_TIMEOUT,
    EMBEDDING_DIM,
    GLOVE_FILE,
    GLOVE_MIRRORS,
    GLOVE_TXT_MIRROR,
    HIDDEN_DIM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradientReversalFunction(torch.autograd.Function):
    """Reverse the upstream gradient through the representation."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    """Module that applies gradient reversal with a schedule-controlled lambda."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambd)


class StyleDiscriminator(nn.Module):
    """Two-class head that tries to tell originals apart from style variants."""

    def __init__(self, dim_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainDiscriminator(StyleDiscriminator):
    """Two-class head that tries to tell Human-ID from MegaFake-OOD domains.

    Gradients are reversed through the shared encoder, so training it drives
    the representation to be invariant to the domain/rewrite shift.
    """


class BiLSTM(nn.Module):
    """BiLSTM fake-news backbone, same topology as the original project."""

    def __init__(self, vocab_size: int, embedding_matrix: torch.Tensor, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(HIDDEN_DIM * 2, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        x = self.dropout1(h_cat)
        z = self.relu(self.fc1(x))
        logit = self.fc2(self.dropout2(z))
        return self.sigmoid(logit).squeeze(1), z


class StyleGuardBiLSTM(nn.Module):
    """BiLSTM backbone plus adversarial discriminators.

    ``discriminator`` is trained on the style head (original vs augmented
    variant); ``domain_discriminator`` is trained on the domain head
    (Human-ID vs MegaFake-OOD, pseudo-labels when OOD is unlabeled). Both
    receive the same gradient-reversed representation and push the shared
    encoder toward representations that are invariant to surface style.
    """

    def __init__(self, vocab_size: int, embedding_matrix: torch.Tensor, disc_hidden: int = 64):
        super().__init__()
        self.backbone = BiLSTM(vocab_size, embedding_matrix)
        self.reversal = GradientReversal()
        self.discriminator = StyleDiscriminator(32, hidden=disc_hidden)
        self.domain_discriminator = DomainDiscriminator(32, hidden=disc_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(x)

    def style_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.reversal(z))

    def domain_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.domain_discriminator(self.reversal(z))

    def forward_domain(self, x_ood: torch.Tensor) -> torch.Tensor:
        """Forward an unlabeled OOD batch, returning only its ``z``."""
        with torch.no_grad():
            _, z = self.backbone(x_ood)
        return z


def embed_from_glove(word2idx: dict[str, int], glove_path: str | Path | None = None) -> torch.Tensor:
    """Build the embedding matrix from GloVe 6B 100-d, downloading if needed."""
    glove_path = Path(glove_path) if glove_path else Path(GLOVE_FILE)
    if not glove_path.exists():
        _download_glove(glove_path)
    index: dict[str, np.ndarray] = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            index[values[0]] = np.asarray(values[1:], dtype="float32")
    vocab_size = len(word2idx)
    matrix = np.random.normal(0, 0.1, (vocab_size, EMBEDDING_DIM)).astype("float32")
    matrix[0] = 0.0
    for word, idx in word2idx.items():
        vec = index.get(word)
        if vec is not None:
            matrix[idx] = vec
    return torch.tensor(matrix)


def _fetch(url: str) -> bytes:
    import socket
    from urllib.error import URLError

    try:
        socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)
        with urlopen(url) as response:
            return response.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"failed to reach {url}: {exc}") from exc


def _download_glove(target: Path) -> None:
    """Fetch GloVe 6B 100-d, preferring a direct mirror and then the zip archive.

    Tries several mirrors so a dead host cannot hang the run forever.
    """
    import io

    if target.name == GLOVE_FILE and target.resolve() != Path(GLOVE_FILE).resolve():
        try:
            print(f"Downloading GloVe from {GLOVE_TXT_MIRROR} ...")
            target.write_bytes(_fetch(GLOVE_TXT_MIRROR))
            print(f"Saved embeddings to {target}")
            return
        except Exception as exc:
            print(f"Direct download failed ({exc}); falling back to zip mirrors ...")

    last_error: Exception | None = None
    for url in GLOVE_MIRRORS:
        try:
            print(f"Downloading GloVe zip from {url} ...")
            archive = ZipFile(io.BytesIO(_fetch(url)))
            member = next(m for m in archive.namelist() if m.startswith(GLOVE_FILE))
            target.write_bytes(archive.read(member))
            print(f"Saved embeddings to {target}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Mirror {url} failed ({exc}); trying next ...")
    raise RuntimeError(
        "Could not download the GloVe 6B 100-d embeddings from any mirror. "
        "Download the file manually and pass its path via "
        "run_experiment.py --glove-path <file>."
    ) from last_error