"""Shared configuration and constants for the StyleGuard framework."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def auto_cpu_cores() -> int:
    """Return the number of usable logical CPU cores."""
    return os.cpu_count() or 1


NUM_THREADS = auto_cpu_cores()
NUM_WORKERS = min(max(1, NUM_THREADS // 2), 8)

DEFAULT_DATA_DIR = PROJECT_ROOT / "classical_ML_result" / "Split_data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "styleguard_results"

LABEL_REAL = 0
LABEL_FAKE = 1

RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

MAX_VOCAB_SIZE = 50_000
EMBEDDING_DIM = 100
MAX_SEQ_LEN = 300
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
PATIENCE = 5

GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"
GLOVE_MIRRORS = [
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    "https://nlp.stanford.edu/data/glove.6B.zip",
]
GLOVE_TXT_MIRROR = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.100d.txt"
DOWNLOAD_TIMEOUT = 120

STYLE_ALPHA = 0.2
STYLE_BETA = 0.0
STYLE_GAMMA = 0.3
AUG_PER_SAMPLE = 2