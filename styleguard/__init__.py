"""StyleGuard: consistency-regularized, style-adversarial fine-tuning for
robust cross-domain fake news detection.

The package provides modular building blocks (data, augmentation, models,
training, evaluation) that replace the original notebook-only workflow while
preserving the exact input splits, metrics, and label conventions.
"""

from .augmentation import StyleAugmentor, canonicalize
from .data import (
    NewsSplits,
    Vocabulary,
    combine_human_isot,
    light_clean,
    load_splits,
    make_dataloader,
    simple_tokenize,
)
from .evaluation import (
    build_summary_table,
    generalization_gap,
    lime_top_feature_overlap,
    save_all,
)
from .models import (
    BiLSTM,
    StyleDiscriminator,
    StyleGuardBiLSTM,
    embed_from_glove,
)
from .training import (
    PairedNewsDataset,
    classical_predict_proba,
    evaluate_predictions,
    style_ensemble_proba,
    threshold,
    train_bilstm,
    train_classical,
)

__all__ = [
    "NewsSplits",
    "Vocabulary",
    "StyleAugmentor",
    "canonicalize",
    "StyleGuardBiLSTM",
    "BiLSTM",
    "StyleDiscriminator",
    "load_splits",
    "combine_human_isot",
    "light_clean",
    "simple_tokenize",
    "make_dataloader",
    "embed_from_glove",
    "train_bilstm",
    "train_classical",
    "classical_predict_proba",
    "evaluate_predictions",
    "style_ensemble_proba",
    "threshold",
    "generalization_gap",
    "build_summary_table",
    "lime_top_feature_overlap",
    "save_all",
]