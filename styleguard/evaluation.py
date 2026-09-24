"""Evaluation utilities: generalization-gap accounting and LIME analysis."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .training import PredictProba


def generalization_gap(id_f1: float, ood_f1: float) -> float:
    """F1_Human-ID minus F1_MegaFake-OOD."""
    return round(float(id_f1) - float(ood_f1), 6)


def build_summary_table(id_records, ood_records, id_f1, ood_f1, model_name: str) -> pd.DataFrame:
    """Combine ID/OOD metric records with the generalization gap."""
    return pd.DataFrame([
        {"Model": model_name, "F1_Human-ID": id_f1, "F1_OOD": ood_f1,
         "Generalization_Gap": generalization_gap(id_f1, ood_f1)},
    ])


def lime_top_feature_overlap(
    predict: PredictProba,
    id_text: str,
    ood_text: str,
    k: int = 10,
    labels: tuple[str, str] = ("REAL", "FAKE"),
) -> dict[str, Any]:
    """Explain one ID and one OOD example and quantify top-k overlap.

    Returns a dict with per-example top features and the overlap percentage,
    the diagnostic used to show whether the evidence the model relies on
    survives the LLM style shift.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lime is required for explainability analysis") from exc

    def proba(texts):
        probs_fake = predict(list(texts))
        probs_real = 1.0 - np.asarray(probs_fake)
        return np.column_stack([probs_real, np.asarray(probs_fake)])

    explainer = LimeTextExplainer(class_names=list(labels), random_state=42)
    exp_id = explainer.explain_instance(id_text, proba, num_features=k)
    exp_ood = explainer.explain_instance(ood_text, proba, num_features=k)

    id_features = [w for w, _ in exp_id.as_list()[:k]]
    ood_features = [w for w, _ in exp_ood.as_list()[:k]]
    common = set(id_features) & set(ood_features)

    return {
        "Top_K": k,
        "ID_Features": ", ".join(id_features),
        "OOD_Features": ", ".join(ood_features),
        "Common_Features": ", ".join(sorted(common)),
        "Overlap_Percent": round(len(common) / k * 100, 1),
    }


def save_all(df: pd.DataFrame, out_dir: str | Path, stem: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{stem}.csv"
    df.to_csv(dest, index=False)
    return dest


def describe_results(records: dict[str, pd.DataFrame], model_name: str) -> str:
    """Render a compact human-readable summary block for logging."""
    try:
        id_rec = records["id"]
        ood_rec = records["ood"]
        gap = records["gap"]
        line_id = id_rec.iloc[0]
        line_ood = ood_rec.iloc[0]
        line_gap = gap.iloc[0]
    except (KeyError, IndexError):
        return ""
    return textwrap.dedent(
        f"""
        === {model_name} ===
        Human-ID : F1={line_id['F1']:.4f} Acc={line_id['Accuracy']:.4f} AUC={line_id['ROC-AUC']:.4f}
        MegaFake : F1={line_ood['F1']:.4f} Acc={line_ood['Accuracy']:.4f} AUC={line_ood['ROC-AUC']:.4f}
        Gen-Gap  : {line_gap['Generalization_Gap']:.4f}
        """
    )