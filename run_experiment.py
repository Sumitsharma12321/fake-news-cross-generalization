#!/usr/bin/env python3
"""StyleGuard experiment runner.

Two protocols are supported.

Fully-unsupervised (original project protocol):
    train detectors on Human-ID only, evaluate in-distribution (ISOT test)
    and out-of-distribution (MegaFake-OOD).

Semi-supervised target adaptation (--sweep / --target-frac):
    a *labeled* fraction p of the MegaFake target set is added to the
    training corpus; the untouched remainder is the OOD evaluation set.
    This sweeps how much target supervision is needed to break the
    "predict everything as fake" ceiling. For the BiLSTM the unlabeled
    remainder additionally drives domain-adversarial alignment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_cpus = os.cpu_count() or 1
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, str(_cpus))

import numpy as np
import pandas as pd
import torch

from styleguard import (
    StyleAugmentor,
    Vocabulary,
    canonicalize,
    embed_from_glove,
    evaluate_predictions,
    load_splits,
    style_ensemble_proba,
    threshold,
    train_bilstm,
    train_classical,
)
from styleguard.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    EPOCHS,
    LABEL_FAKE,
    RANDOM_STATE,
    STYLE_ALPHA,
    STYLE_BETA,
    STYLE_GAMMA,
    AUG_PER_SAMPLE,
)
from styleguard.evaluation import (
    build_summary_table,
    lime_top_feature_overlap,
    save_all,
)

CLASSICAL_MODELS = ["logreg", "svm", "nb"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "svm", "nb", "bilstm"],
        choices=["logreg", "svm", "nb", "bilstm"],
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--aug-k", type=int, default=AUG_PER_SAMPLE,
                        help="style variants per training sample")
    parser.add_argument("--alpha", type=float, default=STYLE_ALPHA,
                        help="consistency regularizer weight")
    parser.add_argument("--beta", type=float, default=STYLE_BETA,
                        help="style-adversarial weight")
    parser.add_argument("--gamma", type=float, default=STYLE_GAMMA,
                        help="unlabeled-domain alignment (DANN) weight")
    parser.add_argument("--no-dann", action="store_true",
                        help="disable DANN alignment on unlabeled OOD samples")
    parser.add_argument("--ensemble-k", type=int, default=2,
                        help="variants used for test-time style ensembling")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--no-canonicalize", action="store_true",
                        help="disable style canonicalization of inputs")
    parser.add_argument("--target-frac", type=float, default=0.0,
                        help="labeled fraction of MegaFake added to training "
                             "(0.0 = fully-unsupervised protocol)")
    parser.add_argument("--sweep", type=float, nargs="+", default=None,
                        metavar="P",
                        help="instead of a single run, sweep labeled target "
                             "fractions; 0.0 is added automatically")
    parser.add_argument("--run-lime", action="store_true",
                        help="run LIME overlap analysis on fine-tuned models")
    parser.add_argument("--glove-path", type=Path, default=None,
                        help="path to an existing glove.6B.100d.txt to avoid downloading")
    parser.add_argument("--num-cpu", type=int, default=_cpus,
                        help="number of CPU threads to use (default: all cores)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def _target_split(ood_texts, ood_labels, frac, seed):
    """Stratified labeled-train slice (p) vs held-out evaluation remainder."""
    from sklearn.model_selection import train_test_split
    if frac <= 0.0:
        return [], np.array([], dtype=int), ood_texts, np.asarray(ood_labels)
    train_t, test_t, train_l, test_l = train_test_split(
        ood_texts, np.asarray(ood_labels), train_size=frac,
        stratify=ood_labels, random_state=seed,
    )
    return list(train_t), train_l, list(test_t), test_l


def _run_model(model_name, splits, target_texts, target_labels,
               ood_unlabeled_texts, ood_eval_texts, ood_eval_labels,
               args, augmentor):
    train_texts = splits.train["text"].tolist()
    train_labels = splits.train["label"].values
    test_texts = splits.test["text"].tolist()
    test_labels = splits.test["label"].values

    if model_name in CLASSICAL_MODELS:
        corpus_texts = train_texts + list(target_texts)
        corpus_labels = np.concatenate([train_labels, target_labels])
        if args.aug_k > 0:
            df = pd.DataFrame({"text": corpus_texts, "label": corpus_labels})
            corpus = augmentor.augment_frame(df, text_col="text", per_sample=args.aug_k)
            corpus_texts = corpus["text"].tolist()
            corpus_labels = corpus["label"].values
        model, bag = train_classical(corpus_texts, corpus_labels, model_name)
        predict = _classical_predictor(model_name, model, bag)
        label = model_name.upper()
    else:
        vocab = Vocabulary.build(train_texts)
        if args.glove_path is not None:
            glove_path = args.glove_path
        else:
            glove_path = args.data_dir / "glove.6B.100d.txt"
        embedding = embed_from_glove(vocab.word2idx, glove_path)
        aug = StyleAugmentor(seed=args.seed)
        tr_texts = train_texts + list(target_texts)
        tr_labels = np.concatenate([train_labels, target_labels])
        variants = [aug.variants(t)[0] for t in tr_texts]
        model = train_bilstm(
            tr_texts, variants, tr_labels, vocab, embedding,
            epochs=args.epochs,
            alpha=args.alpha, beta=args.beta,
            gamma=0.0 if args.no_dann else args.gamma,
            num_threads=args.num_cpu,
            num_workers=min(max(1, args.num_cpu // 2), 8),
            val_texts=splits.val["text"].tolist(),
            val_labels=splits.val["label"].values,
            ood_texts=ood_unlabeled_texts,
            verbose=True,
        )
        predict = _bilstm_predictor(model, vocab)
        label = "BiLSTM"

    if not args.no_ensemble:
        predict = _wrap_ensemble(predict, augmentor, args.ensemble_k)

    id_proba = predict(test_texts)
    id_pred = threshold(id_proba)
    ood_proba = predict(ood_eval_texts)
    ood_pred = threshold(ood_proba)

    id_rec = evaluate_predictions(test_labels, id_pred, id_proba, label, "Human-ID")
    ood_rec = evaluate_predictions(ood_eval_labels, ood_pred, ood_proba, label, "MegaFake-OOD")
    gap_rec = build_summary_table(
        [id_rec], [ood_rec], id_rec["F1"], ood_rec["F1"], label
    )
    print(f"Human-ID: F1={id_rec['F1']:.4f}  OOD: F1={ood_rec['F1']:.4f}  "
          f"Gap={gap_rec.iloc[0]['Generalization_Gap']:.4f}")
    return id_rec, ood_rec, gap_rec, predict


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.num_cpu)
    os.environ["MKL_NUM_THREADS"] = str(args.num_cpu)
    torch.set_num_threads(args.num_cpu)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.data_dir)

    if not args.no_canonicalize:
        print("Canonicalizing all inputs (strip datelines/dates, expand abbreviations) ...")
        splits.train["text"] = splits.train["text"].apply(canonicalize)
        splits.val["text"] = splits.val["text"].apply(canonicalize)
        splits.test["text"] = splits.test["text"].apply(canonicalize)
        splits.ood["content_clean"] = splits.ood["content_clean"].apply(canonicalize)

    augmentor = StyleAugmentor(seed=args.seed)
    ood_texts = splits.ood["content_clean"].tolist()
    ood_labels = splits.ood["label"].values

    sweep_vals = list(dict.fromkeys([0.0] + (args.sweep or [args.target_frac])))
    sweeping = len(sweep_vals) > 1 or sweep_vals[0] > 0.0

    all_rows: list[dict] = []
    gap_rows: list[dict] = []

    for frac in sweep_vals:
        print(f"\n########## labeled target fraction p = {frac:.2f} ##########")
        tgt_texts, tgt_labels, eval_texts, eval_labels = _target_split(
            ood_texts, ood_labels, frac, args.seed
        )
        unlabeled = ood_texts if frac <= 0.0 else eval_texts

        for model_name in args.models:
            print(f"\n========== {model_name} ==========")
            id_rec, ood_rec, gap_rec, predict = _run_model(
                model_name, splits, tgt_texts, tgt_labels,
                unlabeled, eval_texts, eval_labels, args, augmentor
            )
            id_rec = {**id_rec, "Target_Frac": frac}
            ood_rec = {**ood_rec, "Target_Frac": frac}
            all_rows.append(id_rec)
            all_rows.append(ood_rec)
            gap_rows.append({**gap_rec.iloc[0].to_dict(), "Target_Frac": frac})

            if args.run_lime and model_name == "bilstm" and frac == sweep_vals[-1]:
                _lime_analysis(predict, splits.test["text"].tolist(),
                               splits.test["label"].values,
                               eval_texts, eval_labels, "BiLSTM", out_dir)

    summary = pd.DataFrame(all_rows)
    gaps = pd.DataFrame(gap_rows)
    stem = f"styleguard_target_sweep_{args.seed}" if sweeping else f"styleguard_full_comparison_{args.seed}"
    save_all(summary, out_dir, stem)
    save_all(gaps, out_dir, f"{stem}_gap")

    config = vars(args)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({k: str(v) for k, v in config.items()}, f, indent=2)

    print("\n=== Summary (F1) ===")
    pivot = summary.pivot_table(index=["Target_Frac", "Model"], columns="Dataset", values="F1")
    print(pivot)


def _classical_predictor(name: str, model, bag):
    from styleguard.training import classical_predict_proba
    return classical_predict_proba(model, bag)


def _bilstm_predictor(model, vocab):
    from styleguard.training import bilstm_predict_proba
    return bilstm_predict_proba(model, vocab)


def _wrap_ensemble(predict, augmentor, k: int):
    def wrapped(texts):
        return style_ensemble_proba(predict, list(texts), augmentor, k=k)

    return wrapped


def _lime_analysis(predict, test_texts, test_labels, ood_texts, ood_labels, label, out_dir):
    id_example = next(t for t, l in zip(test_texts, test_labels) if l == LABEL_FAKE)
    ood_fake_idx = int(np.where(np.asarray(ood_labels) == LABEL_FAKE)[0][0])
    ood_fake_example = ood_texts[ood_fake_idx]
    result = lime_top_feature_overlap(predict, id_example, ood_fake_example, k=10)
    save_all(pd.DataFrame([{"Model": label, **result}]), out_dir, "styleguard_lime_overlap")
    print(f"LIME overlap {label}: {result['Overlap_Percent']}%")


if __name__ == "__main__":
    main()