# StyleGuard: Robust Fake-News Detection Against LLM-Generated Content

Cross-generalization and explainability analysis of fake news detectors under
LLM-generated style attacks, extended with a fine-tuning framework
(**StyleGuard**) that improves OOD robustness across all detector backbones.

## Problem

Detectors trained only on human-written news (ISOT) achieve near-perfect
in-distribution accuracy but lose ~0.33 `F1` on LLM style-transferred news
(MegaFake), because they exploit surface writing artifacts (wire datelines,
agency markers) that LLM rewriting removes — confirmed by LIME showing 0%
ID/OOD feature overlap.

## Method (StyleGuard)

- **SS-Data** — model-free style-transfer augmentation (dateline stripping,
  date scrubbing, abbreviation expansion, synonym substitution, function-word
  deletion, clause permutation) that emulates LLM rewrites without calling an
  LLM, applied to the training corpus without touching labels.
- **Canonicalization** — the deterministic subset of SS-Data (dateline strip,
  date scrub, abbreviation expansion) is applied to *every* input so the
  detector cannot rely on wire-style markers that LLM rewriting removes.
- **SS-Reg** — consistency regularizer forcing an article and its style
  variants to receive identical predictions.
- **Style-adversarial training** — gradient-reversal discriminator drives the
  shared encoder to discard style-discriminative features.
- **Unsupervised domain alignment (DANN)** — a second gradient-reversal
  discriminator aligns Human-ID representations with *unlabeled* MegaFake
  representations, removing the residual domain shift.
- **SS-Ensemble** — test-time averaging of predictions over style variants;
  works with any already-trained detector (including classical models).

Treating neural and classical models in one framework: SS-Reg and the
adversarial/domain terms apply to the BiLSTM; SS-Data, canonicalization, and
SS-Ensemble apply to TF-IDF logistic regression, linear SVM, and multinomial
naive Bayes. A semi-supervised target sweep (`--sweep`) measures how much
*labeled* MegaFake supervision (in addition to the unlabeled alignment)
is needed to break the "predict everything as fake" ceiling.

## Modular code (replaces the notebooks)

```
styleguard/
  config.py        # shared constants (seed, TF-IDF, BiLSTM, StyleGuard hparams)
  data.py          # split loading, cleaning, vocabulary, dataloaders
  augmentation.py  # StyleAugmentor: style-transferred text transforms
  models.py        # BiLSTM backbone + gradient reversal + style discriminator
  training.py      # train_classical / train_bilstm, ensemble, metrics
  evaluation.py    # generalization gap, LIME overlap, artifact export
run_experiment.py  # CLI: baselines + StyleGuard + ablations + LIME
paper/paper.tex    # full manuscript describing StyleGuard
```

## Reproduce

The repository ships pre-split data under `classical_ML_result/Split_data/`
(`train.csv` must point at the actual ISOT training splits; `val.csv`,
`test.csv`, and `megafake_ood_test.csv` are present).

```bash
pip install -r requirements.txt

# Baselines + StyleGuard fine-tuning + metrics + ablation + LIME analysis
python run_experiment.py \
    --data-dir classical_ML_result/Split_data \
    --ensemble-k 2 \
    --run-lime

# Fine-tune only the BiLSTM with custom regularization weights
python run_experiment.py --models bilstm --alpha 1.0 --beta 0.3 --epochs 15

# Semi-supervised target supervision sweep (p = 0, 10, 25, 50% of MegaFake labels)
# p=0 is the fully-unsupervised protocol; p>0 adds labeled MegaFake to training
# and evaluates on the untouched remainder.
python run_experiment.py \
    --data-dir classical_ML_result/Split_data \
    --sweep 0.1 0.25 0.5 \
    --run-lime
```

Results are written to `styleguard_results/`:
`styleguard_full_comparison_<seed>.csv` (ID + OOD metrics),
`styleguard_generalization_gap_<seed>.csv`, and
`styleguard_lime_overlap.csv`. In sweep mode the tables are
`styleguard_target_sweep_<seed>.csv` and carry a `Target_Frac` column.

## Paper

`paper/paper.tex` contains the full manuscript. Build with
`pdflatex`/`latexmk`. Result tables are populated by running
`run_experiment.py` (single command) so the paper reproduces exactly the
numbers it reports.
