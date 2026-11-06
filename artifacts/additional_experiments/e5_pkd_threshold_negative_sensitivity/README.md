# E5 — Sensitivity to pKd threshold and negative subtype

This directory contains a reproducible E5 runner and raw numeric outputs only.
Aggregation, reviewer-ready tables, and interpretation are intentionally not
generated.

Response to Reviewer gXd3 Q4.

## Design

### Negative provenance restoration

The released table does not store `negative_provenance`. It is reconstructed
on `dataset/AptaBench_dataset.csv` for rows with `label == 0`:

| Rule | Value |
|---|---|
| `source == Specificity` | `other_reported_inactive` |
| missing `pKd_value` | `explicit_nonbinding` |
| otherwise (numeric pKd, not Specificity) | `low_affinity` |

Positives keep `negative_provenance` empty.

### E5.1 Threshold sensitivity

Three classification labelings:

- pKd threshold = 3
- pKd threshold = 4
- pKd threshold = 5

Relabel rules:

- `explicit_nonbinding` and `other_reported_inactive` stay negative at every
  threshold;
- affinity measurements with numeric pKd are labeled by the chosen threshold
  (`pKd <= threshold` → negative);
- rows without pKd that have experimentally confirmed binding keep their
  original positive label;
- released splits are reused without rebuilding
  (`dataset/splits/{stratified,disjoint_molecule,disjoint_aptamer}.json`).

Model setup (same as E3/E4):

- encoder pair: Kmer4 + MorganFP (1024 bits, radius 2);
- LightGBM: 500 trees, learning rate 0.05, 96 leaves, subsample 0.8,
  feature subsample 0.8, L2 0.1;
- seed 42, five folds;
- metrics: ROC-AUC, MCC.

### E5.2 Negative subtype difficulty

Using OOF predictions at the main threshold = 4, compute per split and
negative subtype:

- `n`
- false-positive rate (`y_pred == 1` at score threshold 0.5)
- mean predicted probability
- median predicted probability
- ROC-AUC of the subtype negatives against all positives in the same OOF split

## Run

From this directory (Python 3.10 + root `environment.yml` deps):

```bash
python run_experiment.py --force
```

Or from the repository root:

```bash
python artifacts/additional_experiments/e5_pkd_threshold_negative_sensitivity/run_experiment.py --force
```

Annotations / label counts only:

```bash
python run_experiment.py --annotations-only
```

Smoke check:

```bash
python run_experiment.py --only-split stratified --only-threshold 4 --force
```

Results are written incrementally. Without `--force`, completed
`(threshold, split)` fold blocks are skipped. OOF / subtype files are reused
when present unless `--force` is set.

## Inputs

- `dataset/AptaBench_dataset.csv`
- `dataset/splits/*.json`
- repository modules under `src/` (`FeatureCache`, encoders)

## Raw outputs

Only raw files are saved under `results/`:

- `e5_row_annotations.csv` — `row_idx`, provenance, original label, and
  `label_t3` / `label_t4` / `label_t5`
- `e5_threshold_label_counts.csv` — positives / negatives per threshold
- `e5_threshold_fold_results.csv` — fold-level ROC-AUC / MCC for every
  threshold × split × fold
- `e5_oof_threshold4.csv` — row-level OOF at threshold 4
- `e5_negative_subtype_metrics.csv` — subtype FPR / scores / ROC-AUC
