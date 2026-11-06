# E1 v2 — Source ablation on original fixed folds

Primary rebuttal variant of E1. Removes each source from the **released**
train/test fold indices and retrains the headline configuration. Folds are
never rebuilt.

The earlier reconstructed-fold experiment remains at
`../e1_source_ablation` as a robustness check.

## Design

- load `dataset/AptaBench_dataset.csv` and verify its SHA-256;
- load the original paper split JSON indices:
  - `stratified` (in-distribution);
  - `disjoint_molecule`;
  - `disjoint_aptamer`;
- **Full**: use those indices unchanged;
- **without source S**: keep fold membership, drop rows with `source == S`
  from both train and test;
- retrain fixed headline LightGBM configs (no per-ablation HPO);
- classification encoders / expected Full ROC-AUC:
  - stratified: Kmer4 + CLAMP → 0.952 ± 0.002
  - disjoint_molecule: OneHot + MorganFP → 0.886 ± 0.038
  - disjoint_aptamer: Kmer4 + ChemBERTa → 0.867 ± 0.044
- regression encoders / expected Full R²:
  - stratified: DNABERT2 + MolFormer → 0.668 ± 0.038
  - disjoint_molecule: Kmer4 + PhysChem → 0.298 ± 0.191
  - disjoint_aptamer: Kmer4 + CLAMP → 0.323 ± 0.123

Small residual differences vs Table 2 can remain from LightGBM / embedding /
library versions, but Full is evaluated on the same fold indices as the paper.

## Run

From the repository root:

```bash
python artifacts/additional_experiments/e1_v2/run_experiment.py
```

Or from this directory:

```bash
python run_experiment.py
```

Resume skips completed `results/<task>/scopes/<scope>/e1_v2_fold_results.csv`
unless `--force` is passed.

Short check:

```bash
python run_experiment.py --only-scope full --task classification --force
```

## Saved artifacts

| Path | Contents |
|---|---|
| `configs/e1_v2_classification_best_params.json` | Fixed classification HPO winners |
| `splits/original/*.json` | Local copy of released paper fold indices |
| `splits/filtered/<scope>/*.json` | Filtered indices after source removal |
| `results/e1_v2_manifest.json` | Dataset hash, params, package versions |
| `results/e1_v2_classification_fold_results.csv` | Combined classification fold metrics |
| `results/e1_v2_regression_fold_results.csv` | Combined regression fold metrics |
| `results/<task>/scopes/<scope>/e1_v2_fold_results.csv` | Per-scope fold metrics |

No summary / interpretation tables are written.
