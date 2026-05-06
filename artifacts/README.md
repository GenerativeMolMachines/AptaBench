# Artifacts Map

This folder is organized for reviewer-facing reproducibility.

## Directory roles

- `artifacts/final/`: final tables/figures used for paper claims.
- `artifacts/intermediate/`: auxiliary files generated during runs.
- `artifacts/ml/`: classical ML outputs (screening/regression).
- `artifacts/dl/`: deep learning outputs (screening/ablations/regression).
- `artifacts/figures/`: generated figures.
- `archive/`: legacy files that are not required for the main reviewer path.

## Paper mapping (primary files)

| Paper item | Main file |
|---|---|
| ML screening summary | `artifacts/ml/classification/screening/all_splits/results.csv` |
| ML regression summary | `artifacts/ml/regression/results.csv` |
| DL screening leaderboard | `artifacts/dl/screening/dl_leaderboard.csv` |
| DL comprehensive screening | `artifacts/dl/screening/comprehensive_screening_all_splits.csv` |
| DL ablation | `artifacts/dl/ablations/ablation_study_gena_unimol.csv` |
| Dataset distributions figure | `artifacts/figures/dataset/dataset_distributions.png` |
| Split strategy figure | `artifacts/figures/dataset/split_strategies.png` |

See `REPRODUCE.md` for detailed notebook-to-output mapping.
