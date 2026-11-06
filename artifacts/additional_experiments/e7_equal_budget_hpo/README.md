# E7 — Equal-budget DL HPO and regression-specific HPO

Self-contained entry point for reproducing Reviewer gXd3 W2/Q6. The runner
locates the AptaBench repository root automatically, so commands can be run
from this directory.

Only raw numeric outputs are written. Interpretation is left to the user.

## Design

### E7.1 Classification equal-budget comparison

- Features: OneHot + MorganFP for every model
- Models: TwoTowerMLP, CNNDualEncoder, CrossAttention
- LightGBM excluded here (already tuned in the main screening); MultiTask excluded
- Budget per model × split:
  - 15 Optuna trials
  - search on folds 0–2
  - final evaluation of the winner on all 5 fixed folds
  - objective: mean validation ROC-AUC
  - max 80 epochs, early stopping patience 10
  - shared seed `42`
- Compute columns: Optuna search wall-time and final five-fold refit wall-time

### E7.2 Regression HPO robustness

- Top-3 encoder pairs per split from
  `inputs/regression_leaderboard.csv` by lowest `RMSE mean`
- 20 Optuna trials of LightGBMRegressor on the same five fixed folds
- objective: mean RMSE
- Compact best-per-split table keeps only the tuned winner

## Contents

- `run_experiment.py` — CLI entry point
- `e7_core.py` — models and fold evaluation
- `inputs/regression_leaderboard.csv` — frozen baseline regression grid
- `results/` — raw outputs
- `studies/` — Optuna SQLite studies for resume

Released data used from the repository root:

- `dataset/AptaBench_dataset.csv`
- `dataset/splits/*.json`
- `artifacts/cache/embeddings/` (E7 also writes `e7_*` feature caches there)

## Reproduce

From this directory (Python 3.10 + root `environment.yml` deps):

```powershell
$env:PYTHONUTF8='1'
python run_experiment.py
```

Or from the repository root:

```bash
python artifacts/additional_experiments/e7_equal_budget_hpo/run_experiment.py
```

Resume is incremental. Completed Optuna studies keep their trial budget.
`--force` recomputes final fold outputs only.

Smoke checks:

```bash
python run_experiment.py --task classification --only-split stratified --only-model TwoTowerMLP --classification-trials 2 --max-epochs 5
python run_experiment.py --task regression --only-split stratified --regression-trials 2
```

## Raw outputs

- `results/e7_classification_raw_results.csv`
- `results/e7_classification_fold_results.csv`
- `results/e7_regression_raw_results.csv`
- `results/e7_regression_best_per_split.csv`
- `results/e7_regression_selected_baselines.csv`
- `results/e7_regression_fold_results.csv`
- `results/e7_best_parameters.json`
- `results/e7_provenance.json`
