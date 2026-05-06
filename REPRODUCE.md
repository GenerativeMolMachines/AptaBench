# Reproduce Paper Claims

Use this file to map each paper-facing table/figure to the notebook that generates it and the output artifact you should verify.

| Claim/Table/Figure | Notebook | Output file |
|---|---|---|
| Unified benchmark dataset | `notebooks/01_dataset_construction.ipynb` | `dataset/AptaBench_dataset.csv` |
| Dataset split definitions | `dataset/make_splits.py` | `dataset/splits/stratified.json` |
| Dataset distribution figure | `notebooks/02_data_exploration.ipynb` | `artifacts/figures/dataset/dataset_distributions.png` |
| Split strategy figure | `notebooks/02_data_exploration.ipynb` | `artifacts/figures/dataset/split_strategies.png` |
| Diversity violin figure | `notebooks/02_data_exploration.ipynb` | `artifacts/figures/dataset/dataset_violin_diversity.png` |
| ML screening table | `notebooks/03_ml_benchmark.ipynb` | `artifacts/ml/classification/screening/all_splits/results.csv` |
| ML regression table | `notebooks/03_ml_benchmark.ipynb` | `artifacts/ml/regression/results.csv` |
| DL screening summary | `notebooks/04_dl_model.ipynb` | `artifacts/dl/screening/comprehensive_screening_all_splits.csv` |
| DL leaderboard | `notebooks/04_dl_model.ipynb` | `artifacts/dl/screening/dl_leaderboard.csv` |
| DL ablation table | `notebooks/04_dl_model.ipynb` | `artifacts/dl/ablations/ablation_study_gena_unimol.csv` |

## Recommended flow

1. `python test_installation.py`
2. `python -m pytest -q`
3. `python scripts/reproduce_review.py --quick`
4. Compare your generated outputs with the table above.
