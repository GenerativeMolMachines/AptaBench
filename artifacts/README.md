# Artifacts Map

Published **result tables** only. Figures are generated locally from tables/scripts.

## Directory roles

- `artifacts/ml/`: classical ML screening and regression CSVs
- `artifacts/dl/`: deep learning screening and ablation CSVs
- `artifacts/additional_experiments/`: rebuttal runners and summary CSVs
- `artifacts/figures/`: local plot outputs (gitignored); rebuild with notebooks / `scripts/plot_screening_heatmaps.py`

## Paper mapping (primary files)

| Paper item | Main file |
|---|---|
| ML screening summary | `artifacts/ml/classification/screening/all_splits/results.csv` |
| ML regression summary | `artifacts/ml/regression/results_reg.csv` |
| DL screening leaderboard | `artifacts/dl/screening/dl_leaderboard.csv` |
| DL comprehensive screening | `artifacts/dl/screening/comprehensive_screening_all_splits.csv` |
| DL ablation | `artifacts/dl/ablations/ablation_study_gena_unimol.csv` |
| E1 source ablation | `artifacts/additional_experiments/e1_source_ablation/results/e1_classification_fold_results.csv` |
| E1 v2 source ablation | `artifacts/additional_experiments/e1_v2/results/e1_v2_classification_fold_results.csv` |
| E4 harmonization audit | `artifacts/additional_experiments/e4_duplicate_conflict_audit/results/e4_harmonization_stats.csv` |
| E4 sensitivity | `artifacts/additional_experiments/e4_duplicate_conflict_audit/results/e4_sensitivity_fold_results.csv` |
| E5 threshold sensitivity | `artifacts/additional_experiments/e5_pkd_threshold_negative_sensitivity/results/e5_threshold_fold_results.csv` |
| E8 within-assay ranking | `artifacts/additional_experiments/e8_within_assay_ranking/results/e8_summary.csv` |
| E9 pharmacophore similarity | `artifacts/additional_experiments/e9_pharmacophore_similarity_diagnostic/results/e9_fold_similarity_summary.csv` |

See `REPRODUCE.md` and `REBUTTAL_RESULTS.md`.
