# Artifacts Map

Use this folder to inspect reproducibility artifacts.

## Directory roles

- `artifacts/final/`: final tables/figures used for paper claims.
- `artifacts/intermediate/`: auxiliary files generated during runs.
- `artifacts/ml/`: classical ML outputs (screening/regression).
- `artifacts/dl/`: deep learning outputs (screening/ablations/regression).
- `artifacts/figures/`: generated figures.
- `artifacts/metrics/`: supplementary analyses.
- `artifacts/additional_experiments/`: self-contained reviewer experiments
  (E1, E1 v2, E2, E3, E4, E5, E6, E7, E8, E9) with runners, README, and raw results.

## Paper mapping (primary files)

| Paper item | Main file |
|---|---|
| ML screening summary | `artifacts/ml/classification/screening/all_splits/results.csv` |
| ML regression summary | `artifacts/ml/regression/results_reg.csv` |
| DL screening leaderboard | `artifacts/dl/screening/dl_leaderboard.csv` |
| DL comprehensive screening | `artifacts/dl/screening/comprehensive_screening_all_splits.csv` |
| DL ablation | `artifacts/dl/ablations/ablation_study_gena_unimol.csv` |
| Dataset distributions figure | `artifacts/figures/dataset/dataset_distributions.png` |
| Split strategy figure | `artifacts/figures/dataset/split_strategies.png` |
| E1 source ablation (reconstructed folds) | `artifacts/additional_experiments/e1_source_ablation/results/e1_classification_fold_results.csv` |
| E1 v2 source ablation (fixed folds) | `artifacts/additional_experiments/e1_v2/results/e1_v2_classification_fold_results.csv` |
| E4 harmonization audit | `artifacts/additional_experiments/e4_duplicate_conflict_audit/results/e4_harmonization_stats.csv` |
| E4 deduplicated sensitivity | `artifacts/additional_experiments/e4_duplicate_conflict_audit/results/e4_sensitivity_fold_results.csv` |
| E5 pKd-threshold / negative-subtype sensitivity | `artifacts/additional_experiments/e5_pkd_threshold_negative_sensitivity/results/e5_threshold_fold_results.csv` |
| E8 within-assay ranking | `artifacts/additional_experiments/e8_within_assay_ranking/results/e8_summary.csv` |
| E9 pharmacophore similarity diagnostic | `artifacts/additional_experiments/e9_pharmacophore_similarity_diagnostic/results/e9_fold_similarity_summary.csv` |

See `REPRODUCE.md` for detailed notebook-to-output mapping and `REBUTTAL_RESULTS.md` for rebuttal headlines and reproduce commands.
