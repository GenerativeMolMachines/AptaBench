# Reproduce Paper Claims

Use this file to map each paper-facing table/figure to the notebook that generates it and the output artifact you should verify.

For a short summary of rebuttal analyses, see [REBUTTAL_RESULTS.md](REBUTTAL_RESULTS.md).

| Claim/Table/Figure | Notebook | Output file |
|---|---|---|
| Unified benchmark dataset | `notebooks/01_dataset_construction.ipynb` | `dataset/AptaBench_dataset.csv` |
| Dataset split definitions | `dataset/make_splits.py` | `dataset/splits/stratified.json` |
| Dataset distribution figure | `notebooks/02_data_exploration.ipynb` | `artifacts/figures/dataset/dataset_distributions.png` |
| Split strategy figure | `notebooks/02_data_exploration.ipynb` | `artifacts/figures/dataset/split_strategies.png` |
| Diversity violin figure | `notebooks/02_data_exploration.ipynb` | `artifacts/figures/dataset/dataset_violin_diversity.png` |
| ML screening table | `notebooks/03_ml_benchmark.ipynb` | `artifacts/ml/classification/screening/all_splits/results.csv` |
| ML regression table | `notebooks/03_ml_benchmark.ipynb` | `artifacts/ml/regression/results_reg.csv` |
| DL screening summary | `notebooks/04_dl_model.ipynb` | `artifacts/dl/screening/comprehensive_screening_all_splits.csv` |
| DL leaderboard | `notebooks/04_dl_model.ipynb` | `artifacts/dl/screening/dl_leaderboard.csv` |
| DL ablation table | `notebooks/04_dl_model.ipynb` | `artifacts/dl/ablations/ablation_study_gena_unimol.csv` |
| E1 leave-one-source-out ablation (reconstructed folds) | `artifacts/additional_experiments/e1_source_ablation/run_experiment.py` | `artifacts/additional_experiments/e1_source_ablation/results/e1_classification_fold_results.csv` |
| E1 v2 source ablation on original fixed folds | `artifacts/additional_experiments/e1_v2/run_experiment.py` | `artifacts/additional_experiments/e1_v2/results/e1_v2_classification_fold_results.csv` |
| E2 DNA/RNA and RSAPred subgroup analysis | `artifacts/additional_experiments/e2_dna_rna_rsapred/run_e2_subgroup_analysis.py` | `artifacts/additional_experiments/e2_dna_rna_rsapred/results/e2_dna_rna_summary.csv` |
| E4 duplicate/conflict audit and sensitivity | `artifacts/additional_experiments/e4_duplicate_conflict_audit/run_experiment.py` | `artifacts/additional_experiments/e4_duplicate_conflict_audit/results/e4_sensitivity_fold_results.csv` |
| E5 pKd-threshold and negative-subtype sensitivity | `artifacts/additional_experiments/e5_pkd_threshold_negative_sensitivity/run_experiment.py` | `artifacts/additional_experiments/e5_pkd_threshold_negative_sensitivity/results/e5_threshold_fold_results.csv` |
| E7 equal-budget optimization | `artifacts/additional_experiments/e7_equal_budget_hpo/run_experiment.py` | `artifacts/additional_experiments/e7_equal_budget_hpo/results/e7_classification_raw_results.csv` |
| E8 within-assay ranking evaluation | `artifacts/additional_experiments/e8_within_assay_ranking/run_experiment.py` | `artifacts/additional_experiments/e8_within_assay_ranking/results/e8_summary.csv` |
| E9 pharmacophore similarity diagnostic | `artifacts/additional_experiments/e9_pharmacophore_similarity_diagnostic/run_experiment.py` | `artifacts/additional_experiments/e9_pharmacophore_similarity_diagnostic/results/e9_fold_similarity_summary.csv` |

## E1 leave-one-source-out source ablation

Primary rebuttal variant (filter sources inside the original fixed folds):

```bash
cd artifacts/additional_experiments/e1_v2
python run_experiment.py
```

Robustness variant (remove source, then rebuild folds):

```bash
cd artifacts/additional_experiments/e1_source_ablation
python run_experiment.py
```

Raw fold-level outputs live under each experiment's `results/` directory.

## E2 subgroup analysis

Run the fixed best-per-split ML configurations without repeating HPO:

```bash
cd artifacts/additional_experiments/e2_dna_rna_rsapred
python run_e2_subgroup_analysis.py
```

The first run creates encoder caches and row-level OOF predictions. Later runs
reuse `results/oof/`; pass `--analysis-only` to require those files and
regenerate only the subgroup tables. The experiment directory contains its
runner, a dedicated README, and the following files under `results/`:

- `e2_dna_rna_fold_metrics.csv` and `e2_dna_rna_summary.csv`: all requested metrics;
- `e2_rsapred_category_descriptives.csv`: reconstructed source categories;
- `e2_rsapred_subtype_metrics.csv`: pooled OOF metrics for categories with at
  least 20 active and 20 inactive records.

## E4 duplicate/conflict audit

Run the audit and fixed Kmer4 + MorganFP LightGBM sensitivity from the
self-contained experiment directory:

```bash
cd artifacts/additional_experiments/e4_duplicate_conflict_audit
python run_experiment.py
```

Use `--audit-only` to regenerate only the raw audit tables. See the directory
`README.md` for design details and the list of raw outputs under `results/`.

## E5 pKd-threshold and negative-subtype sensitivity

Restore `negative_provenance`, relabel classification at pKd thresholds 3/4/5,
and evaluate fixed Kmer4 + MorganFP LightGBM on the released splits:

```bash
cd artifacts/additional_experiments/e5_pkd_threshold_negative_sensitivity
python run_experiment.py --force
```

Use `--annotations-only` to write provenance / label-count tables without
fitting. See the directory `README.md` for design details and raw outputs
under `results/` (`e5_threshold_fold_results.csv`, `e5_oof_threshold4.csv`,
`e5_negative_subtype_metrics.csv`).

## E8 within-assay ranking evaluation

Evaluate within-DOI ranking of headline regression OOF predictions:

```bash
cd artifacts/additional_experiments/e8_within_assay_ranking
python run_experiment.py
```

Use `--analysis-only` to rebuild tables from `results/oof/e8_regression_oof.csv`
(or a reusable E2 regression OOF). See the directory `README.md` for DOI-only
batch rules, RSAPred exclusion, and raw outputs under `results/`.

## E9 pharmacophore similarity diagnostic

Compute per-fold max Morgan and Gobbi Pharm2D Tanimoto similarity from each
test molecule to train molecules on the released molecule-disjoint and
in-distribution splits:

```bash
cd artifacts/additional_experiments/e9_pharmacophore_similarity_diagnostic
python run_experiment.py --force
```

Raw outputs under `results/`:

- `e9_test_molecule_max_similarity.csv`: one row per test molecule;
- `e9_fold_similarity_summary.csv`: fold-level mean/median/p90/p95/max and
  threshold fractions.

See the directory `README.md` for fingerprint parameters and CLI flags.

## E7 equal-budget classification and regression HPO

Run the controlled OneHot + MorganFP classification comparison and the
regression-specific robustness search from the self-contained experiment
directory:

```bash
cd artifacts/additional_experiments/e7_equal_budget_hpo
python run_experiment.py
```

Classification uses 15 Optuna trials for every model/split, folds 0-2 for the
search objective, and all five released folds for final evaluation. Deep
models are capped at 80 epochs with patience 10. Regression selects the three
lowest-RMSE encoder pairs per split from the frozen
`inputs/regression_leaderboard.csv` and gives every pair 20 trials over the
five released folds.

Only raw numeric outputs are written under `results/`:

- `e7_classification_raw_results.csv`
- `e7_regression_raw_results.csv`
- `e7_regression_best_per_split.csv`
- `e7_best_parameters.json`
- `e7_provenance.json`

See the directory `README.md` for design details, resume behavior, and smoke
flags.

## Recommended flow

1. `python test_installation.py`
2. `python -m pytest -q`
3. `python scripts/reproduce_review.py --quick`
4. Compare your generated outputs with the table above.
