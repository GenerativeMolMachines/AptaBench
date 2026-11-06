# AptaBench rebuttal results

Tables below are the main quantitative responses used in the NeurIPS 2026 rebuttal
(reviewers **18Xi**, **gXd3**, and **Q2dA**). Full runners live under
`artifacts/additional_experiments/`; paper baselines are in [REPRODUCE.md](REPRODUCE.md).

---

## Paper baselines (Table 2)

| Task | In-distribution | Molecule-disjoint | Aptamer-disjoint |
|------|-----------------|-------------------|------------------|
| Classification ROC-AUC | 0.952 ± 0.002 | 0.886 ± 0.038 | 0.867 ± 0.044 |
| Regression R² | 0.668 ± 0.038 | 0.298 ± 0.191 | 0.323 ± 0.123 |

Sources: `artifacts/ml/classification/screening/all_splits/results.csv`,
`artifacts/ml/regression/results_reg.csv`.

---

## Source removal (18Xi Q1; gXd3 Q1)

### Reconstructed folds (response to 18Xi)

Fixed LightGBM configs; each source removed, then folds rebuilt.

| Dataset | In-distribution ROC-AUC / MCC | Molecule-disjoint | Aptamer-disjoint |
|---------|-------------------------------|-------------------|------------------|
| Full dataset | 0.952 ± 0.002 / 0.778 ± 0.006 | 0.886 ± 0.038 / 0.638 ± 0.063 | 0.867 ± 0.044 / 0.552 ± 0.111 |
| Without Manual | 0.947 ± 0.005 / 0.763 ± 0.027 | 0.851 ± 0.075 / 0.566 ± 0.102 | 0.851 ± 0.079 / 0.552 ± 0.137 |
| Without RSAPred | 0.946 ± 0.007 / 0.756 ± 0.013 | 0.838 ± 0.053 / 0.544 ± 0.084 | 0.851 ± 0.106 / 0.558 ± 0.196 |
| Without Specificity | 0.936 ± 0.012 / 0.666 ± 0.040 | 0.820 ± 0.057 / 0.469 ± 0.156 | 0.798 ± 0.041 / 0.225 ± 0.104 |

Reproduce: `python artifacts/additional_experiments/e1_source_ablation/run_experiment.py`  
Results: `artifacts/additional_experiments/e1_source_ablation/results/`

### Fixed original folds, without Specificity (response to gXd3)

| Task | In-distribution | Molecule-disjoint | Aptamer-disjoint |
|------|-----------------|-------------------|------------------|
| Classification ROC-AUC | 0.936 ± 0.020 | 0.750 ± 0.200 | 0.801 ± 0.029 |
| Regression R² | 0.686 ± 0.059 | 0.207 ± 0.230 | 0.284 ± 0.197 |

Reproduce: `python artifacts/additional_experiments/e1_v2/run_experiment.py`  
Results: `artifacts/additional_experiments/e1_v2/results/`

### DNA / RNA subgroups

| Protocol | DNA ROC-AUC | RNA ROC-AUC |
|----------|-------------|-------------|
| In-distribution | 0.938 ± 0.005 | 0.941 ± 0.017 |
| Molecule-disjoint | 0.806 ± 0.079 | 0.834 ± 0.033 |
| Aptamer-disjoint | 0.850 ± 0.098 | 0.784 ± 0.077 |

Reproduce: `python artifacts/additional_experiments/e2_dna_rna_rsapred/run_e2_subgroup_analysis.py`  
Results: `artifacts/additional_experiments/e2_dna_rna_rsapred/results/e2_dna_rna_summary.csv`

---

## Curated vs shuffled negatives (gXd3 Q2)

Matched positives/negatives, eight seeds.

| Protocol | Curated ROC-AUC | Shuffled ROC-AUC | Difference |
|----------|-----------------|------------------|------------|
| In-distribution | 0.948 ± 0.004 | 0.762 ± 0.014 | +0.186 |
| Molecule-disjoint | 0.837 ± 0.063 | 0.637 ± 0.047 | +0.200 |
| Aptamer-disjoint | 0.830 ± 0.055 | 0.590 ± 0.072 | +0.240 |

Without Specificity (650 balanced records):

| Protocol | Curated ROC-AUC | Shuffled ROC-AUC | Difference |
|----------|-----------------|------------------|------------|
| In-distribution | 0.898 ± 0.016 | 0.657 ± 0.033 | +0.242 |
| Molecule-disjoint | 0.815 ± 0.077 | 0.584 ± 0.093 | +0.231 |
| Aptamer-disjoint | 0.848 ± 0.052 | 0.424 ± 0.238 | +0.424 |

Reproduce: `python artifacts/additional_experiments/e3_curated_vs_shuffled_negatives/run_experiment.py`  
Results: `artifacts/additional_experiments/e3_curated_vs_shuffled_negatives/results/e3_curated_vs_shuffled_raw_results.csv`

---

## Harmonization / conflicts (18Xi; gXd3 Q3)

| Stage | Repeated groups | Cross-source groups | Conflicting binary labels |
|-------|-----------------|---------------------|---------------------------|
| Before harmonization | 305 | 197 | 36 |
| After harmonization | 208 | 128 (2.12% of 6,052 unique pairs) | 23 |

Collapsing classification to unique pairs changed mean ROC-AUC by **+0.0025** /
**−0.0020** / **−0.0131** (in-distribution / molecule-disjoint / aptamer-disjoint).

Reproduce: `python artifacts/additional_experiments/e4_duplicate_conflict_audit/run_experiment.py`  
Results: `e4_harmonization_stats.csv`, `e4_sensitivity_fold_results.csv`

---

## pKd threshold and negative subtypes (gXd3 Q4)

| Active definition | In-distribution ROC-AUC | Molecule-disjoint | Aptamer-disjoint |
|-------------------|------------------------|-------------------|------------------|
| pKd > 3 | 0.955 ± 0.002 | 0.871 ± 0.055 | 0.842 ± 0.086 |
| pKd > 4 | 0.946 ± 0.001 | 0.855 ± 0.056 | 0.837 ± 0.074 |
| pKd > 5 | 0.931 ± 0.003 | 0.829 ± 0.055 | 0.804 ± 0.038 |

Negative subtypes vs positives at pKd = 4 (AUC):

| Protocol | Explicit non-binding | Low affinity | Other reported inactive |
|----------|----------------------|--------------|-------------------------|
| In-distribution | 0.825 | 0.840 | 0.959 |
| Molecule-disjoint | 0.788 | 0.685 | 0.874 |
| Aptamer-disjoint | 0.742 | 0.640 | 0.868 |

Reproduce: `python artifacts/additional_experiments/e5_pkd_threshold_negative_sensitivity/run_experiment.py`  
Results: `e5_threshold_fold_results.csv`, `e5_negative_subtype_metrics.csv`

---

## RNA-native encoders (gXd3 Q5)

RNA-only classification subset with Morgan fingerprints.

| Aptamer representation | In-distribution ROC-AUC | Molecule-disjoint | Aptamer-disjoint |
|------------------------|------------------------|-------------------|------------------|
| Original RNA baseline | 0.941 ± 0.017 | 0.834 ± 0.033 | 0.784 ± 0.077 |
| RNA-FM | 0.928 ± 0.029 | 0.846 ± 0.055 | 0.753 ± 0.052 |
| RiNALMo-Mega | 0.931 ± 0.035 | 0.840 ± 0.065 | 0.748 ± 0.070 |

Reproduce: `python artifacts/additional_experiments/e6_rna_native_baselines/run_experiment.py`  
Results: `e6_rna_native_raw_results.csv`

---

## Equal-budget neural HPO (gXd3 Q6)

Same OneHot + MorganFP inputs; neural models: 15 Optuna trials per split.

| Protocol | LightGBM | TwoTowerMLP | CNN dual encoder | CrossAttention |
|----------|----------|-------------|------------------|----------------|
| In-distribution | 0.949 ± 0.003 | 0.944 ± 0.005 | 0.944 ± 0.003 | 0.945 ± 0.004 |
| Molecule-disjoint | 0.886 ± 0.038 | 0.861 ± 0.024 | 0.869 ± 0.038 | 0.865 ± 0.042 |
| Aptamer-disjoint | 0.843 ± 0.054 | 0.849 ± 0.056 | 0.868 ± 0.057 | 0.859 ± 0.062 |

Reproduce: `python artifacts/additional_experiments/e7_equal_budget_hpo/run_experiment.py`  
Results: `e7_classification_raw_results.csv`

---

## Within-study ranking (Q2dA W1)

DOI-grouped quantitative records; 33 studies / 544 pKd measurements.

| Protocol | Macro Spearman, all 33 | Macro Spearman, n≥10 | Pairwise concordance, n≥10 |
|----------|------------------------|----------------------|----------------------------|
| In-distribution | 0.124 | 0.498 | 0.686 |
| Molecule-disjoint | −0.095 | 0.291 | 0.614 |
| Aptamer-disjoint | −0.112 | 0.298 | 0.618 |

Reproduce: `python artifacts/additional_experiments/e8_within_assay_ranking/run_experiment.py`  
Results: `e8_summary.csv`, `e8_batch_metrics.csv`

---

## Pharmacophore similarity (Q2dA W2)

Max train–test ligand similarity.

| Protocol | Fingerprint | Median max similarity | Fraction ≥0.9 |
|----------|-------------|----------------------|---------------|
| In-distribution | Morgan | 1.000 | 68.7% |
| Molecule-disjoint | Morgan | 0.515 | 7.9% |
| In-distribution | Gobbi Pharm2D | 1.000 | 74.2% |
| Molecule-disjoint | Gobbi Pharm2D | 0.498 | 18.3% |

Reproduce: `python artifacts/additional_experiments/e9_pharmacophore_similarity_diagnostic/run_experiment.py`  
Results: `e9_fold_similarity_summary.csv`

---

## Multitask deep learning (18Xi)

Appendix multitask reference: ROC-AUC **0.913 ± 0.020** in-distribution,
**0.805 ± 0.044** molecule-disjoint, **0.762 ± 0.058** aptamer-disjoint
(`artifacts/dl/screening/dl_leaderboard.csv`).
