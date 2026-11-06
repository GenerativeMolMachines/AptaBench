# AptaBench rebuttal results

Additional analyses confirming that the main conclusions are robust to source composition, negative-label construction, and evaluation choices.

**Baselines.** Classification ROC-AUC is 0.952 ± 0.002 in-distribution, 0.886 ± 0.038 molecule-disjoint, and 0.867 ± 0.044 aptamer-disjoint. Regression R² is 0.668 ± 0.038, 0.298 ± 0.191, and 0.323 ± 0.123 under the same protocols.

**Source removal.** Dropping individual sources changes absolute scores, but disjoint protocols remain harder than in-distribution in every case. Removing Specificity has the largest effect on classification (ROC-AUC 0.936 / 0.750 / 0.801 on fixed folds) because it supplies most negatives; removing RSAPred mainly hurts molecule-disjoint regression coverage. The DNA/RNA split shows the same transfer gap (DNA 0.938 → 0.806 / 0.850; RNA 0.941 → 0.834 / 0.784).

**Curated vs shuffled negatives.** Curated inactive labels beat random cross-pairing under all three protocols (+0.19 to +0.24 ROC-AUC) and after removing Specificity (+0.23 to +0.42).

**Harmonization.** After canonicalization there are 6,052 unique pairs; 128 (2.1%) are cross-source and 23 retain conflicting binary labels. Collapsing to unique pairs changes ROC-AUC by at most ~0.01, so overlap does not explain the transfer gap.

**pKd threshold.** Relabeling actives at pKd cutoffs 3 / 4 / 5 shifts absolute ROC-AUC, but the transfer gap persists. Low-affinity negatives are harder than other reported inactives, especially under distribution shift.

**RNA-native encoders.** On the RNA-only subset, RNA-FM and RiNALMo-Mega match DNA-style baselines (~0.93 in-distribution) and still drop to ~0.75 aptamer-disjoint; they do not close the sequence-family gap.

**Equal-budget HPO.** With matched OneHot + MorganFP inputs and 15 neural HPO trials per split, LightGBM and tuned deep models are competitive; there is no universal ranking across protocols.

**Within-study ranking.** DOI-grouped ranking over 33 studies is weak in aggregate (macro Spearman ~0.12 / −0.10 / −0.11) but stronger in larger studies (n≥10: 0.50 / 0.29 / 0.30), consistent with assay heterogeneity.

**Pharmacophore similarity.** Molecule-disjoint splits cut median max Morgan similarity from 1.00 to 0.52 (fraction ≥0.9: 68.7% → 7.9%); Gobbi Pharm2D shows the same pattern. The protocol tests scaffold-level transfer, not complete pharmacophore novelty.
