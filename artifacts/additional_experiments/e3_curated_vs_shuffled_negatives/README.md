# E3 — Curated vs shuffled negatives

This directory contains a reproducible E3 runner and raw seed-level results
only. Aggregation, statistical interpretation, and reviewer-ready tables are
intentionally not generated.

## Design

- datasets: Full AptaBench and AptaBench without Specificity;
- negatives: curated and shuffled cross-pairs in train only;
- test: shared curated test for paired comparison;
- seeds: 42–49;
- protocols: in-distribution, molecule-disjoint, aptamer-disjoint;
- for each scope/seed, keep all curated negatives and randomly sample the same
  number of positives;
- fold 0 of five folds;
- encoder pair: Kmer4 + MorganFP (1024 bits, radius 2);
- LightGBM: 500 trees, learning rate 0.05, 96 leaves, subsample 0.8,
  feature subsample 0.8, L2 0.1;
- metrics: ROC-AUC, PR-AUC, MCC.

Curated and shuffled variants use the same original-row assignment, aptamer
clusters, and molecule scaffolds in train/test. Shuffled negatives are built
only from train-positive entity pools, excluding known positive and test pairs.

## Run

From the repository root:

```bash
python artifacts/additional_experiments/e3_curated_vs_shuffled_negatives/run_experiment.py --force
```

Or directly from this directory:

```bash
python run_experiment.py --force
```

The script uses `dataset/AptaBench_dataset.csv` and the main repository modules,
so the AptaBench environment from the root `environment.yml` is required.

A full run includes 96 model fits. Aptamer clustering makes it long-running.
Results are written incrementally, so an interrupted run can be resumed with
the same command without `--force`.

Short check:

```bash
python run_experiment.py --only-scope without_specificity --only-split stratified --n-seeds 1 --force
```

## Result

The only saved artifact:

`results/e3_curated_vs_shuffled_raw_results.csv`

Each row corresponds to one seed, dataset scope, split protocol, and negative
type, and contains train/test sizes, class counts, ROC-AUC, PR-AUC, and MCC.
