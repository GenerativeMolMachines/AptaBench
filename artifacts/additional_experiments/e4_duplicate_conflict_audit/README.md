# E4 — Duplicate/conflict audit and collapsed sensitivity

This directory contains a reproducible E4 runner and raw results only.
Aggregation, reviewer-ready tables, example selection, and textual
interpretation are intentionally not generated.

## Design

- pre-harmonization: concatenation of the eight `data/interim/*_processed.csv`
  tables before the final global deduplication;
- post-harmonization: released `dataset/AptaBench_dataset.csv`;
- canonical pair: `(sequence, canonical_smiles)`;
- classification collapse: identical labels → one record; conflicting labels →
  exclude; `source`/`origin`/`buffer` kept only as provenance;
- regression collapse (available in code when needed; not written to raw
  outputs): median of valid pKd values, `n_measurements`, `pKd_range`; label
  conflicts do not exclude valid pKd groups;
- sensitivity: current measurement-level vs collapsed unique pairs;
- protocols: in-distribution, molecule-disjoint, aptamer-disjoint;
- fold control: measurement-level folds are frozen first (matching the released
  benchmark splits under `dataset/splits/`); each collapsed pair is assigned to
  the same fold that held its pair group in the measurement-level data, so the
  comparison isolates collapsing rather than re-splitting;
- seed 42, 5 folds;
- encoder pair: Kmer4 + MorganFP (1024 bits, radius 2);
- LightGBM: 500 trees, learning rate 0.05, 96 leaves, subsample 0.8,
  feature subsample 0.8, L2 0.1;
- metrics: ROC-AUC, MCC.

### Multi-pKd definitions

The harmonization summary reports both counts:

- `n_multi_pkd_measurement_groups`: groups with more than one valid pKd row
  (values may be identical);
- `n_multi_pkd_unique_value_groups`: groups with more than one distinct pKd
  value.

pKd-range bins (`<=0.5`, `0.5-1.0`, `>1.0`) are counted only among
unique-value multi-pKd groups, so identical repeated measurements do not
inflate the `<=0.5` bin.

The pre-stage is not a full raw-ingestion audit: some upstream raw files are
unavailable in the repository.

## Run

From the repository root:

```bash
python artifacts/additional_experiments/e4_duplicate_conflict_audit/run_experiment.py --force
```

Or directly from this directory:

```bash
python run_experiment.py --force
```

The script uses `dataset/AptaBench_dataset.csv`, `data/interim/`, and the main
repository modules, so the AptaBench environment from the root
`environment.yml` is required.

Results are written incrementally, so an interrupted run can be resumed with
the same command without `--force`.

Audit only, without LightGBM:

```bash
python run_experiment.py --audit-only
```

Short check on one split:

```bash
python run_experiment.py --only-split stratified --force
```

Rebuild projected collapsed folds from the frozen measurement-level splits:

```bash
python run_experiment.py --generate-splits-only
```

Regenerate measurement-level splits from scratch (slow for aptamer-disjoint):

```bash
python run_experiment.py --generate-splits-only --force-splits
```

## Results

Only raw files are saved:

- `results/e4_harmonization_stats.csv` — before/after counts for rows, unique
  pairs, repeat groups, cross-source groups, conflicting labels, both
  multi-pKd definitions, and unique-value pKd-range bins;
- `results/e4_pair_group_audit.csv` — per pre-harmonization pair: sources,
  labels, pKd values, `n_measurements`, `n_unique_pkd`, range, origins/buffers,
  and applied rule;
- `results/e4_sensitivity_fold_results.csv` — fold-level ROC-AUC/MCC for two
  dataset versions × three splits, with `split_provenance` marking projected
  collapsed folds;
- `splits/current/*.json` — frozen measurement-level fold assignments;
- `splits/collapsed/*.json` — collapsed folds projected from the corresponding
  current pair-fold map (`provenance=projected_from_current_pair_folds`).
