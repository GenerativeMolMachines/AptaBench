# E8: Within-assay ranking evaluation

Self-contained entry point for Reviewer Q2dA / W1 within-assay ranking.
Run from this directory; the script locates the AptaBench repository root
automatically.

## Design

- Dataset: released `dataset/AptaBench_dataset.csv`.
- Only regression rows with an **extractable DOI** in `origin` participate.
  Rows without a DOI are excluded entirely (eligibility and metrics).
- **RSAPred** cannot enter: all RSAPred rows have missing `origin`, so no DOI
  can be extracted.
- Buffer is **not** used for filtering or batching.
- Batch key: `batch_id = normalized_doi` (lowercase DOI string from `origin`).
- Eligible batches (full regression subset, before split metrics): DOI present,
  at least 3 pKd measurements, at least 2 distinct pKd values.
- Predictions: out-of-fold headline LightGBM regression configs (same selection
  as E2). Prefer reuse of E2 `e2_regression_oof.csv` when present; otherwise
  regenerate OOF.
- Per eligible test batch (pooled OOF within each split): Spearman ρ,
  Kendall τ-b, pairwise concordance, record count.
- Summary per split: macro means, pooled pairwise concordance, bootstrap 95%
  CIs (1000 resamples over batches, seed 42), batch/record counts.

Splits evaluated: in-distribution (`stratified`), molecule-disjoint,
aptamer-disjoint.

## Reproduce

From this directory:

```bash
python run_experiment.py
```

Regenerate ranking tables from a saved OOF file without fitting models:

```bash
python run_experiment.py --analysis-only
```

Force rebuild of embeddings and OOF:

```bash
python run_experiment.py --force-oof
```

`--analysis-only` needs `results/oof/e8_regression_oof.csv` or a reusable
`../e2_dna_rna_rsapred/results/oof/e2_regression_oof.csv`. Full OOF rebuild
requires the optional pretrained encoder dependencies from the repository root
`README.md` / `environment.yml`.

Frozen fold assignments live in local `splits/`.

## Main outputs

Raw tables only (no interpretation):

- `results/e8_eligible_batches.csv` — DOI batches meeting size/diversity rules
- `results/e8_batch_metrics.csv` — per split × batch ranking metrics
- `results/e8_summary.csv` — macro / pooled metrics and bootstrap CIs
- `results/oof/e8_regression_oof.csv` — row-level regression OOF predictions
