# E2: DNA/RNA and RSAPred subtype analysis

This directory is a self-contained entry point for reproducing E2. The runner
locates the AptaBench repository root automatically, so commands can be run
directly from this directory.

## Contents

- `run_e2_subgroup_analysis.py` — selects the best published ML configuration
  independently for each split, restores OOF predictions without repeating
  HPO, and generates all subgroup tables.
- `results/` — raw OOF predictions and metric tables without interpretation.

## Reproduce

From this directory:

```bash
python run_e2_subgroup_analysis.py
```

The full run reuses compatible embeddings from
`../../cache/embeddings/` when available. It requires the optional pretrained
encoder dependencies documented in the repository root `README.md`.

To regenerate tables from the saved OOF predictions without fitting models:

```bash
python run_e2_subgroup_analysis.py --analysis-only
```

Use `--force-oof` to rebuild embeddings and OOF predictions. Outputs are
written to `results/` by default; `--output-dir` can override that location.

## Main outputs

- `results/e2_dna_rna_fold_metrics.csv`
- `results/e2_dna_rna_summary.csv`
- `results/e2_rsapred_category_descriptives.csv`
- `results/e2_rsapred_subtype_metrics.csv`
- `results/oof/e2_classification_oof.csv`
- `results/oof/e2_regression_oof.csv`
