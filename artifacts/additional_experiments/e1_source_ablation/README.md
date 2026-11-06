# E1 — Leave-one-source-out source ablation (reconstructed folds)

**Robustness check.** After removing a source, this variant rebuilds five folds
from scratch. It is valid for stability under new split construction, but Full
is not guaranteed to match Table 2 exactly because folds are regenerated.

For the primary rebuttal comparison against Table 2, use
`../e1_v2` (source removal inside the original fixed folds).

## Design

- remove one source, then rebuild five folds;
- protocols: in-distribution (`stratified`), molecule-disjoint, aptamer-disjoint;
- classification sources: RSAPred, Aptamer Base, AptaGen, UTexas, Ribocentre,
  AptaDB, Manual, Specificity;
- regression sources with pKd: RSAPred, Aptamer Base, AptaGen, UTexas, Manual,
  Specificity;
- fixed full-dataset headline LightGBM configs (no per-ablation HPO);
- classification encoders:
  - stratified: Kmer4 + CLAMP
  - disjoint_molecule: OneHot + MorganFP
  - disjoint_aptamer: Kmer4 + ChemBERTa
- regression encoders:
  - stratified: DNABERT2 + MolFormer
  - disjoint_molecule: Kmer4 + PhysChem
  - disjoint_aptamer: Kmer4 + CLAMP
- seed 42, five folds, StandardScaler per fold.

Fixed classification hyperparameters are stored in
`configs/e1_classification_best_params.json`. Regression uses the shared
headline defaults from the main screening pipeline.

## Run

From the repository root:

```bash
python artifacts/additional_experiments/e1_source_ablation/run_experiment.py
```

Or from this directory:

```bash
python run_experiment.py
```

Requires the AptaBench environment from the repository root `environment.yml`
(or the working Python 3.10 setup with the project packages). Embeddings are
cached under `artifacts/cache/embeddings/`.

Resume is supported: completed `results/<task>/scopes/<scope>/e1_fold_results.csv`
files are skipped unless `--force` is passed.

Short check:

```bash
python run_experiment.py --only-scope full --task classification --force
```

## Saved artifacts

| Path | Contents |
|---|---|
| `configs/e1_classification_best_params.json` | Fixed classification HPO winners |
| `results/e1_manifest.json` | Dataset hash, seeds, fixed params, package versions |
| `results/e1_sequence_similarity_graph.json` | Cached aptamer-similarity graph used for disjoint aptamer splits |
| `results/e1_existing_split_source_composition.csv` | Source ratios in the released full-data folds |
| `results/e1_existing_split_fold_composition.csv` | Dominant-source / balance diagnostics for released folds |
| `results/e1_classification_fold_results.csv` | Combined raw classification fold metrics |
| `results/e1_regression_fold_results.csv` | Combined raw regression fold metrics |
| `results/<task>/scopes/<scope>/e1_fold_results.csv` | Per-scope fold metrics written during the run |
| `splits/<scope>/{stratified,disjoint_molecule,disjoint_aptamer}.json` | Rebuilt fold indices (original dataset row ids) |

Each fold-result row includes metrics plus train/test sizes, positive fraction,
unique aptamers/molecules, and group counts. No summary or interpretation files
are written.
