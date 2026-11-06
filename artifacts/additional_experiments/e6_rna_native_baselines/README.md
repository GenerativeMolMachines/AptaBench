# E6 — RNA-native vs DNA-style baselines (shared fingerprints)

This directory contains a reproducible E6 runner and raw fold-level results
only. Aggregation, transfer-gap interpretation, and reviewer-ready tables are
intentionally not generated.

## Design (fair comparison)

All aptamer encoders are evaluated on the **same RNA-only pairs** with the
**same molecule fingerprint**, so DNA-style vs RNA-native differences are not
confounded by subset or molecule representation.

- subset: RNA only (`type == "RNA"`); regression further requires non-null
  `pKd_value`;
- molecule encoder: **MorganFP** for both classification and regression;
- DNA-style aptamer encoders (U→T): Kmer4, DNABERT2;
- RNA-native aptamer encoders (U preserved): RNA-FM, RiNALMo-Mega;
- protocols: in-distribution, aptamer-disjoint, molecule-disjoint;
- splits are rebuilt on each task subset (full-dataset indices are not reused);
- LightGBM + Optuna: 20 trials for each task × split × aptamer encoder;
- classification HPO metric: ROC-AUC (maximize);
- regression HPO metric: RMSE (minimize);
- seed: 42, 5 folds.

## Dependencies

Use the AptaBench environment from the root `environment.yml` / `pyproject.toml`,
then install the official RNA encoders.

### RNA-FM (official `fm` package)

```bash
pip install rna-fm
```

Weights are loaded via `fm.pretrained.rna_fm_t12(...)`. If the CUHK download
endpoint is blocked, the runner fetches the same official checkpoint from
Hugging Face `cuhkaih/rnafm` (`RNA-FM_pretrained.pth`).

### RiNALMo (official lbcb-sci package, Mega / 150M)

```bash
git clone https://github.com/lbcb-sci/RiNALMo
cd RiNALMo
pip install .
pip install einops ml_collections gdown
# optional CUDA speedup; not required
# pip install flash-attn==2.3.2
```

This experiment keeps a local editable clone under
`artifacts/cache/RiNALMo` with small Windows/CUDA compatibility patches
(optional flash-attn fallback + non-JIT RoPE). Mega weights are downloaded
from Zenodo (`rinalmo_mega_pretrained.pt`) on first use.

DNABERT2 uses the existing AptaBench Hugging Face integration
(`zhihan1996/DNABERT-2-117M`).

## Run

From the repository root:

```bash
python artifacts/additional_experiments/e6_rna_native_baselines/run_experiment.py
```

Or directly from this directory:

```bash
python run_experiment.py
```

DNA-style only / RNA-native only:

```bash
python run_experiment.py --only-encoder Kmer4 --only-encoder DNABERT2
python run_experiment.py --only-encoder RNA-FM --only-encoder RiNALMo-Mega
```

A full run includes 24 HPO studies × 20 trials
(2 tasks × 3 splits × 4 encoders). Results are written incrementally.

Short check:

```bash
python run_experiment.py --only-encoder Kmer4 --only-split stratified --n-trials 2 --task classification --force
```

## Result

Main artifact:

`results/e6_rna_native_raw_results.csv`

Each row is one fold for one task / split / aptamer encoder combination.
Columns include fold sizes, fold metrics (`roc_auc`/`mcc` or `r2`/`rmse`),
`encoder_family` (`dna_style` or `rna_native`), completed Optuna trial count,
and `best_params`.

For reproducibility the following are also saved:

- `splits/{classification,regression}/{stratified,disjoint_aptamer,disjoint_molecule}.json`
- `studies/*.log` — Optuna journal checkpoints for resume;
- `run_manifest.json` — dataset hash, subset sizes, encoder provenance;
- embeddings cache defaults to `../../cache/e6_embeddings/`.
