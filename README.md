# AptaBench

Benchmark for predicting aptamer–small molecule interactions.

## Dataset (hosted)

The released tabular benchmark is on Hugging Face:

- [https://huggingface.co/datasets/aptabench-anonymous/AptaBench_dataset](https://huggingface.co/datasets/aptabench-anonymous/AptaBench_dataset)

**Croissant metadata (core + RAI):** [dataset/croissant.json](dataset/croissant.json).

## Quickstart

```bash
# 0) From the repository root
cd AptaBench

# 1) Canonical environment (single source of truth)
conda env create -f environment.yml
conda activate aptabench

# 2) Install package in editable mode
pip install -e .

# 3) Smoke check (network required for Hugging Face pull)
python test_installation.py

# 4) Minimal health-check
python -m pytest -q

# 5) One-command quick reproduction (splits + key notebooks + output checks)
python scripts/reproduce_review.py --quick
# or
make reproduce-review
```

Additional installation files:
- `requirements.txt`: pip export for non-conda workflows (secondary).
- `envs/aptabench.yml`: compatibility mirror for previous setups (secondary).
- `environment.yml`: canonical file for reproducing the results.

### Optional encoders (heavy)

Some notebooks and encoders need extra packages (large downloads at import time):

```bash
pip install -e ".[full]"
# CLAMP (vendored): from repository root
pip install -e src/clamp
```

## Environment Compatibility

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.10.4 | Pinned in `environment.yml` |
| OS | Linux / macOS / Windows | `Makefile` commands are cross-platform via Python scripts |
| CUDA | Optional (12.4 build in canonical env) | Needed only for GPU acceleration |
| GPU | Optional | DL notebook runs faster on GPU, not required for quick reproduction |
| Network | Required for smoke check | `test_installation.py` fetches a small dataset split |

Pinned core versions for reproducibility (`environment.yml`): `numpy==2.2.6`, `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `rdkit==2025.9.6`.

## Repository layout

```
AptaBench/
├── dataset/           # Splits, Croissant metadata, local table copies (optional)
├── notebooks/         # Jupyter workflows
├── src/               # Library code (import as `src.*`)
├── artifacts/         # Canonical experiment outputs (ML, DL, figures, cache)
├── envs/              # Extra conda stubs (e.g. CLAMP)
└── environment.yml    # Same as canonical conda spec
```

Canonical output layout:

- `artifacts/ml/classification/screening/` - classification screening tables per split and combined
- `artifacts/ml/classification/checkpoints/` - incremental Optuna/checkpoint outputs for ML screening
- `artifacts/ml/regression/` - ML regression results
- `artifacts/dl/screening/` - DL screening results and model parameter summaries
- `artifacts/dl/ablations/` - DL ablation results
- `artifacts/dl/regression/` - DL regression outputs
- `artifacts/figures/` - dataset and model figures used in analysis/paper
- `artifacts/cache/embeddings/` - reusable embedding caches

## Notebooks

- `01_dataset_construction.ipynb` — dataset construction from public sources  
- `02_data_exploration.ipynb` — exploration  
- `03_ml_benchmark.ipynb` — classical ML benchmarks  
- `04_dl_model.ipynb` — deep learning models

## Notebook Responsibilities

- `notebooks/01_dataset_construction.ipynb`  
  Builds and cleans the unified benchmark dataset and writes:
  - `dataset/AptaBench_dataset.csv`
  - `dataset/AptaBench_dataset.parquet`
- `notebooks/02_data_exploration.ipynb`  
  Computes descriptive analysis and figures and writes:
  - `artifacts/figures/dataset/dataset_distributions.png`
  - `artifacts/figures/dataset/split_strategies.png`
  - `artifacts/figures/dataset/dataset_violin_diversity.png`
- `notebooks/03_ml_benchmark.ipynb`  
  Runs classical ML screening/regression baselines and writes:
  - `artifacts/ml/classification/screening/all_splits/results.csv`
  - `artifacts/ml/regression/results.csv`
- `notebooks/04_dl_model.ipynb`  
  Runs deep learning benchmarks/ablations and writes:
  - `artifacts/dl/screening/comprehensive_screening_all_splits.csv`
  - `artifacts/dl/screening/dl_leaderboard.csv`
  - `artifacts/dl/ablations/ablation_study_gena_unimol.csv`

## Expected outputs

After `python scripts/reproduce_review.py --quick`, verify these files exist:

- `dataset/splits/stratified.json`
- `dataset/splits/disjoint_aptamer.json`
- `dataset/splits/disjoint_molecule.json`
- `artifacts/ml/classification/screening/all_splits/results.csv`
- `artifacts/ml/regression/results.csv`
- `artifacts/dl/screening/comprehensive_screening_all_splits.csv`
- `artifacts/dl/screening/dl_leaderboard.csv`
- `artifacts/dl/ablations/ablation_study_gena_unimol.csv`
- `artifacts/figures/dataset/dataset_distributions.png`
- `artifacts/figures/dataset/split_strategies.png`

For paper-specific mapping, see `REPRODUCE.md`. For artifact layout, see `artifacts/README.md`.

## License

Source code in this repository is licensed under the MIT License — see [LICENSE](LICENSE). The **dataset** on Hugging Face is distributed under **CC BY 4.0** as stated on the dataset card (code license ≠ data license).
