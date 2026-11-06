# AptaBench: A Benchmark for Aptamer–Small Molecule Binding

[![Paper](https://img.shields.io/badge/NeurIPS%202026-Evaluations%20%26%20Datasets-blue)](https://neurips.cc/virtual/2026/poster/139036)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-AptaBench__dataset-yellow)](https://huggingface.co/datasets/GenerativeMolMachines/AptaBench_dataset)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

**Mariia Eremeyeva, Nikita Serov**

Published: 25 Sept 2026 · Last Modified: 25 Sept 2026  
**NeurIPS 2026 Evaluations and Datasets Track** (poster)

AptaBench is a curated, standardized, and leakage-aware benchmark for predicting aptamer–small-molecule binding. It integrates eight sources into **6,289** interaction pairs covering **1,610** DNA/RNA aptamers and **942** ligands, with experimentally reported actives/inactives, optional pKd affinities, and fixed in-distribution, molecule-disjoint, and aptamer-disjoint evaluation protocols.

## Links

| Resource | URL |
|---|---|
| Paper (NeurIPS 2026 poster) | https://neurips.cc/virtual/2026/poster/139036 |
| Dataset (Hugging Face) | https://huggingface.co/datasets/GenerativeMolMachines/AptaBench_dataset |
| Croissant metadata | [dataset/croissant.json](dataset/croissant.json) |
| Reproduce paper tables/figures | [REPRODUCE.md](REPRODUCE.md) |
| Rebuttal results (summary) | [REBUTTAL_RESULTS.md](REBUTTAL_RESULTS.md) |

## Citation

```bibtex
@inproceedings{eremeyeva2026aptabench,
  title     = {AptaBench: A Benchmark for Aptamer-Small Molecule Binding},
  author    = {Eremeyeva, Mariia and Serov, Nikita},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year      = {2026}
}
```

## Dataset

The released tabular benchmark (CSV/Parquet + fixed splits) is hosted on Hugging Face:

- https://huggingface.co/datasets/GenerativeMolMachines/AptaBench_dataset

Local copies used by the notebooks live under `dataset/` and match the Hugging Face release (SHA-256 of `dataset/AptaBench_dataset.csv` checked against the published file).

```python
from datasets import load_dataset

ds = load_dataset("GenerativeMolMachines/AptaBench_dataset", split="train")
```

Or load the in-repo table and released folds:

```python
import json
import pandas as pd

data = pd.read_csv("dataset/AptaBench_dataset.csv")
with open("dataset/splits/stratified.json") as f:
    splits = json.load(f)
```

## Quickstart

```bash
# 0) From the repository root
cd AptaBench

# 1) Canonical environment
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
- `environment.yml`: canonical file for reproducing the results.

### Optional encoders (heavy)

```bash
pip install -e ".[full]"
# CLAMP (vendored): from repository root
pip install -e src/clamp
```

## Environment compatibility

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.10.4 | Pinned in `environment.yml` |
| OS | Linux / macOS / Windows | `Makefile` commands are cross-platform via Python scripts |
| CUDA | Optional (12.4 build in canonical env) | Needed only for GPU acceleration |
| GPU | Optional | DL notebook runs faster on GPU, not required for quick reproduction |
| Network | Required for smoke check | `test_installation.py` fetches a small dataset split |

Pinned core versions (`environment.yml`): `numpy==2.2.6`, `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `rdkit==2025.9.6`.

## Repository layout

```
AptaBench/
├── dataset/                         # Released table, splits, Croissant metadata
├── notebooks/                       # Dataset construction, exploration, ML/DL benchmarks
├── src/                             # Library code (import as src.*)
├── artifacts/                       # Canonical paper outputs + rebuttal experiments
│   ├── ml/ dl/ figures/             # Main paper results
│   └── additional_experiments/      # E1–E9 rebuttal runners and raw results
├── data/                            # Raw and interim source tables used for construction
├── REPRODUCE.md                     # Paper claim → notebook/artifact map
├── REBUTTAL_RESULTS.md              # Short summary of rebuttal analyses
└── environment.yml                  # Canonical conda environment
```

Canonical paper outputs:

- `artifacts/ml/classification/screening/` — ML classification screening
- `artifacts/ml/regression/` — ML regression results
- `artifacts/dl/screening/` — DL screening and leaderboard
- `artifacts/dl/ablations/` — DL ablation results
- `artifacts/figures/` — figures used in the paper
- `artifacts/additional_experiments/` — self-contained rebuttal experiments

## Notebooks

- `01_dataset_construction.ipynb` — build the unified benchmark
- `02_data_exploration.ipynb` — descriptive analysis and figures
- `03_ml_benchmark.ipynb` — classical ML baselines
- `04_dl_model.ipynb` — deep learning baselines and ablations

## Expected outputs

After `python scripts/reproduce_review.py --quick`, verify:

- `dataset/splits/stratified.json`
- `dataset/splits/disjoint_aptamer.json`
- `dataset/splits/disjoint_molecule.json`
- `artifacts/ml/classification/screening/all_splits/results.csv`
- `artifacts/ml/regression/results_reg.csv`
- `artifacts/figures/dataset/dataset_distributions.png`
- `artifacts/figures/dataset/split_strategies.png`

Paper claim mapping: [REPRODUCE.md](REPRODUCE.md). Rebuttal summary: [REBUTTAL_RESULTS.md](REBUTTAL_RESULTS.md). Artifact layout: [artifacts/README.md](artifacts/README.md).

## License

Source code is licensed under the MIT License — see [LICENSE](LICENSE). The **dataset** on Hugging Face is distributed under **CC BY 4.0** as stated on the dataset card.
