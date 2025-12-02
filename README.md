# AptaBench 🧬

Benchmark for predicting aptamer-small molecule interactions.

## Quick Installation

```bash
# Clone repository
git clone https://github.com/your-username/AptaBench.git
cd AptaBench

# Create environment (conda recommended)
conda env create -f environment.yml
conda activate aptabench

# Install project
pip install -e .

# Verify installation
python test_installation.py
```

## Requirements

- Python >= 3.8
- 8+ GB RAM
- GPU optional (for DL models)
```


## Structure

```
AptaBench/
├── dataset/              # Data and splits
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data/            # Data processing
│   ├── encoders/        # Aptamer and molecule encoders
│   └── models/          # ML/DL models
└── artifacts/           # Results
```

## Notebooks

- `01_dataset_construction.ipynb` - Dataset construction
- `02_data_exploration.ipynb` - Data exploration
- `03_ml_benchmark.ipynb` - ML benchmarks
- `04_dl_model.ipynb` - DL models


