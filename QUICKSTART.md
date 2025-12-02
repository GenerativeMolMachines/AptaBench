# AptaBench Quick Start ⚡

## 🚀 Installation (5 minutes)

```bash
# 1. Clone and navigate
git clone https://github.com/your-username/AptaBench.git
cd AptaBench

# 2. Create environment (choose one)
conda env create -f environment.yml && conda activate aptabench
# or
python -m venv venv && source venv/bin/activate

# 3. Install
pip install -r requirements.txt
pip install -e .

# 4. Verify
python test_installation.py
```

## 📊 First Run

```python
import pandas as pd
from src.models.screening import precompute_and_save_embeddings, screen_lgbm_optuna
from src.encoders.aptamer_encoders import kmer_freq_with_type_bit
from src.encoders.molecule_encoders import morgan_fp_with_validity

# Load data
df = pd.read_csv("dataset/AptaBench_dataset_v2.csv")

# Configure encoders
apt_cfgs = [{"name": "kmer6", "func": kmer_freq_with_type_bit, "kwargs": {"k": 6}}]
mol_cfgs = [{"name": "morgan2048", "func": morgan_fp_with_validity, "kwargs": {"radius": 2, "n_bits": 2048}}]

# Run
apt_map, mol_map, _ = precompute_and_save_embeddings(df, apt_cfgs, mol_cfgs)
results = screen_lgbm_optuna(df, apt_cfgs, mol_cfgs, n_trials=5)
```

## 📓 Jupyter Notebooks

```bash
jupyter notebook
# Open: notebooks/03_ml_benchmark.ipynb
```

## 🐛 Issues?

**RDKit:** `conda install -c conda-forge rdkit`  
**GPU:** `pip install torch --index-url https://download.pytorch.org/whl/cu118`  
**Memory:** Reduce `n_trials` and `batch_size`

