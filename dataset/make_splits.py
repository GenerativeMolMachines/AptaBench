import os
import sys
import json
import numpy as np
import pandas as pd

# make src/ available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data.split import (
    stratified_group_splits,
    disjoint_aptamer_splits,
    disjoint_molecule_splits,
)

# ======================================================
# CONFIG
# ======================================================
DATASET_PATH = os.path.join(os.path.dirname(__file__), "AptaBench_dataset.csv")
N_SPLITS = 5
RANDOM_STATE = 42

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATASET_PATH)

# ======================================================
# MAKE OUTPUT DIR
# ======================================================
outdir = os.path.join(os.path.dirname(__file__), "splits")
os.makedirs(outdir, exist_ok=True)

# ======================================================
# HELPER TO SAVE SPLITS
# ======================================================
def save_splits(name, splits):
    outpath = os.path.join(outdir, f"{name}.json")
    data = []
    for i, (tr, va) in enumerate(splits):
        data.append({
            "fold": i,
            "train_idx": tr.tolist(),
            "val_idx": va.tolist(),
        })
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {name} splits -> {outpath}")


# ======================================================
# STRATIFIED
# ======================================================
splits = stratified_group_splits(df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
save_splits("stratified", splits)

# ======================================================
# DISJOINT APTAMER
# ======================================================
splits = disjoint_aptamer_splits(df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
save_splits("disjoint_aptamer", splits)

# ======================================================
# DISJOINT MOLECULE
# ======================================================
splits = disjoint_molecule_splits(df, n_splits=N_SPLITS, random_state=RANDOM_STATE)
save_splits("disjoint_molecule", splits)

print("All splits successfully saved in dataset/splits/")
