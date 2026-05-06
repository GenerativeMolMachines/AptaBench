import os
import json
import numpy as np
import pandas as pd

from src.data.split import (
    stratified_group_splits,
    disjoint_aptamer_splits,
    disjoint_molecule_splits,
)

# ======================================================
# CONFIG
# ======================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.abspath(os.getcwd()) != REPO_ROOT:
    raise RuntimeError(
        "Run dataset/make_splits.py from repository root: "
        f"expected cwd={REPO_ROOT}, got cwd={os.path.abspath(os.getcwd())}"
    )

DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "AptaBench_dataset.csv")
N_SPLITS = 5
RANDOM_STATE = 42
FRAC_TOLERANCE = 0.02  # acceptable abs deviation from expected test fraction
MAX_TRIES = 200  # retry different seeds to meet ratio

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATASET_PATH)

# ======================================================
# MAKE OUTPUT DIR
# ======================================================
outdir = os.path.join(REPO_ROOT, "dataset", "splits")
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
splits = stratified_group_splits(
    df,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    frac_tolerance=FRAC_TOLERANCE,
    max_tries=MAX_TRIES,
)
save_splits("stratified", splits)

# ======================================================
# DISJOINT APTAMER
# ======================================================
splits = disjoint_aptamer_splits(
    df,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    frac_tolerance=FRAC_TOLERANCE,
    max_tries=MAX_TRIES,
)
save_splits("disjoint_aptamer", splits)

# ======================================================
# DISJOINT MOLECULE
# ======================================================
splits = disjoint_molecule_splits(
    df,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    scaffold_missing="separate",
    frac_tolerance=FRAC_TOLERANCE,
    max_tries=MAX_TRIES,
)
save_splits("disjoint_molecule", splits)

print("All splits successfully saved in dataset/splits/")
