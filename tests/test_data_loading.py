import pandas as pd
from pathlib import Path


def test_dataset_loads():
    dataset_path = Path("dataset/AptaBench_dataset.csv")
    assert dataset_path.exists(), "dataset/AptaBench_dataset.csv is missing"
    df = pd.read_csv(dataset_path)
    assert len(df) > 0
    for col in ["sequence", "canonical_smiles", "label"]:
        assert col in df.columns
