import json
from pathlib import Path


def test_released_splits_are_well_formed():
    """Validate committed paper folds; do not regenerate (avoids overwriting them)."""
    dataset_rows = sum(1 for _ in open("dataset/AptaBench_dataset.csv", encoding="utf-8")) - 1
    for name in ["stratified", "disjoint_aptamer", "disjoint_molecule"]:
        path = Path("dataset/splits") / f"{name}.json"
        assert path.exists(), f"missing split file: {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list) and len(payload) == 5
        all_test_indices = []
        for fold in payload:
            train = set(fold["train_idx"])
            test = set(fold["val_idx"])
            assert train.isdisjoint(test)
            assert train | test == set(range(dataset_rows))
            all_test_indices.extend(test)
        assert len(all_test_indices) == len(set(all_test_indices))
        assert len(all_test_indices) == dataset_rows
