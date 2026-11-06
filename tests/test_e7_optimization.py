import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "additional_experiments"
    / "e7_equal_budget_hpo"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from e7_core import (  # noqa: E402
    MODEL_NAMES,
    aggregate_folds,
    build_dl_model,
    evaluate_regression_fold,
    load_released_splits,
    select_top_regression_combinations,
    set_global_seed,
    suggest_classification_params,
)
from run_experiment import _optimize_exact  # noqa: E402


class FixedTrial:
    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_select_top_regression_combinations_uses_rmse_and_stable_ties():
    rows = []
    for split in ("stratified", "disjoint_molecule", "disjoint_aptamer"):
        rows.extend(
            [
                {
                    "split": split,
                    "aptamer_encoder": "B",
                    "molecule_encoder": "M",
                    "RMSE mean": 0.8,
                    "R2 mean": 0.1,
                },
                {
                    "split": split,
                    "aptamer_encoder": "A",
                    "molecule_encoder": "M",
                    "RMSE mean": 0.8,
                    "R2 mean": 0.9,
                },
                {
                    "split": split,
                    "aptamer_encoder": "C",
                    "molecule_encoder": "M",
                    "RMSE mean": 0.7,
                    "R2 mean": 0.2,
                },
                {
                    "split": split,
                    "aptamer_encoder": "D",
                    "molecule_encoder": "M",
                    "RMSE mean": 0.9,
                    "R2 mean": 0.95,
                },
            ]
        )
    selected = select_top_regression_combinations(pd.DataFrame(rows))
    for split in selected["split"].unique():
        subset = selected[selected["split"] == split]
        assert subset["aptamer_encoder"].tolist() == ["C", "A", "B"]
        assert subset["baseline_rank"].tolist() == [1, 2, 3]


def test_released_split_loader_requires_five_ordered_folds(tmp_path):
    records = []
    for fold in range(5):
        records.append({"fold": fold, "train_idx": [0, 1], "val_idx": [2, 3]})
    (tmp_path / "stratified.json").write_text(json.dumps(records), encoding="utf-8")
    folds = load_released_splits(tmp_path, "stratified")
    assert len(folds[:3]) == 3
    assert len(folds) == 5


def test_all_dl_models_have_binary_logit_shape():
    params = {
        "hidden_dim": 64,
        "n_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 4,
        "seq_len": 8,
        "kernel_size": 3,
        "n_heads": 2,
    }
    aptamer = torch.randn(4, 12)
    molecule = torch.randn(4, 10)
    for model_name in MODEL_NAMES:
        model = build_dl_model(model_name, 12, 10, params)
        assert model(aptamer, molecule).shape == (4,)


def test_search_spaces_contain_common_and_architecture_parameters():
    trial = FixedTrial()
    for model_name in MODEL_NAMES:
        params = suggest_classification_params(trial, model_name)
        assert {
            "learning_rate",
            "weight_decay",
            "dropout",
            "hidden_dim",
            "n_layers",
            "batch_size",
        }.issubset(params)
    assert "seq_len" in suggest_classification_params(trial, "CNNDualEncoder")
    assert "n_heads" in suggest_classification_params(trial, "CrossAttention")


def test_regression_fold_filters_missing_targets():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(30, 6)).astype(np.float32)
    targets = rng.normal(size=30)
    targets[[1, 4, 22]] = np.nan
    params = {
        "n_estimators": 10,
        "learning_rate": 0.05,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 2,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 1e-5,
        "reg_lambda": 1e-5,
    }
    result = evaluate_regression_fold(
        params,
        features,
        targets,
        np.arange(20),
        np.arange(20, 30),
        seed=42,
    )
    assert result["n_train"] == 18
    assert result["n_val"] == 9
    assert np.isfinite(result["rmse"])


def test_seed_and_fold_aggregation_are_deterministic():
    set_global_seed(42)
    first = torch.rand(3)
    set_global_seed(42)
    second = torch.rand(3)
    assert torch.equal(first, second)
    result = aggregate_folds(
        [{"roc_auc": 0.6, "mcc": 0.2}, {"roc_auc": 0.8, "mcc": 0.4}],
        ("roc_auc", "mcc"),
    )
    assert result["roc_auc_mean"] == 0.7
    assert np.isclose(result["mcc_std"], 0.1)


def test_exact_budget_resume_does_not_repeat_completed_trials():
    study = optuna.create_study(direction="maximize")
    calls = []

    def objective(trial):
        calls.append(trial.number)
        return float(trial.number)

    _optimize_exact(study, objective, 2)
    _optimize_exact(study, objective, 2)
    assert calls == [0, 1]
    assert len(study.trials) == 2
