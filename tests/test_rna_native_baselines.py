import numpy as np
import pandas as pd
import pytest
import torch

from src.encoders import aptamer_encoders
from src.models.rna_native_baselines import (
    load_task_splits,
    objective_direction,
    prepare_task_subsets,
    reviewer_table,
    save_task_splits,
    summarize_fold_results,
    transfer_gaps,
)


def test_mean_pool_excludes_bos_eos_tokens():
    hidden = torch.tensor(
        [[[100.0, 100.0], [1.0, 3.0], [5.0, 7.0], [200.0, 200.0]]],
    )
    pooled = aptamer_encoders._mean_pool_token_rows(hidden, [2])
    np.testing.assert_allclose(pooled.numpy(), [[3.0, 5.0]])


def test_rna_native_embedding_rejects_dna_alphabet():
    with pytest.raises(ValueError, match="unsupported symbols"):
        aptamer_encoders._prepare_rna_sequences(["ATGC"])


def test_prepare_rna_sequences_preserves_u():
    assert aptamer_encoders._prepare_rna_sequences(["aUgC"]) == ["AUGC"]


def test_prepare_task_subsets_filters_rna_and_pkd():
    frame = pd.DataFrame(
        {
            "sequence": ["AUGC", "ACGU", "ATGC"],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "type": ["RNA", "RNA", "DNA"],
            "label": [1, 0, 1],
            "pKd_value": [7.0, np.nan, 6.0],
        }
    )
    subsets = prepare_task_subsets(frame)
    assert subsets["classification"]["original_index"].tolist() == [0, 1]
    assert subsets["regression"]["original_index"].tolist() == [0]
    assert all("T" not in value for value in subsets["classification"]["sequence"])


def test_task_split_cache_round_trip_and_validation(tmp_path):
    subset = pd.DataFrame(
        {
            "sequence": ["AUGC", "UUAA", "CCGG", "AGUC"],
            "canonical_smiles": ["CCO", "CCN", "c1ccccc1", "O=C=O"],
            "label": [0, 1, 0, 1],
            "original_index": [10, 11, 12, 13],
        }
    )
    bundle = {}
    for mode in ("stratified", "disjoint_aptamer", "disjoint_molecule"):
        bundle[mode] = {
            "groups": np.arange(4),
            "splits": [
                (np.array([2, 3]), np.array([0, 1])),
                (np.array([0, 1]), np.array([2, 3])),
            ],
        }
    save_task_splits(subset, bundle, tmp_path, "classification")
    loaded = load_task_splits(subset, tmp_path, "classification")
    assert loaded is not None
    for mode in bundle:
        test_rows = np.concatenate([fold[1] for fold in loaded[mode]["splits"]])
        np.testing.assert_array_equal(np.sort(test_rows), np.arange(4))


def _fold_rows():
    rows = []
    values = {
        ("classification", "stratified", "Kmer4"): (0.80, 0.40),
        ("classification", "disjoint_aptamer", "Kmer4"): (0.60, 0.20),
        ("classification", "stratified", "RNA-FM"): (0.82, 0.42),
        ("classification", "disjoint_aptamer", "RNA-FM"): (0.70, 0.30),
        ("regression", "stratified", "Kmer4"): (0.40, 1.00),
        ("regression", "disjoint_aptamer", "Kmer4"): (0.10, 1.40),
        ("regression", "stratified", "RNA-FM"): (0.45, 0.90),
        ("regression", "disjoint_aptamer", "RNA-FM"): (0.25, 1.10),
    }
    for (task, split, encoder), metrics in values.items():
        for fold in range(2):
            row = {
                "task": task,
                "split": split,
                "fold": fold,
                "aptamer_encoder": encoder,
                "molecule_encoder": (
                    "MorganFP" if task == "classification" else "MolFormer"
                ),
                "n_train": 6,
                "n_test": 2,
            }
            if task == "classification":
                row.update(roc_auc=metrics[0], mcc=metrics[1])
            else:
                row.update(r2=metrics[0], rmse=metrics[1])
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_reviewer_schema_and_gap_orientation():
    summary = summarize_fold_results(_fold_rows())
    table = reviewer_table(summary)
    assert table.columns.tolist() == [
        "Task",
        "Split",
        "Aptamer encoder",
        "n",
        "ROC-AUC/R²",
        "MCC/RMSE",
    ]
    gaps = transfer_gaps(summary)
    rna_auc = gaps.loc[
        (gaps["task"] == "classification")
        & (gaps["aptamer_encoder"] == "RNA-FM")
        & (gaps["metric"] == "roc_auc")
    ].iloc[0]
    rna_rmse = gaps.loc[
        (gaps["task"] == "regression")
        & (gaps["aptamer_encoder"] == "RNA-FM")
        & (gaps["metric"] == "rmse")
    ].iloc[0]
    assert rna_auc["degradation_change_vs_kmer4"] == pytest.approx(-0.08)
    assert rna_rmse["degradation_change_vs_kmer4"] == pytest.approx(-0.20)


def test_hpo_objective_directions_are_explicit():
    assert objective_direction("classification") == "maximize"
    assert objective_direction("regression") == "minimize"
    with pytest.raises(ValueError, match="Unknown task"):
        objective_direction("other")
