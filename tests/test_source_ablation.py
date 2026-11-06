import json

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    build_sequence_similarity_edges,
    sequence_groups_from_edges,
)
from src.models.source_ablation import (
    CLASSIFICATION_ENCODERS,
    _classification_scores,
    _regression_scores,
    load_classification_params,
    make_scope,
    scope_name,
    validate_split_bundle,
)


def _small_dataset():
    return pd.DataFrame(
        {
            "sequence": [f"ACGT{i}" for i in range(10)],
            "canonical_smiles": [f"C{i}" for i in range(10)],
            "label": [0, 1] * 5,
            "pKd_value": np.linspace(4.0, 8.0, 10),
            "source": ["A"] * 4 + ["B"] * 6,
        }
    )


def _valid_bundle(n_rows=10):
    groups = np.arange(n_rows)
    splits = []
    for fold in range(5):
        test = np.array([fold * 2, fold * 2 + 1])
        train = np.setdiff1d(np.arange(n_rows), test)
        splits.append((train, test))
    payload = {"splits": splits, "groups": groups}
    return {
        "stratified": payload,
        "disjoint_molecule": payload,
        "disjoint_aptamer": payload,
    }


def test_make_scope_removes_source_and_retains_original_indices():
    scoped = make_scope(_small_dataset(), "A")
    assert set(scoped["source"]) == {"B"}
    assert scoped["original_index"].tolist() == list(range(4, 10))
    assert scope_name("Aptamer Base") == "without_aptamer_base"


def test_validate_split_bundle_checks_coverage_and_group_disjointness():
    df = make_scope(_small_dataset(), None)
    validate_split_bundle(df, _valid_bundle(), removed_source=None)

    broken = _valid_bundle()
    train, test = broken["disjoint_aptamer"]["splits"][0]
    broken["disjoint_aptamer"]["groups"][train[0]] = broken[
        "disjoint_aptamer"
    ]["groups"][test[0]]
    with pytest.raises(ValueError, match="Group leakage"):
        validate_split_bundle(df, broken, removed_source=None)


def test_metric_edge_cases_are_explicitly_nan():
    classification = _classification_scores(
        np.ones(3, dtype=int),
        np.array([0.7, 0.8, 0.9]),
        np.ones(3, dtype=int),
    )
    assert np.isnan(classification["roc_auc"])
    assert np.isnan(classification["mcc"])
    assert classification["f1"] == 1.0

    regression = _regression_scores(
        np.array([1.0, 2.0, 3.0]),
        np.ones(3),
    )
    assert np.isnan(regression["pearson"])
    assert np.isnan(regression["spearman"])
    assert regression["rmse"] > 0


def test_load_classification_params_requires_each_headline_row(tmp_path):
    rows = []
    for split_mode, (aptamer, molecule) in CLASSIFICATION_ENCODERS.items():
        rows.append(
            {
                "split": split_mode,
                "aptamer_encoder": aptamer,
                "molecule_encoder": molecule,
                "best_params": json.dumps({"n_estimators": 5}),
            }
        )
    path = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    params = load_classification_params(path)
    assert set(params) == set(CLASSIFICATION_ENCODERS)
    assert all(value["n_estimators"] == 5 for value in params.values())


def test_cached_sequence_graph_preserves_induced_subset_components():
    reference = ["AAAA", "AAAT", "TTTT"]
    edges = build_sequence_similarity_edges(
        reference,
        threshold=0.75,
        n_jobs=1,
    )
    connected = sequence_groups_from_edges(
        ["AAAA", "AAAT"],
        reference,
        edges,
    )
    separated = sequence_groups_from_edges(
        ["AAAA", "TTTT"],
        reference,
        edges,
    )
    assert connected[0] == connected[1]
    assert separated[0] != separated[1]
