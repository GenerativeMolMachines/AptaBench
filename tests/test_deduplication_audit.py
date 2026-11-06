import numpy as np
import pandas as pd

from src.models.deduplication_audit import (
    add_pair_ids,
    build_pair_group_audit,
    collapse_classification,
    collapse_regression,
    generate_splits,
    pair_fold_assignment,
    project_splits_by_pair_folds,
    restrict_to_pair_universe,
    select_representative_examples,
    sensitivity_deltas,
    stable_pair_id,
    summarize_sensitivity,
    summarize_stage,
    validate_splits,
)


def _row(sequence, smiles, label, pkd, source, origin="", buffer=""):
    return {
        "type": "DNA",
        "sequence": sequence,
        "canonical_smiles": smiles,
        "pKd_value": pkd,
        "label": label,
        "buffer": buffer,
        "origin": origin,
        "source": source,
    }


def _audit_dataset():
    return pd.DataFrame(
        [
            _row("AAAA", "C", 1, 5.0, "A", "paper-1", "buffer-1"),
            _row("AAAA", "C", 1, 5.5, "B", "paper-2", "buffer-2"),
            _row("CCCC", "CC", 0, 4.0, "A"),
            _row("CCCC", "CC", 1, 6.0, "A"),
            _row("GGGG", "CCC", 1, 7.0, "A"),
            _row("GGGG", "CCC", 1, 7.0, "A"),
            _row("TTTT", "CCCC", 1, 4.0, "A"),
            _row("TTTT", "CCCC", 1, 5.0001, "A"),
            _row("ACAC", "N", 0, np.nan, "A"),
        ]
    )


def test_pair_audit_detects_duplicates_conflicts_and_provenance():
    audit = build_pair_group_audit(_audit_dataset())
    cross_source = audit.loc[audit["sequence"] == "AAAA"].iloc[0]
    conflict = audit.loc[audit["sequence"] == "CCCC"].iloc[0]

    assert cross_source["n_records"] == 2
    assert cross_source["sources"] == "A | B"
    assert cross_source["pkd_range"] == 0.5
    assert cross_source["origins"] == "paper-1 | paper-2"
    assert bool(cross_source["cross_source"])
    assert bool(conflict["conflicting_labels"])
    assert stable_pair_id("AAAA", "C") == stable_pair_id("AAAA", "C")
    assert stable_pair_id("AAAA", "C") != stable_pair_id("AAAA", "CC")


def test_stage_summary_uses_non_overlapping_pkd_range_bins():
    summary = summarize_stage(_audit_dataset(), stage="before").iloc[0]

    assert summary["n_rows"] == 9
    assert summary["n_unique_canonical_pairs"] == 5
    assert summary["n_repeat_pair_groups"] == 4
    assert summary["n_conflicting_label_groups"] == 1
    # GGGG has two identical pKd rows: multi-measurement but not unique-value.
    assert summary["n_multi_pkd_measurement_groups"] == 4
    assert summary["n_multi_pkd_unique_value_groups"] == 3
    assert summary["n_pkd_range_le_0_5"] == 1
    assert summary["n_pkd_range_gt_0_5_le_1_0"] == 0
    assert summary["n_pkd_range_gt_1_0"] == 2


def test_collapsed_folds_are_projected_from_measurement_level_pairs():
    current = add_pair_ids(
        pd.DataFrame(
            [
                _row("AAAA", "C", 1, 5.0, "A"),
                _row("AAAA", "C", 1, 5.5, "B"),
                _row("CCCC", "CC", 0, 3.0, "A"),
                _row("GGGG", "CCC", 1, 7.0, "A"),
                _row("TTTT", "CCCC", 0, 2.0, "A"),
                _row("ACAC", "N", 1, 6.0, "A"),
                _row("AGAG", "NN", 0, 2.5, "A"),
                _row("ATAT", "NO", 1, 6.5, "A"),
            ]
        )
    )
    # Artificial two-fold assignment on measurement-level rows.
    current_splits = [
        (np.array([2, 3, 4, 5]), np.array([0, 1, 6, 7])),
        (np.array([0, 1, 6, 7]), np.array([2, 3, 4, 5])),
    ]
    pair_to_fold = pair_fold_assignment(current, current_splits)
    assert pair_to_fold[stable_pair_id("AAAA", "C")] == 0
    assert pair_to_fold[stable_pair_id("CCCC", "CC")] == 1

    collapsed = collapse_classification(current)
    projected = project_splits_by_pair_folds(
        collapsed, pair_to_fold, n_splits=2
    )
    validate_splits(collapsed, projected)
    test_pairs_fold0 = set(
        collapsed.iloc[projected[0][1]]["pair_id"].astype(str)
    )
    assert test_pairs_fold0 == {
        stable_pair_id("AAAA", "C"),
        stable_pair_id("AGAG", "NN"),
        stable_pair_id("ATAT", "NO"),
    }
    assert all(pair_to_fold[pair_id] == 0 for pair_id in test_pairs_fold0)


def test_classification_collapse_excludes_conflicts_and_aggregates_sources():
    collapsed = collapse_classification(_audit_dataset())

    assert len(collapsed) == 4
    assert "CCCC" not in set(collapsed["sequence"])
    same_label = collapsed.loc[collapsed["sequence"] == "AAAA"].iloc[0]
    assert same_label["label"] == 1
    assert same_label["source"] == "A | B"
    assert same_label["pKd_value"] == 5.25
    assert same_label["n_measurements"] == 2


def test_restrict_to_pair_universe_preserves_all_reference_pairs():
    full = _audit_dataset()
    reference = full.loc[full["sequence"].isin(["AAAA", "CCCC"])].iloc[[0, 2]]
    restricted = restrict_to_pair_universe(full, reference)

    assert set(restricted["sequence"]) == {"AAAA", "CCCC"}
    assert len(restricted) == 4


def test_regression_collapse_uses_median_and_ignores_label_conflict():
    collapsed = collapse_regression(_audit_dataset())

    assert "ACAC" not in set(collapsed["sequence"])
    conflict = collapsed.loc[collapsed["sequence"] == "CCCC"].iloc[0]
    assert conflict["pKd_value"] == 5.0
    assert conflict["n_measurements"] == 2
    assert conflict["pKd_range"] == 2.0
    assert np.isnan(conflict["label"])


def test_representative_examples_have_reviewer_columns():
    examples = select_representative_examples(_audit_dataset())

    assert 3 <= len(examples) <= 5
    assert list(examples.columns) == [
        "Pair ID",
        "Sources",
        "Labels",
        "pKd values",
        "Origin/buffer",
        "Applied rule",
    ]


def _split_dataset():
    rows = []
    for index in range(30):
        rows.append(
            _row(
                f"ACGTACGT{index:02d}",
                "C" * (index % 7 + 1),
                index % 2,
                5.0 + index / 100,
                "A",
            )
        )
    return pd.DataFrame(rows)


def test_generated_stratified_splits_are_deterministic_and_disjoint():
    df = _split_dataset()
    first, groups = generate_splits(
        df,
        split_mode="stratified",
        seed=42,
        frac_tolerance=0.2,
        max_tries=5,
    )
    second, second_groups = generate_splits(
        df,
        split_mode="stratified",
        seed=42,
        frac_tolerance=0.2,
        max_tries=5,
    )

    validate_splits(df, first, groups)
    assert np.array_equal(groups, second_groups)
    for (first_train, first_test), (second_train, second_test) in zip(first, second):
        assert np.array_equal(first_train, second_train)
        assert np.array_equal(first_test, second_test)


def test_sensitivity_summary_and_fold_aligned_deltas():
    rows = []
    for version, offset in (
        ("Current measurement-level data", 0.0),
        ("Collapsed unique pairs", 0.1),
    ):
        for fold in range(5):
            rows.append(
                {
                    "dataset_version": version,
                    "split_mode": "stratified",
                    "fold": fold,
                    "roc_auc": 0.6 + offset + fold / 100,
                    "mcc": 0.3 + offset + fold / 100,
                }
            )
    fold_results = pd.DataFrame(rows)
    summary = summarize_sensitivity(fold_results)
    deltas = sensitivity_deltas(fold_results)

    assert len(summary) == 2
    assert set(summary["n_folds"]) == {5}
    assert np.isclose(deltas.iloc[0]["roc_auc_mean_delta"], 0.1)
    assert np.isclose(deltas.iloc[0]["mcc_mean_delta"], 0.1)
