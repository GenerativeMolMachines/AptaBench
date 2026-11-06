import numpy as np
import pandas as pd

from src.models.negative_ablation import (
    aggregate_seed_results,
    build_shuffled_negatives,
    generate_first_fold,
    make_balanced_scope,
    make_train_variants,
    positive_pairs,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for index in range(20):
        rows.append(
            {
                "sequence": f"ACGTACGT{index:02d}",
                "canonical_smiles": "C" * (index % 6 + 1),
                "label": 1,
                "source": "Manual",
                "pKd_value": 7.0,
            }
        )
    for index in range(8):
        rows.append(
            {
                "sequence": f"TTTTGGGG{index:02d}",
                "canonical_smiles": "N" + "C" * (index % 5 + 1),
                "label": 0,
                "source": "Specificity" if index < 3 else "Manual",
                "pKd_value": np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_balanced_scope_is_deterministic_and_removes_specificity():
    df = _dataset()
    first = make_balanced_scope(df, scope="full", seed=42)
    second = make_balanced_scope(df, scope="full", seed=42)
    different = make_balanced_scope(df, scope="full", seed=43)

    assert first["label"].value_counts().to_dict() == {0: 8, 1: 8}
    assert first["original_index"].tolist() == second["original_index"].tolist()
    assert first["original_index"].tolist() != different["original_index"].tolist()

    without = make_balanced_scope(df, scope="without_specificity", seed=42)
    assert "Specificity" not in set(without["source"])
    assert without["label"].value_counts().to_dict() == {0: 5, 1: 5}


def test_shuffled_negatives_are_unique_and_exclude_all_known_positives():
    balanced = make_balanced_scope(_dataset(), scope="full", seed=42)
    held_out_pair = {
        (
            balanced.loc[balanced["label"] == 1, "sequence"].iloc[0],
            balanced.loc[balanced["label"] == 1, "canonical_smiles"].iloc[1],
        )
    }
    shuffled = build_shuffled_negatives(
        balanced,
        seed=42,
        known_positive_pairs=positive_pairs(balanced),
        additional_forbidden_pairs=held_out_pair,
    )
    pairs = set(
        shuffled[["sequence", "canonical_smiles"]].itertuples(
            index=False,
            name=None,
        )
    )

    assert len(shuffled) == int((balanced["label"] == 0).sum())
    assert len(pairs) == len(shuffled)
    assert pairs.isdisjoint(positive_pairs(balanced))
    assert pairs.isdisjoint(held_out_pair)
    assert set(shuffled["label"]) == {0}


def test_train_only_replacement_preserves_counts_and_curated_input():
    balanced = make_balanced_scope(_dataset(), scope="full", seed=42)
    original = balanced.copy(deep=True)
    variants = make_train_variants(
        balanced,
        seed=42,
        known_positive_pairs=positive_pairs(balanced),
    )

    pd.testing.assert_frame_equal(balanced, original)
    assert len(variants["Curated"]) == len(variants["Shuffled"])
    for variant in variants.values():
        assert variant["label"].value_counts().to_dict() == {0: 8, 1: 8}
    assert set(variants["Shuffled"].loc[
        variants["Shuffled"]["label"] == 0, "source"
    ]) == {"negative_ablation"}


def test_first_fold_is_deterministic_and_group_disjoint():
    balanced = make_balanced_scope(_dataset(), scope="full", seed=42)
    first = generate_first_fold(
        balanced,
        split_mode="stratified",
        seed=42,
        frac_tolerance=0.2,
        max_tries=10,
    )
    second = generate_first_fold(
        balanced,
        split_mode="stratified",
        seed=42,
        frac_tolerance=0.2,
        max_tries=10,
    )
    train_idx, test_idx, groups = first

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.intersect1d(groups[train_idx], groups[test_idx]).size == 0
    assert np.intersect1d(train_idx, test_idx).size == 0


def test_seed_aggregation_includes_paired_difference():
    rows = []
    for seed, curated, shuffled in ((42, 0.8, 0.6), (43, 0.6, 0.5)):
        for negative_type, value in (
            ("Curated", curated),
            ("Shuffled", shuffled),
        ):
            rows.append(
                {
                    "dataset": "Full",
                    "scope": "full",
                    "split": "In-distribution",
                    "seed": seed,
                    "negative_type": negative_type,
                    "roc_auc": value,
                    "pr_auc": value - 0.1,
                    "mcc": value - 0.2,
                }
            )

    summary = aggregate_seed_results(pd.DataFrame(rows))
    delta = summary.loc[
        summary["negative_type"] == "Curated − Shuffled"
    ].iloc[0]
    assert len(summary) == 3
    assert delta["n_seeds"] == 2
    assert np.isclose(delta["roc_auc_mean"], 0.15)
    assert np.isclose(delta["roc_auc_std"], 0.05)
    assert delta["roc_auc"] == "+0.150 ± 0.050"
