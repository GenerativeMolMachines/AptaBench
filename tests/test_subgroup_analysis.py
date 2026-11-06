import numpy as np
import pandas as pd

from src.models.subgroup_analysis import (
    RSAPRED_FILES,
    annotate_rsapred,
    build_rsapred_category_map,
    classification_scores,
    dna_rna_metrics,
    rsapred_subtype_metrics,
)


def test_classification_single_class_is_explicitly_undefined():
    group = pd.DataFrame(
        {
            "y_true": [1, 1],
            "y_score": [0.8, 0.9],
            "y_pred": [1, 1],
        }
    )
    scores = classification_scores(group)
    assert np.isnan(scores["roc_auc"])
    assert np.isnan(scores["pr_auc"])
    assert np.isnan(scores["mcc"])


def test_dna_rna_summary_marks_pooled_fallback():
    df = pd.DataFrame(
        {
            "type": ["DNA"] * 4,
            "label": [1, 1, 0, 0],
            "pKd_value": [5.0, 6.0, 7.0, 8.0],
        }
    )
    classification = pd.DataFrame(
        {
            "split": ["stratified"] * 4,
            "fold": [0, 0, 1, 1],
            "row_idx": range(4),
            "y_true": [1, 1, 0, 0],
            "y_score": [0.9, 0.8, 0.2, 0.1],
            "y_pred": [1, 1, 0, 0],
        }
    )
    regression = pd.DataFrame(
        {
            "split": ["stratified"] * 4,
            "fold": [0, 0, 1, 1],
            "row_idx": range(4),
            "y_true": [5.0, 6.0, 7.0, 8.0],
            "y_pred": [5.1, 5.9, 7.1, 7.9],
        }
    )
    folds, summary = dna_rna_metrics(classification, regression, df)
    cls = summary.loc[summary["task"] == "classification"].iloc[0]
    assert cls["aggregation"] == "pooled_oof_fallback"
    assert cls["roc_auc_pooled"] == 1.0
    assert cls["roc_auc_valid_folds"] == 0
    assert len(folds) == 4


def test_rsapred_threshold_suppresses_small_categories():
    rows = []
    metadata = []
    row_idx = 0
    for category, active, inactive in (("large", 20, 20), ("small", 20, 19)):
        labels = [1] * active + [0] * inactive
        for label in labels:
            metadata.append(
                {"row_idx": row_idx, "category": category, "label": label}
            )
            rows.append(
                {
                    "split": "stratified",
                    "fold": row_idx % 5,
                    "row_idx": row_idx,
                    "y_true": label,
                    "y_score": 0.8 if label else 0.2,
                    "y_pred": label,
                }
            )
            row_idx += 1
    metrics = rsapred_subtype_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(metadata),
        min_per_class=20,
    ).set_index("category")
    assert metrics.loc["large", "status"] == "reported"
    assert metrics.loc["large", "roc_auc"] == 1.0
    assert metrics.loc["small", "status"] == "insufficient_class_support"
    assert np.isnan(metrics.loc["small", "roc_auc"])


def test_rsapred_category_reconstruction_uses_deterministic_priority(tmp_path):
    columns = [
        "Entry_ID",
        "SMILES",
        "Target_RNA_sequence",
        "Molecule_name",
        "Molecule_ID",
        "Target_RNA_name",
        "Target_RNA_ID",
        "pKd",
    ]
    for index, (filename, _) in enumerate(RSAPRED_FILES):
        row = [
            index + 1,
            "CCO",
            "AUGC" if index < 2 else f"AUGC{'A' * index}",
            "compound",
            f"M{index}",
            "target",
            f"T{index}",
            6.0,
        ]
        pd.DataFrame([row], columns=columns).to_csv(
            tmp_path / filename,
            sep="\t",
            index=False,
        )

    category_map, diagnostics = build_rsapred_category_map(tmp_path)
    ambiguous = diagnostics.set_index("diagnostic").loc[
        "ambiguous_cross_category_keys", "value"
    ]
    assert ambiguous == 1
    duplicate = category_map.loc[
        (category_map["sequence"] == "AUGC")
        & (category_map["canonical_smiles"] == "CCO")
    ]
    assert duplicate.iloc[0]["category"] == "aptamer"

    benchmark = pd.DataFrame(
        {
            "source": ["RSAPred"],
            "sequence": ["AUGC"],
            "canonical_smiles": ["CCO"],
            "pKd_value": [6.0],
            "label": [1],
        }
    )
    annotated, mapping_diagnostics = annotate_rsapred(benchmark, category_map)
    assert annotated.iloc[0]["category"] == "aptamer"
    assert (
        mapping_diagnostics.set_index("diagnostic").loc[
            "unmapped_benchmark_rows", "value"
        ]
        == 0
    )
