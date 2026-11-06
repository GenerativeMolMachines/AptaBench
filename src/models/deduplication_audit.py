"""Utilities for E4 duplicate/conflict auditing and sensitivity analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import (
    disjoint_aptamer_splits,
    disjoint_molecule_splits,
    stratified_group_splits,
)
from src.models.negative_ablation import FeatureCache, evaluate_variant


PAIR_COLUMNS = ("sequence", "canonical_smiles")
SPLIT_MODES = ("stratified", "disjoint_molecule", "disjoint_aptamer")
METRICS = ("roc_auc", "mcc")
REQUIRED_COLUMNS = {
    "type",
    "sequence",
    "canonical_smiles",
    "pKd_value",
    "label",
    "buffer",
    "origin",
    "source",
}


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy containing valid canonical pair identifiers."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    normalized = df.copy()
    normalized["pKd_value"] = pd.to_numeric(
        normalized["pKd_value"], errors="coerce"
    )
    normalized["label"] = pd.to_numeric(normalized["label"], errors="coerce")
    normalized = normalized.dropna(subset=list(PAIR_COLUMNS)).reset_index(drop=True)
    normalized["sequence"] = normalized["sequence"].astype(str)
    normalized["canonical_smiles"] = normalized["canonical_smiles"].astype(str)
    return normalized


def stable_pair_id(sequence: str, canonical_smiles: str) -> str:
    """Create a compact deterministic identifier for one canonical pair."""
    key = f"{sequence}\0{canonical_smiles}".encode("utf-8")
    return f"pair_{hashlib.sha256(key).hexdigest()[:16]}"


def add_pair_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["pair_id"] = [
        stable_pair_id(sequence, smiles)
        for sequence, smiles in result[list(PAIR_COLUMNS)].itertuples(
            index=False, name=None
        )
    ]
    return result


def restrict_to_pair_universe(
    df: pd.DataFrame, reference_df: pd.DataFrame
) -> pd.DataFrame:
    """Keep records whose canonical pair occurs in the reference dataset."""
    normalized = validate_dataset(df)
    reference = validate_dataset(reference_df)
    reference_pairs = set(
        reference[list(PAIR_COLUMNS)].itertuples(index=False, name=None)
    )
    mask = [
        pair in reference_pairs
        for pair in normalized[list(PAIR_COLUMNS)].itertuples(index=False, name=None)
    ]
    restricted = normalized.loc[mask].reset_index(drop=True)
    observed_pairs = set(
        restricted[list(PAIR_COLUMNS)].itertuples(index=False, name=None)
    )
    missing = reference_pairs - observed_pairs
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} canonical pairs from the reference"
        )
    return restricted


def _unique_values(values: pd.Series, *, numeric: bool = False) -> list:
    non_missing = values.dropna()
    if numeric:
        return sorted({float(value) for value in non_missing})
    return sorted({str(value).strip() for value in non_missing if str(value).strip()})


def _joined(values: pd.Series) -> str:
    return " | ".join(_unique_values(values))


def _applied_rule(row: pd.Series) -> str:
    rules = []
    if row["conflicting_labels"]:
        rules.append("classification: exclude conflicting labels")
    elif row["n_records"] > 1:
        rules.append("classification: collapse identical label")
    else:
        rules.append("classification: retain unique pair")
    if row["n_measurements"] > 0:
        rules.append("regression: median valid pKd")
    else:
        rules.append("regression: exclude (no valid pKd)")
    if row["cross_source"]:
        rules.append("aggregate source as provenance")
    return "; ".join(rules)


def build_pair_group_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize every canonical pair and all duplicate/conflict dimensions."""
    normalized = add_pair_ids(validate_dataset(df))
    rows: list[dict] = []
    for (sequence, smiles), part in normalized.groupby(
        list(PAIR_COLUMNS), sort=True, dropna=False
    ):
        labels = _unique_values(part["label"], numeric=True)
        pkd_values = _unique_values(part["pKd_value"], numeric=True)
        valid_pkd = part["pKd_value"].dropna().astype(float)
        sources = _unique_values(part["source"])
        row = {
            "pair_id": stable_pair_id(sequence, smiles),
            "sequence": sequence,
            "canonical_smiles": smiles,
            "n_records": int(len(part)),
            "n_sources": int(len(sources)),
            "sources": " | ".join(sources),
            "labels": " | ".join(str(int(x)) if x.is_integer() else str(x) for x in labels),
            "n_labels": int(len(labels)),
            "conflicting_labels": len(labels) > 1,
            "n_measurements": int(len(valid_pkd)),
            "n_unique_pkd": int(len(pkd_values)),
            "pkd_values": " | ".join(f"{value:.12g}" for value in pkd_values),
            "pkd_median": (
                float(valid_pkd.median()) if not valid_pkd.empty else np.nan
            ),
            "pkd_range": (
                float(valid_pkd.max() - valid_pkd.min())
                if not valid_pkd.empty
                else np.nan
            ),
            "cross_source": len(sources) > 1,
            "origins": _joined(part["origin"]),
            "buffers": _joined(part["buffer"]),
        }
        row["applied_rule"] = _applied_rule(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)


def summarize_stage(df: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    """Return the requested one-row harmonization summary.

    Multi-pKd groups are reported under both definitions:
    - measurement: more than one valid pKd row (values may be identical);
    - unique_value: more than one distinct pKd value.

    pKd-range bins are counted only among unique-value multi-pKd groups, so
    identical repeated measurements do not inflate the <=0.5 bin.
    """
    normalized = validate_dataset(df)
    groups = build_pair_group_audit(normalized)
    multi_measurement = groups["n_measurements"] > 1
    multi_unique = groups["n_unique_pkd"] > 1
    ranges = groups.loc[multi_unique, "pkd_range"]
    row = {
        "stage": stage,
        "n_rows": int(len(normalized)),
        "n_unique_canonical_pairs": int(len(groups)),
        "n_repeat_pair_groups": int((groups["n_records"] > 1).sum()),
        "n_cross_source_pair_groups": int(groups["cross_source"].sum()),
        "n_conflicting_label_groups": int(groups["conflicting_labels"].sum()),
        "n_multi_pkd_measurement_groups": int(multi_measurement.sum()),
        "n_multi_pkd_unique_value_groups": int(multi_unique.sum()),
        "n_pkd_range_le_0_5": int((ranges <= 0.5).sum()),
        "n_pkd_range_gt_0_5_le_1_0": int(
            ((ranges > 0.5) & (ranges <= 1.0)).sum()
        ),
        "n_pkd_range_gt_1_0": int((ranges > 1.0).sum()),
    }
    return pd.DataFrame([row])


def _representative_row(part: pd.DataFrame, audit_row: pd.Series) -> dict:
    origin_buffer = []
    for origin, buffer in part[["origin", "buffer"]].itertuples(index=False, name=None):
        fields = []
        if pd.notna(origin) and str(origin).strip():
            fields.append(f"origin={origin}")
        if pd.notna(buffer) and str(buffer).strip():
            fields.append(f"buffer={buffer}")
        if fields:
            origin_buffer.append(", ".join(fields))
    return {
        "Pair ID": audit_row["pair_id"],
        "Sources": audit_row["sources"],
        "Labels": audit_row["labels"],
        "pKd values": audit_row["pkd_values"],
        "Origin/buffer": " | ".join(dict.fromkeys(origin_buffer)),
        "Applied rule": audit_row["applied_rule"],
    }


def select_representative_examples(
    df: pd.DataFrame, audit: pd.DataFrame | None = None, *, max_examples: int = 5
) -> pd.DataFrame:
    """Select deterministic examples spanning the requested E4 cases."""
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    normalized = add_pair_ids(validate_dataset(df))
    audit = build_pair_group_audit(normalized) if audit is None else audit.copy()
    candidates = [
        audit.loc[audit["conflicting_labels"]].sort_values(
            ["n_records", "pair_id"], ascending=[False, True]
        ),
        audit.loc[audit["cross_source"]].sort_values(
            ["n_sources", "pair_id"], ascending=[False, True]
        ),
        audit.loc[
            (audit["n_records"] > 1) & ~audit["conflicting_labels"]
        ].sort_values(["n_records", "pair_id"], ascending=[False, True]),
        audit.loc[audit["n_measurements"] > 1].sort_values(
            ["n_measurements", "pair_id"], ascending=[False, True]
        ),
        audit.loc[audit["pkd_range"] > 1.0].sort_values(
            ["pkd_range", "pair_id"], ascending=[False, True]
        ),
    ]
    selected: list[pd.Series] = []
    seen: set[str] = set()
    for candidate in candidates:
        for _, row in candidate.iterrows():
            if row["pair_id"] not in seen:
                selected.append(row)
                seen.add(row["pair_id"])
                break
        if len(selected) >= max_examples:
            break
    if len(selected) < min(3, max_examples):
        fallback = audit.loc[audit["n_records"] > 1].sort_values("pair_id")
        for _, row in fallback.iterrows():
            if row["pair_id"] not in seen:
                selected.append(row)
                seen.add(row["pair_id"])
            if len(selected) >= min(3, max_examples):
                break
    rows = []
    for audit_row in selected[:max_examples]:
        part = normalized.loc[normalized["pair_id"] == audit_row["pair_id"]]
        rows.append(_representative_row(part, audit_row))
    return pd.DataFrame(rows)


def _provenance_record(part: pd.DataFrame) -> dict:
    type_values = _unique_values(part["type"])
    valid_pkd = part["pKd_value"].dropna().astype(float)
    return {
        "type": type_values[0] if type_values else np.nan,
        "sequence": str(part["sequence"].iloc[0]),
        "canonical_smiles": str(part["canonical_smiles"].iloc[0]),
        "buffer": _joined(part["buffer"]),
        "origin": _joined(part["origin"]),
        "source": _joined(part["source"]),
        "pair_id": str(part["pair_id"].iloc[0]),
        "n_records": int(len(part)),
        "n_measurements": int(len(valid_pkd)),
        "pKd_range": (
            float(valid_pkd.max() - valid_pkd.min())
            if not valid_pkd.empty
            else np.nan
        ),
    }


def collapse_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-label pairs and remove every conflicting-label pair."""
    normalized = add_pair_ids(validate_dataset(df))
    rows = []
    for _, part in normalized.groupby("pair_id", sort=True):
        labels = _unique_values(part["label"], numeric=True)
        if len(labels) != 1:
            continue
        row = _provenance_record(part)
        row["label"] = int(labels[0])
        valid_pkd = part["pKd_value"].dropna().astype(float)
        row["pKd_value"] = (
            float(valid_pkd.median()) if not valid_pkd.empty else np.nan
        )
        rows.append(row)
    columns = [
        "type",
        "sequence",
        "canonical_smiles",
        "pKd_value",
        "label",
        "buffer",
        "origin",
        "source",
        "pair_id",
        "n_records",
        "n_measurements",
        "pKd_range",
    ]
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def collapse_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse valid pKd values by median, independent of label conflicts."""
    normalized = add_pair_ids(validate_dataset(df))
    rows = []
    for _, part in normalized.groupby("pair_id", sort=True):
        valid_pkd = part["pKd_value"].dropna().astype(float)
        if valid_pkd.empty:
            continue
        row = _provenance_record(part)
        labels = _unique_values(part["label"], numeric=True)
        row["label"] = int(labels[0]) if len(labels) == 1 else np.nan
        row["pKd_value"] = float(valid_pkd.median())
        rows.append(row)
    columns = [
        "type",
        "sequence",
        "canonical_smiles",
        "pKd_value",
        "label",
        "buffer",
        "origin",
        "source",
        "pair_id",
        "n_records",
        "n_measurements",
        "pKd_range",
    ]
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def generate_splits(
    df: pd.DataFrame,
    *,
    split_mode: str,
    seed: int,
    n_splits: int = 5,
    frac_tolerance: float = 0.02,
    max_tries: int = 200,
    sequence_n_jobs: int = 1,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    common = {
        "n_splits": n_splits,
        "random_state": seed,
        "frac_tolerance": frac_tolerance,
        "max_tries": max_tries,
        "return_groups": True,
    }
    if split_mode == "stratified":
        return stratified_group_splits(df, **common)
    if split_mode == "disjoint_molecule":
        return disjoint_molecule_splits(
            df, scaffold_missing="separate", **common
        )
    if split_mode == "disjoint_aptamer":
        return disjoint_aptamer_splits(
            df, sequence_n_jobs=sequence_n_jobs, **common
        )
    raise ValueError(f"Unknown split mode: {split_mode}")


def validate_splits(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray | None = None,
) -> None:
    for fold, (train_idx, test_idx) in enumerate(splits):
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"Index leakage in fold {fold}")
        if groups is not None and np.intersect1d(
            groups[train_idx], groups[test_idx]
        ).size:
            raise ValueError(f"Group leakage in fold {fold}")
        for name, indices in (("train", train_idx), ("test", test_idx)):
            if df.iloc[indices]["label"].nunique() != 2:
                raise ValueError(f"Fold {fold} {name} lacks one class")


def pair_fold_assignment(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, int]:
    """Map each canonical pair to the fold where its rows are held out."""
    if "pair_id" not in df.columns:
        raise ValueError("Dataset must contain pair_id before fold projection")
    pair_ids = df["pair_id"].astype(str).to_numpy()
    assignment: dict[str, int] = {}
    seen_indices: set[int] = set()
    for fold, (_train_idx, test_idx) in enumerate(splits):
        for idx in np.asarray(test_idx, dtype=int):
            if idx in seen_indices:
                raise ValueError(f"Row {idx} appears in multiple test folds")
            seen_indices.add(int(idx))
            pair_id = pair_ids[idx]
            previous = assignment.get(pair_id)
            if previous is not None and previous != fold:
                raise ValueError(
                    f"Pair {pair_id} is split across folds {previous} and {fold}"
                )
            assignment[pair_id] = fold
    if len(seen_indices) != len(df):
        raise ValueError("Split folds do not cover every dataset row exactly once")
    return assignment


def project_splits_by_pair_folds(
    df: pd.DataFrame,
    pair_to_fold: dict[str, int],
    *,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Assign each row to the source fold of its canonical pair."""
    if "pair_id" not in df.columns:
        raise ValueError("Dataset must contain pair_id before fold projection")
    pair_ids = df["pair_id"].astype(str)
    missing = sorted(set(pair_ids) - set(pair_to_fold))
    if missing:
        raise ValueError(
            f"{len(missing)} collapsed pairs lack a source fold assignment"
        )
    fold_labels = pair_ids.map(pair_to_fold).to_numpy(dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        test_idx = np.flatnonzero(fold_labels == fold)
        train_idx = np.flatnonzero(fold_labels != fold)
        splits.append((train_idx, test_idx))
    return splits


def evaluate_folds(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    dataset_version: str,
    split_mode: str,
    seed: int,
    feature_cache: FeatureCache | None = None,
    split_provenance: str | None = None,
) -> pd.DataFrame:
    """Evaluate fixed Kmer4 + MorganFP LightGBM on all supplied folds."""
    cache = FeatureCache() if feature_cache is None else feature_cache
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        metrics = evaluate_variant(train, test, seed=seed, feature_cache=cache)
        row = {
            "dataset_version": dataset_version,
            "split_mode": split_mode,
            "fold": fold,
            "seed": seed,
            "n_dataset": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "roc_auc": metrics["roc_auc"],
            "mcc": metrics["mcc"],
        }
        if split_provenance is not None:
            row["split_provenance"] = split_provenance
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sensitivity(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (version, split_mode), part in fold_results.groupby(
        ["dataset_version", "split_mode"], sort=False
    ):
        row = {
            "dataset_version": version,
            "split_mode": split_mode,
            "n_folds": int(part["fold"].nunique()),
        }
        for metric in METRICS:
            values = part[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
            row[metric] = (
                f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_deltas(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Compute fold-aligned collapsed-minus-current descriptive deltas."""
    pivot = fold_results.pivot(
        index=["split_mode", "fold"], columns="dataset_version", values=list(METRICS)
    )
    required = {"Current measurement-level data", "Collapsed unique pairs"}
    if not required.issubset(set(pivot.columns.get_level_values(1))):
        raise ValueError("Both dataset versions are required for deltas")
    rows = []
    for split_mode, part in pivot.groupby(level="split_mode"):
        row = {"split_mode": split_mode, "comparison": "collapsed_minus_current"}
        for metric in METRICS:
            values = (
                part[(metric, "Collapsed unique pairs")]
                - part[(metric, "Current measurement-level data")]
            ).to_numpy(dtype=float)
            row[f"{metric}_mean_delta"] = float(np.mean(values))
            row[f"{metric}_std_delta"] = float(np.std(values, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def load_interim_tables(interim_dir: Path) -> pd.DataFrame:
    paths = sorted(interim_dir.glob("*_processed.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_processed.csv files in {interim_dir}")
    tables = [pd.read_csv(path) for path in paths]
    return validate_dataset(pd.concat(tables, ignore_index=True))
