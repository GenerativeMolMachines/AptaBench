"""Utilities for the E1 remove-one-source benchmark ablation."""

from __future__ import annotations

import ast
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data.split import (
    build_sequence_similarity_edges,
    disjoint_aptamer_splits,
    disjoint_molecule_splits,
    sequence_groups_from_edges,
    stratified_group_splits,
)


SPLIT_MODES = ("stratified", "disjoint_molecule", "disjoint_aptamer")
CLASSIFICATION_SOURCES = (
    "RSAPred",
    "Aptamer Base",
    "AptaGen",
    "UTexas",
    "Ribocentre",
    "AptaDB",
    "Manual",
    "Specificity",
)
REGRESSION_SOURCES = (
    "RSAPred",
    "Aptamer Base",
    "AptaGen",
    "UTexas",
    "Manual",
    "Specificity",
)
CLASSIFICATION_ENCODERS = {
    "stratified": ("Kmer4", "CLAMP"),
    "disjoint_molecule": ("OneHot", "MorganFP"),
    "disjoint_aptamer": ("Kmer4", "ChemBERTa"),
}
REGRESSION_ENCODERS = {
    "stratified": ("DNABERT2", "MolFormer"),
    "disjoint_molecule": ("Kmer4", "PhysChem"),
    "disjoint_aptamer": ("Kmer4", "CLAMP"),
}
REGRESSION_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 96,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.1,
}
CLASSIFICATION_METRICS = ("roc_auc", "pr_auc", "mcc", "f1")
REGRESSION_METRICS = ("r2", "rmse", "mae", "pearson", "spearman")


def dataset_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scope_name(removed_source: str | None) -> str:
    if removed_source is None:
        return "full"
    safe = "".join(
        char.lower() if char.isalnum() else "_"
        for char in removed_source
    ).strip("_")
    return f"without_{safe}"


def make_scope(df: pd.DataFrame, removed_source: str | None) -> pd.DataFrame:
    """Remove one source while retaining canonical row ids."""
    scoped = df.copy()
    scoped["original_index"] = np.arange(len(scoped), dtype=int)
    if removed_source is not None:
        scoped = scoped.loc[scoped["source"] != removed_source]
    return scoped.reset_index(drop=True)


def load_classification_params(
    results_path: str | Path,
) -> dict[str, dict]:
    """Read the three full-dataset HPO winners required by E1."""
    results = pd.read_csv(results_path)
    params = {}
    for split_mode, (aptamer_encoder, molecule_encoder) in (
        CLASSIFICATION_ENCODERS.items()
    ):
        rows = results.loc[
            (results["split"] == split_mode)
            & (results["aptamer_encoder"] == aptamer_encoder)
            & (results["molecule_encoder"] == molecule_encoder)
        ]
        if len(rows) != 1:
            raise ValueError(
                "Expected exactly one classification config for "
                f"{split_mode}/{aptamer_encoder}/{molecule_encoder}, "
                f"found {len(rows)}"
            )
        parsed = ast.literal_eval(rows.iloc[0]["best_params"])
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(f"Invalid best_params for {split_mode}")
        params[split_mode] = parsed
    return params


def generate_scope_splits(
    scoped_df: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    frac_tolerance: float = 0.02,
    max_tries: int = 200,
    aptamer_groups: np.ndarray | None = None,
) -> dict[str, dict]:
    """Rebuild all three five-fold protocols for one ablation scope."""
    common = {
        "n_splits": n_splits,
        "random_state": random_state,
        "frac_tolerance": frac_tolerance,
        "max_tries": max_tries,
        "return_groups": True,
    }
    stratified, stratified_groups = stratified_group_splits(
        scoped_df,
        **common,
    )
    aptamer, aptamer_groups = disjoint_aptamer_splits(
        scoped_df,
        precomputed_groups=aptamer_groups,
        **common,
    )
    molecule, molecule_groups = disjoint_molecule_splits(
        scoped_df,
        scaffold_missing="separate",
        **common,
    )
    return {
        "stratified": {"splits": stratified, "groups": stratified_groups},
        "disjoint_molecule": {"splits": molecule, "groups": molecule_groups},
        "disjoint_aptamer": {"splits": aptamer, "groups": aptamer_groups},
    }


def prepare_sequence_graph(
    df: pd.DataFrame,
    cache_path: str | Path,
    *,
    threshold: float = 0.8,
    n_jobs: int = -1,
) -> dict:
    """Build once and reuse the full-data sequence graph across ablations."""
    path = Path(cache_path)
    reference_sequences = df["sequence"].astype(str).drop_duplicates().tolist()
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("threshold") == threshold
            and cached.get("reference_sequences") == reference_sequences
        ):
            return cached
    edges = build_sequence_similarity_edges(
        reference_sequences,
        threshold=threshold,
        n_jobs=n_jobs,
    )
    cached = {
        "threshold": threshold,
        "reference_sequences": reference_sequences,
        "edges": edges,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cached), encoding="utf-8")
    return cached


def scope_aptamer_groups(
    scoped_df: pd.DataFrame,
    sequence_graph: dict,
) -> np.ndarray:
    unique_sequences = (
        scoped_df["sequence"].astype(str).drop_duplicates().tolist()
    )
    unique_groups = sequence_groups_from_edges(
        unique_sequences,
        sequence_graph["reference_sequences"],
        sequence_graph["edges"],
    )
    mapping = {
        sequence: unique_groups[index]
        for index, sequence in enumerate(unique_sequences)
    }
    return scoped_df["sequence"].astype(str).map(mapping).to_numpy()


def validate_split_bundle(
    scoped_df: pd.DataFrame,
    split_bundle: dict[str, dict],
    *,
    removed_source: str | None,
) -> None:
    """Fail fast on source-removal, coverage, or group-leakage errors."""
    if removed_source is not None and (
        scoped_df["source"] == removed_source
    ).any():
        raise ValueError(f"Removed source is still present: {removed_source}")
    expected = np.arange(len(scoped_df))
    for split_mode in SPLIT_MODES:
        payload = split_bundle[split_mode]
        groups = np.asarray(payload["groups"])
        test_indices = []
        for train_idx, test_idx in payload["splits"]:
            if np.intersect1d(train_idx, test_idx).size:
                raise ValueError(f"Index leakage in {split_mode}")
            if np.intersect1d(
                np.unique(groups[train_idx]),
                np.unique(groups[test_idx]),
            ).size:
                raise ValueError(f"Group leakage in {split_mode}")
            test_indices.extend(np.asarray(test_idx, dtype=int).tolist())
        if not np.array_equal(np.sort(test_indices), expected):
            raise ValueError(
                f"Test folds do not cover scope exactly once: {split_mode}"
            )


def save_scope_splits(
    scoped_df: pd.DataFrame,
    split_bundle: dict[str, dict],
    output_dir: str | Path,
    removed_source: str | None,
) -> None:
    destination = Path(output_dir) / "splits" / scope_name(removed_source)
    destination.mkdir(parents=True, exist_ok=True)
    original = scoped_df["original_index"].to_numpy(dtype=int)
    for split_mode, payload in split_bundle.items():
        groups = np.asarray(payload["groups"])
        records = []
        for fold, (train_idx, test_idx) in enumerate(payload["splits"]):
            records.append(
                {
                    "fold": fold,
                    "removed_source": removed_source,
                    "train_idx": original[train_idx].tolist(),
                    "test_idx": original[test_idx].tolist(),
                    "n_train_groups": int(np.unique(groups[train_idx]).size),
                    "n_test_groups": int(np.unique(groups[test_idx]).size),
                }
            )
        (destination / f"{split_mode}.json").write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )


def load_original_splits(
    splits_dir: str | Path,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load the released paper fold indices (train_idx / val_idx)."""
    splits_dir = Path(splits_dir)
    loaded = {}
    for split_mode in SPLIT_MODES:
        payload = json.loads(
            (splits_dir / f"{split_mode}.json").read_text(encoding="utf-8")
        )
        folds = []
        for item in payload:
            folds.append(
                (
                    np.asarray(item["train_idx"], dtype=int),
                    np.asarray(item["val_idx"], dtype=int),
                )
            )
        loaded[split_mode] = folds
    return loaded


def filter_indices_by_source(
    indices: np.ndarray,
    sources: np.ndarray,
    removed_source: str | None,
) -> np.ndarray:
    """Keep original fold membership; drop rows of the removed source."""
    if removed_source is None:
        return np.asarray(indices, dtype=int)
    indices = np.asarray(indices, dtype=int)
    return indices[sources[indices] != removed_source]


def save_filtered_fixed_splits(
    original_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sources: np.ndarray,
    output_dir: str | Path,
    removed_source: str | None,
) -> None:
    """Persist filtered original-fold indices for one ablation scope."""
    destination = Path(output_dir) / scope_name(removed_source)
    destination.mkdir(parents=True, exist_ok=True)
    for split_mode, folds in original_splits.items():
        records = []
        for fold, (train_idx, test_idx) in enumerate(folds):
            train_f = filter_indices_by_source(
                train_idx, sources, removed_source
            )
            test_f = filter_indices_by_source(
                test_idx, sources, removed_source
            )
            records.append(
                {
                    "fold": fold,
                    "removed_source": removed_source,
                    "train_idx": train_f.tolist(),
                    "test_idx": test_f.tolist(),
                    "n_train": int(len(train_f)),
                    "n_test": int(len(test_f)),
                }
            )
        (destination / f"{split_mode}.json").write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )


def evaluate_classification_fixed_folds(
    df: pd.DataFrame,
    original_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    embeddings: dict[str, np.ndarray],
    params_by_split: dict[str, dict],
    *,
    removed_source: str | None,
    groups_by_split: dict[str, np.ndarray] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Ablate sources inside the released fixed folds (no split rebuild)."""
    fold_records = []
    y = df["label"].astype(int).to_numpy()
    sources = df["source"].to_numpy()

    for split_mode in SPLIT_MODES:
        aptamer_encoder, molecule_encoder = CLASSIFICATION_ENCODERS[split_mode]
        X = np.concatenate(
            [
                embeddings[aptamer_encoder],
                embeddings[molecule_encoder],
            ],
            axis=1,
        ).astype(np.float32)
        groups = (
            groups_by_split[split_mode]
            if groups_by_split is not None
            else np.arange(len(df))
        )
        for fold, (train_idx, test_idx) in enumerate(
            original_splits[split_mode]
        ):
            print(
                f"    cls {split_mode} fold {fold + 1}/5",
                flush=True,
            )
            train_idx = filter_indices_by_source(
                train_idx, sources, removed_source
            )
            test_idx = filter_indices_by_source(
                test_idx, sources, removed_source
            )
            if len(train_idx) < 2 or len(test_idx) < 1:
                raise ValueError(
                    f"Insufficient rows in {scope_name(removed_source)}/"
                    f"{split_mode}/fold_{fold}"
                )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = LGBMClassifier(
                **params_by_split[split_mode],
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(X_train, y[train_idx])
            probability = model.predict_proba(X_test)[:, 1]
            prediction = model.predict(X_test)
            record = {
                "task": "classification",
                "scope": scope_name(removed_source),
                "removed_source": removed_source,
                "split": split_mode,
                "fold": fold,
                "aptamer_encoder": aptamer_encoder,
                "molecule_encoder": molecule_encoder,
                "protocol": "fixed_original_folds",
            }
            record.update(_subset_stats(df, train_idx, groups, "train"))
            record.update(_subset_stats(df, test_idx, groups, "test"))
            record.update(
                _classification_scores(
                    y[test_idx],
                    probability,
                    prediction,
                )
            )
            fold_records.append(record)
    return pd.DataFrame(fold_records)


def evaluate_regression_fixed_folds(
    df: pd.DataFrame,
    original_splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    embeddings: dict[str, np.ndarray],
    *,
    removed_source: str | None,
    groups_by_split: dict[str, np.ndarray] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Ablate pKd sources inside the released fixed folds (no split rebuild)."""
    fold_records = []
    y = df["pKd_value"].to_numpy(dtype=float)
    valid = np.isfinite(y)
    sources = df["source"].to_numpy()

    for split_mode in SPLIT_MODES:
        aptamer_encoder, molecule_encoder = REGRESSION_ENCODERS[split_mode]
        X = np.concatenate(
            [
                embeddings[aptamer_encoder],
                embeddings[molecule_encoder],
            ],
            axis=1,
        ).astype(np.float32)
        groups = (
            groups_by_split[split_mode]
            if groups_by_split is not None
            else np.arange(len(df))
        )
        for fold, (train_idx, test_idx) in enumerate(
            original_splits[split_mode]
        ):
            print(
                f"    reg {split_mode} fold {fold + 1}/5",
                flush=True,
            )
            train_idx = filter_indices_by_source(
                train_idx, sources, removed_source
            )
            test_idx = filter_indices_by_source(
                test_idx, sources, removed_source
            )
            train_valid = train_idx[valid[train_idx]]
            test_valid = test_idx[valid[test_idx]]
            if len(train_valid) < 2 or len(test_valid) < 1:
                raise ValueError(
                    f"Insufficient pKd rows in {scope_name(removed_source)}/"
                    f"{split_mode}/fold_{fold}"
                )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_valid])
            X_test = scaler.transform(X[test_valid])
            model = LGBMRegressor(
                **REGRESSION_PARAMS,
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(X_train, y[train_valid])
            prediction = model.predict(X_test)
            record = {
                "task": "regression",
                "scope": scope_name(removed_source),
                "removed_source": removed_source,
                "split": split_mode,
                "fold": fold,
                "aptamer_encoder": aptamer_encoder,
                "molecule_encoder": molecule_encoder,
                "protocol": "fixed_original_folds",
                "scope_train_rows": int(len(train_idx)),
                "scope_test_rows": int(len(test_idx)),
            }
            record.update(_subset_stats(df, train_valid, groups, "train"))
            record.update(_subset_stats(df, test_valid, groups, "test"))
            record.update(_regression_scores(y[test_valid], prediction))
            fold_records.append(record)
    return pd.DataFrame(fold_records)


def _subset_stats(
    scoped_df: pd.DataFrame,
    indices: np.ndarray,
    groups: np.ndarray,
    prefix: str,
) -> dict:
    part = scoped_df.iloc[indices]
    return {
        f"n_{prefix}": int(len(part)),
        f"{prefix}_positive_fraction": float(part["label"].mean()),
        f"{prefix}_unique_aptamers": int(part["sequence"].nunique()),
        f"{prefix}_unique_molecules": int(
            part["canonical_smiles"].nunique()
        ),
        f"n_{prefix}_groups": int(np.unique(groups[indices]).size),
    }


def _classification_scores(
    y_true: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    has_both_classes = np.unique(y_true).size > 1
    return {
        "roc_auc": (
            float(roc_auc_score(y_true, probability))
            if has_both_classes
            else np.nan
        ),
        "pr_auc": (
            float(average_precision_score(y_true, probability))
            if np.any(y_true == 1)
            else np.nan
        ),
        "mcc": (
            float(matthews_corrcoef(y_true, prediction))
            if has_both_classes
            else np.nan
        ),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def _safe_correlation(
    function,
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.nanstd(y_true) == 0 or np.nanstd(prediction) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = function(y_true, prediction)
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def _regression_scores(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)) if len(y_true) >= 2 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
        "pearson": _safe_correlation(pearsonr, y_true, prediction),
        "spearman": _safe_correlation(spearmanr, y_true, prediction),
    }


def evaluate_classification_scope(
    scoped_df: pd.DataFrame,
    split_bundle: dict[str, dict],
    embeddings: dict[str, np.ndarray],
    params_by_split: dict[str, dict],
    *,
    removed_source: str | None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train fixed full-dataset classifiers on every rebuilt fold."""
    fold_records = []
    prediction_records = []
    y = scoped_df["label"].astype(int).to_numpy()
    original = scoped_df["original_index"].to_numpy(dtype=int)

    for split_mode in SPLIT_MODES:
        aptamer_encoder, molecule_encoder = CLASSIFICATION_ENCODERS[split_mode]
        X = np.concatenate(
            [
                embeddings[aptamer_encoder][original],
                embeddings[molecule_encoder][original],
            ],
            axis=1,
        ).astype(np.float32)
        groups = np.asarray(split_bundle[split_mode]["groups"])
        for fold, (train_idx, test_idx) in enumerate(
            split_bundle[split_mode]["splits"]
        ):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = LGBMClassifier(
                **params_by_split[split_mode],
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(X_train, y[train_idx])
            probability = model.predict_proba(X_test)[:, 1]
            prediction = model.predict(X_test)
            record = {
                "task": "classification",
                "scope": scope_name(removed_source),
                "removed_source": removed_source,
                "split": split_mode,
                "fold": fold,
                "aptamer_encoder": aptamer_encoder,
                "molecule_encoder": molecule_encoder,
            }
            record.update(
                _subset_stats(scoped_df, train_idx, groups, "train")
            )
            record.update(
                _subset_stats(scoped_df, test_idx, groups, "test")
            )
            record.update(
                _classification_scores(
                    y[test_idx],
                    probability,
                    prediction,
                )
            )
            fold_records.append(record)
            prediction_records.extend(
                {
                    "task": "classification",
                    "scope": scope_name(removed_source),
                    "removed_source": removed_source,
                    "split": split_mode,
                    "fold": fold,
                    "original_index": int(row_id),
                    "y_true": int(target),
                    "y_score": float(score),
                    "y_pred": int(predicted),
                }
                for row_id, target, score, predicted in zip(
                    original[test_idx],
                    y[test_idx],
                    probability,
                    prediction,
                )
            )
    return pd.DataFrame(fold_records), pd.DataFrame(prediction_records)


def evaluate_regression_scope(
    scoped_df: pd.DataFrame,
    split_bundle: dict[str, dict],
    embeddings: dict[str, np.ndarray],
    *,
    removed_source: str | None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train fixed headline regressors after filtering pKd within each fold."""
    fold_records = []
    prediction_records = []
    y = scoped_df["pKd_value"].to_numpy(dtype=float)
    valid = np.isfinite(y)
    original = scoped_df["original_index"].to_numpy(dtype=int)

    for split_mode in SPLIT_MODES:
        aptamer_encoder, molecule_encoder = REGRESSION_ENCODERS[split_mode]
        X = np.concatenate(
            [
                embeddings[aptamer_encoder][original],
                embeddings[molecule_encoder][original],
            ],
            axis=1,
        ).astype(np.float32)
        groups = np.asarray(split_bundle[split_mode]["groups"])
        for fold, (train_idx, test_idx) in enumerate(
            split_bundle[split_mode]["splits"]
        ):
            train_valid = train_idx[valid[train_idx]]
            test_valid = test_idx[valid[test_idx]]
            if len(train_valid) < 2 or len(test_valid) < 1:
                raise ValueError(
                    f"Insufficient pKd rows in {scope_name(removed_source)}/"
                    f"{split_mode}/fold_{fold}"
                )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_valid])
            X_test = scaler.transform(X[test_valid])
            model = LGBMRegressor(
                **REGRESSION_PARAMS,
                random_state=random_state,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(X_train, y[train_valid])
            prediction = model.predict(X_test)
            record = {
                "task": "regression",
                "scope": scope_name(removed_source),
                "removed_source": removed_source,
                "split": split_mode,
                "fold": fold,
                "aptamer_encoder": aptamer_encoder,
                "molecule_encoder": molecule_encoder,
                "scope_train_rows": int(len(train_idx)),
                "scope_test_rows": int(len(test_idx)),
            }
            record.update(
                _subset_stats(scoped_df, train_valid, groups, "train")
            )
            record.update(
                _subset_stats(scoped_df, test_valid, groups, "test")
            )
            record.update(_regression_scores(y[test_valid], prediction))
            fold_records.append(record)
            prediction_records.extend(
                {
                    "task": "regression",
                    "scope": scope_name(removed_source),
                    "removed_source": removed_source,
                    "split": split_mode,
                    "fold": fold,
                    "original_index": int(row_id),
                    "y_true": float(target),
                    "y_pred": float(predicted),
                }
                for row_id, target, predicted in zip(
                    original[test_valid],
                    y[test_valid],
                    prediction,
                )
            )
    return pd.DataFrame(fold_records), pd.DataFrame(prediction_records)


def aggregate_fold_results(
    fold_results: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    keys = [
        "task",
        "scope",
        "removed_source",
        "split",
        "aptamer_encoder",
        "molecule_encoder",
    ]
    data = fold_results.copy()
    data["removed_source"] = data["removed_source"].fillna("Full AptaBench")
    aggregations = {}
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, lambda x: x.std(ddof=0))
    for column in (
        "n_train",
        "n_test",
        "train_positive_fraction",
        "test_positive_fraction",
        "train_unique_aptamers",
        "test_unique_aptamers",
        "train_unique_molecules",
        "test_unique_molecules",
        "n_train_groups",
        "n_test_groups",
    ):
        if column in data:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_min"] = (column, "min")
            aggregations[f"{column}_max"] = (column, "max")
    return data.groupby(keys, dropna=False).agg(**aggregations).reset_index()


def ablation_ranges(
    summary: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    ablated = summary.loc[summary["scope"] != "full"]
    records = []
    for split_mode, part in ablated.groupby("split"):
        for metric in metrics:
            column = f"{metric}_mean"
            valid = part.dropna(subset=[column])
            if valid.empty:
                continue
            low = valid.loc[valid[column].idxmin()]
            high = valid.loc[valid[column].idxmax()]
            records.append(
                {
                    "task": valid.iloc[0]["task"],
                    "split": split_mode,
                    "metric": metric,
                    "min": float(low[column]),
                    "min_scope": low["scope"],
                    "max": float(high[column]),
                    "max_scope": high["scope"],
                    "range": float(high[column] - low[column]),
                }
            )
    return pd.DataFrame(records)


def transfer_gaps(
    summary: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    records = []
    for scope, part in summary.groupby("scope"):
        indexed = part.set_index("split")
        if "stratified" not in indexed.index:
            continue
        for disjoint in ("disjoint_molecule", "disjoint_aptamer"):
            if disjoint not in indexed.index:
                continue
            for metric in metrics:
                column = f"{metric}_mean"
                in_distribution = indexed.loc["stratified", column]
                disjoint_value = indexed.loc[disjoint, column]
                records.append(
                    {
                        "task": part.iloc[0]["task"],
                        "scope": scope,
                        "removed_source": part.iloc[0]["removed_source"],
                        "disjoint_split": disjoint,
                        "metric": metric,
                        "in_distribution": in_distribution,
                        "disjoint": disjoint_value,
                        "gap_disjoint_minus_id": (
                            disjoint_value - in_distribution
                        ),
                    }
                )
    return pd.DataFrame(records)


def headline_comparison(
    summary: pd.DataFrame,
    removed_source: str,
) -> pd.DataFrame:
    wanted = {"full", scope_name(removed_source)}
    return summary.loc[summary["scope"].isin(wanted)].reset_index(drop=True)


def existing_split_source_composition(
    df: pd.DataFrame,
    splits_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe source ratios in the three released full-data split files."""
    rows = []
    fold_rows = []
    for split_mode in SPLIT_MODES:
        payload = json.loads(
            (Path(splits_dir) / f"{split_mode}.json").read_text(
                encoding="utf-8"
            )
        )
        for item in payload:
            for partition, key in (("train", "train_idx"), ("test", "val_idx")):
                part = df.iloc[item[key]]
                counts = part["source"].value_counts()
                for source, count in counts.items():
                    source_part = part.loc[part["source"] == source]
                    rows.append(
                        {
                            "split": split_mode,
                            "fold": item["fold"],
                            "partition": partition,
                            "source": source,
                            "n": int(count),
                            "fraction": float(count / len(part)),
                            "positive_fraction": float(
                                source_part["label"].mean()
                            ),
                            "pkd_n": int(
                                source_part["pKd_value"].notna().sum()
                            ),
                            "pkd_fraction": float(
                                source_part["pKd_value"].notna().mean()
                            ),
                            "unique_aptamers": int(
                                source_part["sequence"].nunique()
                            ),
                            "unique_molecules": int(
                                source_part["canonical_smiles"].nunique()
                            ),
                        }
                    )
                fold_rows.append(
                    {
                        "split": split_mode,
                        "fold": item["fold"],
                        "partition": partition,
                        "n": int(len(part)),
                        "positive_fraction": float(part["label"].mean()),
                        "pkd_n": int(part["pKd_value"].notna().sum()),
                        "pkd_fraction": float(
                            part["pKd_value"].notna().mean()
                        ),
                        "dominant_source": counts.index[0],
                        "dominant_source_fraction": float(
                            counts.iloc[0] / len(part)
                        ),
                        "n_sources": int(counts.size),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)
