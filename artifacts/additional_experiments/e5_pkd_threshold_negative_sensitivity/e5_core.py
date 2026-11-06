"""Core utilities for E5 pKd-threshold and negative-subtype sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.negative_ablation import LGBM_PARAMS, FeatureCache


SPLIT_MODES = ("stratified", "disjoint_molecule", "disjoint_aptamer")
THRESHOLDS = (3, 4, 5)
NEGATIVE_SUBTYPES = (
    "explicit_nonbinding",
    "low_affinity",
    "other_reported_inactive",
)
LOCKED_NEGATIVE = frozenset({"explicit_nonbinding", "other_reported_inactive"})
MAIN_THRESHOLD = 4


def load_released_splits(
    splits_dir: str | Path,
    *,
    split_modes: tuple[str, ...] = SPLIT_MODES,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load released folds while preserving canonical row indices."""
    splits_dir = Path(splits_dir)
    result: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for split_mode in split_modes:
        payload = json.loads(
            (splits_dir / f"{split_mode}.json").read_text(encoding="utf-8")
        )
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected split payload in {split_mode}.json")
        result[split_mode] = [
            (
                np.asarray(item["train_idx"], dtype=int),
                np.asarray(item["val_idx"], dtype=int),
            )
            for item in payload
        ]
    return result


def assign_negative_provenance(df: pd.DataFrame) -> pd.Series:
    """Restore negative provenance on the released AptaBench table."""
    required = {"source", "pKd_value", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    provenance = pd.Series(pd.NA, index=df.index, dtype="object")
    is_negative = df["label"].astype(int) == 0
    is_specificity = df["source"].astype(str) == "Specificity"
    has_pkd = df["pKd_value"].notna()

    provenance.loc[is_negative & is_specificity] = "other_reported_inactive"
    provenance.loc[is_negative & ~is_specificity & ~has_pkd] = "explicit_nonbinding"
    provenance.loc[is_negative & ~is_specificity & has_pkd] = "low_affinity"
    return provenance


def relabel_for_threshold(
    df: pd.DataFrame,
    *,
    threshold: float,
    provenance: pd.Series | None = None,
) -> pd.Series:
    """Assign classification labels for a pKd threshold."""
    if provenance is None:
        provenance = assign_negative_provenance(df)

    original = df["label"].astype(int)
    pkd = pd.to_numeric(df["pKd_value"], errors="coerce")
    labels = original.copy()

    locked = provenance.isin(LOCKED_NEGATIVE)
    labels.loc[locked] = 0

    affinity = (~locked) & pkd.notna()
    labels.loc[affinity] = np.where(
        pkd.loc[affinity].to_numpy() <= threshold, 0, 1
    )

    # Remaining rows keep original_label (no-pKd confirmed binders stay positive).
    return labels.astype(int)


def build_row_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level provenance and labels at thresholds 3/4/5."""
    provenance = assign_negative_provenance(df)
    annotations = pd.DataFrame(
        {
            "row_idx": np.arange(len(df), dtype=int),
            "source": df["source"].astype(str).to_numpy(),
            "pKd_value": pd.to_numeric(df["pKd_value"], errors="coerce").to_numpy(),
            "original_label": df["label"].astype(int).to_numpy(),
            "negative_provenance": provenance.to_numpy(),
        }
    )
    for threshold in THRESHOLDS:
        annotations[f"label_t{threshold}"] = relabel_for_threshold(
            df, threshold=threshold, provenance=provenance
        ).to_numpy()
    return annotations


def threshold_label_counts(annotations: pd.DataFrame) -> pd.DataFrame:
    """Count positives/negatives (and low-affinity negatives) per threshold."""
    rows: list[dict] = []
    for threshold in THRESHOLDS:
        label_col = f"label_t{threshold}"
        labels = annotations[label_col].astype(int)
        negatives = labels == 0
        locked = annotations["negative_provenance"].isin(LOCKED_NEGATIVE)
        # Affinity-threshold negatives: not locked, numeric pKd, label 0.
        n_low_affinity = int(
            (negatives & ~locked & annotations["pKd_value"].notna()).sum()
        )
        rows.append(
            {
                "threshold": int(threshold),
                "n_negatives": int(negatives.sum()),
                "n_positives": int((labels == 1).sum()),
                "n_low_affinity_negatives": n_low_affinity,
                "n_explicit_nonbinding": int(
                    (
                        negatives
                        & (
                            annotations["negative_provenance"]
                            == "explicit_nonbinding"
                        )
                    ).sum()
                ),
                "n_other_reported_inactive": int(
                    (
                        negatives
                        & (
                            annotations["negative_provenance"]
                            == "other_reported_inactive"
                        )
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _fit_predict_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cache: FeatureCache,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_score, y_pred) for one fold."""
    x_train = feature_cache.transform(train_df)
    x_test = feature_cache.transform(test_df)
    y_train = train_df["label"].astype(int).to_numpy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = LGBMClassifier(
        **LGBM_PARAMS,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return y_score, y_pred


def evaluate_threshold_folds(
    df: pd.DataFrame,
    labels: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    threshold: int,
    split_mode: str,
    feature_cache: FeatureCache,
    seed: int = 42,
) -> pd.DataFrame:
    """Run five-fold classification and return fold-level ROC-AUC / MCC."""
    working = df.copy()
    working["label"] = labels.astype(int).to_numpy()
    records: list[dict] = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_df = working.iloc[train_idx]
        val_df = working.iloc[val_idx]
        y_true = val_df["label"].astype(int).to_numpy()
        y_score, y_pred = _fit_predict_fold(
            train_df,
            val_df,
            feature_cache=feature_cache,
            seed=seed,
        )
        both_classes = len(np.unique(y_true)) > 1
        records.append(
            {
                "threshold": int(threshold),
                "split": split_mode,
                "fold": int(fold),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_val_positives": int((y_true == 1).sum()),
                "n_val_negatives": int((y_true == 0).sum()),
                "roc_auc": (
                    float(roc_auc_score(y_true, y_score)) if both_classes else np.nan
                ),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
            }
        )
    return pd.DataFrame(records)


def generate_oof_predictions(
    df: pd.DataFrame,
    labels: pd.Series,
    provenance: pd.Series,
    splits_by_mode: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    *,
    feature_cache: FeatureCache,
    seed: int = 42,
    split_modes: tuple[str, ...] = SPLIT_MODES,
) -> pd.DataFrame:
    """Return row-level OOF predictions for the requested split modes."""
    working = df.copy()
    working["label"] = labels.astype(int).to_numpy()
    provenance_values = provenance.to_numpy()
    records: list[dict] = []

    for split_mode in split_modes:
        folds = splits_by_mode[split_mode]
        for fold, (train_idx, val_idx) in enumerate(folds):
            train_df = working.iloc[train_idx]
            val_df = working.iloc[val_idx]
            y_true = val_df["label"].astype(int).to_numpy()
            y_score, y_pred = _fit_predict_fold(
                train_df,
                val_df,
                feature_cache=feature_cache,
                seed=seed,
            )
            for row_idx, target, score, predicted, prov in zip(
                val_idx,
                y_true,
                y_score,
                y_pred,
                provenance_values[val_idx],
            ):
                records.append(
                    {
                        "split": split_mode,
                        "fold": int(fold),
                        "row_idx": int(row_idx),
                        "y_true": int(target),
                        "y_score": float(score),
                        "y_pred": int(predicted),
                        "negative_provenance": (
                            None if pd.isna(prov) else str(prov)
                        ),
                    }
                )
    return pd.DataFrame(records)


def negative_subtype_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """Compute FPR / score stats / ROC-AUC vs positives per subtype."""
    required = {
        "split",
        "y_true",
        "y_score",
        "y_pred",
        "negative_provenance",
    }
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF table is missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for split_mode, part in oof.groupby("split", sort=False):
        positives = part.loc[part["y_true"].astype(int) == 1]
        for subtype in NEGATIVE_SUBTYPES:
            negatives = part.loc[
                (part["y_true"].astype(int) == 0)
                & (part["negative_provenance"] == subtype)
            ]
            n = int(len(negatives))
            if n == 0:
                rows.append(
                    {
                        "split": split_mode,
                        "negative_subtype": subtype,
                        "n": 0,
                        "fpr": np.nan,
                        "mean_score": np.nan,
                        "median_score": np.nan,
                        "roc_auc_vs_positives": np.nan,
                    }
                )
                continue

            y_pred = negatives["y_pred"].astype(int).to_numpy()
            scores = negatives["y_score"].to_numpy(dtype=float)
            fpr = float(np.mean(y_pred == 1))
            mean_score = float(np.mean(scores))
            median_score = float(np.median(scores))

            if positives.empty:
                roc = np.nan
            else:
                y_true = np.concatenate(
                    [
                        np.zeros(n, dtype=int),
                        np.ones(len(positives), dtype=int),
                    ]
                )
                y_score = np.concatenate(
                    [
                        scores,
                        positives["y_score"].to_numpy(dtype=float),
                    ]
                )
                roc = float(roc_auc_score(y_true, y_score))

            rows.append(
                {
                    "split": split_mode,
                    "negative_subtype": subtype,
                    "n": n,
                    "fpr": fpr,
                    "mean_score": mean_score,
                    "median_score": median_score,
                    "roc_auc_vs_positives": roc,
                }
            )
    return pd.DataFrame(rows)


def condition_complete(
    results: pd.DataFrame,
    *,
    threshold: int,
    split_mode: str,
    n_splits: int,
) -> bool:
    if results.empty:
        return False
    rows = results.loc[
        (results["threshold"].astype(int) == int(threshold))
        & (results["split"] == split_mode)
    ]
    return set(rows["fold"].astype(int)) == set(range(n_splits))


def remove_condition(
    results: pd.DataFrame,
    *,
    threshold: int,
    split_mode: str,
) -> pd.DataFrame:
    if results.empty:
        return results
    mask = (results["threshold"].astype(int) == int(threshold)) & (
        results["split"] == split_mode
    )
    return results.loc[~mask].reset_index(drop=True)
