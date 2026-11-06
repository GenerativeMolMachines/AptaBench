"""Utilities for E2 DNA/RNA and RSAPred subgroup analysis."""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.data.processing import (
    canonicalize_smiles_column,
    clean_and_validate_sequences,
    process_binding_constants,
)


SPLIT_MODES = ("stratified", "disjoint_aptamer", "disjoint_molecule")
SPLIT_LABELS = {
    "stratified": "In-distribution",
    "disjoint_aptamer": "Aptamer-disjoint",
    "disjoint_molecule": "Molecule-disjoint",
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
RSAPRED_FILES = (
    ("Aptamers_dataset_v1.csv", "aptamer"),
    ("Riboswitch_dataset_v1.csv", "riboswitch"),
    ("miRNA_dataset_v1.csv", "miRNA"),
    ("Repeats_dataset_v1.csv", "repeat"),
    ("Ribosomal_dataset_v1.csv", "ribosomal RNA"),
    ("Viral_RNA_dataset_v1.csv", "viral RNA"),
)
JOIN_COLUMNS = ("sequence", "canonical_smiles", "pKd_value")


def select_headline_configs(
    classification_results: str | Path,
    regression_results: str | Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Select the best published ML row independently for each split."""
    classification = pd.read_csv(classification_results)
    regression = pd.read_csv(regression_results)
    classification_configs: dict[str, dict] = {}
    regression_configs: dict[str, dict] = {}
    for split_mode in SPLIT_MODES:
        cls_rows = classification.loc[classification["split"] == split_mode]
        reg_rows = regression.loc[regression["split"] == split_mode]
        if cls_rows.empty or reg_rows.empty:
            raise ValueError(f"Missing screening rows for {split_mode}")
        cls = cls_rows.loc[cls_rows["ROC-AUC mean"].idxmax()]
        reg = reg_rows.loc[reg_rows["R2 mean"].idxmax()]
        params = ast.literal_eval(cls["best_params"])
        if not isinstance(params, dict) or not params:
            raise ValueError(f"Invalid classification best_params for {split_mode}")
        classification_configs[split_mode] = {
            "aptamer_encoder": cls["aptamer_encoder"],
            "molecule_encoder": cls["molecule_encoder"],
            "params": params,
        }
        regression_configs[split_mode] = {
            "aptamer_encoder": reg["aptamer_encoder"],
            "molecule_encoder": reg["molecule_encoder"],
            "params": REGRESSION_PARAMS.copy(),
        }
    return classification_configs, regression_configs


def load_released_splits(
    splits_dir: str | Path,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load released folds while preserving their canonical row indices."""
    result = {}
    for split_mode in SPLIT_MODES:
        payload = json.loads(
            (Path(splits_dir) / f"{split_mode}.json").read_text(encoding="utf-8")
        )
        result[split_mode] = [
            (
                np.asarray(item["train_idx"], dtype=int),
                np.asarray(item["val_idx"], dtype=int),
            )
            for item in payload
        ]
    return result


def required_encoder_names(
    classification_configs: dict[str, dict],
    regression_configs: dict[str, dict],
) -> tuple[set[str], set[str]]:
    aptamer = {
        config["aptamer_encoder"]
        for config in (*classification_configs.values(), *regression_configs.values())
    }
    molecule = {
        config["molecule_encoder"]
        for config in (*classification_configs.values(), *regression_configs.values())
    }
    return aptamer, molecule


def generate_oof_predictions(
    df: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    embeddings: dict[str, np.ndarray],
    classification_configs: dict[str, dict],
    regression_configs: dict[str, dict],
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fixed headline configurations and return canonical row-level OOF."""
    classification_records = []
    regression_records = []
    labels = df["label"].astype(int).to_numpy()
    targets = df["pKd_value"].to_numpy(dtype=float)
    valid_targets = np.isfinite(targets)

    for split_mode in SPLIT_MODES:
        cls_config = classification_configs[split_mode]
        cls_features = np.concatenate(
            [
                embeddings[cls_config["aptamer_encoder"]],
                embeddings[cls_config["molecule_encoder"]],
            ],
            axis=1,
        ).astype(np.float32)
        reg_config = regression_configs[split_mode]
        reg_features = np.concatenate(
            [
                embeddings[reg_config["aptamer_encoder"]],
                embeddings[reg_config["molecule_encoder"]],
            ],
            axis=1,
        ).astype(np.float32)

        for fold, (train_idx, val_idx) in enumerate(splits[split_mode]):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(cls_features[train_idx])
            x_val = scaler.transform(cls_features[val_idx])
            classifier = LGBMClassifier(
                **cls_config["params"],
                random_state=random_state,
                n_jobs=2,
                verbosity=-1,
            )
            classifier.fit(x_train, labels[train_idx])
            probability = classifier.predict_proba(x_val)[:, 1]
            prediction = (probability >= 0.5).astype(int)
            classification_records.extend(
                {
                    "task": "classification",
                    "split": split_mode,
                    "fold": fold,
                    "row_idx": int(row_idx),
                    "aptamer_encoder": cls_config["aptamer_encoder"],
                    "molecule_encoder": cls_config["molecule_encoder"],
                    "y_true": int(target),
                    "y_score": float(score),
                    "y_pred": int(predicted),
                }
                for row_idx, target, score, predicted in zip(
                    val_idx, labels[val_idx], probability, prediction
                )
            )

            train_valid = train_idx[valid_targets[train_idx]]
            val_valid = val_idx[valid_targets[val_idx]]
            if len(train_valid) < 2 or len(val_valid) < 1:
                raise ValueError(
                    f"Insufficient regression rows in {split_mode}/fold_{fold}"
                )
            scaler = StandardScaler()
            x_train = scaler.fit_transform(reg_features[train_valid])
            x_val = scaler.transform(reg_features[val_valid])
            regressor = LGBMRegressor(
                **reg_config["params"],
                random_state=random_state,
                n_jobs=2,
                verbosity=-1,
            )
            regressor.fit(x_train, targets[train_valid])
            prediction = regressor.predict(x_val)
            regression_records.extend(
                {
                    "task": "regression",
                    "split": split_mode,
                    "fold": fold,
                    "row_idx": int(row_idx),
                    "aptamer_encoder": reg_config["aptamer_encoder"],
                    "molecule_encoder": reg_config["molecule_encoder"],
                    "y_true": float(target),
                    "y_pred": float(predicted),
                }
                for row_idx, target, predicted in zip(
                    val_valid, targets[val_valid], prediction
                )
            )

    classification_oof = pd.DataFrame(classification_records)
    regression_oof = pd.DataFrame(regression_records)
    validate_oof_coverage(classification_oof, regression_oof, df, splits)
    return classification_oof, regression_oof


def validate_oof_coverage(
    classification_oof: pd.DataFrame,
    regression_oof: pd.DataFrame,
    df: pd.DataFrame,
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Ensure every eligible validation row appears exactly once per split."""
    valid_regression = set(np.flatnonzero(df["pKd_value"].notna().to_numpy()))
    for split_mode in SPLIT_MODES:
        expected = np.concatenate([val for _, val in splits[split_mode]])
        if not np.array_equal(np.sort(expected), np.arange(len(df))):
            raise ValueError(f"Released folds do not cover dataset: {split_mode}")
        cls = classification_oof.loc[classification_oof["split"] == split_mode]
        reg = regression_oof.loc[regression_oof["split"] == split_mode]
        if cls["row_idx"].duplicated().any() or set(cls["row_idx"]) != set(expected):
            raise ValueError(f"Invalid classification OOF coverage: {split_mode}")
        if reg["row_idx"].duplicated().any() or set(reg["row_idx"]) != valid_regression:
            raise ValueError(f"Invalid regression OOF coverage: {split_mode}")


def classification_scores(group: pd.DataFrame) -> dict[str, float]:
    y_true = group["y_true"].to_numpy(dtype=int)
    y_score = group["y_score"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=int)
    both_classes = np.unique(y_true).size == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else np.nan,
        "pr_auc": (
            float(average_precision_score(y_true, y_score))
            if both_classes
            else np.nan
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if both_classes else np.nan,
    }


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(y_true, y_pred)
    return float(result.statistic)


def regression_scores(group: pd.DataFrame) -> dict[str, float]:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def dna_rna_metrics(
    classification_oof: pd.DataFrame,
    regression_oof: pd.DataFrame,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fold rows and fold/pooled summaries by nucleic-acid type."""
    metadata = df[["type", "label"]].reset_index().rename(columns={"index": "row_idx"})
    cls = classification_oof.merge(metadata, on="row_idx", validate="many_to_one")
    reg = regression_oof.merge(metadata, on="row_idx", validate="many_to_one")
    fold_rows = []
    summaries = []
    task_specs = (
        ("classification", cls, ("roc_auc", "pr_auc", "mcc"), classification_scores),
        ("regression", reg, ("r2", "rmse", "mae", "spearman"), regression_scores),
    )

    for task, predictions, metrics, score_function in task_specs:
        for (split_mode, subtype), pooled in predictions.groupby(["split", "type"]):
            current_fold_rows = []
            for fold, part in pooled.groupby("fold"):
                row = {
                    "task": task,
                    "split": split_mode,
                    "split_label": SPLIT_LABELS[split_mode],
                    "type": subtype,
                    "fold": int(fold),
                    "n": int(len(part)),
                    "active": int((part["label"] == 1).sum()) if task == "classification" else np.nan,
                    "inactive": int((part["label"] == 0).sum()) if task == "classification" else np.nan,
                    "positive_fraction": float(part["label"].mean()) if task == "classification" else np.nan,
                    **score_function(part),
                }
                fold_rows.append(row)
                current_fold_rows.append(row)
            fold_frame = pd.DataFrame(current_fold_rows)
            pooled_scores = score_function(pooled)
            summary = {
                "task": task,
                "split": split_mode,
                "split_label": SPLIT_LABELS[split_mode],
                "type": subtype,
                "n": int(len(pooled)),
                "active": int((pooled["label"] == 1).sum()) if task == "classification" else np.nan,
                "inactive": int((pooled["label"] == 0).sum()) if task == "classification" else np.nan,
                "positive_fraction": float(pooled["label"].mean()) if task == "classification" else np.nan,
            }
            fallback_metrics = []
            for metric in metrics:
                valid = fold_frame[metric].dropna()
                summary[f"{metric}_fold_mean"] = float(valid.mean()) if len(valid) else np.nan
                summary[f"{metric}_fold_std"] = float(valid.std(ddof=0)) if len(valid) else np.nan
                summary[f"{metric}_valid_folds"] = int(len(valid))
                summary[f"{metric}_pooled"] = pooled_scores[metric]
                if len(valid) < pooled["fold"].nunique():
                    fallback_metrics.append(metric)
            summary["aggregation"] = (
                "pooled_oof_fallback" if fallback_metrics else "fold_mean"
            )
            summary["fallback_metrics"] = ",".join(fallback_metrics)
            summaries.append(summary)
    return pd.DataFrame(fold_rows), pd.DataFrame(summaries)


def build_rsapred_category_map(
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct categories from source filenames using benchmark processing."""
    processed_frames = []
    for priority, (filename, category) in enumerate(RSAPRED_FILES):
        path = Path(raw_dir) / filename
        raw = pd.read_csv(path, sep="\t")
        raw = canonicalize_smiles_column(raw, "SMILES")
        raw = clean_and_validate_sequences(raw, seq_col="Target_RNA_sequence")
        raw = process_binding_constants(raw, column="pKd", unit="pKd")
        raw["category"] = category
        raw["category_priority"] = priority
        processed_frames.append(
            raw[
                [
                    *JOIN_COLUMNS,
                    "category",
                    "category_priority",
                    "Entry_ID",
                    "Molecule_ID",
                    "Target_RNA_ID",
                ]
            ]
        )
    combined = pd.concat(processed_frames, ignore_index=True).dropna(
        subset=list(JOIN_COLUMNS)
    )
    ambiguity = (
        combined.groupby(list(JOIN_COLUMNS), dropna=False)["category"]
        .nunique()
        .rename("category_count")
        .reset_index()
    )
    ambiguous_keys = ambiguity.loc[ambiguity["category_count"] > 1]
    category_map = (
        combined.sort_values("category_priority")
        .drop_duplicates(list(JOIN_COLUMNS), keep="first")
        .reset_index(drop=True)
    )
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "raw_rows", "value": sum(len(frame) for frame in processed_frames)},
            {"diagnostic": "unique_processed_keys", "value": len(category_map)},
            {"diagnostic": "ambiguous_cross_category_keys", "value": len(ambiguous_keys)},
        ]
    )
    return category_map, diagnostics


def annotate_rsapred(
    df: pd.DataFrame,
    category_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rsapred = (
        df.loc[df["source"] == "RSAPred"]
        .reset_index()
        .rename(columns={"index": "row_idx"})
    )
    annotated = rsapred.merge(
        category_map[[*JOIN_COLUMNS, "category"]],
        on=list(JOIN_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "benchmark_rsapred_rows", "value": len(annotated)},
            {
                "diagnostic": "mapped_benchmark_rows",
                "value": int(annotated["category"].notna().sum()),
            },
            {
                "diagnostic": "unmapped_benchmark_rows",
                "value": int(annotated["category"].isna().sum()),
            },
        ]
    )
    return annotated, diagnostics


def rsapred_descriptives(annotated: pd.DataFrame) -> pd.DataFrame:
    """Produce the reviewer-requested category count table."""
    data = annotated.copy()
    data["category"] = data["category"].fillna("unmapped")
    rows = []
    for category, part in data.groupby("category", sort=False):
        rows.append(
            {
                "Category": category,
                "Records": int(len(part)),
                "Unique sequences": int(part["sequence"].nunique()),
                "Unique molecules": int(part["canonical_smiles"].nunique()),
                "Active": int((part["label"] == 1).sum()),
                "Inactive": int((part["label"] == 0).sum()),
                "With pKd": int(part["pKd_value"].notna().sum()),
            }
        )
    order = {category: index for index, (_, category) in enumerate(RSAPRED_FILES)}
    result = pd.DataFrame(rows)
    result["_order"] = result["Category"].map(order).fillna(len(order))
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def rsapred_subtype_metrics(
    classification_oof: pd.DataFrame,
    annotated: pd.DataFrame,
    *,
    min_per_class: int = 20,
) -> pd.DataFrame:
    """Compute pooled OOF subtype metrics only for supported categories."""
    metadata = annotated[["row_idx", "category", "label"]]
    predictions = classification_oof.merge(
        metadata, on="row_idx", how="inner", validate="many_to_one"
    )
    rows = []
    for (split_mode, category), part in predictions.groupby(["split", "category"]):
        active = int((part["label"] == 1).sum())
        inactive = int((part["label"] == 0).sum())
        eligible = active >= min_per_class and inactive >= min_per_class
        scores = (
            classification_scores(part)
            if eligible
            else {"roc_auc": np.nan, "pr_auc": np.nan, "mcc": np.nan}
        )
        rows.append(
            {
                "split": split_mode,
                "split_label": SPLIT_LABELS[split_mode],
                "category": category,
                "n": int(len(part)),
                "active": active,
                "inactive": inactive,
                "status": "reported" if eligible else "insufficient_class_support",
                "min_per_class": min_per_class,
                "aggregation": "pooled_oof",
                **scores,
            }
        )
    return pd.DataFrame(rows)


def reviewer_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create the compact headline table requested by the reviewers."""
    rows = []
    for _, row in summary.iterrows():
        classification = row["task"] == "classification"
        main = "ROC-AUC" if classification else "R²"
        secondary = "MCC" if classification else "RMSE"
        main_key = "roc_auc" if classification else "r2"
        secondary_key = "mcc" if classification else "rmse"
        use_pooled = row["aggregation"] == "pooled_oof_fallback"
        suffix = "pooled" if use_pooled else "fold_mean"
        rows.append(
            {
                "Task": row["task"].capitalize(),
                "Split": row["split_label"],
                "Type": row["type"],
                "n": int(row["n"]),
                "Class balance": (
                    f"{int(row['active'])}/{int(row['inactive'])}"
                    if classification
                    else ""
                ),
                "Main metric": main,
                "Main value": row[f"{main_key}_{suffix}"],
                "Secondary metric": secondary,
                "Secondary value": row[f"{secondary_key}_{suffix}"],
                "Aggregation": row["aggregation"],
            }
        )
    return pd.DataFrame(rows)
