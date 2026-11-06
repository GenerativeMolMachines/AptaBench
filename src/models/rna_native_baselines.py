"""E6 RNA-native representation baselines."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.models.source_ablation import (
    generate_scope_splits,
    validate_split_bundle,
)


SPLIT_MODES = ("stratified", "disjoint_aptamer", "disjoint_molecule")
SPLIT_LABELS = {
    "stratified": "In-distribution",
    "disjoint_aptamer": "Aptamer-disjoint",
    "disjoint_molecule": "Molecule-disjoint",
}
APTAMER_ENCODERS = ("Kmer4", "DNABERT2", "RNA-FM", "RiNALMo-Mega")
TASKS = ("classification", "regression")
# Same fingerprint for both tasks so DNA vs RNA aptamer encoders are comparable.
TASK_MOLECULE_ENCODER = {
    "classification": "MorganFP",
    "regression": "MorganFP",
}
ENCODER_FAMILY = {
    "Kmer4": "dna_style",
    "DNABERT2": "dna_style",
    "RNA-FM": "rna_native",
    "RiNALMo-Mega": "rna_native",
}
TASK_PRIMARY_METRIC = {
    "classification": "roc_auc",
    "regression": "r2",
}
TASK_SECONDARY_METRIC = {
    "classification": "mcc",
    "regression": "rmse",
}
MODEL_PROVENANCE = {
    "RNA-FM": {
        "package": "rna-fm",
        "import": "fm",
        "loader": "fm.pretrained.rna_fm_t12",
        "weights": "cuhkaih/rnafm:RNA-FM_pretrained.pth",
        "repr_layer": 12,
        "family": "rna_native",
    },
    "RiNALMo-Mega": {
        "package": "rinalmo",
        "source": "https://github.com/lbcb-sci/RiNALMo",
        "model_name": "mega-v1",
        "config": "mega",
        "weights": "https://zenodo.org/records/15043668/files/rinalmo_mega_pretrained.pt",
        "use_flash_attn": False,
        "family": "rna_native",
    },
    "DNABERT2": {
        "model_id": "zhihan1996/DNABERT-2-117M",
        "alphabet": "U->T",
        "family": "dna_style",
    },
    "Kmer4": {
        "alphabet": "U->T",
        "family": "dna_style",
    },
    "MorganFP": {
        "bits": 1024,
        "radius": 2,
    },
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    hashed = pd.util.hash_pandas_object(df[columns], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]


def prepare_task_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create RNA-only classification and RNA/pKd regression frames."""
    required = {"sequence", "canonical_smiles", "type", "label", "pKd_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing E6 columns: {sorted(missing)}")

    rna = df.loc[df["type"].astype(str).str.upper() == "RNA"].copy()
    rna["original_index"] = rna.index.astype(int)
    rna["sequence"] = rna["sequence"].astype(str).str.upper()
    invalid = rna["sequence"].map(lambda value: sorted(set(value) - set("ACGUN")))
    if invalid.map(bool).any():
        row = invalid.map(bool).idxmax()
        raise ValueError(
            f"RNA row {row} contains unsupported symbols: {invalid.loc[row]}"
        )
    if rna.empty:
        raise ValueError("RNA subset is empty")

    classification = rna.reset_index(drop=True)
    regression = rna.loc[pd.to_numeric(rna["pKd_value"], errors="coerce").notna()]
    regression = regression.copy()
    regression["pKd_value"] = pd.to_numeric(
        regression["pKd_value"], errors="raise"
    )
    regression = regression.reset_index(drop=True)
    if regression.empty:
        raise ValueError("RNA pKd subset is empty")
    return {"classification": classification, "regression": regression}


def generate_task_splits(
    subset: pd.DataFrame,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict]:
    """Generate and validate all protocols directly on one task subset."""
    bundle = generate_scope_splits(
        subset,
        n_splits=n_splits,
        random_state=random_state,
        frac_tolerance=0.02,
        max_tries=200,
    )
    validate_split_bundle(subset, bundle, removed_source=None)
    return bundle


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_task_splits(
    subset: pd.DataFrame,
    bundle: dict[str, dict],
    output_dir: str | Path,
    task: str,
) -> None:
    destination = Path(output_dir) / "splits" / task
    destination.mkdir(parents=True, exist_ok=True)
    original = subset["original_index"].to_numpy(dtype=int)
    fingerprint = frame_fingerprint(
        subset, ["sequence", "canonical_smiles", "label"]
    )
    for split_mode, payload in bundle.items():
        records = []
        for fold, (train_idx, test_idx) in enumerate(payload["splits"]):
            records.append(
                {
                    "fold": fold,
                    "train_idx": np.asarray(train_idx, dtype=int).tolist(),
                    "test_idx": np.asarray(test_idx, dtype=int).tolist(),
                    "train_original_index": original[train_idx].tolist(),
                    "test_original_index": original[test_idx].tolist(),
                }
            )
        document = {
            "task": task,
            "split": split_mode,
            "subset_fingerprint": fingerprint,
            "groups": [_json_scalar(item) for item in payload["groups"]],
            "folds": records,
        }
        (destination / f"{split_mode}.json").write_text(
            json.dumps(document, indent=2),
            encoding="utf-8",
        )


def load_task_splits(
    subset: pd.DataFrame,
    output_dir: str | Path,
    task: str,
) -> dict[str, dict] | None:
    destination = Path(output_dir) / "splits" / task
    expected = frame_fingerprint(
        subset, ["sequence", "canonical_smiles", "label"]
    )
    bundle = {}
    for split_mode in SPLIT_MODES:
        path = destination / f"{split_mode}.json"
        if not path.exists():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("subset_fingerprint") != expected:
            return None
        bundle[split_mode] = {
            "groups": np.asarray(document["groups"]),
            "splits": [
                (
                    np.asarray(fold["train_idx"], dtype=int),
                    np.asarray(fold["test_idx"], dtype=int),
                )
                for fold in document["folds"]
            ],
        }
    validate_split_bundle(subset, bundle, removed_source=None)
    return bundle


def load_or_create_task_splits(
    subset: pd.DataFrame,
    output_dir: str | Path,
    task: str,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    force: bool = False,
) -> dict[str, dict]:
    if not force:
        cached = load_task_splits(subset, output_dir, task)
        if cached is not None:
            return cached
    bundle = generate_task_splits(
        subset,
        n_splits=n_splits,
        random_state=random_state,
    )
    save_task_splits(subset, bundle, output_dir, task)
    return bundle


def load_or_compute_embedding(
    values: list[str],
    name: str,
    encoder: Callable[[list[str]], np.ndarray],
    cache_dir: str | Path,
    *,
    force: bool = False,
) -> tuple[np.ndarray, dict]:
    """Cache one frozen encoder matrix under a content-derived key."""
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    key = digest.hexdigest()[:16]
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / f"{name.lower().replace('-', '_')}_{key}.npy"
    metadata_path = matrix_path.with_suffix(".json")
    if matrix_path.exists() and metadata_path.exists() and not force:
        matrix = np.load(matrix_path)
    else:
        matrix = np.asarray(encoder(values), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(values):
            raise ValueError(
                f"{name} returned shape {matrix.shape} for {len(values)} rows"
            )
        np.save(matrix_path, matrix)
        metadata_path.write_text(
            json.dumps(
                {
                    "encoder": name,
                    "cache_key": key,
                    "rows": len(values),
                    "dimensions": int(matrix.shape[1]),
                    "provenance": MODEL_PROVENANCE.get(name),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return matrix.astype(np.float32, copy=False), {
        "path": str(matrix_path),
        "cache_key": key,
        "shape": list(matrix.shape),
    }


def _suggest_lgbm_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    }


def objective_direction(task: str) -> str:
    if task == "classification":
        return "maximize"
    if task == "regression":
        return "minimize"
    raise ValueError(f"Unknown task: {task}")


def _fit_score(
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict,
    *,
    random_state: int,
    n_jobs: int,
) -> float:
    scores = []
    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        common = {
            **params,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": -1,
        }
        if task == "classification":
            model = LGBMClassifier(**common)
            model.fit(X_train, y[train_idx])
            probability = model.predict_proba(X_test)[:, 1]
            score = (
                roc_auc_score(y[test_idx], probability)
                if np.unique(y[test_idx]).size > 1
                else np.nan
            )
        else:
            model = LGBMRegressor(**common)
            model.fit(X_train, y[train_idx])
            prediction = model.predict(X_test)
            score = np.sqrt(mean_squared_error(y[test_idx], prediction))
        scores.append(score)
    value = float(np.nanmean(scores))
    if not np.isfinite(value):
        raise ValueError(f"No finite HPO score for {task}")
    return value


def optimize_lgbm(
    task: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    study_path: str | Path,
    study_name: str,
    *,
    n_trials: int = 20,
    random_state: int = 42,
    n_jobs: int = 2,
    force: bool = False,
) -> tuple[dict, dict]:
    """Run an equal completed-trial budget with resumable journal storage."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    path = Path(study_path).resolve()
    if path.suffix.lower() == ".db":
        path = path.with_suffix(".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.exists():
        path.unlink()
    try:
        from optuna.storages.journal import JournalFileBackend

        file_storage = JournalFileBackend(str(path))
    except ImportError:  # pragma: no cover - older Optuna
        file_storage = optuna.storages.JournalFileStorage(str(path))
    storage = optuna.storages.JournalStorage(file_storage)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=objective_direction(task),
        sampler=optuna.samplers.TPESampler(seed=random_state),
        load_if_exists=True,
    )
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )

    def objective(trial):
        return _fit_score(
            task,
            X,
            y,
            splits,
            _suggest_lgbm_params(trial),
            random_state=random_state,
            n_jobs=n_jobs,
        )

    remaining = max(0, n_trials - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    if completed < n_trials:
        raise RuntimeError(
            f"{study_name} completed {completed}/{n_trials} Optuna trials"
        )
    details = {
        "study_name": study_name,
        "study_path": str(path),
        "direction": objective_direction(task),
        "n_trials_completed": completed,
        "best_value": float(study.best_value),
    }
    return dict(study.best_params), details


def evaluate_best_params(
    task: str,
    split_mode: str,
    aptamer_encoder: str,
    molecule_encoder: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict,
    *,
    random_state: int = 42,
    n_jobs: int = 2,
) -> pd.DataFrame:
    records = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        common = {
            **params,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": -1,
        }
        record = {
            "task": task,
            "split": split_mode,
            "fold": fold,
            "aptamer_encoder": aptamer_encoder,
            "molecule_encoder": molecule_encoder,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }
        if task == "classification":
            model = LGBMClassifier(**common)
            model.fit(X_train, y[train_idx])
            probability = model.predict_proba(X_test)[:, 1]
            prediction = model.predict(X_test)
            record["roc_auc"] = (
                float(roc_auc_score(y[test_idx], probability))
                if np.unique(y[test_idx]).size > 1
                else np.nan
            )
            record["mcc"] = (
                float(matthews_corrcoef(y[test_idx], prediction))
                if np.unique(y[test_idx]).size > 1
                else np.nan
            )
        else:
            model = LGBMRegressor(**common)
            model.fit(X_train, y[train_idx])
            prediction = model.predict(X_test)
            record["r2"] = (
                float(r2_score(y[test_idx], prediction))
                if len(test_idx) >= 2
                else np.nan
            )
            record["rmse"] = float(
                np.sqrt(mean_squared_error(y[test_idx], prediction))
            )
        records.append(record)
    return pd.DataFrame(records)


def summarize_fold_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "split", "aptamer_encoder", "molecule_encoder"]
    records = []
    for key, part in fold_results.groupby(keys, sort=False):
        task = key[0]
        metrics = (
            ("roc_auc", "mcc")
            if task == "classification"
            else ("r2", "rmse")
        )
        record = dict(zip(keys, key))
        record["n"] = int(part["n_test"].sum())
        record["n_folds"] = int(len(part))
        for metric in metrics:
            record[f"{metric}_mean"] = float(part[metric].mean())
            record[f"{metric}_std"] = float(part[metric].std(ddof=0))
        records.append(record)
    return pd.DataFrame(records)


def reviewer_table(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in summary.itertuples(index=False):
        primary = TASK_PRIMARY_METRIC[row.task]
        secondary = TASK_SECONDARY_METRIC[row.task]
        records.append(
            {
                "Task": row.task.capitalize(),
                "Split": SPLIT_LABELS[row.split],
                "Aptamer encoder": row.aptamer_encoder,
                "n": row.n,
                "ROC-AUC/R²": (
                    f"{getattr(row, primary + '_mean'):.3f} ± "
                    f"{getattr(row, primary + '_std'):.3f}"
                ),
                "MCC/RMSE": (
                    f"{getattr(row, secondary + '_mean'):.3f} ± "
                    f"{getattr(row, secondary + '_std'):.3f}"
                ),
            }
        )
    return pd.DataFrame(records)


def transfer_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    """Quantify ID-to-aptamer-disjoint degradation and change vs Kmer4."""
    records = []
    for (task, encoder), part in summary.groupby(
        ["task", "aptamer_encoder"], sort=False
    ):
        indexed = part.set_index("split")
        if not {"stratified", "disjoint_aptamer"} <= set(indexed.index):
            continue
        for metric in (
            TASK_PRIMARY_METRIC[task],
            TASK_SECONDARY_METRIC[task],
        ):
            id_value = float(indexed.loc["stratified", f"{metric}_mean"])
            disjoint_value = float(
                indexed.loc["disjoint_aptamer", f"{metric}_mean"]
            )
            raw_gap = disjoint_value - id_value
            degradation = raw_gap if metric == "rmse" else -raw_gap
            records.append(
                {
                    "task": task,
                    "aptamer_encoder": encoder,
                    "metric": metric,
                    "in_distribution": id_value,
                    "aptamer_disjoint": disjoint_value,
                    "gap_disjoint_minus_id": raw_gap,
                    "absolute_gap": abs(raw_gap),
                    "degradation": degradation,
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    controls = (
        result.loc[result["aptamer_encoder"] == "Kmer4"]
        .set_index(["task", "metric"])["degradation"]
        .to_dict()
    )
    result["degradation_change_vs_kmer4"] = result.apply(
        lambda row: row["degradation"] - controls[(row["task"], row["metric"])],
        axis=1,
    )
    return result


def package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "lightgbm", "optuna", "torch", "rna-fm", "rinalmo")
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    # Official RNA-FM package is installed as rna-fm but imported as fm.
    try:
        import fm

        versions["fm"] = getattr(fm, "__version__", versions.get("rna-fm"))
    except ImportError:
        versions["fm"] = None
    return versions
