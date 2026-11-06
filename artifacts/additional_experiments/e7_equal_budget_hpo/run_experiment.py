#!/usr/bin/env python
"""Reproduce E7: equal-budget DL HPO and regression-specific HPO.

Writes only raw numeric outputs into this experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUTF8", "1")

EXPERIMENT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "dataset" / "AptaBench_dataset.csv").exists() and (
            candidate / "src"
        ).is_dir():
            return candidate
    raise RuntimeError("Could not locate the AptaBench repository root")


REPO_ROOT = find_repo_root(EXPERIMENT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import torch  # noqa: E402

from e7_core import (  # noqa: E402
    MODEL_NAMES,
    SEARCH_FOLD_COUNT,
    SPLIT_MODES,
    aggregate_folds,
    evaluate_dl_fold,
    evaluate_lgbm_classification_fold,
    evaluate_regression_fold,
    load_released_splits,
    select_top_regression_combinations,
    set_global_seed,
    suggest_classification_params,
    suggest_regression_params,
)


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_SPLITS = REPO_ROOT / "dataset" / "splits"
DEFAULT_BASELINE = EXPERIMENT_DIR / "inputs" / "regression_leaderboard.csv"
DEFAULT_CACHE = REPO_ROOT / "artifacts" / "cache" / "embeddings"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results"

SPLIT_LABELS = {
    "stratified": "In-distribution",
    "disjoint_molecule": "Molecule-disjoint",
    "disjoint_aptamer": "Aptamer-disjoint",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _storage_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _load_study(path: Path, study_name: str, seed: int, direction: str) -> optuna.Study:
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=study_name,
        storage=_storage_url(path),
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        load_if_exists=True,
    )


def _complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def _optimize_exact(study: optuna.Study, objective: Callable, budget: int) -> None:
    complete = len(_complete_trials(study))
    if complete > budget:
        raise ValueError(f"Study {study.study_name} already exceeds budget {budget}")
    while complete < budget:
        before = len(study.trials)
        study.optimize(objective, n_trials=1, gc_after_trial=True)
        complete = len(_complete_trials(study))
        if len(study.trials) == before or study.trials[-1].state == optuna.trial.TrialState.FAIL:
            raise RuntimeError(
                f"Trial failed in {study.study_name}; equal completed-trial budget not met"
            )


def _feature_cache_path(cache_dir: Path, kind: str, name: str, dataset_hash: str) -> Path:
    return cache_dir / f"e7_{kind}_{name}_{dataset_hash[:16]}.npy"


def _compute_encoder(name: str, values: list[str], kind: str) -> np.ndarray:
    if kind == "apt":
        from src.encoders.aptamer_encoders import (
            dnabert2_embed,
            gena_embed,
            kmer_freq_with_type_bit,
            onehot_with_type_bit,
        )

        functions: dict[str, tuple[Callable, dict[str, Any]]] = {
            "OneHot": (onehot_with_type_bit, {}),
            "Kmer3": (kmer_freq_with_type_bit, {"k": 3}),
            "Kmer4": (kmer_freq_with_type_bit, {"k": 4}),
            "GENA": (gena_embed, {}),
            "DNABERT2": (dnabert2_embed, {}),
        }
    else:
        from src.encoders.molecule_encoders import (
            chemberta_embed,
            clamp_embed,
            molformer_embed,
            morgan_fp,
            physchem_descriptors,
        )

        functions = {
            "MorganFP": (morgan_fp, {}),
            "PhysChem": (physchem_descriptors, {}),
            "ChemBERTa": (chemberta_embed, {}),
            "MolFormer": (molformer_embed, {}),
            "CLAMP": (clamp_embed, {}),
        }
    if name not in functions:
        raise ValueError(f"Unsupported E7 {kind} encoder: {name}")
    function, kwargs = functions[name]
    return np.asarray(function(values, **kwargs))


def load_or_compute_feature(
    df: pd.DataFrame,
    *,
    name: str,
    kind: str,
    dataset_hash: str,
    cache_dir: Path,
    force: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _feature_cache_path(cache_dir, kind, name, dataset_hash)
    if path.exists() and not force:
        features = np.load(path)
    else:
        column = "sequence" if kind == "apt" else "canonical_smiles"
        features = _compute_encoder(
            name,
            df[column].fillna("").astype(str).tolist(),
            kind,
        )
        np.save(path, features)
    if len(features) != len(df) or features.ndim != 2:
        raise ValueError(f"Invalid {kind}/{name} feature shape: {features.shape}")
    return features.astype(np.float32, copy=False)


def _trial_seconds(study: optuna.Study) -> float:
    return float(
        sum(
            trial.duration.total_seconds()
            for trial in _complete_trials(study)
            if trial.duration is not None
        )
    )


def run_classification(
    args: argparse.Namespace,
    df: pd.DataFrame,
    dataset_hash: str,
) -> list[optuna.Study]:
    output_dir = args.output_dir
    fold_path = output_dir / "e7_classification_fold_results.csv"
    existing = (
        pd.read_csv(fold_path)
        if fold_path.exists() and not args.force
        else pd.DataFrame()
    )
    labels = df["label"].astype(int).to_numpy()
    apt = load_or_compute_feature(
        df,
        name="OneHot",
        kind="apt",
        dataset_hash=dataset_hash,
        cache_dir=args.cache_dir,
        force=args.force_features,
    )
    mol = load_or_compute_feature(
        df,
        name="MorganFP",
        kind="mol",
        dataset_hash=dataset_hash,
        cache_dir=args.cache_dir,
        force=args.force_features,
    )
    concatenated = np.concatenate((apt, mol), axis=1).astype(np.float32)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    studies: list[optuna.Study] = []
    final_rows: list[dict[str, Any]] = existing.to_dict("records")
    split_modes = args.only_split or list(SPLIT_MODES)
    models = args.only_model or list(MODEL_NAMES)

    for split_mode in split_modes:
        folds = load_released_splits(args.splits_dir, split_mode)
        for model_name in models:
            study_name = f"e7_cls_{split_mode}_{model_name.lower()}"
            study_path = args.studies_dir / f"{study_name}.db"
            study = _load_study(study_path, study_name, args.seed, "maximize")
            if not args.analysis_only:

                def objective(trial: optuna.Trial) -> float:
                    params = suggest_classification_params(trial, model_name)
                    fold_metrics = []
                    for fold_index, (train_idx, val_idx) in enumerate(
                        folds[:SEARCH_FOLD_COUNT]
                    ):
                        fold_seed = args.seed + fold_index
                        if model_name == "LightGBM":
                            metrics = evaluate_lgbm_classification_fold(
                                params,
                                concatenated,
                                labels,
                                train_idx,
                                val_idx,
                                seed=fold_seed,
                            )
                        else:
                            metrics = evaluate_dl_fold(
                                model_name,
                                params,
                                apt,
                                mol,
                                labels,
                                train_idx,
                                val_idx,
                                seed=fold_seed,
                                device=device,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                            )
                        fold_metrics.append(metrics)
                        trial.report(
                            float(np.mean([row["roc_auc"] for row in fold_metrics])),
                            fold_index,
                        )
                    return float(np.mean([row["roc_auc"] for row in fold_metrics]))

                _optimize_exact(study, objective, args.classification_trials)

            studies.append(study)
            if len(_complete_trials(study)) != args.classification_trials:
                raise ValueError(
                    f"{study_name} does not have the required completed-trial budget"
                )

            already_final = (
                not existing.empty
                and (
                    (existing["split"] == split_mode)
                    & (existing["model"] == model_name)
                ).sum()
                == len(folds)
            )
            if args.analysis_only or already_final:
                continue

            # Drop previous rows for this pair when --force.
            final_rows = [
                row
                for row in final_rows
                if not (row["split"] == split_mode and row["model"] == model_name)
            ]
            best_params = study.best_params
            for fold_index, (train_idx, val_idx) in enumerate(folds):
                fold_seed = args.seed + fold_index
                if model_name == "LightGBM":
                    metrics = evaluate_lgbm_classification_fold(
                        best_params,
                        concatenated,
                        labels,
                        train_idx,
                        val_idx,
                        seed=fold_seed,
                    )
                else:
                    metrics = evaluate_dl_fold(
                        model_name,
                        best_params,
                        apt,
                        mol,
                        labels,
                        train_idx,
                        val_idx,
                        seed=fold_seed,
                        device=device,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                    )
                final_rows.append(
                    {
                        "split": split_mode,
                        "model": model_name,
                        "fold": fold_index,
                        **metrics,
                    }
                )
            pd.DataFrame(final_rows).to_csv(fold_path, index=False)
            print(f"finished {split_mode}/{model_name}")

    publish_classification(output_dir, studies, args.classification_trials)
    return studies


def publish_classification(
    output_dir: Path,
    studies: list[optuna.Study],
    trials: int,
) -> None:
    folds = pd.read_csv(output_dir / "e7_classification_fold_results.csv")
    rows = []
    for (split_mode, model_name), group in folds.groupby(
        ["split", "model"], sort=False
    ):
        study_name = f"e7_cls_{split_mode}_{model_name.lower()}"
        study = next(study for study in studies if study.study_name == study_name)
        summary = aggregate_folds(group.to_dict("records"), ("roc_auc", "mcc"))
        rows.append(
            {
                "Split": SPLIT_LABELS.get(split_mode, split_mode),
                "split": split_mode,
                "Model": model_name,
                "Trials": trials,
                "ROC-AUC": summary["roc_auc_mean"],
                "ROC-AUC_std": summary["roc_auc_std"],
                "MCC": summary["mcc_mean"],
                "MCC_std": summary["mcc_std"],
                "Search compute": _trial_seconds(study),
                "Final compute": float(group["seconds"].sum()),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "e7_classification_raw_results.csv",
        index=False,
    )


def run_regression(
    args: argparse.Namespace,
    df: pd.DataFrame,
    dataset_hash: str,
) -> list[optuna.Study]:
    baseline = pd.read_csv(args.regression_baseline)
    selected = select_top_regression_combinations(baseline, top_k=3)
    selected.to_csv(
        args.output_dir / "e7_regression_selected_baselines.csv",
        index=False,
    )
    targets = pd.to_numeric(df["pKd_value"], errors="coerce").to_numpy(dtype=float)
    apt_names = selected["aptamer_encoder"].drop_duplicates().tolist()
    mol_names = selected["molecule_encoder"].drop_duplicates().tolist()
    apt_features = {
        name: load_or_compute_feature(
            df,
            name=name,
            kind="apt",
            dataset_hash=dataset_hash,
            cache_dir=args.cache_dir,
            force=args.force_features,
        )
        for name in apt_names
    }
    mol_features = {
        name: load_or_compute_feature(
            df,
            name=name,
            kind="mol",
            dataset_hash=dataset_hash,
            cache_dir=args.cache_dir,
            force=args.force_features,
        )
        for name in mol_names
    }
    fold_path = args.output_dir / "e7_regression_fold_results.csv"
    existing = (
        pd.read_csv(fold_path)
        if fold_path.exists() and not args.force
        else pd.DataFrame()
    )
    final_rows: list[dict[str, Any]] = existing.to_dict("records")
    studies: list[optuna.Study] = []
    split_modes = args.only_split or list(SPLIT_MODES)

    for row in selected.itertuples(index=False):
        split_mode = row.split
        if split_mode not in split_modes:
            continue
        apt_name = row.aptamer_encoder
        mol_name = row.molecule_encoder
        features = np.concatenate(
            (apt_features[apt_name], mol_features[mol_name]),
            axis=1,
        ).astype(np.float32)
        folds = load_released_splits(args.splits_dir, split_mode)
        safe_pair = f"{apt_name}_{mol_name}".lower()
        study_name = f"e7_reg_{split_mode}_{safe_pair}"
        study_path = args.studies_dir / f"{study_name}.db"
        study = _load_study(study_path, study_name, args.seed, "minimize")
        if not args.analysis_only:

            def objective(trial: optuna.Trial) -> float:
                params = suggest_regression_params(trial)
                metrics = [
                    evaluate_regression_fold(
                        params,
                        features,
                        targets,
                        train_idx,
                        val_idx,
                        seed=args.seed + fold_index,
                    )
                    for fold_index, (train_idx, val_idx) in enumerate(folds)
                ]
                return float(np.mean([metric["rmse"] for metric in metrics]))

            _optimize_exact(study, objective, args.regression_trials)

        studies.append(study)
        if len(_complete_trials(study)) != args.regression_trials:
            raise ValueError(
                f"{study_name} does not have the required completed-trial budget"
            )

        already_final = (
            not existing.empty
            and (
                (existing["split"] == split_mode)
                & (existing["aptamer_encoder"] == apt_name)
                & (existing["molecule_encoder"] == mol_name)
            ).sum()
            == len(folds)
        )
        if args.analysis_only or already_final:
            continue

        final_rows = [
            item
            for item in final_rows
            if not (
                item["split"] == split_mode
                and item["aptamer_encoder"] == apt_name
                and item["molecule_encoder"] == mol_name
            )
        ]
        for fold_index, (train_idx, val_idx) in enumerate(folds):
            metrics = evaluate_regression_fold(
                study.best_params,
                features,
                targets,
                train_idx,
                val_idx,
                seed=args.seed + fold_index,
            )
            final_rows.append(
                {
                    "split": split_mode,
                    "aptamer_encoder": apt_name,
                    "molecule_encoder": mol_name,
                    "fold": fold_index,
                    **metrics,
                }
            )
        pd.DataFrame(final_rows).to_csv(fold_path, index=False)
        print(f"finished regression {split_mode}/{apt_name}x{mol_name}")

    publish_regression(args.output_dir, selected, studies)
    return studies


def publish_regression(
    output_dir: Path,
    selected: pd.DataFrame,
    studies: list[optuna.Study],
) -> None:
    folds = pd.read_csv(output_dir / "e7_regression_fold_results.csv")
    candidates = []
    for keys, group in folds.groupby(
        ["split", "aptamer_encoder", "molecule_encoder"],
        sort=False,
    ):
        split_mode, apt_name, mol_name = keys
        summary = aggregate_folds(group.to_dict("records"), ("rmse", "r2", "mae"))
        study_name = f"e7_reg_{split_mode}_{apt_name}_{mol_name}".lower()
        study = next(study for study in studies if study.study_name == study_name)
        baseline = selected.loc[
            (selected["split"] == split_mode)
            & (selected["aptamer_encoder"] == apt_name)
            & (selected["molecule_encoder"] == mol_name)
        ].iloc[0]
        candidates.append(
            {
                "Split": SPLIT_LABELS.get(split_mode, split_mode),
                "split": split_mode,
                "aptamer_encoder": apt_name,
                "molecule_encoder": mol_name,
                "Previous R2": float(baseline["R2 mean"]),
                "Previous RMSE": float(baseline["RMSE mean"]),
                "Tuned R2": summary["r2_mean"],
                "Tuned R2_std": summary["r2_std"],
                "Tuned RMSE": summary["rmse_mean"],
                "Tuned RMSE_std": summary["rmse_std"],
                "Tuned MAE": summary["mae_mean"],
                "Search compute": _trial_seconds(study),
                "Final compute": float(group["seconds"].sum()),
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_csv(
        output_dir / "e7_regression_raw_results.csv",
        index=False,
    )
    rows = []
    for split_mode in candidate_frame["split"].unique():
        tuned = candidate_frame.loc[
            candidate_frame["split"] == split_mode
        ].sort_values("Tuned RMSE").iloc[0]
        rows.append(
            {
                "Split": tuned["Split"],
                "Previous R2": tuned["Previous R2"],
                "Previous RMSE": tuned["Previous RMSE"],
                "Tuned R2": tuned["Tuned R2"],
                "Tuned RMSE": tuned["Tuned RMSE"],
                "aptamer_encoder": tuned["aptamer_encoder"],
                "molecule_encoder": tuned["molecule_encoder"],
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "e7_regression_best_per_split.csv",
        index=False,
    )


def write_best_parameters(studies: list[optuna.Study], output_dir: Path) -> None:
    payload = {
        study.study_name: {
            "direction": study.direction.name,
            "value": study.best_value,
            "params": study.best_params,
            "completed_trials": len(_complete_trials(study)),
        }
        for study in studies
    }
    (output_dir / "e7_best_parameters.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_provenance(
    args: argparse.Namespace,
    dataset_hash: str,
    df: pd.DataFrame,
    studies: list[optuna.Study],
    started: float,
) -> None:
    split_hashes = {
        split_mode: file_sha256(args.splits_dir / f"{split_mode}.json")
        for split_mode in (args.only_split or SPLIT_MODES)
    }
    payload = {
        "experiment": "E7 equal-budget DL and regression HPO",
        "reviewer": "gXd3 W2/Q6",
        "dataset": str(args.dataset.relative_to(REPO_ROOT)),
        "dataset_sha256": dataset_hash,
        "n_rows": len(df),
        "split_sha256": split_hashes,
        "regression_leaderboard": str(
            args.regression_baseline.relative_to(REPO_ROOT)
            if args.regression_baseline.is_relative_to(REPO_ROOT)
            else args.regression_baseline
        ),
        "regression_leaderboard_sha256": file_sha256(args.regression_baseline),
        "seed": args.seed,
        "classification": {
            "features": ["OneHot", "MorganFP"],
            "models": list(MODEL_NAMES),
            "trials_per_model_split": args.classification_trials,
            "search_folds": [0, 1, 2],
            "final_folds": [0, 1, 2, 3, 4],
            "objective": "mean validation ROC-AUC",
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "multitask_included": False,
        },
        "regression": {
            "selection": "three lowest baseline RMSE mean per split",
            "trials_per_encoder_split": args.regression_trials,
            "folds": [0, 1, 2, 3, 4],
            "objective": "mean validation RMSE",
            "model": "LightGBMRegressor",
        },
        "timing": (
            "wall-clock seconds; Search compute sums Optuna trial durations; "
            "Final compute sums five-fold refit seconds"
        ),
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "python": sys.version,
        "package_versions": {
            package: _version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "optuna",
                "torch",
                "rdkit",
            )
        },
        "studies": {
            study.study_name: len(_complete_trials(study)) for study in studies
        },
        "total_wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "e7_provenance.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "classification", "regression"),
        default="all",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--regression-baseline",
        type=Path,
        default=DEFAULT_BASELINE,
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--studies-dir",
        type=Path,
        default=EXPERIMENT_DIR / "studies",
    )
    parser.add_argument("--only-split", action="append", choices=SPLIT_MODES)
    parser.add_argument("--only-model", action="append", choices=MODEL_NAMES)
    parser.add_argument("--classification-trials", type=int, default=15)
    parser.add_argument("--regression-trials", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute final fold outputs; studies remain budget-protected.",
    )
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.dataset = args.dataset.resolve()
    args.splits_dir = args.splits_dir.resolve()
    args.regression_baseline = args.regression_baseline.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.studies_dir = args.studies_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.studies_dir.mkdir(parents=True, exist_ok=True)
    if args.classification_trials < 1 or args.regression_trials < 1:
        raise ValueError("Trial budgets must be positive")
    set_global_seed(args.seed)
    df = pd.read_csv(args.dataset)
    dataset_hash = file_sha256(args.dataset)
    studies: list[optuna.Study] = []
    if args.task in ("all", "classification"):
        studies.extend(run_classification(args, df, dataset_hash))
    if args.task in ("all", "regression"):
        studies.extend(run_regression(args, df, dataset_hash))
    write_best_parameters(studies, args.output_dir)
    write_provenance(args, dataset_hash, df, studies, started)
    print(f"Wrote raw outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
