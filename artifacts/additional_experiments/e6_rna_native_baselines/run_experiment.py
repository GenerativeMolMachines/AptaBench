#!/usr/bin/env python
"""Reproduce E6: RNA-native aptamer representation baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.encoders.aptamer_encoders import (  # noqa: E402
    dnabert2_embed,
    kmer_freq_with_type_bit,
    release_rna_native_models,
    rinalmo_embed,
    rnafm_embed,
)
from src.encoders.molecule_encoders import morgan_fp  # noqa: E402
from src.models.rna_native_baselines import (  # noqa: E402
    APTAMER_ENCODERS,
    ENCODER_FAMILY,
    MODEL_PROVENANCE,
    SPLIT_MODES,
    TASK_MOLECULE_ENCODER,
    TASKS,
    evaluate_best_params,
    file_sha256,
    frame_fingerprint,
    load_or_compute_embedding,
    load_or_create_task_splits,
    optimize_lgbm,
    package_versions,
    prepare_task_subsets,
)


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "results" / "e6_rna_native_raw_results.csv"
DEFAULT_SPLITS_DIR = EXPERIMENT_DIR
DEFAULT_STUDIES_DIR = EXPERIMENT_DIR / "studies"
DEFAULT_CACHE = REPO_ROOT / "artifacts" / "cache" / "e6_embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help="Directory that stores/reuses frozen task-specific splits/.",
    )
    parser.add_argument("--studies-dir", type=Path, default=DEFAULT_STUDIES_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument(
        "--only-encoder",
        action="append",
        choices=APTAMER_ENCODERS,
        default=None,
    )
    parser.add_argument(
        "--only-split",
        action="append",
        choices=SPLIT_MODES,
        default=None,
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun embeddings, splits, Optuna studies, and overwrite raw results.",
    )
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--force-studies", action="store_true")
    parser.add_argument(
        "--generate-splits-only",
        action="store_true",
        help="Create frozen RNA-subset splits without fitting models.",
    )
    return parser.parse_args()


def _selected(values: list[str] | None, available: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(values) if values else available


def _load_existing(output_file: Path, force: bool) -> pd.DataFrame:
    if force or not output_file.exists():
        return pd.DataFrame()
    return pd.read_csv(output_file)


def _condition_complete(
    results: pd.DataFrame,
    *,
    task: str,
    split_mode: str,
    encoder: str,
    n_splits: int,
    n_trials: int,
) -> bool:
    if results.empty:
        return False
    rows = results.loc[
        (results["task"] == task)
        & (results["split"] == split_mode)
        & (results["aptamer_encoder"] == encoder)
    ]
    if len(rows) != n_splits:
        return False
    if set(rows["fold"].astype(int)) != set(range(n_splits)):
        return False
    return set(rows["n_trials_completed"].astype(int)) == {n_trials}


def _remove_condition(
    results: pd.DataFrame,
    *,
    task: str,
    split_mode: str,
    encoder: str,
) -> pd.DataFrame:
    if results.empty:
        return results
    mask = (
        (results["task"] == task)
        & (results["split"] == split_mode)
        & (results["aptamer_encoder"] == encoder)
    )
    return results.loc[~mask].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    studies_dir = args.studies_dir.resolve()
    studies_dir.mkdir(parents=True, exist_ok=True)
    splits_root = args.splits_dir.resolve()
    splits_root.mkdir(parents=True, exist_ok=True)

    tasks = TASKS if args.task == "all" else (args.task,)
    encoders = _selected(args.only_encoder, APTAMER_ENCODERS)
    split_modes = _selected(args.only_split, SPLIT_MODES)
    force_embeddings = args.force or args.force_embeddings
    force_splits = args.force or args.force_splits
    force_studies = args.force or args.force_studies

    results = _load_existing(output_file, args.force)
    source = pd.read_csv(args.dataset)
    subsets = prepare_task_subsets(source)
    rna = subsets["classification"]
    original_to_rna = {
        int(original): position
        for position, original in enumerate(rna["original_index"])
    }

    if args.generate_splits_only:
        for task in tasks:
            load_or_create_task_splits(
                subsets[task],
                splits_root,
                task,
                n_splits=args.n_splits,
                random_state=args.seed,
                force=force_splits,
            )
            print(f"splits ready task={task}")
        return

    aptamer_functions = {
        "Kmer4": lambda values: kmer_freq_with_type_bit(values, k=4),
        "DNABERT2": dnabert2_embed,
        "RNA-FM": rnafm_embed,
        "RiNALMo-Mega": rinalmo_embed,
    }
    embeddings = {}
    cache_manifest = {}
    for name in encoders:
        embeddings[name], cache_manifest[name] = load_or_compute_embedding(
            rna["sequence"].tolist(),
            name,
            aptamer_functions[name],
            args.embeddings_dir,
            force=force_embeddings,
        )
        if name in {"RNA-FM", "RiNALMo-Mega"}:
            release_rna_native_models()

    # Fixed molecule fingerprint for both tasks (fair DNA vs RNA comparison).
    molecule_names = {
        TASK_MOLECULE_ENCODER[task] for task in tasks
    }
    if "MorganFP" in molecule_names:
        embeddings["MorganFP"], cache_manifest["MorganFP"] = (
            load_or_compute_embedding(
                rna["canonical_smiles"].tolist(),
                "MorganFP",
                morgan_fp,
                args.embeddings_dir,
                force=force_embeddings,
            )
        )

    split_fingerprints = {}
    for task in tasks:
        subset = subsets[task]
        subset_positions = np.asarray(
            [original_to_rna[int(index)] for index in subset["original_index"]],
            dtype=int,
        )
        split_bundle = load_or_create_task_splits(
            subset,
            splits_root,
            task,
            n_splits=args.n_splits,
            random_state=args.seed,
            force=force_splits,
        )
        split_fingerprints[task] = frame_fingerprint(
            subset, ["sequence", "canonical_smiles", "label"]
        )
        y = (
            subset["label"].astype(int).to_numpy()
            if task == "classification"
            else subset["pKd_value"].astype(float).to_numpy()
        )
        molecule_name = TASK_MOLECULE_ENCODER[task]
        molecule = embeddings[molecule_name][subset_positions]

        for split_mode in split_modes:
            splits = split_bundle[split_mode]["splits"]
            for encoder_name in encoders:
                if _condition_complete(
                    results,
                    task=task,
                    split_mode=split_mode,
                    encoder=encoder_name,
                    n_splits=args.n_splits,
                    n_trials=args.n_trials,
                ):
                    print(
                        "skip "
                        f"task={task} split={split_mode} encoder={encoder_name}"
                    )
                    continue

                print(
                    "run "
                    f"task={task} split={split_mode} encoder={encoder_name}"
                )
                aptamer = embeddings[encoder_name][subset_positions]
                X = np.concatenate([aptamer, molecule], axis=1).astype(
                    np.float32, copy=False
                )
                study_name = (
                    f"e6_{task}_{split_mode}_"
                    f"{encoder_name.lower().replace('-', '_')}_"
                    f"{split_fingerprints[task]}"
                )
                study_path = studies_dir / f"{study_name}.log"
                best_params, study_details = optimize_lgbm(
                    task,
                    X,
                    y,
                    splits,
                    study_path,
                    study_name,
                    n_trials=args.n_trials,
                    random_state=args.seed,
                    n_jobs=args.n_jobs,
                    force=force_studies,
                )
                fold_frame = evaluate_best_params(
                    task,
                    split_mode,
                    encoder_name,
                    molecule_name,
                    X,
                    y,
                    splits,
                    best_params,
                    random_state=args.seed,
                    n_jobs=args.n_jobs,
                )
                fold_frame["n_trials_completed"] = study_details[
                    "n_trials_completed"
                ]
                fold_frame["encoder_family"] = ENCODER_FAMILY[encoder_name]
                fold_frame["best_params"] = json.dumps(
                    best_params, sort_keys=True
                )
                fold_frame["hpo_best_value"] = study_details["best_value"]

                results = _remove_condition(
                    results,
                    task=task,
                    split_mode=split_mode,
                    encoder=encoder_name,
                )
                results = pd.concat(
                    [results, fold_frame],
                    ignore_index=True,
                )
                results = results.sort_values(
                    ["task", "split", "aptamer_encoder", "fold"]
                ).reset_index(drop=True)
                results.to_csv(output_file, index=False)

    results.to_csv(output_file, index=False)

    provenance = {
        "experiment": "E6 RNA-native baselines",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "subsets": {task: int(len(frame)) for task, frame in subsets.items()},
        "subset_fingerprints": split_fingerprints,
        "tasks": list(tasks),
        "split_modes": list(split_modes),
        "aptamer_encoders": list(encoders),
        "encoder_families": {
            name: ENCODER_FAMILY[name] for name in encoders
        },
        "comparison": {
            "shared_subset": "RNA-only (type==RNA); regression further requires pKd",
            "shared_molecule_encoder": "MorganFP for classification and regression",
            "dna_style_encoders": "Kmer4, DNABERT2 (U->T)",
            "rna_native_encoders": "RNA-FM, RiNALMo-Mega (U preserved)",
        },
        "molecule_encoders": {
            task: TASK_MOLECULE_ENCODER[task] for task in tasks
        },
        "model_provenance": MODEL_PROVENANCE,
        "embedding_cache": cache_manifest,
        "n_splits": args.n_splits,
        "n_trials_per_task_split_encoder": args.n_trials,
        "random_state": args.seed,
        "classification_optimization": "ROC-AUC (maximize)",
        "regression_optimization": "RMSE (minimize)",
        "u_to_t_conversion_for_rna_native_encoders": False,
        "raw_results": str(output_file),
        "package_versions": package_versions(),
    }
    (EXPERIMENT_DIR / "run_manifest.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(f"Raw E6 results written to {output_file}")


if __name__ == "__main__":
    main()
