#!/usr/bin/env python
"""Reproduce E1: leave-one-source-out source ablation (raw fold results only)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.screening import precompute_and_save_embeddings  # noqa: E402
from src.models.source_ablation import (  # noqa: E402
    CLASSIFICATION_SOURCES,
    REGRESSION_PARAMS,
    REGRESSION_SOURCES,
    dataset_sha256,
    evaluate_classification_scope,
    evaluate_regression_scope,
    existing_split_source_composition,
    generate_scope_splits,
    make_scope,
    prepare_sequence_graph,
    save_scope_splits,
    scope_aptamer_groups,
    scope_name,
    validate_split_bundle,
)


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_PARAMS = EXPERIMENT_DIR / "configs" / "e1_classification_best_params.json"
DEFAULT_RESULTS = EXPERIMENT_DIR / "results"
DEFAULT_SPLITS = EXPERIMENT_DIR / "splits"
DEFAULT_EMBEDDINGS = REPO_ROOT / "artifacts" / "cache" / "embeddings"
SCOPE_FOLD_FILENAME = "e1_fold_results.csv"


def build_encoder_configs(task: str):
    from src.encoders.aptamer_encoders import (
        dnabert2_embed,
        kmer_freq_with_type_bit,
        onehot_with_type_bit,
    )
    from src.encoders.molecule_encoders import (
        chemberta_embed,
        clamp_embed,
        molformer_embed,
        morgan_fp,
        physchem_descriptors,
    )

    aptamer = {
        "Kmer4": {
            "name": "Kmer4",
            "func": kmer_freq_with_type_bit,
            "kwargs": {"k": 4},
        },
        "OneHot": {"name": "OneHot", "func": onehot_with_type_bit},
        "DNABERT2": {"name": "DNABERT2", "func": dnabert2_embed},
    }
    molecule = {
        "CLAMP": {"name": "CLAMP", "func": clamp_embed},
        "MorganFP": {"name": "MorganFP", "func": morgan_fp},
        "ChemBERTa": {"name": "ChemBERTa", "func": chemberta_embed},
        "MolFormer": {"name": "MolFormer", "func": molformer_embed},
        "PhysChem": {"name": "PhysChem", "func": physchem_descriptors},
    }
    if task == "classification":
        aptamer_names = ("Kmer4", "OneHot")
        molecule_names = ("CLAMP", "MorganFP", "ChemBERTa")
    elif task == "regression":
        aptamer_names = ("DNABERT2", "Kmer4")
        molecule_names = ("MolFormer", "PhysChem", "CLAMP")
    else:
        aptamer_names = ("Kmer4", "OneHot", "DNABERT2")
        molecule_names = (
            "CLAMP",
            "MorganFP",
            "ChemBERTa",
            "MolFormer",
            "PhysChem",
        )
    return (
        [aptamer[name] for name in aptamer_names],
        [molecule[name] for name in molecule_names],
    )


def load_local_classification_params(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = {}
    for split_mode, item in payload.items():
        if "best_params" not in item:
            raise ValueError(f"Missing best_params for {split_mode} in {path}")
        params[split_mode] = dict(item["best_params"])
    return params


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_manifest(
    results_dir: Path,
    dataset_path: Path,
    classification_params: dict[str, dict],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "experiment": "E1 leave-one-source-out source ablation",
        "semantics": "remove source, then rebuild five folds",
        "dataset": str(dataset_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": dataset_sha256(dataset_path),
        "random_state": args.seed,
        "n_splits": 5,
        "frac_tolerance": args.frac_tolerance,
        "max_tries": args.max_tries,
        "split_jobs": args.split_jobs,
        "classification_sources": list(CLASSIFICATION_SOURCES),
        "regression_sources": list(REGRESSION_SOURCES),
        "classification_params_by_split": classification_params,
        "regression_params": REGRESSION_PARAMS,
        "hpo_per_ablation": False,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: installed_version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "torch",
                "transformers",
                "rdkit",
                "clamp-bio",
                "mhnreact",
            )
        },
    }
    (results_dir / "e1_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def combine_scope_fold_results(task_dir: Path) -> pd.DataFrame:
    paths = sorted((task_dir / "scopes").glob(f"*/{SCOPE_FOLD_FILENAME}"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "classification", "regression"),
        default="all",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--classification-params",
        type=Path,
        default=DEFAULT_PARAMS,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frac-tolerance", type=float, default=0.02)
    parser.add_argument("--max-tries", type=int, default=200)
    parser.add_argument(
        "--split-jobs",
        type=int,
        default=-1,
        help="Processes used once to build the sequence-similarity graph.",
    )
    parser.add_argument(
        "--only-scope",
        action="append",
        default=None,
        help="Run only full or one exact source name; repeat as needed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed scope outputs and embeddings.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding precompute/load (useful for split-only checks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    results_dir = args.results_dir.resolve()
    splits_dir = args.splits_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    expected_sources = set(CLASSIFICATION_SOURCES)
    actual_sources = set(df["source"].unique())
    if actual_sources != expected_sources:
        raise ValueError(
            f"Source mismatch: expected {sorted(expected_sources)}, "
            f"found {sorted(actual_sources)}"
        )

    classification_params = load_local_classification_params(
        args.classification_params
    )
    write_manifest(results_dir, dataset_path, classification_params, args)

    source_rows, fold_rows = existing_split_source_composition(
        df,
        REPO_ROOT / "dataset" / "splits",
    )
    source_rows.to_csv(
        results_dir / "e1_existing_split_source_composition.csv",
        index=False,
    )
    fold_rows.to_csv(
        results_dir / "e1_existing_split_fold_composition.csv",
        index=False,
    )

    sequence_graph = prepare_sequence_graph(
        df,
        results_dir / "e1_sequence_similarity_graph.json",
        threshold=0.8,
        n_jobs=args.split_jobs,
    )

    embeddings = {}
    if not args.skip_embeddings:
        aptamer_configs, molecule_configs = build_encoder_configs(args.task)
        aptamer_map, molecule_map, _ = precompute_and_save_embeddings(
            df,
            aptamer_configs,
            molecule_configs,
            embeddings_dir=args.embeddings_dir,
            force_recompute=args.force,
        )
        embeddings = {**aptamer_map, **molecule_map}

    tasks = (
        ("classification", "regression")
        if args.task == "all"
        else (args.task,)
    )
    requested = args.only_scope
    if requested:
        invalid = set(requested) - (expected_sources | {"full"})
        if invalid:
            raise ValueError(f"Unknown --only-scope values: {sorted(invalid)}")
        removed_sources = [
            None if source == "full" else source for source in requested
        ]
    else:
        removed_sources = [None, *CLASSIFICATION_SOURCES]

    for removed_source in removed_sources:
        run_classification = (
            "classification" in tasks
            and (
                removed_source is None
                or removed_source in CLASSIFICATION_SOURCES
            )
        )
        run_regression = (
            "regression" in tasks
            and (
                removed_source is None
                or removed_source in REGRESSION_SOURCES
            )
        )
        if not run_classification and not run_regression:
            continue

        scoped_df = make_scope(df, removed_source)
        name = scope_name(removed_source)
        print(
            f"\nE1 scope={name}: {len(scoped_df)} rows "
            f"({scoped_df['pKd_value'].notna().sum()} pKd)"
        )
        split_bundle = generate_scope_splits(
            scoped_df,
            random_state=args.seed,
            frac_tolerance=args.frac_tolerance,
            max_tries=args.max_tries,
            aptamer_groups=scope_aptamer_groups(
                scoped_df,
                sequence_graph,
            ),
        )
        validate_split_bundle(
            scoped_df,
            split_bundle,
            removed_source=removed_source,
        )
        # save_scope_splits writes under <output_dir>/splits/<scope>/
        save_scope_splits(
            scoped_df,
            split_bundle,
            EXPERIMENT_DIR,
            removed_source,
        )

        if args.skip_embeddings:
            continue

        if run_classification:
            scope_dir = results_dir / "classification" / "scopes" / name
            scope_dir.mkdir(parents=True, exist_ok=True)
            fold_path = scope_dir / SCOPE_FOLD_FILENAME
            if args.force or not fold_path.exists():
                folds, _predictions = evaluate_classification_scope(
                    scoped_df,
                    split_bundle,
                    embeddings,
                    classification_params,
                    removed_source=removed_source,
                    random_state=args.seed,
                )
                folds.to_csv(fold_path, index=False)

        if run_regression:
            scope_dir = results_dir / "regression" / "scopes" / name
            scope_dir.mkdir(parents=True, exist_ok=True)
            fold_path = scope_dir / SCOPE_FOLD_FILENAME
            if args.force or not fold_path.exists():
                folds, _predictions = evaluate_regression_scope(
                    scoped_df,
                    split_bundle,
                    embeddings,
                    removed_source=removed_source,
                    random_state=args.seed,
                )
                folds.to_csv(fold_path, index=False)

    for task in tasks:
        folds = combine_scope_fold_results(results_dir / task)
        if not folds.empty:
            folds.to_csv(
                results_dir / f"e1_{task}_fold_results.csv",
                index=False,
            )

    print(f"\nE1 raw results written to {results_dir}")


if __name__ == "__main__":
    main()
