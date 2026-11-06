#!/usr/bin/env python
"""Reproduce E1 v2: source ablation on the original fixed paper folds."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="force_all_finite was renamed to ensure_all_finite",
)
warnings.filterwarnings("ignore", category=FutureWarning)


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.screening import precompute_and_save_embeddings  # noqa: E402
from src.models.source_ablation import (  # noqa: E402
    CLASSIFICATION_SOURCES,
    REGRESSION_PARAMS,
    REGRESSION_SOURCES,
    SPLIT_MODES,
    dataset_sha256,
    evaluate_classification_fixed_folds,
    evaluate_regression_fixed_folds,
    existing_split_source_composition,
    load_original_splits,
    save_filtered_fixed_splits,
    scope_name,
)


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_PARAMS = (
    EXPERIMENT_DIR / "configs" / "e1_v2_classification_best_params.json"
)
DEFAULT_RESULTS = EXPERIMENT_DIR / "results"
DEFAULT_ORIGINAL_SPLITS = EXPERIMENT_DIR / "splits" / "original"
DEFAULT_EMBEDDINGS = REPO_ROOT / "artifacts" / "cache" / "embeddings"
SCOPE_FOLD_FILENAME = "e1_v2_fold_results.csv"

EXPECTED_FULL = {
    "classification": {
        "stratified": {"roc_auc": 0.952, "std": 0.002},
        "disjoint_molecule": {"roc_auc": 0.886, "std": 0.038},
        "disjoint_aptamer": {"roc_auc": 0.867, "std": 0.044},
    },
    "regression": {
        "stratified": {"r2": 0.668, "std": 0.038},
        "disjoint_molecule": {"r2": 0.298, "std": 0.191},
        "disjoint_aptamer": {"r2": 0.323, "std": 0.123},
    },
}


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


def ensure_original_splits(original_splits_dir: Path) -> None:
    """Sync local copies from dataset/splits (always refresh; avoid stale dirty folds)."""
    original_splits_dir.mkdir(parents=True, exist_ok=True)
    source_dir = REPO_ROOT / "dataset" / "splits"
    for split_mode in SPLIT_MODES:
        source = source_dir / f"{split_mode}.json"
        destination = original_splits_dir / f"{split_mode}.json"
        shutil.copy2(source, destination)


def write_manifest(
    results_dir: Path,
    dataset_path: Path,
    classification_params: dict[str, dict],
    args: argparse.Namespace,
    dataset_hash: str,
) -> None:
    manifest = {
        "experiment": "E1 v2 source ablation on original fixed folds",
        "semantics": (
            "filter removed source out of the released train/test indices; "
            "do not rebuild folds"
        ),
        "dataset": str(dataset_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": dataset_hash,
        "original_splits": "splits/original/{stratified,disjoint_molecule,disjoint_aptamer}.json",
        "random_state": args.seed,
        "n_splits": 5,
        "classification_sources": list(CLASSIFICATION_SOURCES),
        "regression_sources": list(REGRESSION_SOURCES),
        "classification_params_by_split": classification_params,
        "regression_params": REGRESSION_PARAMS,
        "hpo_per_ablation": False,
        "expected_full": EXPECTED_FULL,
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
    (results_dir / "e1_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def combine_scope_fold_results(task_dir: Path) -> pd.DataFrame:
    paths = sorted((task_dir / "scopes").glob(f"*/{SCOPE_FOLD_FILENAME}"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def print_full_check(folds: pd.DataFrame, task: str) -> None:
    metric = "roc_auc" if task == "classification" else "r2"
    expected = EXPECTED_FULL[task]
    part = folds.loc[folds["scope"] == "full"]
    if part.empty:
        return
    print(f"\nFull-check ({task}, mean±std vs expected):")
    for split_mode in SPLIT_MODES:
        cur = part.loc[part["split"] == split_mode, metric]
        mean = float(cur.mean())
        std = float(cur.std(ddof=0))
        target = expected[split_mode]
        print(
            f"  {split_mode:18s} {mean:.3f} ± {std:.3f} "
            f"(expected {target[metric]:.3f} ± {target['std']:.3f})"
        )


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
    parser.add_argument(
        "--original-splits-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_SPLITS,
    )
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--seed", type=int, default=42)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    results_dir = args.results_dir.resolve()
    original_splits_dir = args.original_splits_dir.resolve()
    filtered_splits_dir = EXPERIMENT_DIR / "splits" / "filtered"
    results_dir.mkdir(parents=True, exist_ok=True)
    filtered_splits_dir.mkdir(parents=True, exist_ok=True)

    ensure_original_splits(original_splits_dir)
    df = pd.read_csv(dataset_path)
    dataset_hash = dataset_sha256(dataset_path)
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
    write_manifest(
        results_dir,
        dataset_path,
        classification_params,
        args,
        dataset_hash,
    )

    source_rows, fold_rows = existing_split_source_composition(
        df,
        original_splits_dir,
    )
    source_rows.to_csv(
        results_dir / "e1_v2_existing_split_source_composition.csv",
        index=False,
    )
    fold_rows.to_csv(
        results_dir / "e1_v2_existing_split_fold_composition.csv",
        index=False,
    )

    original_splits = load_original_splits(original_splits_dir)
    # Avoid recomputing expensive aptamer clusters; unique aptamer/molecule
    # counts are already recorded in fold rows.
    groups_by_split = None

    aptamer_configs, molecule_configs = build_encoder_configs(args.task)
    aptamer_map, molecule_map, _ = precompute_and_save_embeddings(
        df,
        aptamer_configs,
        molecule_configs,
        embeddings_dir=args.embeddings_dir,
        force_recompute=args.force,
    )
    embeddings = {**aptamer_map, **molecule_map}
    sources = df["source"].to_numpy()

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

        name = scope_name(removed_source)
        kept = (
            len(df)
            if removed_source is None
            else int((df["source"] != removed_source).sum())
        )
        pkd_kept = (
            int(df["pKd_value"].notna().sum())
            if removed_source is None
            else int(
                df.loc[df["source"] != removed_source, "pKd_value"].notna().sum()
            )
        )
        print(f"\nE1 v2 scope={name}: {kept} rows ({pkd_kept} pKd)", flush=True)
        save_filtered_fixed_splits(
            original_splits,
            sources,
            filtered_splits_dir,
            removed_source,
        )

        if run_classification:
            scope_dir = results_dir / "classification" / "scopes" / name
            scope_dir.mkdir(parents=True, exist_ok=True)
            fold_path = scope_dir / SCOPE_FOLD_FILENAME
            if args.force or not fold_path.exists():
                print(f"  training classification/{name} ...", flush=True)
                folds = evaluate_classification_fixed_folds(
                    df,
                    original_splits,
                    embeddings,
                    classification_params,
                    removed_source=removed_source,
                    groups_by_split=groups_by_split,
                    random_state=args.seed,
                )
                folds.to_csv(fold_path, index=False)
                print(f"  wrote {fold_path.name}", flush=True)
            else:
                print(f"  skip classification/{name}", flush=True)

        if run_regression:
            scope_dir = results_dir / "regression" / "scopes" / name
            scope_dir.mkdir(parents=True, exist_ok=True)
            fold_path = scope_dir / SCOPE_FOLD_FILENAME
            if args.force or not fold_path.exists():
                print(f"  training regression/{name} ...", flush=True)
                folds = evaluate_regression_fixed_folds(
                    df,
                    original_splits,
                    embeddings,
                    removed_source=removed_source,
                    groups_by_split=groups_by_split,
                    random_state=args.seed,
                )
                folds.to_csv(fold_path, index=False)
                print(f"  wrote {fold_path.name}", flush=True)
            else:
                print(f"  skip regression/{name}", flush=True)

    for task in tasks:
        folds = combine_scope_fold_results(results_dir / task)
        if folds.empty:
            continue
        out = results_dir / f"e1_v2_{task}_fold_results.csv"
        folds.to_csv(out, index=False)
        print_full_check(folds, task)

    print(f"\nE1 v2 raw results written to {results_dir}")


if __name__ == "__main__":
    main()
