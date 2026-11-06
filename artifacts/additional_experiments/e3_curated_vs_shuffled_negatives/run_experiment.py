#!/usr/bin/env python
"""Reproduce E3: curated versus train-only shuffled negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.negative_ablation import (  # noqa: E402
    SCOPES,
    SPLIT_MODES,
    FeatureCache,
    evaluate_variant,
    generate_first_fold,
    make_balanced_scope,
    make_train_variants,
    positive_pairs,
)


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_OUTPUT = (
    EXPERIMENT_DIR
    / "results"
    / "e3_curated_vs_shuffled_raw_results.csv"
)
DEFAULT_SPLITS_DIR = EXPERIMENT_DIR / "splits"
SCOPE_LABELS = {
    "full": "Full",
    "without_specificity": "Without Specificity",
}
SPLIT_LABELS = {
    "stratified": "In-distribution",
    "disjoint_molecule": "Molecule-disjoint",
    "disjoint_aptamer": "Aptamer-disjoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--frac-tolerance", type=float, default=0.02)
    parser.add_argument("--max-tries", type=int, default=200)
    parser.add_argument(
        "--only-scope",
        action="append",
        choices=SCOPES,
        default=None,
    )
    parser.add_argument(
        "--only-split",
        action="append",
        choices=SPLIT_MODES,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard the existing raw output and rerun requested conditions.",
    )
    parser.add_argument(
        "--generate-splits-only",
        action="store_true",
        help="Generate missing frozen split assignments without fitting models.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_path(
    splits_dir: Path,
    *,
    scope: str,
    split_mode: str,
    seed: int,
) -> Path:
    return splits_dir / scope / split_mode / f"seed_{seed}.json"


def save_split(
    path: Path,
    *,
    dataset_sha256: str,
    scope: str,
    split_mode: str,
    seed: int,
    balanced: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    original = balanced["original_index"].to_numpy(dtype=int)
    payload = {
        "dataset_sha256": dataset_sha256,
        "scope": scope,
        "split_mode": split_mode,
        "seed": seed,
        "n_splits": 5,
        "fold": 0,
        "balanced_original_indices": original.tolist(),
        "train_original_indices": original[train_idx].tolist(),
        "test_original_indices": original[test_idx].tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(
    path: Path,
    *,
    dataset_sha256: str,
    balanced: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["dataset_sha256"] != dataset_sha256:
        raise ValueError(f"Dataset hash does not match frozen split {path}")
    original_to_local = {
        int(original): local
        for local, original in enumerate(balanced["original_index"])
    }
    if set(payload["balanced_original_indices"]) != set(original_to_local):
        raise ValueError(f"Balanced scope does not match frozen split {path}")
    return (
        np.asarray(
            [original_to_local[int(value)] for value in payload["train_original_indices"]],
            dtype=int,
        ),
        np.asarray(
            [original_to_local[int(value)] for value in payload["test_original_indices"]],
            dtype=int,
        ),
    )


def condition_complete(
    results: pd.DataFrame,
    *,
    scope: str,
    split_mode: str,
    seed: int,
) -> bool:
    if results.empty:
        return False
    rows = results.loc[
        (results["scope"] == scope)
        & (results["split_mode"] == split_mode)
        & (results["seed"] == seed)
    ]
    return set(rows["negative_type"]) == {"Curated", "Shuffled"}


def remove_condition(
    results: pd.DataFrame,
    *,
    scope: str,
    split_mode: str,
    seed: int,
) -> pd.DataFrame:
    if results.empty:
        return results
    mask = (
        (results["scope"] == scope)
        & (results["split_mode"] == split_mode)
        & (results["seed"] == seed)
    )
    return results.loc[~mask].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be positive")

    dataset_path = args.dataset.resolve()
    dataset_hash = file_sha256(dataset_path)
    output_file = args.output_file.resolve()
    splits_dir = args.splits_dir.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not args.force:
        results = pd.read_csv(output_file)
    else:
        results = pd.DataFrame()

    df = pd.read_csv(dataset_path)
    feature_cache = FeatureCache()
    scopes = tuple(args.only_scope or SCOPES)
    split_modes = tuple(args.only_split or SPLIT_MODES)
    seeds = range(args.first_seed, args.first_seed + args.n_seeds)

    for scope in scopes:
        for seed in seeds:
            balanced = make_balanced_scope(df, scope=scope, seed=seed)
            known_positives = positive_pairs(balanced)
            for split_mode in split_modes:
                frozen_path = split_path(
                    splits_dir,
                    scope=scope,
                    split_mode=split_mode,
                    seed=seed,
                )
                frozen = load_split(
                    frozen_path,
                    dataset_sha256=dataset_hash,
                    balanced=balanced,
                )
                if frozen is None:
                    train_idx, test_idx, _ = generate_first_fold(
                        balanced,
                        split_mode=split_mode,
                        seed=seed,
                        frac_tolerance=args.frac_tolerance,
                        max_tries=args.max_tries,
                    )
                    save_split(
                        frozen_path,
                        dataset_sha256=dataset_hash,
                        scope=scope,
                        split_mode=split_mode,
                        seed=seed,
                        balanced=balanced,
                        train_idx=train_idx,
                        test_idx=test_idx,
                    )
                else:
                    train_idx, test_idx = frozen

                if args.generate_splits_only:
                    print(f"split scope={scope} split={split_mode} seed={seed}")
                    continue
                if condition_complete(
                    results,
                    scope=scope,
                    split_mode=split_mode,
                    seed=seed,
                ):
                    print(f"skip scope={scope} split={split_mode} seed={seed}")
                    continue

                print(f"run scope={scope} split={split_mode} seed={seed}")
                train = balanced.iloc[train_idx].reset_index(drop=True)
                test = balanced.iloc[test_idx].reset_index(drop=True)
                variants = make_train_variants(
                    train,
                    seed=seed,
                    known_positive_pairs=known_positives,
                    test_pairs=set(
                        test[["sequence", "canonical_smiles"]].itertuples(
                            index=False,
                            name=None,
                        )
                    ),
                )

                records = []
                for negative_type, train_variant in variants.items():
                    metrics = evaluate_variant(
                        train_variant,
                        test,
                        seed=seed,
                        feature_cache=feature_cache,
                    )
                    records.append(
                        {
                            "dataset": SCOPE_LABELS[scope],
                            "scope": scope,
                            "split": SPLIT_LABELS[split_mode],
                            "split_mode": split_mode,
                            "seed": seed,
                            "negative_type": negative_type,
                            "n_balanced": int(len(balanced)),
                            "n_train": int(len(train_variant)),
                            "n_test": int(len(test)),
                            "n_train_positive": int(
                                (train_variant["label"] == 1).sum()
                            ),
                            "n_train_negative": int(
                                (train_variant["label"] == 0).sum()
                            ),
                            "n_test_positive": int((test["label"] == 1).sum()),
                            "n_test_negative": int((test["label"] == 0).sum()),
                            **metrics,
                        }
                    )

                results = remove_condition(
                    results,
                    scope=scope,
                    split_mode=split_mode,
                    seed=seed,
                )
                results = pd.concat(
                    [results, pd.DataFrame(records)],
                    ignore_index=True,
                )
                results = results.sort_values(
                    ["scope", "split_mode", "seed", "negative_type"]
                ).reset_index(drop=True)
                results.to_csv(output_file, index=False)

    if not args.generate_splits_only:
        results.to_csv(output_file, index=False)
        print(f"Raw E3 results written to {output_file}")


if __name__ == "__main__":
    main()
