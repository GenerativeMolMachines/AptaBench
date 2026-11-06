#!/usr/bin/env python
"""Reproduce E5: pKd-threshold sensitivity and negative-subtype difficulty."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    """Find the checkout root so this script works from its experiment folder."""
    for candidate in (start, *start.parents):
        if (
            (candidate / "dataset" / "AptaBench_dataset.csv").exists()
            and (candidate / "src").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate the AptaBench repository root")


REPO_ROOT = find_repo_root(SCRIPT_DIR)
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e5_core import (  # noqa: E402
    MAIN_THRESHOLD,
    SPLIT_MODES,
    THRESHOLDS,
    assign_negative_provenance,
    build_row_annotations,
    condition_complete,
    evaluate_threshold_folds,
    generate_oof_predictions,
    load_released_splits,
    negative_subtype_metrics,
    relabel_for_threshold,
    remove_condition,
    threshold_label_counts,
)
from src.models.negative_ablation import FeatureCache  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_SPLITS_DIR = REPO_ROOT / "dataset" / "splits"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only-split",
        action="append",
        choices=SPLIT_MODES,
        default=None,
    )
    parser.add_argument(
        "--only-threshold",
        action="append",
        type=int,
        choices=list(THRESHOLDS),
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard existing fold/OOF results and recompute.",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Write provenance annotations and label counts without fitting.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    splits_dir = args.splits_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    split_modes = tuple(args.only_split or SPLIT_MODES)
    thresholds = tuple(args.only_threshold or THRESHOLDS)

    df = pd.read_csv(dataset_path)
    required = {
        "type",
        "sequence",
        "canonical_smiles",
        "pKd_value",
        "label",
        "source",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    dataset_hash = file_sha256(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"SHA-256: {dataset_hash}")
    print(f"Rows: {len(df)}")

    annotations = build_row_annotations(df)
    annotations_path = results_dir / "e5_row_annotations.csv"
    annotations.to_csv(annotations_path, index=False)

    counts = threshold_label_counts(annotations)
    counts_path = results_dir / "e5_threshold_label_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"Wrote {annotations_path.name} and {counts_path.name}")

    if args.annotations_only:
        return

    splits_by_mode = load_released_splits(splits_dir, split_modes=split_modes)
    for split_mode, folds in splits_by_mode.items():
        n_indexed = max(int(idx.max()) for train, val in folds for idx in (train, val))
        if n_indexed >= len(df):
            raise ValueError(
                f"Split {split_mode} references row {n_indexed} but dataset "
                f"has {len(df)} rows"
            )

    provenance = assign_negative_provenance(df)
    feature_cache = FeatureCache()

    fold_path = results_dir / "e5_threshold_fold_results.csv"
    fold_columns = [
        "threshold",
        "split",
        "fold",
        "n_train",
        "n_val",
        "n_val_positives",
        "n_val_negatives",
        "roc_auc",
        "mcc",
    ]
    fold_results = read_csv_or_empty(fold_path, fold_columns)
    if args.force:
        for threshold in thresholds:
            for split_mode in split_modes:
                fold_results = remove_condition(
                    fold_results, threshold=threshold, split_mode=split_mode
                )

    for threshold in thresholds:
        labels = relabel_for_threshold(
            df, threshold=threshold, provenance=provenance
        )
        for split_mode in split_modes:
            n_folds = len(splits_by_mode[split_mode])
            if condition_complete(
                fold_results,
                threshold=threshold,
                split_mode=split_mode,
                n_splits=n_folds,
            ):
                print(f"Skip threshold={threshold} split={split_mode} (complete)")
                continue

            print(f"Evaluate threshold={threshold} split={split_mode}")
            fold_results = remove_condition(
                fold_results, threshold=threshold, split_mode=split_mode
            )
            new_rows = evaluate_threshold_folds(
                df,
                labels,
                splits_by_mode[split_mode],
                threshold=threshold,
                split_mode=split_mode,
                feature_cache=feature_cache,
                seed=args.seed,
            )
            fold_results = pd.concat([fold_results, new_rows], ignore_index=True)
            fold_results.to_csv(fold_path, index=False)

    fold_results = fold_results.sort_values(
        ["threshold", "split", "fold"]
    ).reset_index(drop=True)
    fold_results.to_csv(fold_path, index=False)
    print(f"Wrote {fold_path}")

    # E5.2: OOF + subtype metrics at the main threshold.
    oof_path = results_dir / "e5_oof_threshold4.csv"
    subtype_path = results_dir / "e5_negative_subtype_metrics.csv"
    need_oof = args.force or not oof_path.exists() or not subtype_path.exists()
    if need_oof:
        print(f"Generate OOF predictions at threshold={MAIN_THRESHOLD}")
        labels_main = relabel_for_threshold(
            df, threshold=MAIN_THRESHOLD, provenance=provenance
        )
        oof = generate_oof_predictions(
            df,
            labels_main,
            provenance,
            splits_by_mode,
            feature_cache=feature_cache,
            seed=args.seed,
            split_modes=split_modes,
        )
        oof.to_csv(oof_path, index=False)
        subtype = negative_subtype_metrics(oof)
        subtype.to_csv(subtype_path, index=False)
        print(f"Wrote {oof_path.name} and {subtype_path.name}")
    else:
        print(f"Skip OOF / subtype metrics (existing at {results_dir})")


if __name__ == "__main__":
    main()
