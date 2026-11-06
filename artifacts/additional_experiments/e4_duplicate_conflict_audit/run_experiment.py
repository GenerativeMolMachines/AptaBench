#!/usr/bin/env python
"""Reproduce E4: duplicate/conflict audit and deduplicated sensitivity."""

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

from src.models.deduplication_audit import (  # noqa: E402
    SPLIT_MODES,
    add_pair_ids,
    build_pair_group_audit,
    collapse_classification,
    evaluate_folds,
    generate_splits,
    load_interim_tables,
    pair_fold_assignment,
    project_splits_by_pair_folds,
    restrict_to_pair_universe,
    summarize_stage,
    validate_dataset,
    validate_splits,
)
from src.models.negative_ablation import FeatureCache  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_INTERIM_DIR = REPO_ROOT / "data" / "interim"
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_SPLITS_DIR = EXPERIMENT_DIR / "splits"
VERSION_LABELS = {
    "current": "Current measurement-level data",
    "collapsed": "Collapsed unique pairs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--frac-tolerance", type=float, default=0.02)
    parser.add_argument("--max-tries", type=int, default=200)
    parser.add_argument("--sequence-jobs", type=int, default=1)
    parser.add_argument(
        "--only-split",
        action="append",
        choices=SPLIT_MODES,
        default=None,
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write audit tables without fitting LightGBM.",
    )
    parser.add_argument(
        "--generate-splits-only",
        action="store_true",
        help="Generate current splits and project collapsed splits without fitting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard existing sensitivity results and re-evaluate.",
    )
    parser.add_argument(
        "--force-splits",
        action="store_true",
        help="Regenerate current measurement-level splits before projection.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(dataset_sha256: str, pair_ids: list[str]) -> str:
    pair_digest = hashlib.sha256("\n".join(pair_ids).encode("utf-8")).hexdigest()
    return f"{dataset_sha256}:{pair_digest}"


def split_path(splits_dir: Path, *, version_key: str, split_mode: str) -> Path:
    return splits_dir / version_key / f"{split_mode}.json"


def save_split(
    path: Path,
    *,
    dataset_sha256: str,
    version_key: str,
    split_mode: str,
    seed: int,
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    provenance: str,
    source_split: str | None = None,
) -> None:
    pair_ids = df["pair_id"].astype(str).tolist()
    payload = {
        "dataset_sha256": dataset_sha256,
        "dataset_version": VERSION_LABELS[version_key],
        "version_key": version_key,
        "split_mode": split_mode,
        "seed": seed,
        "n_splits": len(splits),
        "n_rows": int(len(df)),
        "provenance": provenance,
        "pair_ids": pair_ids,
        "folds": [
            {
                "fold": fold,
                "train_indices": np.asarray(train_idx, dtype=int).tolist(),
                "test_indices": np.asarray(test_idx, dtype=int).tolist(),
            }
            for fold, (train_idx, test_idx) in enumerate(splits)
        ],
    }
    if source_split is not None:
        payload["source_split"] = source_split
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(
    path: Path,
    *,
    dataset_sha256: str,
    version_key: str,
    split_mode: str,
    seed: int,
    df: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_pairs = df["pair_id"].astype(str).tolist()
    if "pair_ids" in payload:
        if payload.get("dataset_sha256") != dataset_sha256:
            raise ValueError(f"Dataset hash does not match frozen split {path}")
        if payload.get("version_key") != version_key:
            raise ValueError(f"Version key does not match frozen split {path}")
        if payload.get("split_mode") != split_mode:
            raise ValueError(f"Split mode does not match frozen split {path}")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Seed does not match frozen split {path}")
        if int(payload.get("n_rows", -1)) != len(df):
            raise ValueError(f"Row count does not match frozen split {path}")
        if payload.get("pair_ids") != expected_pairs:
            raise ValueError(f"Pair order does not match frozen split {path}")
    else:
        if payload.get("dataset_version") != VERSION_LABELS[version_key]:
            raise ValueError(f"Dataset version does not match frozen split {path}")
        if payload.get("split_mode") != split_mode:
            raise ValueError(f"Split mode does not match frozen split {path}")
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Seed does not match frozen split {path}")
        if int(payload.get("n_splits", -1)) != len(payload.get("folds", [])):
            raise ValueError(f"Incomplete folds in frozen split {path}")
        for fold in payload["folds"]:
            fold_pairs = fold.get("train_pair_ids", []) + fold.get("test_pair_ids", [])
            if sorted(fold_pairs) != sorted(
                expected_pairs[i]
                for i in fold["train_indices"] + fold["test_indices"]
            ):
                raise ValueError(f"Legacy pair ids do not match dataset for {path}")

    return [
        (
            np.asarray(fold["train_indices"], dtype=int),
            np.asarray(fold["test_indices"], dtype=int),
        )
        for fold in payload["folds"]
    ]


def condition_complete(
    results: pd.DataFrame,
    *,
    version: str,
    split_mode: str,
    n_splits: int,
) -> bool:
    if results.empty:
        return False
    rows = results.loc[
        (results["dataset_version"] == version)
        & (results["split_mode"] == split_mode)
    ]
    return set(rows["fold"].astype(int)) == set(range(n_splits))


def remove_condition(
    results: pd.DataFrame, *, version: str, split_mode: str
) -> pd.DataFrame:
    if results.empty:
        return results
    mask = (
        (results["dataset_version"] == version)
        & (results["split_mode"] == split_mode)
    )
    return results.loc[~mask].reset_index(drop=True)


def write_audit(
    *,
    interim_dir: Path,
    dataset_path: Path,
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interim_paths = sorted(interim_dir.glob("*_processed.csv"))
    if len(interim_paths) != 8:
        raise ValueError(
            f"Expected exactly 8 processed interim tables, found {len(interim_paths)}"
        )
    pre = load_interim_tables(interim_dir)
    post = validate_dataset(pd.read_csv(dataset_path))
    eligible_pre = restrict_to_pair_universe(pre, post)
    stats = pd.concat(
        [
            summarize_stage(pre, stage="pre_global_harmonization"),
            summarize_stage(post, stage="post_global_harmonization"),
        ],
        ignore_index=True,
    )
    audit = build_pair_group_audit(pre)
    classification = collapse_classification(eligible_pre)
    results_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(results_dir / "e4_harmonization_stats.csv", index=False)
    audit.to_csv(results_dir / "e4_pair_group_audit.csv", index=False)
    return post, classification


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = add_pair_ids(df).dropna(
        subset=["sequence", "canonical_smiles", "label"]
    ).reset_index(drop=True)
    prepared["label"] = prepared["label"].astype(int)
    return prepared


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2")

    dataset_path = args.dataset.resolve()
    interim_dir = args.interim_dir.resolve()
    results_dir = args.results_dir.resolve()
    splits_dir = args.splits_dir.resolve()
    released_hash = file_sha256(dataset_path)

    post, classification = write_audit(
        interim_dir=interim_dir,
        dataset_path=dataset_path,
        results_dir=results_dir,
    )
    print(f"Audit tables written to {results_dir}")

    if args.audit_only:
        return

    fold_path = results_dir / "e4_sensitivity_fold_results.csv"
    if fold_path.exists() and not args.force:
        results = pd.read_csv(fold_path)
    else:
        results = pd.DataFrame()

    current = prepare_dataset(post)
    collapsed = prepare_dataset(classification)
    current_hash = content_hash(
        released_hash, current["pair_id"].astype(str).tolist()
    )
    collapsed_hash = content_hash(
        released_hash, collapsed["pair_id"].astype(str).tolist()
    )
    cache = FeatureCache()
    split_modes = tuple(args.only_split or SPLIT_MODES)

    for split_mode in split_modes:
        current_path = split_path(
            splits_dir, version_key="current", split_mode=split_mode
        )
        current_splits = None if args.force_splits else load_split(
            current_path,
            dataset_sha256=current_hash,
            version_key="current",
            split_mode=split_mode,
            seed=args.seed,
            df=current,
        )
        if current_splits is None:
            print(f"split version=current split={split_mode}")
            current_splits, groups = generate_splits(
                current,
                split_mode=split_mode,
                seed=args.seed,
                n_splits=args.n_splits,
                frac_tolerance=args.frac_tolerance,
                max_tries=args.max_tries,
                sequence_n_jobs=args.sequence_jobs,
            )
            validate_splits(current, current_splits, groups)
            save_split(
                current_path,
                dataset_sha256=current_hash,
                version_key="current",
                split_mode=split_mode,
                seed=args.seed,
                df=current,
                splits=current_splits,
                provenance="generated_measurement_level",
            )
        else:
            print(f"reuse split version=current split={split_mode}")

        pair_to_fold = pair_fold_assignment(current, current_splits)
        collapsed_splits = project_splits_by_pair_folds(
            collapsed,
            pair_to_fold,
            n_splits=args.n_splits,
        )
        validate_splits(collapsed, collapsed_splits)
        collapsed_path = split_path(
            splits_dir, version_key="collapsed", split_mode=split_mode
        )
        save_split(
            collapsed_path,
            dataset_sha256=collapsed_hash,
            version_key="collapsed",
            split_mode=split_mode,
            seed=args.seed,
            df=collapsed,
            splits=collapsed_splits,
            provenance="projected_from_current_pair_folds",
            source_split=str(current_path.relative_to(EXPERIMENT_DIR)),
        )
        print(f"project split version=collapsed split={split_mode}")

        if args.generate_splits_only:
            continue

        for version_key, dataset, splits, provenance in (
            (
                "current",
                current,
                current_splits,
                "measurement_level",
            ),
            (
                "collapsed",
                collapsed,
                collapsed_splits,
                "projected_from_current_pair_folds",
            ),
        ):
            version = VERSION_LABELS[version_key]
            if condition_complete(
                results,
                version=version,
                split_mode=split_mode,
                n_splits=args.n_splits,
            ):
                print(f"skip version={version_key} split={split_mode}")
                continue

            print(f"run version={version_key} split={split_mode}")
            fold_results = evaluate_folds(
                dataset,
                splits,
                dataset_version=version,
                split_mode=split_mode,
                seed=args.seed,
                feature_cache=cache,
                split_provenance=provenance,
            )
            results = remove_condition(
                results, version=version, split_mode=split_mode
            )
            results = pd.concat([results, fold_results], ignore_index=True)
            results = results.sort_values(
                ["dataset_version", "split_mode", "fold"]
            ).reset_index(drop=True)
            results.to_csv(fold_path, index=False)

    if not args.generate_splits_only:
        results.to_csv(fold_path, index=False)
        print(f"Raw E4 sensitivity results written to {fold_path}")


if __name__ == "__main__":
    main()
