#!/usr/bin/env python
"""Reproduce E9: pharmacophore / Morgan max-similarity diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_SPLITS_DIR = EXPERIMENT_DIR / "splits"
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"

SPLIT_MODES = ("stratified", "disjoint_molecule")
FINGERPRINTS = ("Morgan", "GobbiPharm2D")
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
MORGAN_RADIUS = 2
MORGAN_N_BITS = 1024

MOLECULE_CSV = "e9_test_molecule_max_similarity.csv"
SUMMARY_CSV = "e9_fold_similarity_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--only-split",
        action="append",
        choices=SPLIT_MODES,
        default=None,
    )
    parser.add_argument(
        "--only-fingerprint",
        action="append",
        choices=FINGERPRINTS,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing result CSVs.",
    )
    return parser.parse_args()


def load_splits(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list of folds in {path}")
    return data


def unique_smiles(df: pd.DataFrame, indices: list[int]) -> list[str]:
    part = df.iloc[indices]
    smiles = (
        part["canonical_smiles"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.len() > 0]
        .unique()
        .tolist()
    )
    return smiles


def build_fingerprints(smiles_list: list[str], fingerprint: str) -> dict[str, object]:
    fps: dict[str, object] = {}
    morgan_gen = None
    if fingerprint == "Morgan":
        morgan_gen = rfg.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if fingerprint == "Morgan":
            fps[smi] = morgan_gen.GetFingerprint(mol)
        elif fingerprint == "GobbiPharm2D":
            fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
            if fp.GetNumOnBits() == 0:
                fps[smi] = None
            else:
                fps[smi] = fp
        else:
            raise ValueError(f"Unknown fingerprint: {fingerprint}")
    return fps


def max_tanimoto(query_fp, train_fps: list) -> float:
    if not train_fps:
        return float("nan")
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
    if not sims:
        return float("nan")
    return float(max(sims))


def summarize_max_sims(max_sims: np.ndarray, n_test: int, n_empty: int) -> dict:
    valid = max_sims[np.isfinite(max_sims)]
    n_valid = int(valid.size)
    row = {
        "n_test_molecules": int(n_test),
        "n_valid": n_valid,
        "n_empty_fp": int(n_empty),
        "mean_max_sim": float(np.mean(valid)) if n_valid else float("nan"),
        "median": float(np.median(valid)) if n_valid else float("nan"),
        "p90": float(np.percentile(valid, 90)) if n_valid else float("nan"),
        "p95": float(np.percentile(valid, 95)) if n_valid else float("nan"),
        "maximum": float(np.max(valid)) if n_valid else float("nan"),
    }
    for threshold in THRESHOLDS:
        key = f"frac_ge_{threshold}"
        if n_valid:
            row[key] = float(np.mean(valid >= threshold))
        else:
            row[key] = float("nan")
    return row


def evaluate_fold(
    *,
    split_mode: str,
    fingerprint: str,
    fold: int,
    train_smiles: list[str],
    test_smiles: list[str],
    fps: dict[str, object],
) -> tuple[list[dict], dict]:
    train_fps = [fps[s] for s in train_smiles if s in fps and fps[s] is not None]
    n_train = len(train_fps)

    molecule_rows: list[dict] = []
    max_sims: list[float] = []
    n_empty = 0

    for smi in test_smiles:
        if smi not in fps:
            continue
        query = fps[smi]
        if query is None:
            n_empty += 1
            sim = float("nan")
        else:
            sim = max_tanimoto(query, train_fps)
        max_sims.append(sim)
        molecule_rows.append(
            {
                "split": split_mode,
                "fingerprint": fingerprint,
                "fold": fold,
                "test_smiles": smi,
                "max_sim": sim,
                "n_train_molecules": n_train,
            }
        )

    summary = {
        "split": split_mode,
        "fingerprint": fingerprint,
        "fold": fold,
        **summarize_max_sims(np.asarray(max_sims, dtype=float), len(molecule_rows), n_empty),
    }
    return molecule_rows, summary


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    dataset_path = args.dataset.resolve()
    splits_dir = args.splits_dir.resolve()
    results_dir = args.results_dir.resolve()
    molecule_path = results_dir / MOLECULE_CSV
    summary_path = results_dir / SUMMARY_CSV

    if (
        not args.force
        and molecule_path.exists()
        and summary_path.exists()
        and args.only_split is None
        and args.only_fingerprint is None
    ):
        print(f"Results already present in {results_dir}; pass --force to regenerate.")
        return

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "canonical_smiles" not in df.columns:
        raise ValueError("Dataset must contain canonical_smiles")

    split_modes = tuple(args.only_split) if args.only_split else SPLIT_MODES
    fingerprints = (
        tuple(args.only_fingerprint) if args.only_fingerprint else FINGERPRINTS
    )

    all_smiles = (
        df["canonical_smiles"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.len() > 0]
        .unique()
        .tolist()
    )
    fp_cache: dict[str, dict[str, object]] = {}
    for fingerprint in fingerprints:
        print(f"Building {fingerprint} fingerprints for {len(all_smiles)} molecules...")
        fp_cache[fingerprint] = build_fingerprints(all_smiles, fingerprint)
        print(f"  done ({len(fp_cache[fingerprint])} parsed)")

    molecule_rows: list[dict] = []
    summary_rows: list[dict] = []

    for split_mode in split_modes:
        split_path = splits_dir / f"{split_mode}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        folds = load_splits(split_path)
        for fold_data in folds:
            fold = int(fold_data["fold"])
            train_idx = fold_data["train_idx"]
            test_idx = fold_data["val_idx"]
            train_smiles = unique_smiles(df, train_idx)
            test_smiles = unique_smiles(df, test_idx)
            for fingerprint in fingerprints:
                mol_part, summary = evaluate_fold(
                    split_mode=split_mode,
                    fingerprint=fingerprint,
                    fold=fold,
                    train_smiles=train_smiles,
                    test_smiles=test_smiles,
                    fps=fp_cache[fingerprint],
                )
                molecule_rows.extend(mol_part)
                summary_rows.append(summary)
                mean_val = summary["mean_max_sim"]
                mean_txt = f"{mean_val:.4f}" if np.isfinite(mean_val) else "nan"
                print(
                    f"{split_mode} {fingerprint} fold={fold} "
                    f"n_test={summary['n_test_molecules']} mean={mean_txt}",
                    flush=True,
                )

    molecule_df = pd.DataFrame(molecule_rows)
    summary_df = pd.DataFrame(summary_rows)

    # Keep a stable column order for the reviewer-facing summary.
    summary_cols = [
        "split",
        "fingerprint",
        "fold",
        "n_test_molecules",
        "n_valid",
        "n_empty_fp",
        "mean_max_sim",
        "median",
        "p90",
        "p95",
        "maximum",
        *[f"frac_ge_{t}" for t in THRESHOLDS],
    ]
    summary_df = summary_df[summary_cols]

    write_csv(molecule_path, molecule_df)
    write_csv(summary_path, summary_df)
    print(f"Wrote {molecule_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
