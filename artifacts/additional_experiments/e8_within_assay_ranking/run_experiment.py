#!/usr/bin/env python
"""Run E8: within-assay (DOI-batch) ranking evaluation on regression OOF."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_SPLITS = SCRIPT_DIR / "splits"
DEFAULT_CLASSIFICATION_RESULTS = (
    REPO_ROOT
    / "artifacts"
    / "ml"
    / "classification"
    / "screening"
    / "all_splits"
    / "results.csv"
)
DEFAULT_REGRESSION_RESULTS = (
    REPO_ROOT / "artifacts" / "ml" / "regression" / "results_reg.csv"
)
DEFAULT_E2_OOF = (
    REPO_ROOT
    / "artifacts"
    / "additional_experiments"
    / "e2_dna_rna_rsapred"
    / "results"
    / "oof"
    / "e2_regression_oof.csv"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "results"

SPLIT_MODES = ("stratified", "disjoint_molecule", "disjoint_aptamer")
SPLIT_LABELS = {
    "stratified": "In-distribution",
    "disjoint_molecule": "Molecule-disjoint",
    "disjoint_aptamer": "Aptamer-disjoint",
}
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
MIN_BATCH_N = 3
MIN_UNIQUE_PKD = 2
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42


def build_encoder_configs(
    aptamer_names: set[str],
    molecule_names: set[str],
) -> tuple[list[dict], list[dict]]:
    """Instantiate only the encoders selected by the screening tables."""
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
        "OneHot": {"name": "OneHot", "func": onehot_with_type_bit},
        "Kmer4": {
            "name": "Kmer4",
            "func": kmer_freq_with_type_bit,
            "kwargs": {"k": 4},
        },
        "DNABERT2": {"name": "DNABERT2", "func": dnabert2_embed},
    }
    molecule = {
        "MorganFP": {"name": "MorganFP", "func": morgan_fp},
        "PhysChem": {"name": "PhysChem", "func": physchem_descriptors},
        "ChemBERTa": {"name": "ChemBERTa", "func": chemberta_embed},
        "MolFormer": {"name": "MolFormer", "func": molformer_embed},
        "CLAMP": {"name": "CLAMP", "func": clamp_embed},
    }
    unknown_aptamer = aptamer_names - set(aptamer)
    unknown_molecule = molecule_names - set(molecule)
    if unknown_aptamer or unknown_molecule:
        raise ValueError(
            f"Unsupported selected encoders: aptamer={sorted(unknown_aptamer)}, "
            f"molecule={sorted(unknown_molecule)}"
        )
    return (
        [aptamer[name] for name in sorted(aptamer_names)],
        [molecule[name] for name in sorted(molecule_names)],
    )


def extract_doi(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = DOI_RE.search(text)
    if match is None:
        return None
    return match.group(0).lower().rstrip(".")


def annotate_doi_batches(df: pd.DataFrame) -> pd.DataFrame:
    """Attach DOI batch ids; rows without DOI get NaN batch_id."""
    out = df.copy()
    out["row_idx"] = np.arange(len(out), dtype=int)
    out["doi"] = out["origin"].map(extract_doi)
    out["batch_id"] = out["doi"]
    return out


def eligible_batches(df: pd.DataFrame) -> pd.DataFrame:
    """DOI batches with enough pKd measurements and distinct values."""
    reg = df.loc[df["pKd_value"].notna() & df["batch_id"].notna()].copy()
    rows = []
    for batch_id, part in reg.groupby("batch_id", sort=True):
        n = int(len(part))
        n_unique = int(part["pKd_value"].nunique(dropna=True))
        if n < MIN_BATCH_N or n_unique < MIN_UNIQUE_PKD:
            continue
        sources = sorted({str(s) for s in part["source"].dropna().unique()})
        rows.append(
            {
                "batch_id": batch_id,
                "doi": batch_id,
                "n": n,
                "n_unique_pkd": n_unique,
                "sources": ";".join(sources),
            }
        )
    return pd.DataFrame(rows)


def pairwise_concordance_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, int, float]:
    """Return concordance, number of comparable pairs, and concordant mass."""
    n = len(y_true)
    if n < 2:
        return float("nan"), 0, 0.0
    concordant = 0.0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            true_diff = y_true[i] - y_true[j]
            if true_diff == 0:
                continue
            comparable += 1
            pred_diff = y_pred[i] - y_pred[j]
            if pred_diff == 0:
                concordant += 0.5
            elif (true_diff > 0) == (pred_diff > 0):
                concordant += 1.0
    if comparable == 0:
        return float("nan"), 0, 0.0
    return float(concordant / comparable), comparable, concordant


def batch_ranking_metrics(part: pd.DataFrame) -> dict[str, float | int]:
    y_true = part["y_true"].to_numpy(dtype=float)
    y_pred = part["y_pred"].to_numpy(dtype=float)
    n = int(len(part))
    if n < 2 or np.unique(y_true).size < 2:
        return {
            "n_records": n,
            "spearman": float("nan"),
            "kendall_tau_b": float("nan"),
            "pairwise_concordance": float("nan"),
            "n_comparable_pairs": 0,
            "concordant_mass": 0.0,
        }
    spearman = float(spearmanr(y_true, y_pred).statistic)
    kendall = float(kendalltau(y_true, y_pred, variant="b").statistic)
    concordance, n_pairs, mass = pairwise_concordance_stats(y_true, y_pred)
    return {
        "n_records": n,
        "spearman": spearman,
        "kendall_tau_b": kendall,
        "pairwise_concordance": concordance,
        "n_comparable_pairs": int(n_pairs),
        "concordant_mass": float(mass),
    }


def compute_batch_metrics(
    regression_oof: pd.DataFrame,
    annotated: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    meta = annotated[["row_idx", "batch_id", "doi", "source"]].copy()
    eligible_ids = set(eligible["batch_id"])
    records = []
    for split_mode in SPLIT_MODES:
        part = regression_oof.loc[regression_oof["split"] == split_mode].merge(
            meta, on="row_idx", how="inner", validate="one_to_one"
        )
        part = part.loc[part["batch_id"].isin(eligible_ids)]
        for batch_id, batch in part.groupby("batch_id", sort=True):
            metrics = batch_ranking_metrics(batch)
            # Re-check eligibility on the OOF subset present for this split.
            if (
                metrics["n_records"] < MIN_BATCH_N
                or batch["y_true"].nunique(dropna=True) < MIN_UNIQUE_PKD
            ):
                continue
            records.append(
                {
                    "split": split_mode,
                    "split_label": SPLIT_LABELS[split_mode],
                    "batch_id": batch_id,
                    "doi": batch_id,
                    **metrics,
                }
            )
    return pd.DataFrame(records)


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def summarize_split(
    batch_metrics: pd.DataFrame,
    *,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for split_mode in SPLIT_MODES:
        part = batch_metrics.loc[batch_metrics["split"] == split_mode].reset_index(
            drop=True
        )
        n_batches = int(len(part))
        n_records = int(part["n_records"].sum()) if n_batches else 0
        if n_batches == 0:
            rows.append(
                {
                    "split": split_mode,
                    "split_label": SPLIT_LABELS[split_mode],
                    "n_batches": 0,
                    "n_records": 0,
                    "macro_spearman": float("nan"),
                    "macro_spearman_ci_low": float("nan"),
                    "macro_spearman_ci_high": float("nan"),
                    "macro_kendall_tau_b": float("nan"),
                    "macro_kendall_tau_b_ci_low": float("nan"),
                    "macro_kendall_tau_b_ci_high": float("nan"),
                    "macro_pairwise_concordance": float("nan"),
                    "macro_pairwise_concordance_ci_low": float("nan"),
                    "macro_pairwise_concordance_ci_high": float("nan"),
                    "pooled_pairwise_concordance": float("nan"),
                    "pooled_pairwise_concordance_ci_low": float("nan"),
                    "pooled_pairwise_concordance_ci_high": float("nan"),
                }
            )
            continue

        macro_spearman = float(part["spearman"].mean())
        macro_kendall = float(part["kendall_tau_b"].mean())
        macro_conc = float(part["pairwise_concordance"].mean())
        pair_sum = float(part["n_comparable_pairs"].sum())
        mass_sum = float(part["concordant_mass"].sum())
        pooled = float(mass_sum / pair_sum) if pair_sum > 0 else float("nan")

        boot_spearman = np.empty(n_boot, dtype=float)
        boot_kendall = np.empty(n_boot, dtype=float)
        boot_macro_conc = np.empty(n_boot, dtype=float)
        boot_pooled = np.empty(n_boot, dtype=float)
        n = len(part)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = part.iloc[idx]
            boot_spearman[b] = float(sample["spearman"].mean())
            boot_kendall[b] = float(sample["kendall_tau_b"].mean())
            boot_macro_conc[b] = float(sample["pairwise_concordance"].mean())
            s_pairs = float(sample["n_comparable_pairs"].sum())
            s_mass = float(sample["concordant_mass"].sum())
            boot_pooled[b] = float(s_mass / s_pairs) if s_pairs > 0 else float("nan")

        sp_lo, sp_hi = _percentile_ci(boot_spearman)
        ke_lo, ke_hi = _percentile_ci(boot_kendall)
        mc_lo, mc_hi = _percentile_ci(boot_macro_conc)
        po_lo, po_hi = _percentile_ci(boot_pooled)
        rows.append(
            {
                "split": split_mode,
                "split_label": SPLIT_LABELS[split_mode],
                "n_batches": n_batches,
                "n_records": n_records,
                "macro_spearman": macro_spearman,
                "macro_spearman_ci_low": sp_lo,
                "macro_spearman_ci_high": sp_hi,
                "macro_kendall_tau_b": macro_kendall,
                "macro_kendall_tau_b_ci_low": ke_lo,
                "macro_kendall_tau_b_ci_high": ke_hi,
                "macro_pairwise_concordance": macro_conc,
                "macro_pairwise_concordance_ci_low": mc_lo,
                "macro_pairwise_concordance_ci_high": mc_hi,
                "pooled_pairwise_concordance": pooled,
                "pooled_pairwise_concordance_ci_low": po_lo,
                "pooled_pairwise_concordance_ci_high": po_hi,
            }
        )
    return pd.DataFrame(rows)


def load_released_splits(
    splits_dir: str | Path,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load released folds while preserving canonical row indices."""
    import json

    result: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
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


def load_or_reuse_regression_oof(
    path: Path,
    e2_path: Path,
    df: pd.DataFrame,
    splits: dict,
) -> pd.DataFrame | None:
    """Load local OOF or reuse compatible E2 regression OOF."""
    if path.exists():
        oof = pd.read_csv(path)
        _validate_regression_oof(oof, df, splits)
        return oof
    if e2_path.exists():
        oof = pd.read_csv(e2_path)
        _validate_regression_oof(oof, df, splits)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(e2_path, path)
        return oof
    return None


def _validate_regression_oof(
    regression_oof: pd.DataFrame,
    df: pd.DataFrame,
    splits: dict,
) -> None:
    required = {
        "task",
        "split",
        "fold",
        "row_idx",
        "y_true",
        "y_pred",
    }
    missing = required - set(regression_oof.columns)
    if missing:
        raise ValueError(f"Regression OOF missing columns: {sorted(missing)}")
    valid_regression = set(np.flatnonzero(df["pKd_value"].notna().to_numpy()))
    for split_mode in SPLIT_MODES:
        expected = np.concatenate([val for _, val in splits[split_mode]])
        if not np.array_equal(np.sort(expected), np.arange(len(df))):
            raise ValueError(f"Released folds do not cover dataset: {split_mode}")
        reg = regression_oof.loc[regression_oof["split"] == split_mode]
        if reg["row_idx"].duplicated().any() or set(reg["row_idx"]) != valid_regression:
            raise ValueError(f"Invalid regression OOF coverage: {split_mode}")


def generate_regression_oof(
    df: pd.DataFrame,
    splits: dict,
    classification_results: Path,
    regression_results: Path,
    embeddings_dir: Path,
    *,
    force_recompute: bool,
    seed: int,
) -> pd.DataFrame:
    from src.models.screening import precompute_and_save_embeddings
    from src.models.subgroup_analysis import (
        generate_oof_predictions,
        required_encoder_names,
        select_headline_configs,
    )

    classification_configs, regression_configs = select_headline_configs(
        classification_results,
        regression_results,
    )
    aptamer_names, molecule_names = required_encoder_names(
        classification_configs,
        regression_configs,
    )
    aptamer_configs, molecule_configs = build_encoder_configs(
        aptamer_names,
        molecule_names,
    )
    aptamer_embeddings, molecule_embeddings, _ = precompute_and_save_embeddings(
        df,
        aptamer_configs,
        molecule_configs,
        embeddings_dir=embeddings_dir,
        force_recompute=force_recompute,
    )
    _, regression_oof = generate_oof_predictions(
        df,
        splits,
        {**aptamer_embeddings, **molecule_embeddings},
        classification_configs,
        regression_configs,
        random_state=seed,
    )
    return regression_oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--classification-results",
        type=Path,
        default=DEFAULT_CLASSIFICATION_RESULTS,
    )
    parser.add_argument(
        "--regression-results",
        type=Path,
        default=DEFAULT_REGRESSION_RESULTS,
    )
    parser.add_argument("--e2-oof", type=Path, default=DEFAULT_E2_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "cache" / "embeddings",
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    parser.add_argument(
        "--force-oof",
        action="store_true",
        help="Recompute embeddings and OOF even when prediction files exist.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Require existing OOF and only regenerate ranking tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    oof_dir = output_dir / "oof"
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset).reset_index(drop=True)
    splits = load_released_splits(args.splits_dir)
    annotated = annotate_doi_batches(df)
    eligible = eligible_batches(annotated)
    eligible.to_csv(output_dir / "e8_eligible_batches.csv", index=False)

    oof_path = oof_dir / "e8_regression_oof.csv"
    if args.force_oof:
        if args.analysis_only:
            raise ValueError("--analysis-only cannot be combined with --force-oof")
        regression_oof = generate_regression_oof(
            df,
            splits,
            args.classification_results,
            args.regression_results,
            args.embeddings_dir,
            force_recompute=True,
            seed=args.seed,
        )
        regression_oof.to_csv(oof_path, index=False)
    else:
        regression_oof = load_or_reuse_regression_oof(
            oof_path, args.e2_oof, df, splits
        )
        if regression_oof is None:
            if args.analysis_only:
                raise FileNotFoundError(
                    "--analysis-only requires results/oof/e8_regression_oof.csv "
                    f"(or reusable E2 OOF at {args.e2_oof})"
                )
            regression_oof = generate_regression_oof(
                df,
                splits,
                args.classification_results,
                args.regression_results,
                args.embeddings_dir,
                force_recompute=False,
                seed=args.seed,
            )
            regression_oof.to_csv(oof_path, index=False)

    batch_metrics = compute_batch_metrics(regression_oof, annotated, eligible)
    batch_metrics.to_csv(output_dir / "e8_batch_metrics.csv", index=False)
    summary = summarize_split(
        batch_metrics, n_boot=args.bootstrap, seed=args.seed
    )
    summary.to_csv(output_dir / "e8_summary.csv", index=False)


if __name__ == "__main__":
    main()
