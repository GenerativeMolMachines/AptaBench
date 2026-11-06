#!/usr/bin/env python
"""Run E2 DNA/RNA and RSAPred subtype analyses."""

from __future__ import annotations

import argparse
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.screening import precompute_and_save_embeddings
from src.models.subgroup_analysis import (
    annotate_rsapred,
    build_rsapred_category_map,
    dna_rna_metrics,
    generate_oof_predictions,
    load_released_splits,
    required_encoder_names,
    rsapred_descriptives,
    rsapred_subtype_metrics,
    select_headline_configs,
    validate_oof_coverage,
)


DEFAULT_DATASET = REPO_ROOT / "dataset" / "AptaBench_dataset.csv"
DEFAULT_SPLITS = REPO_ROOT / "dataset" / "splits"
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
DEFAULT_RAW_RSAPRED = REPO_ROOT / "data" / "raw" / "RSAPred"
DEFAULT_OUTPUT = SCRIPT_DIR / "results"


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
    parser.add_argument("--rsapred-raw-dir", type=Path, default=DEFAULT_RAW_RSAPRED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "cache" / "embeddings",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument(
        "--force-oof",
        action="store_true",
        help="Recompute embeddings and OOF even when prediction files exist.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Require existing OOF files and only regenerate subgroup tables.",
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
    classification_configs, regression_configs = select_headline_configs(
        args.classification_results,
        args.regression_results,
    )
    classification_path = oof_dir / "e2_classification_oof.csv"
    regression_path = oof_dir / "e2_regression_oof.csv"
    have_oof = classification_path.exists() and regression_path.exists()

    if args.analysis_only and not have_oof:
        raise FileNotFoundError(
            "--analysis-only requires e2_classification_oof.csv and "
            "e2_regression_oof.csv"
        )
    if have_oof and not args.force_oof:
        classification_oof = pd.read_csv(classification_path)
        regression_oof = pd.read_csv(regression_path)
        validate_oof_coverage(classification_oof, regression_oof, df, splits)
    else:
        if args.analysis_only:
            raise FileNotFoundError("OOF files are missing")
        aptamer_names, molecule_names = required_encoder_names(
            classification_configs,
            regression_configs,
        )
        aptamer_configs, molecule_configs = build_encoder_configs(
            aptamer_names,
            molecule_names,
        )
        aptamer_embeddings, molecule_embeddings, _ = (
            precompute_and_save_embeddings(
                df,
                aptamer_configs,
                molecule_configs,
                embeddings_dir=args.embeddings_dir,
                force_recompute=args.force_oof,
            )
        )
        classification_oof, regression_oof = generate_oof_predictions(
            df,
            splits,
            {**aptamer_embeddings, **molecule_embeddings},
            classification_configs,
            regression_configs,
            random_state=args.seed,
        )
        classification_oof.to_csv(classification_path, index=False)
        regression_oof.to_csv(regression_path, index=False)

    fold_metrics, summary = dna_rna_metrics(
        classification_oof,
        regression_oof,
        df,
    )
    fold_metrics.to_csv(output_dir / "e2_dna_rna_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "e2_dna_rna_summary.csv", index=False)

    category_map, _ = build_rsapred_category_map(args.rsapred_raw_dir)
    annotated, _ = annotate_rsapred(df, category_map)
    descriptives = rsapred_descriptives(annotated)
    subtype_metrics = rsapred_subtype_metrics(
        classification_oof,
        annotated,
        min_per_class=args.min_per_class,
    )
    descriptives.to_csv(
        output_dir / "e2_rsapred_category_descriptives.csv",
        index=False,
    )
    subtype_metrics.to_csv(
        output_dir / "e2_rsapred_subtype_metrics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
