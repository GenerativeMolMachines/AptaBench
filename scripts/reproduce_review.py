"""Reviewer-oriented reproducibility entrypoint.

Runs split generation, executes key notebooks, and validates expected outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

QUICK_NOTEBOOKS = [
    NOTEBOOK_DIR / "02_data_exploration.ipynb",
    NOTEBOOK_DIR / "03_ml_benchmark.ipynb",
]

FULL_EXTRA_NOTEBOOKS = [
    NOTEBOOK_DIR / "04_dl_model.ipynb",
]

EXPECTED_SPLITS = [
    REPO_ROOT / "dataset" / "splits" / "stratified.json",
    REPO_ROOT / "dataset" / "splits" / "disjoint_aptamer.json",
    REPO_ROOT / "dataset" / "splits" / "disjoint_molecule.json",
]

EXPECTED_QUICK_OUTPUTS = [
    REPO_ROOT / "artifacts" / "ml" / "classification" / "screening" / "all_splits" / "results.csv",
    REPO_ROOT / "artifacts" / "ml" / "regression" / "results_reg.csv",
    REPO_ROOT / "artifacts" / "figures" / "dataset" / "dataset_distributions.png",
    REPO_ROOT / "artifacts" / "figures" / "dataset" / "split_strategies.png",
]

EXPECTED_FULL_OUTPUTS = [
    REPO_ROOT / "artifacts" / "dl" / "screening" / "comprehensive_screening_all_splits.csv",
    REPO_ROOT / "artifacts" / "dl" / "screening" / "dl_leaderboard.csv",
    REPO_ROOT / "artifacts" / "dl" / "ablations" / "ablation_study_gena_unimol.csv",
]


def run_command(command: list[str], *, cwd: Path) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def run_splits() -> None:
    run_command([sys.executable, "-m", "dataset.make_splits"], cwd=REPO_ROOT)


def run_notebook(notebook_path: Path, timeout_seconds: int) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={timeout_seconds}",
            str(notebook_path),
        ],
        cwd=REPO_ROOT,
    )


def verify_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        print("\n[error] Missing expected artifacts:")
        for path in missing:
            print(f" - {path.relative_to(REPO_ROOT)}")
        raise SystemExit(1)
    print("\n[ok] All expected artifacts are present.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce reviewer outputs.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick reviewer path (splits + 02/03 notebooks).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-notebook execution timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_splits()

    notebooks = list(QUICK_NOTEBOOKS)
    expected = list(EXPECTED_SPLITS) + list(EXPECTED_QUICK_OUTPUTS)

    if not args.quick:
        notebooks.extend(FULL_EXTRA_NOTEBOOKS)
        expected.extend(EXPECTED_FULL_OUTPUTS)

    for notebook in notebooks:
        run_notebook(notebook, timeout_seconds=args.timeout)

    verify_paths(expected)


if __name__ == "__main__":
    main()
