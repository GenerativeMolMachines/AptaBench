"""Cross-platform cleanup utility used by Makefile."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DIR_PATTERNS = [
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
]

FILE_PATTERNS = ["*.pyc", "*.pyo", "*.pyd"]

EXTRA_ALL_DIRS = [
    "build",
    "dist",
    "checkpoints",
    "quick_test_checkpoints",
    ".cache",
    "notebooks/logs",
    "notebooks/notebooks",
]

EXTRA_ALL_FILES = [
    "artifacts/ml/classification/checkpoints/*.db",
    "artifacts/cache/embeddings/*.npy",
    "*.egg-info",
    "notebooks/lgbm_results_*.csv",
    "notebooks/regression_results.csv",
    "notebooks/results/lgbm_results_online.csv",
]


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def clean_base() -> None:
    for pattern in DIR_PATTERNS:
        for path in REPO_ROOT.rglob(pattern):
            remove_path(path)
    for pattern in FILE_PATTERNS:
        for path in REPO_ROOT.rglob(pattern):
            remove_path(path)


def clean_all() -> None:
    for path_str in EXTRA_ALL_DIRS:
        remove_path(REPO_ROOT / path_str)
    for pattern in EXTRA_ALL_FILES:
        for path in REPO_ROOT.glob(pattern):
            remove_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform cleanup script.")
    parser.add_argument("--all", action="store_true", help="Remove extra generated artifacts.")
    args = parser.parse_args()

    clean_base()
    if args.all:
        clean_all()


if __name__ == "__main__":
    main()
