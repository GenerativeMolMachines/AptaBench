"""Create a clean reviewer snapshot archive from current repository."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "artifacts" / "final" / "reviewer_snapshot"

IGNORE_PATTERNS = shutil.ignore_patterns(
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "*.pyc",
    "*.pyo",
    "*.db",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "logs",
    "notebooks/logs",
    "checkpoints",
    "quick_test_checkpoints",
    "artifacts/cache",
    "notebooks/notebooks",
)


def create_snapshot(output_base: Path) -> Path:
    snapshot_root = output_base / "reviewer_snapshot"
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    shutil.copytree(REPO_ROOT, snapshot_root, ignore=IGNORE_PATTERNS)
    return snapshot_root


def archive_snapshot(snapshot_root: Path) -> Path:
    archive_base = snapshot_root.parent / "reviewer_snapshot"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=snapshot_root)
    return Path(archive_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reviewer snapshot zip.")
    parser.add_argument(
        "--output-dir",
        default=str(SNAPSHOT_DIR.parent),
        help="Directory where reviewer_snapshot folder and zip are written.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_root = create_snapshot(out_dir)
    archive_path = archive_snapshot(snapshot_root)
    print(f"Snapshot directory: {snapshot_root}")
    print(f"Snapshot archive: {archive_path}")


if __name__ == "__main__":
    main()
