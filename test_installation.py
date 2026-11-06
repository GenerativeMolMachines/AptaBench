"""
Minimal smoke checks.
Run: python test_installation.py
"""
from __future__ import annotations

import sys


def main() -> int:
    import numpy as np
    import pandas as pd

    assert np.__version__
    assert pd.__version__

    from datasets import load_dataset

    ds = load_dataset("GenerativeMolMachines/AptaBench_dataset", split="train")
    assert len(ds) > 0
    row = ds[0]
    assert "sequence" in row and "canonical_smiles" in row

    import src  # noqa: F401

    print("OK: core imports, Hugging Face dataset, and package `src` load successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
