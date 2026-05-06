import json
import subprocess
import sys
from pathlib import Path


def test_make_splits_script_generates_expected_files():
    subprocess.run([sys.executable, "-m", "dataset.make_splits"], check=True)
    for name in ["stratified", "disjoint_aptamer", "disjoint_molecule"]:
        path = Path("dataset/splits") / f"{name}.json"
        assert path.exists(), f"missing split file: {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list) and len(payload) > 0
