#!/usr/bin/env python3
"""
Test script to verify AptaBench installation.
Run this after installation to check if all components are working correctly.

Usage:
    python test_installation.py
"""

import sys
from typing import Tuple

def test_core_libraries() -> Tuple[bool, str]:
    """Test core scientific libraries."""
    try:
        import numpy as np
        import pandas as pd
        import scipy
        return True, f"NumPy {np.__version__}, Pandas {pd.__version__}, SciPy {scipy.__version__}"
    except ImportError as e:
        return False, str(e)

def test_machine_learning() -> Tuple[bool, str]:
    """Test machine learning libraries."""
    try:
        import sklearn
        import lightgbm as lgb
        import optuna
        return True, f"Scikit-learn {sklearn.__version__}, LightGBM {lgb.__version__}, Optuna {optuna.__version__}"
    except ImportError as e:
        return False, str(e)

def test_deep_learning() -> Tuple[bool, str]:
    """Test deep learning libraries."""
    try:
        import torch
        info = [f"PyTorch {torch.__version__}"]
        
        if torch.cuda.is_available():
            info.append(f"CUDA {torch.version.cuda}")
            info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info.append("MPS (Apple Silicon) available")
        else:
            info.append("No GPU acceleration (using CPU)")
        
        return True, ", ".join(info)
    except ImportError as e:
        return False, str(e)

def test_transformers() -> Tuple[bool, str]:
    """Test transformers library."""
    try:
        import transformers
        import tokenizers
        return True, f"Transformers {transformers.__version__}, Tokenizers {tokenizers.__version__}"
    except ImportError as e:
        return False, str(e)

def test_chemistry() -> Tuple[bool, str]:
    """Test chemistry libraries."""
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        # Test basic functionality
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            return False, "RDKit installed but not working correctly"
        return True, f"RDKit {rdkit_version}"
    except ImportError as e:
        return False, str(e)

def test_visualization() -> Tuple[bool, str]:
    """Test visualization libraries."""
    try:
        import matplotlib
        import seaborn as sns
        return True, f"Matplotlib {matplotlib.__version__}, Seaborn {sns.__version__}"
    except ImportError as e:
        return False, str(e)

def test_utilities() -> Tuple[bool, str]:
    """Test utility libraries."""
    try:
        import tqdm
        import requests
        return True, f"tqdm {tqdm.__version__}, requests {requests.__version__}"
    except ImportError as e:
        return False, str(e)

def test_project_modules() -> Tuple[bool, str]:
    """Test project modules."""
    try:
        from src.encoders import aptamer_encoders, molecule_encoders
        from src.models import screening
        from src.data import processing
        return True, "All project modules accessible"
    except ImportError as e:
        return False, str(e)

def test_optional_encoders() -> Tuple[bool, str]:
    """Test optional encoder libraries."""
    results = []
    
    # UniMol
    try:
        import unimol_tools
        results.append("✓ UniMol")
    except ImportError:
        results.append("✗ UniMol (optional)")
    
    # CLAMP
    try:
        import clamp
        results.append("✓ CLAMP")
    except ImportError:
        results.append("✗ CLAMP (optional)")
    
    # MHNReact
    try:
        import mhnreact
        results.append("✓ MHNReact")
    except ImportError:
        results.append("✗ MHNReact (optional)")
    
    return True, ", ".join(results)

def test_data_availability() -> Tuple[bool, str]:
    """Test if dataset is available."""
    import os
    from pathlib import Path
    
    dataset_path = Path("dataset/AptaBench_dataset_v2.csv")
    
    if not dataset_path.exists():
        return False, "Dataset file not found"
    
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        return True, f"Dataset loaded: {len(df)} samples"
    except Exception as e:
        return False, f"Error loading dataset: {str(e)}"

def main():
    """Run all tests and print results."""
    print("=" * 70)
    print("AptaBench Installation Test")
    print("=" * 70)
    print()
    
    tests = [
        ("Core Libraries", test_core_libraries),
        ("Machine Learning", test_machine_learning),
        ("Deep Learning", test_deep_learning),
        ("Transformers", test_transformers),
        ("Chemistry (RDKit)", test_chemistry),
        ("Visualization", test_visualization),
        ("Utilities", test_utilities),
        ("Project Modules", test_project_modules),
        ("Optional Encoders", test_optional_encoders),
        ("Data Availability", test_data_availability),
    ]
    
    all_passed = True
    results = []
    
    for test_name, test_func in tests:
        try:
            success, message = test_func()
            status = "✓" if success else "✗"
            color = "\033[92m" if success else "\033[91m"
            reset = "\033[0m"
            
            print(f"{color}{status}{reset} {test_name:.<50} {message}")
            
            results.append((test_name, success))
            if not success:
                all_passed = False
        except Exception as e:
            print(f"✗ {test_name:.<50} Unexpected error: {e}")
            results.append((test_name, False))
            all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✅ All tests passed! AptaBench is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run 'jupyter notebook' to explore the notebooks")
        print("  2. Check QUICKSTART.md for a quick introduction")
        print("  3. Read README.md for detailed documentation")
        return 0
    else:
        print("⚠ Some tests failed. See details above.")
        print()
        failed_tests = [name for name, success in results if not success]
        print("Failed tests:")
        for name in failed_tests:
            print(f"  - {name}")
        print()
        print("Solutions:")
        print("  - For RDKit issues: conda install -c conda-forge rdkit")
        print("  - For other missing packages: pip install -r requirements.txt")
        print("  - For project modules: pip install -e .")
        print()
        print("See INSTALL.md for detailed installation instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

