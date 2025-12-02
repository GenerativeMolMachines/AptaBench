# Makefile for AptaBench
# Simplifies common commands

.PHONY: help install install-dev install-full test clean clean-all format lint notebook splits data

# Default target
help:
	@echo "AptaBench Makefile Commands:"
	@echo ""
	@echo "  make install         - Install basic dependencies"
	@echo "  make install-dev     - Install development dependencies"
	@echo "  make install-full    - Install all dependencies (including optional encoders)"
	@echo "  make test            - Run tests"
	@echo "  make clean           - Clean temporary files"
	@echo "  make clean-all       - Clean all generated files (including checkpoints)"
	@echo "  make format          - Format code (black)"
	@echo "  make lint            - Check code (flake8)"
	@echo "  make notebook        - Launch Jupyter Notebook"
	@echo "  make splits          - Create train/val splits"
	@echo "  make data            - Check data availability"
	@echo ""

# Installation targets
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "✓ Basic installation complete"

install-dev: install
	pip install -e ".[dev]"
	@echo "✓ Development installation complete"

install-full: install
	pip install -e ".[full,viz,notebooks]"
	@echo "✓ Full installation complete"
	@echo "⚠ Note: Some packages (clamp-bio, mhnreact) may require additional setup"

# Data preparation
splits:
	@echo "📊 Creating train/val splits..."
	cd dataset && python make_splits.py
	@echo "✓ Splits created in dataset/splits/"

data:
	@echo "📂 Checking data..."
	@python -c "import pandas as pd; df = pd.read_csv('dataset/AptaBench_dataset_v2.csv'); print(f'✓ Dataset loaded: {len(df)} samples')"

# Development tools
test:
	@echo "🧪 Running tests..."
	@python -c "import sys; print('⚠ Tests not yet configured. Create tests/ directory.')"
	# pytest tests/ -v

format:
	@echo "🎨 Formatting code..."
	black src/ --line-length 120
	@echo "✓ Formatting complete"

lint:
	@echo "🔍 Checking code..."
	flake8 src/ --max-line-length 120 --ignore=E203,W503
	@echo "✓ Check complete"

# Jupyter
notebook:
	@echo "📓 Launching Jupyter Notebook..."
	jupyter notebook

# Cleaning
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaning complete"

clean-all: clean
	@echo "🧹 Full cleanup..."
	rm -rf build/ dist/ *.egg-info
	rm -rf notebooks/checkpoints/*.db
	rm -rf notebooks/data/embeddings/*.npy
	rm -rf checkpoints/
	rm -rf quick_test_checkpoints/
	rm -rf .cache/
	@echo "✓ Full cleanup complete"

# Quick test
quick-test:
	@echo "⚡ Quick installation test..."
	@python -c "import numpy as np; import pandas as pd; import torch; import rdkit; import transformers; import lightgbm; import optuna; print('✅ All core libraries working!')"

# Docker (if added later)
# docker-build:
# 	docker build -t aptabench:latest .

# docker-run:
# 	docker run -it --rm -v $(PWD):/workspace aptabench:latest

