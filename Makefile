# Makefile for AptaBench
# Simplifies common commands

.PHONY: help install install-dev install-full test clean clean-all format lint notebook splits data reproduce-review reviewer-snapshot

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
	@echo "  make reproduce-review - Run quick reviewer reproduction path"
	@echo "  make reviewer-snapshot - Build clean reviewer snapshot archive"
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
	@echo "Creating train/val splits..."
	python -m dataset.make_splits
	@echo "✓ Splits created in dataset/splits/"

data:
	@echo "📂 Checking data..."
	@python -c "import pandas as pd; df = pd.read_csv('dataset/AptaBench_dataset.csv'); print(f'✓ Dataset loaded: {len(df)} samples')"

# Development tools
test:
	@echo "Running tests..."
	python -m pytest -q

reproduce-review:
	@echo "Running reviewer reproducibility workflow..."
	python scripts/reproduce_review.py --quick

reviewer-snapshot:
	@echo "Creating clean reviewer snapshot..."
	python scripts/create_reviewer_snapshot.py

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
	@echo "Cleaning temporary files..."
	python scripts/clean.py
	@echo "✓ Cleaning complete"

clean-all: clean
	@echo "Full cleanup..."
	python scripts/clean.py --all
	@echo "✓ Full cleanup complete"

# Quick test
quick-test:
	@echo "⚡ Quick installation test..."
	@python test_installation.py

# Docker (if added later)
# docker-build:
# 	docker build -t aptabench:latest .

# docker-run:
# 	docker run -it --rm -v $(PWD):/workspace aptabench:latest

