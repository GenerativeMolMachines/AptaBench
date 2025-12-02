"""
AptaBench - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="aptabench",
    version="1.0.0",
    author="AptaBench Team",
    author_email="your.email@example.com",
    description="A comprehensive benchmark for aptamer-molecule interaction prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/AptaBench",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=2.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.3.0",
        "optuna>=3.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "rdkit>=2023.3.1",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "tqdm>=4.60.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipython>=8.0.0",
            "notebook>=6.4.0",
        ],
        "viz": [
            "torchinfo>=1.7.0",
        ],
        "full": [
            "unimol-tools",
            "clamp-bio",
            "mhnreact",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

