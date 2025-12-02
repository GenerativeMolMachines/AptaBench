# Changelog

## [1.0.0] - 2025-12-02

### Added
- Unified aptamer-molecule dataset from multiple sources
- ML models (LightGBM + Optuna)
- DL models (Two-Tower, Cross-Attention, Multi-Task, Pretrained)
- Aptamer encoders (One-Hot, K-mer, GENA-LM, DNABERT-2)
- Molecule encoders (Morgan, MACCS, RDKit descriptors, ChemBERTa, UniMol, CLAMP)
- Three split strategies (stratified, disjoint aptamer, disjoint molecule)
- Embedding caching system
- Jupyter notebooks (dataset construction, exploration, ML/DL benchmarks)

## [Unreleased]

### Planned
- Docker support
- Pre-trained weights
- Web API

### Known Issues
- UniMol/CLAMP may require manual setup
- Large caches need significant disk space

