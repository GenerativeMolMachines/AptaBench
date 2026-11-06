# E9 — Pharmacophore similarity diagnostic

Папка содержит воспроизводимый запуск E9 и только сырые результаты.
Агрегация across folds, интерпретация и reviewer-ready prose намеренно не
генерируются. Новый pharmacophore-disjoint split и LightGBM baseline в эту
фазу не входят.

## Дизайн

- dataset: released `dataset/AptaBench_dataset.csv`;
- protocols: molecule-disjoint (`disjoint_molecule`) и in-distribution
  (`stratified`) для контекста;
- frozen splits: локальные копии released 5-fold assignments из
  `dataset/splits/`;
- unit: уникальные `canonical_smiles` в train / test каждого fold;
- для каждой test molecule: максимальный Tanimoto к train molecules;
- fingerprints:
  - Morgan: radius 2, 1024 bits;
  - Gobbi Pharm2D: RDKit `Gobbi_Pharm2D.factory`;
- thresholds для долей: 0.3, 0.5, 0.7, 0.9;
- молекулы с пустым Gobbi FP (0 on-bits) получают `max_sim = NaN` и
  учитываются отдельно (`n_empty_fp` / исключаются из mean/percentiles).

## Запуск

Из корня репозитория:

```bash
python artifacts/additional_experiments/e9_pharmacophore_similarity_diagnostic/run_experiment.py --force
```

Или непосредственно из этой папки:

```bash
python run_experiment.py --force
```

Скрипт использует `dataset/AptaBench_dataset.csv` и локальные `splits/`,
поэтому требуется окружение AptaBench из корневого `environment.yml`
(нужен `rdkit`). Построение Gobbi Pharm2D fingerprints для ~900 молекул
занимает порядка нескольких минут один раз за запуск; дальше folds
переиспользуют этот кэш.

Без `--force` повторный запуск пропускается, если оба CSV уже существуют.

Короткая проверка одного split / fingerprint:

```bash
python run_experiment.py --only-split disjoint_molecule --only-fingerprint Morgan --force
```

## Результаты

Сохраняются только сырые файлы:

- `results/e9_test_molecule_max_similarity.csv` — по каждой test molecule:
  `split`, `fingerprint`, `fold`, `test_smiles`, `max_sim`,
  `n_train_molecules`;
- `results/e9_fold_similarity_summary.csv` — fold-level:
  `n_test_molecules`, `n_valid`, `n_empty_fp`, `mean_max_sim`, `median`,
  `p90`, `p95`, `maximum`, `frac_ge_0.3` … `frac_ge_0.9`;
- `splits/{stratified,disjoint_molecule}.json` — frozen fold assignments.
