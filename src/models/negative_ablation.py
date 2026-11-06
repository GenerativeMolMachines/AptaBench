"""Utilities for the E3 curated-versus-shuffled negative experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.split import (
    disjoint_aptamer_splits,
    disjoint_molecule_splits,
    stratified_group_splits,
)


SPLIT_MODES = ("stratified", "disjoint_molecule", "disjoint_aptamer")
SCOPES = ("full", "without_specificity")
METRICS = ("roc_auc", "pr_auc", "mcc")
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 96,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.1,
}


def make_balanced_scope(
    df: pd.DataFrame,
    *,
    scope: str,
    seed: int,
) -> pd.DataFrame:
    """Keep all curated negatives and sample the same number of positives."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope: {scope}")

    required = {"sequence", "canonical_smiles", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    scoped = df.copy()
    scoped["original_index"] = np.arange(len(scoped), dtype=int)
    scoped = scoped.dropna(
        subset=["sequence", "canonical_smiles", "label", "source"]
    )
    if scope == "without_specificity":
        scoped = scoped.loc[scoped["source"] != "Specificity"]
    scoped["label"] = scoped["label"].astype(int)

    negatives = scoped.loc[scoped["label"] == 0]
    positives = scoped.loc[scoped["label"] == 1]
    if negatives.empty:
        raise ValueError(f"No curated negatives remain in scope {scope}")
    if len(positives) < len(negatives):
        raise ValueError(
            f"Cannot balance {scope}: {len(positives)} positives for "
            f"{len(negatives)} negatives"
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(
        positives.index.to_numpy(),
        size=len(negatives),
        replace=False,
    )
    balanced = pd.concat([negatives, positives.loc[selected]], axis=0)
    return balanced.sort_values("original_index").reset_index(drop=True)


def positive_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    positives = df.loc[df["label"] == 1, ["sequence", "canonical_smiles"]]
    return set(positives.itertuples(index=False, name=None))


def build_shuffled_negatives(
    train_df: pd.DataFrame,
    *,
    seed: int,
    known_positive_pairs: set[tuple[str, str]] | None = None,
    additional_forbidden_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Generate unique cross-pairs from train-positive entity pools."""
    rng = np.random.default_rng(seed)
    positives = train_df.loc[train_df["label"] == 1].copy()
    negative_count = int((train_df["label"] == 0).sum())
    if positives.empty or negative_count == 0:
        raise ValueError("Training data must contain both classes")

    forbidden = (
        positive_pairs(train_df)
        if known_positive_pairs is None
        else set(known_positive_pairs)
    )
    forbidden.update(additional_forbidden_pairs or set())
    # Keep row-level pools to match the frequency-weighted Table 5 sampler.
    aptamers = positives["sequence"].to_numpy()
    molecules = positives["canonical_smiles"].to_numpy()
    unique_aptamers = set(aptamers)
    unique_molecules = set(molecules)
    available = len(unique_aptamers) * len(unique_molecules) - sum(
        aptamer in unique_aptamers and molecule in unique_molecules
        for aptamer, molecule in forbidden
    )
    if available < negative_count:
        raise RuntimeError(
            f"Only {available} valid cross-pairs are available for "
            f"{negative_count} shuffled negatives"
        )

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(negative_count * 50, 1000)
    while len(rows) < negative_count and attempts < max_attempts:
        attempts += 1
        pair = (
            str(aptamers[int(rng.integers(0, len(aptamers)))]),
            str(molecules[int(rng.integers(0, len(molecules)))]),
        )
        if pair in forbidden or pair in seen:
            continue
        seen.add(pair)
        rows.append(
            {
                "type": np.nan,
                "sequence": pair[0],
                "canonical_smiles": pair[1],
                "pKd_value": np.nan,
                "label": 0,
                "buffer": "shuffled_negative",
                "origin": "synthetic_shuffle",
                "source": "negative_ablation",
                "original_index": np.nan,
            }
        )

    if len(rows) != negative_count:
        raise RuntimeError(
            f"Could not create enough shuffled negatives: "
            f"{len(rows)} / {negative_count}"
        )
    return pd.DataFrame(rows)


def make_train_variants(
    train_df: pd.DataFrame,
    *,
    seed: int,
    known_positive_pairs: set[tuple[str, str]],
    test_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    curated = train_df.copy().reset_index(drop=True)
    positives = curated.loc[curated["label"] == 1].copy()
    shuffled = build_shuffled_negatives(
        curated,
        seed=seed,
        known_positive_pairs=known_positive_pairs,
        additional_forbidden_pairs=test_pairs,
    )
    shuffled_train = pd.concat([positives, shuffled], ignore_index=True)
    if len(curated) != len(shuffled_train):
        raise ValueError("Curated and shuffled train variants differ in size")
    if not np.array_equal(
        curated["label"].value_counts().sort_index().to_numpy(),
        shuffled_train["label"].value_counts().sort_index().to_numpy(),
    ):
        raise ValueError("Curated and shuffled train variants differ in class counts")
    return {"Curated": curated, "Shuffled": shuffled_train}


def generate_first_fold(
    balanced_df: pd.DataFrame,
    *,
    split_mode: str,
    seed: int,
    frac_tolerance: float = 0.02,
    max_tries: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one five-fold bundle and return fold zero plus its fixed groups."""
    common = {
        "n_splits": 5,
        "random_state": seed,
        "frac_tolerance": frac_tolerance,
        "max_tries": max_tries,
        "return_groups": True,
    }
    if split_mode == "stratified":
        splits, groups = stratified_group_splits(balanced_df, **common)
    elif split_mode == "disjoint_molecule":
        splits, groups = disjoint_molecule_splits(
            balanced_df,
            scaffold_missing="separate",
            **common,
        )
    elif split_mode == "disjoint_aptamer":
        splits, groups = disjoint_aptamer_splits(balanced_df, **common)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    train_idx, test_idx = splits[0]
    train_groups = np.unique(np.asarray(groups)[train_idx])
    test_groups = np.unique(np.asarray(groups)[test_idx])
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"Index leakage in {split_mode}")
    if np.intersect1d(train_groups, test_groups).size:
        raise ValueError(f"Group leakage in {split_mode}")
    for name, indices in (("train", train_idx), ("test", test_idx)):
        if balanced_df.iloc[indices]["label"].nunique() != 2:
            raise ValueError(f"{split_mode} fold 0 {name} lacks one class")
    return np.asarray(train_idx), np.asarray(test_idx), np.asarray(groups)


@dataclass
class FeatureCache:
    """Cache canonical Kmer4 and 1024-bit MorganFP vectors by entity."""

    aptamers: dict[str, np.ndarray] = field(default_factory=dict)
    molecules: dict[str, np.ndarray] = field(default_factory=dict)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        from src.encoders.aptamer_encoders import kmer_freq_with_type_bit
        from src.encoders.molecule_encoders import morgan_fp

        sequences = df["sequence"].astype(str).tolist()
        smiles = df["canonical_smiles"].astype(str).tolist()

        missing_sequences = list(dict.fromkeys(
            sequence for sequence in sequences if sequence not in self.aptamers
        ))
        if missing_sequences:
            vectors = kmer_freq_with_type_bit(missing_sequences, k=4)
            self.aptamers.update(zip(missing_sequences, vectors))

        missing_smiles = list(dict.fromkeys(
            molecule for molecule in smiles if molecule not in self.molecules
        ))
        if missing_smiles:
            vectors = morgan_fp(missing_smiles, n_bits=1024, radius=2)
            self.molecules.update(zip(missing_smiles, vectors))

        aptamer_matrix = np.vstack([self.aptamers[value] for value in sequences])
        molecule_matrix = np.vstack([self.molecules[value] for value in smiles])
        return np.concatenate(
            [aptamer_matrix, molecule_matrix],
            axis=1,
        ).astype(np.float32)


def evaluate_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    feature_cache: FeatureCache,
) -> dict[str, float]:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import (
        average_precision_score,
        matthews_corrcoef,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    X_train = feature_cache.transform(train_df)
    X_test = feature_cache.transform(test_df)
    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LGBMClassifier(
        **LGBM_PARAMS,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = model.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "mcc": float(matthews_corrcoef(y_test, prediction)),
    }


def aggregate_seed_results(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Return reviewer rows, including paired Curated - Shuffled deltas."""
    keys = ["dataset", "scope", "split"]
    required_types = {"Curated", "Shuffled"}
    rows: list[dict] = []
    for key, part in seed_results.groupby(keys, sort=False):
        if set(part["negative_type"]) != required_types:
            raise ValueError(f"Incomplete negative types for {key}")
        if part.groupby("negative_type")["seed"].nunique().nunique() != 1:
            raise ValueError(f"Unpaired seed counts for {key}")

        for negative_type in ("Curated", "Shuffled"):
            selected = part.loc[part["negative_type"] == negative_type]
            row = dict(zip(keys, key))
            row["negative_type"] = negative_type
            row["n_seeds"] = int(selected["seed"].nunique())
            for metric in METRICS:
                values = selected.sort_values("seed")[metric].to_numpy()
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=0))
                row[metric] = (
                    f"{row[f'{metric}_mean']:.3f} ± "
                    f"{row[f'{metric}_std']:.3f}"
                )
            rows.append(row)

        pivot = part.pivot(index="seed", columns="negative_type", values=list(METRICS))
        if pivot.isna().any().any():
            raise ValueError(f"Seeds are not paired for {key}")
        row = dict(zip(keys, key))
        row["negative_type"] = "Curated − Shuffled"
        row["n_seeds"] = int(len(pivot))
        for metric in METRICS:
            values = (
                pivot[(metric, "Curated")] - pivot[(metric, "Shuffled")]
            ).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=0))
            row[metric] = (
                f"{row[f'{metric}_mean']:+.3f} ± "
                f"{row[f'{metric}_std']:.3f}"
            )
        rows.append(row)
    return pd.DataFrame(rows)
