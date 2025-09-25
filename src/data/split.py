import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_splits(df, label_col="label", group_cols=("sequence", "canonical_smiles"),
                            n_splits=5, random_state=42):
    """
    Stratified group KFold:
    ensures disjoint groups across folds, while preserving label distribution.

    :param df: DataFrame with data
    :param label_col: column containing labels for stratification
    :param group_cols: columns used to define groups (combined)
    :param n_splits: number of folds
    :param random_state: random seed
    :return: list of (train_idx, val_idx) tuples
    """
    groups = df[list(group_cols)].astype(str).agg("||".join, axis=1)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(tr, va) for tr, va in sgkf.split(X=np.zeros(len(df)), y=df[label_col].values, groups=groups.values)]


def disjoint_aptamer_splits(df, n_splits=5, col="sequence", label_col="label", random_state=42):
    """
    Stratified group KFold split ensuring no aptamer sequences overlap between folds.

    :param df: DataFrame with aptamer sequences
    :param n_splits: number of folds
    :param col: column containing sequences
    :param label_col: column with class labels
    :param random_state: random seed
    :return: list of (train_idx, val_idx) tuples
    """
    groups = df[col].astype(str).values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(tr, va) for tr, va in sgkf.split(X=np.zeros(len(df)), y=df[label_col].values, groups=groups)]


def disjoint_molecule_splits(df, n_splits=5, col="canonical_smiles", label_col="label", random_state=42):
    """
    Stratified group KFold split ensuring no molecules (SMILES) overlap between folds.

    :param df: DataFrame with molecule SMILES
    :param n_splits: number of folds
    :param col: column containing SMILES
    :param label_col: column with class labels
    :param random_state: random seed
    :return: list of (train_idx, val_idx) tuples
    """
    groups = df[col].astype(str).values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(tr, va) for tr, va in sgkf.split(X=np.zeros(len(df)), y=df[label_col].values, groups=groups)]
