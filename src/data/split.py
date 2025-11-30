import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from difflib import SequenceMatcher
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ===========================
# ВНУТРЕННИЕ ВСПОМОГАТЕЛЬНЫЕ
# ===========================

def _seq_identity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)
    except:
        return None


def _cluster_sequences(seqs, threshold=0.8):
    """
    Кластеризация последовательностей по sequence identity >= threshold,
    с учётом транзитивности (A~B и B~C => A,B,C в одном кластере).
    Использует поиск компонент связности в графе схожести.
    """

    n = len(seqs)
    adj = [[] for _ in range(n)]

    # строим граф: ребро между seq[i] и seq[j], если они >= threshold
    for i in range(n):
        for j in range(i + 1, n):
            if _seq_identity(seqs[i], seqs[j]) >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # находим компоненты связности (DFS)
    visited = [False] * n
    groups = [-1] * n
    cluster_id = 0

    def dfs(v):
        stack = [v]
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                groups[u] = cluster_id
                for nei in adj[u]:
                    if not visited[nei]:
                        stack.append(nei)

    for i in range(n):
        if not visited[i]:
            dfs(i)
            cluster_id += 1

    return np.array(groups)


def _cluster_scaffolds(scaffolds):
    """
    Каждому уникальному scaffold присвоить cluster_id.
    """
    uniq = {sc: i for i, sc in enumerate(pd.Series(scaffolds).dropna().unique())}
    return np.array([uniq.get(sc, -1) for sc in scaffolds])


# ==============================================================
#   1) STRATIFIED GROUP BY (sequence_similarity + scaffold)
# ==============================================================

def stratified_group_splits(
    df,
    label_col="label",
    group_cols=("sequence", "canonical_smiles"),
    n_splits=5,
    random_state=42
):
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

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    return [
        (tr, va)
        for tr, va in sgkf.split(
            X=np.zeros(len(df)),
            y=df[label_col].values,
            groups=groups.values
        )
    ]



# ==============================================================
#   2) STRICT DISJOINT APTAMER SPLITS (по кластерам seq)
# ==============================================================

def disjoint_aptamer_splits(
    df,
    n_splits=5,
    col="sequence",
    label_col="label",
    random_state=42,
    seq_similarity_threshold=0.8
):
    """
    Теперь sequence-группы = sequence-similarity кластеры,
    а не просто уникальные строки.
    """

    unique_seqs = df[col].astype(str).drop_duplicates().tolist()
    unique_clusters = _cluster_sequences(unique_seqs)

    # map back to rows
    seq_to_cluster = {seq: unique_clusters[i] for i, seq in enumerate(unique_seqs)}
    groups = df[col].map(seq_to_cluster).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return [(tr, va) for tr, va in sgkf.split(
        X=np.zeros(len(df)), y=df[label_col].values, groups=groups
    )]


# ==============================================================
#   3) STRICT DISJOINT MOLECULE SPLITS (по scaffold)
# ==============================================================

def disjoint_molecule_splits(
    df,
    n_splits=5,
    col="canonical_smiles",
    label_col="label",
    random_state=42
):
    """
    Теперь molecule-группы = scaffold-кластеры,
    а не просто SMILES-строки.
    """

    smiles = df[col].astype(str).tolist()
    scaffolds = [_get_scaffold(smi) for smi in smiles]
    groups = _cluster_scaffolds(scaffolds)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return [(tr, va) for tr, va in sgkf.split(
        X=np.zeros(len(df)), y=df[label_col].values, groups=groups
    )]
