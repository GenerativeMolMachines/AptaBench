import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from difflib import SequenceMatcher
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ===========================
# Internal utilities
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
    Cluster sequences by identity >= threshold with transitivity.
    Uses connected components in similarity graph.
    """

    n = len(seqs)
    adj = [[] for _ in range(n)]

    # Build an undirected similarity graph and extract connected components.
    for i in range(n):
        for j in range(i + 1, n):
            if _seq_identity(seqs[i], seqs[j]) >= threshold:
                adj[i].append(j)
                adj[j].append(i)

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


def _cluster_scaffolds(scaffolds, *, missing="separate"):
    """
    Assign cluster_id to each scaffold.

    Note: returning -1 for missing scaffolds collapses all missing values into a single
    gigantic group which cannot be split across folds and can severely skew pair counts.

    :param missing:
        - "together": collapse all missing into one group id (-1) (old behavior)
        - "separate": give each missing entry its own unique group id (default)
    """
    s = pd.Series(scaffolds)
    uniq = {sc: i for i, sc in enumerate(s.dropna().unique())}

    if missing not in {"together", "separate"}:
        raise ValueError("missing must be one of: {'together', 'separate'}")

    if missing == "together":
        return np.array([uniq.get(sc, -1) for sc in scaffolds])

    # missing == "separate": each missing scaffold is its own group id
    next_id = len(uniq)
    out = []
    for sc in scaffolds:
        if sc in uniq:
            out.append(uniq[sc])
        else:
            out.append(next_id)
            next_id += 1
    return np.array(out, dtype=int)


def _splits_with_target_test_frac(
    *,
    df_len: int,
    make_splits_fn,
    target_test_frac: float,
    frac_tolerance: float,
    random_state: int,
    max_tries: int,
):
    """
    Try different seeds until fold test fractions are within tolerance.

    This keeps the core splitter logic intact (disjoint groups + stratification),
    while preventing pathological fold size imbalance when group sizes are skewed.
    """
    if not (0.0 < target_test_frac < 1.0):
        raise ValueError("target_test_frac must be in (0, 1)")
    if frac_tolerance < 0:
        raise ValueError("frac_tolerance must be >= 0")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")

    rng = np.random.default_rng(random_state)
    best = None
    best_max_abs_err = float("inf")

    # Retry with different seeds and keep the closest fold-size profile.
    for _ in range(max_tries):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        splits = make_splits_fn(seed)

        test_fracs = [len(va) / df_len for _, va in splits]
        max_abs_err = max(abs(f - target_test_frac) for f in test_fracs)

        if max_abs_err < best_max_abs_err:
            best = splits
            best_max_abs_err = max_abs_err

        if max_abs_err <= frac_tolerance:
            return splits

    # return best effort (closest we found)
    return best


# ==============================================================
#   1) STRATIFIED GROUP BY (sequence_similarity + scaffold)
# ==============================================================

def stratified_group_splits(
    df,
    label_col="label",
    group_cols=("sequence", "canonical_smiles"),
    n_splits=5,
    random_state=42,
    target_test_frac=None,
    frac_tolerance=0.02,
    max_tries=50,
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

    def _make(seed: int):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (tr, va)
            for tr, va in sgkf.split(
                X=np.zeros(len(df)),
                y=df[label_col].values,
                groups=groups.values,
            )
        ]

    if target_test_frac is None:
        target_test_frac = 1.0 / n_splits

    return _splits_with_target_test_frac(
        df_len=len(df),
        make_splits_fn=_make,
        target_test_frac=float(target_test_frac),
        frac_tolerance=float(frac_tolerance),
        random_state=int(random_state),
        max_tries=int(max_tries),
    )



# ==============================================================
# 2) STRICT DISJOINT APTAMER SPLITS
# ==============================================================

def disjoint_aptamer_splits(
    df,
    n_splits=5,
    col="sequence",
    label_col="label",
    random_state=42,
    seq_similarity_threshold=0.8,
    target_test_frac=None,
    frac_tolerance=0.02,
    max_tries=50,
):
    """Sequence groups are sequence-similarity clusters."""

    # Cluster unique sequences first, then map cluster IDs back to each row.
    unique_seqs = df[col].astype(str).drop_duplicates().tolist()
    unique_clusters = _cluster_sequences(unique_seqs)

    # map back to rows
    seq_to_cluster = {seq: unique_clusters[i] for i, seq in enumerate(unique_seqs)}
    groups = df[col].map(seq_to_cluster).values

    y = df[label_col].values

    def _make(seed: int):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (tr, va)
            for tr, va in sgkf.split(
                X=np.zeros(len(df)),
                y=y,
                groups=groups,
            )
        ]

    if target_test_frac is None:
        target_test_frac = 1.0 / n_splits

    return _splits_with_target_test_frac(
        df_len=len(df),
        make_splits_fn=_make,
        target_test_frac=float(target_test_frac),
        frac_tolerance=float(frac_tolerance),
        random_state=int(random_state),
        max_tries=int(max_tries),
    )


# ==============================================================
# 3) STRICT DISJOINT MOLECULE SPLITS
# ==============================================================

def disjoint_molecule_splits(
    df,
    n_splits=5,
    col="canonical_smiles",
    label_col="label",
    random_state=42,
    scaffold_missing="separate",
    target_test_frac=None,
    frac_tolerance=0.02,
    max_tries=50,
):
    """Molecule groups are scaffold clusters."""

    smiles = df[col].astype(str).tolist()
    # Group molecules by Murcko scaffold to enforce chemistry-level disjointness.
    scaffolds = [_get_scaffold(smi) for smi in smiles]
    groups = _cluster_scaffolds(scaffolds, missing=scaffold_missing)

    y = df[label_col].values

    def _make(seed: int):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (tr, va)
            for tr, va in sgkf.split(
                X=np.zeros(len(df)),
                y=y,
                groups=groups,
            )
        ]

    if target_test_frac is None:
        target_test_frac = 1.0 / n_splits

    return _splits_with_target_test_frac(
        df_len=len(df),
        make_splits_fn=_make,
        target_test_frac=float(target_test_frac),
        frac_tolerance=float(frac_tolerance),
        random_state=int(random_state),
        max_tries=int(max_tries),
    )
