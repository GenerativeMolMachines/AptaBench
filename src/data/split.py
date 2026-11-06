import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
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


def _sequence_edge_chunk(seqs, start, stop, threshold):
    edges = []
    for i in range(start, stop):
        for j in range(i + 1, len(seqs)):
            if _seq_identity(seqs[i], seqs[j]) >= threshold:
                edges.append((i, j))
    return edges


def build_sequence_similarity_edges(seqs, threshold=0.8, n_jobs=1):
    """Build the reusable similarity graph for a sequence collection."""
    seqs = list(seqs)
    if n_jobs == 1 or len(seqs) < 2:
        return _sequence_edge_chunk(seqs, 0, len(seqs), threshold)
    workers = effective_n_jobs(n_jobs)
    chunk_size = max(1, int(np.ceil(len(seqs) / (workers * 4))))
    chunks = (
        (start, min(start + chunk_size, len(seqs)))
        for start in range(0, len(seqs), chunk_size)
    )
    nested = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_sequence_edge_chunk)(seqs, start, stop, threshold)
        for start, stop in chunks
    )
    return [edge for chunk in nested for edge in chunk]


def sequence_groups_from_edges(seqs, reference_seqs, edges):
    """Cluster a subset using the induced subgraph of cached full-data edges."""
    reference_index = {str(seq): i for i, seq in enumerate(reference_seqs)}
    active_reference = [reference_index[str(seq)] for seq in seqs]
    local_index = {
        reference_id: local_id
        for local_id, reference_id in enumerate(active_reference)
    }
    parent = list(range(len(active_reference)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in edges:
        if left in local_index and right in local_index:
            union(local_index[left], local_index[right])

    root_to_group = {}
    groups = []
    for node in range(len(active_reference)):
        root = find(node)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group)
        groups.append(root_to_group[root])
    return np.asarray(groups, dtype=int)


def _cluster_sequences(seqs, threshold=0.8, n_jobs=1):
    """
    Cluster sequences by identity >= threshold with transitivity.
    Uses connected components in similarity graph.
    """

    edges = build_sequence_similarity_edges(
        seqs,
        threshold=threshold,
        n_jobs=n_jobs,
    )
    return sequence_groups_from_edges(seqs, seqs, edges)


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


def get_split_groups(
    df,
    split_mode,
    *,
    seq_similarity_threshold=0.8,
    scaffold_missing="separate",
    sequence_n_jobs=1,
):
    """Return the group id assigned to every row for a split strategy."""
    if split_mode == "stratified":
        return (
            df[["sequence", "canonical_smiles"]]
            .astype(str)
            .agg("||".join, axis=1)
            .to_numpy()
        )

    if split_mode == "disjoint_aptamer":
        unique_seqs = df["sequence"].astype(str).drop_duplicates().tolist()
        unique_clusters = _cluster_sequences(
            unique_seqs,
            threshold=seq_similarity_threshold,
            n_jobs=sequence_n_jobs,
        )
        seq_to_cluster = {
            seq: unique_clusters[i] for i, seq in enumerate(unique_seqs)
        }
        return df["sequence"].astype(str).map(seq_to_cluster).to_numpy()

    if split_mode == "disjoint_molecule":
        scaffolds = [_get_scaffold(smi) for smi in df["canonical_smiles"].astype(str)]
        return _cluster_scaffolds(scaffolds, missing=scaffold_missing)

    raise ValueError(f"Unknown split mode: {split_mode}")


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
    return_groups=False,
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
    if tuple(group_cols) == ("sequence", "canonical_smiles"):
        groups = get_split_groups(df, "stratified")
    else:
        groups = df[list(group_cols)].astype(str).agg("||".join, axis=1).to_numpy()

    def _make(seed: int):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (tr, va)
            for tr, va in sgkf.split(
                X=np.zeros(len(df)),
                y=df[label_col].values,
                groups=groups,
            )
        ]

    if target_test_frac is None:
        target_test_frac = 1.0 / n_splits

    splits = _splits_with_target_test_frac(
        df_len=len(df),
        make_splits_fn=_make,
        target_test_frac=float(target_test_frac),
        frac_tolerance=float(frac_tolerance),
        random_state=int(random_state),
        max_tries=int(max_tries),
    )
    return (splits, groups) if return_groups else splits



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
    return_groups=False,
    precomputed_groups=None,
    sequence_n_jobs=1,
):
    """Sequence groups are sequence-similarity clusters."""

    # Cluster unique sequences first, then map cluster IDs back to each row.
    if precomputed_groups is not None:
        groups = np.asarray(precomputed_groups)
        if len(groups) != len(df):
            raise ValueError("precomputed_groups must have one value per row")
    elif col == "sequence":
        groups = get_split_groups(
            df,
            "disjoint_aptamer",
            seq_similarity_threshold=seq_similarity_threshold,
            sequence_n_jobs=sequence_n_jobs,
        )
    else:
        unique_seqs = df[col].astype(str).drop_duplicates().tolist()
        unique_clusters = _cluster_sequences(
            unique_seqs,
            threshold=seq_similarity_threshold,
            n_jobs=sequence_n_jobs,
        )
        seq_to_cluster = {
            seq: unique_clusters[i] for i, seq in enumerate(unique_seqs)
        }
        groups = df[col].astype(str).map(seq_to_cluster).to_numpy()

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

    splits = _splits_with_target_test_frac(
        df_len=len(df),
        make_splits_fn=_make,
        target_test_frac=float(target_test_frac),
        frac_tolerance=float(frac_tolerance),
        random_state=int(random_state),
        max_tries=int(max_tries),
    )
    return (splits, groups) if return_groups else splits


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
    return_groups=False,
):
    """Molecule groups are scaffold clusters."""

    if col == "canonical_smiles":
        groups = get_split_groups(
            df,
            "disjoint_molecule",
            scaffold_missing=scaffold_missing,
        )
    else:
        smiles = df[col].astype(str).tolist()
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

    splits = _splits_with_target_test_frac(
        df_len=len(df),
        make_splits_fn=_make,
        target_test_frac=float(target_test_frac),
        frac_tolerance=float(frac_tolerance),
        random_state=int(random_state),
        max_tries=int(max_tries),
    )
    return (splits, groups) if return_groups else splits
