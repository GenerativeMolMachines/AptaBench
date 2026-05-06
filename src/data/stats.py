import os
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
from difflib import SequenceMatcher
from src.models.screening import load_splits


def compute_molecular_weights(smiles_series):
    """
    Calculates the molecular weight for each molecule from SMILES.

    :param smiles_series: pandas.Series containing SMILES strings
    :return: pandas.Series with molecular weights
    """
    return smiles_series.apply(
        lambda smi: Descriptors.MolWt(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else np.nan
    )




def compute_logp(smiles_series):
    """
    Calculates the octanol-water partition coefficient (logP) for each molecule from SMILES.

    :param smiles_series: pandas.Series containing SMILES strings
    :return: pandas.Series with logP values
    """
    return smiles_series.apply(
        lambda smi: Descriptors.MolLogP(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else np.nan
    )




def describe_datasets_with_mw_and_logp(dataset_dict):
    """
    Generates summary statistics for multiple datasets, including aptamer information
    and molecular descriptors (molecular weight and logP).

    :param dataset_dict: dictionary of {dataset_name: DataFrame}
    :return: pandas.DataFrame with summary statistics
    """
    stats = []
    for name, df in dataset_dict.items():
        aptamers = df['sequence'].dropna()
        mols = df['canonical_smiles'].dropna()
        types = df['type'].str.upper()

        # Add molecular weight if not already computed
        if 'molecular_weight' not in df.columns:
            df['molecular_weight'] = compute_molecular_weights(df['canonical_smiles'])

        # Add logP if not already computed
        if 'logP' not in df.columns:
            df['logP'] = compute_logp(df['canonical_smiles'])

        stats.append({
            'Dataset': name,
            'N rows': len(df),
            'N unique aptamers': aptamers.nunique(),
            'DNA:RNA ratio': f"{(types == 'DNA').sum()}:{(types == 'RNA').sum()}",
            'Mean aptamer length ± std': f"{aptamers.str.len().mean():.1f} ± {aptamers.str.len().std():.1f}",
            'N unique molecules': mols.nunique(),
            'Mean MW ± std': f"{df['molecular_weight'].mean():.1f} ± {df['molecular_weight'].std():.1f}",
            'Mean logP ± std': f"{df['logP'].mean():.2f} ± {df['logP'].std():.2f}",
            'N with pKd': df['pKd_value'].notna().sum(),
            'Active:Inactive': f"{(df['label'] == 1).sum()}:{(df['label'] == 0).sum()}"
        })
    return pd.DataFrame(stats)


import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import DataStructs
from Levenshtein import distance as levenshtein_distance



def compute_molecular_weights(smiles_series):
    return smiles_series.apply(
        lambda smi: Descriptors.MolWt(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else np.nan
    )


def compute_logp(smiles_series):
    return smiles_series.apply(
        lambda smi: Descriptors.MolLogP(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else np.nan
    )


def compute_pairwise_levenshtein(sequences):
    dists = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            dists.append(levenshtein_distance(sequences[i], sequences[j]))
    return dists


def compute_tanimoto_similarity(smiles_list):

    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
    fps = [GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    return sims


def describe_datasets_with_distributions(dataset_dict):

    summary = []
    distributions = []

    for name, df in dataset_dict.items():

        df = df.copy()

        aptamers = df["sequence"].dropna()
        mols = df["canonical_smiles"].dropna()
        types = df["type"].str.upper()

        if "molecular_weight" not in df.columns:
            df["molecular_weight"] = compute_molecular_weights(df["canonical_smiles"])

        if "logP" not in df.columns:
            df["logP"] = compute_logp(df["canonical_smiles"])

        apt_lengths = aptamers.str.len()

        # summary table
        summary.append({
            "Dataset": name,
            "N rows": len(df),
            "N unique aptamers": aptamers.nunique(),
            "DNA:RNA ratio": f"{(types=='DNA').sum()}:{(types=='RNA').sum()}",
            "Mean aptamer length ± std": f"{apt_lengths.mean():.1f} ± {apt_lengths.std():.1f}",
            "N unique molecules": mols.nunique(),
            "Mean MW ± std": f"{df['molecular_weight'].mean():.1f} ± {df['molecular_weight'].std():.1f}",
            "Mean logP ± std": f"{df['logP'].mean():.2f} ± {df['logP'].std():.2f}",
            "N with pKd": df["pKd_value"].notna().sum(),
            "Active:Inactive": f"{(df['label']==1).sum()}:{(df['label']==0).sum()}"
        })

        # distributions for plotting

        for v in apt_lengths:
            distributions.append({
                "dataset": name,
                "feature": "aptamer_length",
                "value": v
            })

        for v in df["molecular_weight"].dropna():
            distributions.append({
                "dataset": name,
                "feature": "molecular_weight",
                "value": v
            })

        for v in df["logP"].dropna():
            distributions.append({
                "dataset": name,
                "feature": "logP",
                "value": v
            })

        # pairwise sequence diversity
        seq_dists = compute_pairwise_levenshtein(list(aptamers.unique())[:200])

        for v in seq_dists:
            distributions.append({
                "dataset": name,
                "feature": "aptamer_levenshtein",
                "value": v
            })

        # molecule diversity
        mol_sims = compute_tanimoto_similarity(list(mols.unique())[:200])

        for v in mol_sims:
            distributions.append({
                "dataset": name,
                "feature": "molecule_tanimoto",
                "value": v
            })

    summary_df = pd.DataFrame(summary)
    dist_df = pd.DataFrame(distributions)

    return summary_df, dist_df



def compute_intersections(dataset_dict):
    """
    Computes intersections between multiple datasets in terms of aptamer–molecule pairs,
    aptamer sequences, and molecules.

    For each pair of datasets, the function calculates:
      - the number of common (sequence, SMILES, pKd_value) triplets,
      - the number of common aptamer sequences,
      - the number of common molecules (SMILES).

    The result is a symmetric matrix (DataFrame) with values formatted as
    "common_pairs/common_aptamers/common_molecules".

    :param dataset_dict: dictionary of {dataset_name: DataFrame}, 
                         where each DataFrame contains columns ['sequence', 'canonical_smiles', 'pKd_value']
    :return: pandas.DataFrame, where entry (i, j) shows intersections between dataset_i and dataset_j
    """
    names = list(dataset_dict.keys())
    result_df = pd.DataFrame(index=names, columns=names)

    for name1 in names:
        df1 = dataset_dict[name1]
        set1_pairs = set(zip(df1['sequence'], df1['canonical_smiles'], df1['pKd_value']))
        set1_apt = set(df1['sequence'])
        set1_mol = set(df1['canonical_smiles'])

        for name2 in names:
            df2 = dataset_dict[name2]
            set2_pairs = set(zip(df2['sequence'], df2['canonical_smiles'], df2['pKd_value']))
            set2_apt = set(df2['sequence'])
            set2_mol = set(df2['canonical_smiles'])

            # Count intersections
            common_pairs = len(set1_pairs & set2_pairs)
            common_apt = len(set1_apt & set2_apt)
            common_mol = len(set1_mol & set2_mol)

            # Store result in format "pairs/aptamers/molecules"
            result_df.loc[name1, name2] = f"{common_pairs}/{common_apt}/{common_mol}"

    return result_df


def _compute_scaffold(smiles: str):
    """Compute Bemis-Murcko scaffold for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def _seq_identity(a: str, b: str) -> float:
    """Sequence identity (0-1) using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()


def analyze_generalization_splits(
    df: pd.DataFrame,
    split_modes=("stratified", "disjoint_aptamer", "disjoint_molecule"),
    splits_dir="dataset/splits",
    sequence_col="sequence",
    smiles_col="canonical_smiles",
    identity_threshold=0.8,
    max_train_seqs_for_similarity=2000,
):
    """
    Analyze each fold within each split_mode.
    Returns 15 rows (3 splits × 5 folds) instead of 3.
    """
    df = df.reset_index(drop=True)

    results = []

    for split_mode in split_modes:
        splits = load_splits(split_mode, base_dir=splits_dir)

        for fold_id, (train_idx, test_idx) in enumerate(splits):

            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()

            
            train_pairs = list(zip(train[sequence_col], train[smiles_col]))
            test_pairs = list(zip(test[sequence_col], test[smiles_col]))

            train_pair_dups = len(train_pairs) - len(set(train_pairs))
            test_pair_dups = len(test_pairs) - len(set(test_pairs))

            train_seq_dups = train[sequence_col].duplicated().sum()
            test_seq_dups = test[sequence_col].duplicated().sum()

            train_smiles_dups = train[smiles_col].duplicated().sum()
            test_smiles_dups = test[smiles_col].duplicated().sum()

            
            shared_pairs = set(train_pairs).intersection(set(test_pairs))
            shared_seqs = set(train[sequence_col]).intersection(set(test[sequence_col]))
            shared_smiles = set(train[smiles_col]).intersection(set(test[smiles_col]))

            # --- 3. Scaffold leakage ---
            train_scaffolds = train[smiles_col].apply(_compute_scaffold)
            test_scaffolds = test[smiles_col].apply(_compute_scaffold)
            shared_scaffolds = set(train_scaffolds.dropna()).intersection(
                set(test_scaffolds.dropna())
            )

            # --- 4. Sequence-similarity leakage ---
            unique_train_seqs = list(train[sequence_col].drop_duplicates())
            unique_test_seqs = list(test[sequence_col].drop_duplicates())

            if len(unique_train_seqs) > max_train_seqs_for_similarity:
                unique_train_seqs = unique_train_seqs[:max_train_seqs_for_similarity]

            leakage_count = 0
            for s_test in unique_test_seqs:
                max_id = 0.0
                for s_train in unique_train_seqs:
                    sim = _seq_identity(s_test, s_train)
                    if sim > max_id:
                        max_id = sim
                    if max_id >= identity_threshold:
                        break
                if max_id >= identity_threshold:
                    leakage_count += 1

            frac_leaky_seqs = leakage_count / max(1, len(unique_test_seqs))

            results.append(
                {
                    "split": split_mode,
                    "fold": fold_id,
                    
                    "n_train_pairs": len(train),
                    "n_test_pairs": len(test),
                    "n_unique_train_pairs": len(set(train_pairs)),
                    "n_unique_test_pairs": len(set(test_pairs)),
                    
                    "train_pair_duplicates": train_pair_dups,
                    "test_pair_duplicates": test_pair_dups,
                    "train_seq_duplicates": int(train_seq_dups),
                    "test_seq_duplicates": int(test_seq_dups),
                    "train_smiles_duplicates": int(train_smiles_dups),
                    "test_smiles_duplicates": int(test_smiles_dups),
                    
                    "shared_pairs": len(shared_pairs),
                    "shared_seqs": len(shared_seqs),
                    "shared_smiles": len(shared_smiles),
                    # scaffolds
                    "shared_scaffolds": len(shared_scaffolds),
                    # similarity leakage
                    f"test_seqs_identity>={identity_threshold}": leakage_count,
                    f"frac_test_seqs_identity>={identity_threshold}": frac_leaky_seqs,
                }
            )

    return pd.DataFrame(results)
