import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


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