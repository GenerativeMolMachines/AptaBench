
import requests
import time
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem


COLUMNS_LIST = ['type', 'sequence', 'canonical_smiles', 'pKd_value', 'label', 'buffer', 'origin', 'source']


def canonicalize_smiles_column(df, smiles_column, new_column='canonical_smiles'):
    """
    Transforms SMILES column to canonical form, adds a new column
    
    :param df: pandas.DataFrame, which contains SMILES
    :param smiles_column: str, SMILES column name
    :param new_column: str, new column name for canonical SMILES
    :return: DataFrame with new column added
    """
    def to_canonical(smiles):
        if pd.isna(smiles):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except:
            return None

    df[new_column] = df[smiles_column].apply(to_canonical)
    return df




def clean_and_validate_sequences(df, seq_col='sequence'):
    """
    Processes aptamer sequences and determines their type (DNA or RNA),
    replacing non-standard notations with standard nucleotide letters.

    :param df: DataFrame containing the sequence column
    :param seq_col: name of the column with raw sequences
    :return: DataFrame with additional columns:
            - sequence
            - type ('rna', 'dna', np.nan)
            - is_valid (True/False)
    """
    def normalize(seq):
        if pd.isna(seq):
            return np.nan
        seq = str(seq).upper()
        # Remove 5'/3' start/end labels
        seq = re.sub(r"5'?-[ ]*|[ ]*-3'?", '', seq)
        # Remove spaces and line breaks
        seq = re.sub(r'\s+', '', seq)
        # Replace rA/rU/rC/rG -> A/U/C/G
        seq = re.sub(r'r([AUCG])PR?', r'\1', seq)
        # Replace dA/dT/dC/dG -> A/T/C/G
        seq = re.sub(r'd([ATCG])PD?', r'\1', seq)
        # Remove f, p and other markers
        seq = re.sub(r'[rdfup]*([ATUCG])p[dfru]*', r'\1', seq)
        # Remove extra characters
        seq = re.sub(r'[^ATUCG]', '', seq)
        return seq


    def get_type(s):
        if not isinstance(s, str):
            return np.nan
        if set(s) <= set('ATCG'):
            return 'DNA'
        elif set(s) <= set('AUGC'):
            return 'RNA'
        else:
            return np.nan

    df = df.copy()
    df['sequence'] = df[seq_col].apply(normalize)
    df['type'] = df['sequence'].apply(get_type)
    df['is_valid'] = df['type'].isin(['dna', 'rna'])
    return df




def process_binding_constants(df, column='kd_raw', unit='Kd'):
    """
    Processes binding constants: converts strings and numeric values into pKd and creates a label.

    :param df: DataFrame containing Kd/pKd values
    :param column: name of the column with data (Kd or pKd)
    :param unit: 'Kd' or 'pKd'
    :return: DataFrame with columns 'pKd_value' and 'label'
    """

    def extract_numeric_kd(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float)):
            return value

        value = str(value).lower()
        value = value.replace(",", ".")
        value = re.sub(r"\([^)]*\)", "", value)  # Remove parentheses
        value = value.replace("±", " ")
        value = value.replace("~", "")
        value = value.replace('μ', 'u').replace('µ', 'u')
        value = re.sub(r"[^\d.\-–eE+nmuµμpM\s]", " ", value)  # Remove extra characters

        if 'na' in value or value.strip() == "":
            return np.nan

        # Extract all "number + unit" pairs
        match = re.search(r"(\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?)\s*(pm|nm|µm|um|mm|nM|uM|µM|mM|μM)", value)
        if not match:
            return np.nan

        raw_num, unit = match.groups()
        unit = unit.replace('µ', 'u').lower()
        unit = unit.replace('μ', 'u').lower()

        # Handle ranges
        if '-' in raw_num or '–' in raw_num:
            try:
                parts = re.split(r'[-–]', raw_num)
                nums = [float(p) for p in parts if p.strip()]
                raw_value = np.mean(nums)
            except:
                return np.nan
        else:
            try:
                raw_value = float(raw_num)
            except:
                return np.nan

        # Convert to molar
        factor_map = {
            'pm': 1e-12,
            'nm': 1e-9,
            'um': 1e-6,
            'mm': 1e-3,
            'nm': 1e-9,
            'nM': 1e-9,
            'uM': 1e-6,
            'mM': 1e-3,
        }

        factor = factor_map.get(unit, None)
        if factor is None:
            return np.nan

        return raw_value * factor

    # Calculate pKd
    if unit == 'Kd':
        df['Kd_molar'] = df[column].apply(extract_numeric_kd)
        df['pKd_value'] = df['Kd_molar'].apply(lambda x: -np.log10(x) if pd.notna(x) and x > 0 else np.nan)
    elif unit == 'pKd':
        df['pKd_value'] = pd.to_numeric(df[column], errors='coerce')
    else:
        raise ValueError("unit must be 'Kd' or 'pKd'")

    # Generate label
    def assign_label(row):
        val = str(row[column]).lower()
        if 'inactive' in val:
            return 0
        if pd.notna(row['pKd_value']) and row['pKd_value'] <= 4:
            return 0
        if pd.isna(row['pKd_value']):
            return 1
        return 1

    df['label'] = df.apply(assign_label, axis=1)

    return df




def extract_target_name_from_description(df, description_col):
    """
    Extracts target names from a description column in a DataFrame.

    :param df: DataFrame containing the description column
    :param description_col: name of the column with target descriptions
    :return: DataFrame with an additional column 'target_name'
    """
    df = df.copy()

    def extract(text):
        if not isinstance(text, str) or not text.strip():
            return None

        # Remove everything after "/"
        text = text.split('/')[0].strip()

        # Try extracting an abbreviation from parentheses
        paren_match = re.findall(r'\(([^()]{1,25})\)', text)
        if paren_match:
            # Take the first one if it looks like an abbreviation
            for val in paren_match:
                if len(val.split()) <= 4 and re.search(r'[A-Zα-ωΑ-Ωβγδθλμνξπρστυφχψω]', val):
                    return val.strip()

        # Try extracting text before specific keywords
        match = re.match(
            r'^(.+?)\s+(is|was|are|were|denotes|exemplifies|aka|also known as)\b',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try extracting text before a comma
        if ',' in text:
            return text.split(',')[0].strip()

        # Fallback: first two words
        tokens = text.strip().split()
        return ' '.join(tokens[:2])

    df['target_name'] = df[description_col].apply(extract)
    return df





def resolve_smiles_by_name(df, name_column='name', output_column='canonical_smiles', delay=0.25):
    """
    Queries PubChem for each unique name and retrieves the canonical SMILES.

    :param df: pandas DataFrame containing a column of molecule names
    :param name_column: name of the column with molecule names
    :param output_column: name of the column for the result (SMILES)
    :param delay: delay between requests
    :return: DataFrame with columns 'canonical_smiles' and 'found_smiles'
    """

    def fetch_smiles(name):
        if pd.isna(name) or not isinstance(name, str) or not name.strip():
            return np.nan
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/CanonicalSMILES/TXT"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text.strip()
        except Exception:
            return np.nan
        return np.nan

    # Get unique non-empty names
    unique_names = df[name_column].dropna().astype(str).str.strip().unique()

    # Build a dictionary name -> SMILES
    name_to_smiles = {}
    for name in tqdm(unique_names, desc="Querying PubChem"):
        smiles = fetch_smiles(name)
        name_to_smiles[name] = smiles
        time.sleep(delay)

    # Map results back to the DataFrame
    df[output_column] = df[name_column].astype(str).str.strip().map(name_to_smiles)
    df['found_smiles'] = df[output_column].notna()

    return df




def resolve_smiles_by_name_or_pubchem_id(
    df,
    identifier_column='name',
    identifier_type='name',
    output_column='canonical_smiles',
    delay=1
):
    """
    Queries PubChem for each unique identifier (name or PubChem CID) and retrieves the canonical SMILES.

    :param df: pandas DataFrame containing a column with molecule identifiers
    :param identifier_column: name of the column with names or PubChem CIDs
    :param identifier_type: 'name' or 'pubchem_id'
    :param output_column: name of the column for SMILES
    :param delay: delay between requests (in seconds)
    :return: DataFrame with columns 'canonical_smiles' and 'found_smiles'
    """
    
    def fetch_smiles(identifier):
        if pd.isna(identifier) or not str(identifier).strip():
            return np.nan
        try:
            if identifier_type == 'name':
                encoded = requests.utils.quote(str(identifier).strip())
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/TXT"
            elif identifier_type == 'pubchem_id':
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(identifier)}/property/CanonicalSMILES/TXT"
            else:
                raise ValueError("identifier_type must be 'name' or 'pubchem_id'")
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text.strip()
        except Exception:
            return np.nan
        return np.nan

    # Unique values for querying
    unique_ids = df[identifier_column].dropna().astype(str).str.strip().unique()

    # Dictionary {ID: SMILES}
    id_to_smiles = {}
    for identifier in tqdm(unique_ids, desc=f"Querying PubChem by {identifier_type}"):
        smiles = fetch_smiles(identifier)
        id_to_smiles[identifier] = smiles
        time.sleep(delay)

    # Map results back to the DataFrame
    df[output_column] = df[identifier_column].astype(str).str.strip().map(id_to_smiles)
    df['found_smiles'] = df[output_column].notna()

    return df




def merge_inter_apt_exp(inter, apt, exp):
    """
    Merges interaction, aptamer, and experiment DataFrames into a single table for Aptamerbase.

    :param inter: DataFrame containing interaction data
    :param apt: DataFrame containing aptamer data
    :param exp: DataFrame containing experiment data
    :return: merged DataFrame with unified aptamer and experiment information
    """
    # Join aptamer sequence by has_participant ↔ id
    merged = inter.merge(
        apt,
        left_on="has_participant",
        right_on="id",
        suffixes=("", "_apt"),
        how="left"
    )

    # Join experiment by outcome ↔ int
    merged_outcome = merged.merge(
        exp,
        left_on="int",
        right_on="outcome",
        suffixes=("", "_exp_outcome"),
        how="left"
    )

    # Join experiment by confirms ↔ int
    merged_confirms = merged.merge(
        exp,
        left_on="int",
        right_on="confirms",
        suffixes=("", "_exp_confirms"),
        how="left"
    )

    # Combine both experiment versions — prioritize 'outcome', then 'confirms'
    cols_outcome = [col for col in merged_outcome.columns if col.endswith("_exp_outcome")]
    cols_confirms = [col for col in merged_confirms.columns if col.endswith("_exp_confirms")]

    # Compute the final table by combining values
    for col_out, col_conf in zip(cols_outcome, cols_confirms):
        base_name = col_out.replace("_exp_outcome", "")
        merged_outcome[base_name] = merged_outcome[col_out].combine_first(merged_confirms[col_conf])

    # Drop temporary columns
    merged_final = merged_outcome.drop(columns=cols_outcome + cols_confirms, errors="ignore")

    return merged_final




def extract_clean_target_name(df, column='label'):
    """
    Cleans target names in a DataFrame column and extracts standardized names in Aptamerbase.

    :param df: DataFrame containing raw target labels
    :param column: name of the column with raw labels (default: 'label')
    :return: DataFrame with an additional column 'target_name'
    """

    def clean_label(label):
        if pd.isna(label):
            return np.nan

        label = str(label).strip()

        # Remove prefixes like 8 digits or 8 digits + letter, followed by a hyphen
        label = re.sub(r'^\d{7,8}[a-zA-Z]?-', '', label)

        # Remove suffixes like -number or -Ma-number
        label = re.sub(r'-(Ma-)?\d+$', '', label)

        return label.strip()

    df['target_name'] = df[column].apply(clean_label)
    return df
