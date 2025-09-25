import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib import gridspec
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.encoders.aptamer_encoders import onehot_with_type_bit
from src.encoders.molecule_encoders import morgan_fp


def single_dataset_summary(df, seq_col="sequence", smiles_col="canonical_smiles", 
                    label_col="label", type_col="type", pkd_col="pKd_value"):
    """
    Print basic dataset summary and plot distributions for pKd and type+label.
    """

    print("=== Dataset Summary ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique sequences: {df[seq_col].nunique()}")
    print(f"Unique molecules (SMILES): {df[smiles_col].nunique()}")
    
    if label_col in df.columns:
        label_counts = df[label_col].value_counts(dropna=False)
        print("\nLabel distribution:")
        print(label_counts.to_string())

    # --- Plot 1: pKd distribution ---
    if pkd_col in df.columns and df[pkd_col].notna().sum() > 0:
        plt.figure(figsize=(7,5))
        sns.histplot(df[pkd_col].dropna(), bins=30, kde=True, color="teal")
        plt.title("Distribution of pKd values")
        plt.xlabel("pKd")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # --- Plot 2: type + label distribution ---
    if type_col in df.columns and label_col in df.columns:
        plt.figure(figsize=(7,5))
        sns.countplot(data=df, x=type_col, hue=label_col, palette="Set2")
        plt.title("Distribution by Type and Label")
        plt.xlabel("Type")
        plt.ylabel("Count")
        plt.legend(title="Label")
        plt.tight_layout()
        plt.show()

# src/viz/plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from Levenshtein import distance as levenshtein_distance

# ============================================================
# Molecular descriptors
# ============================================================

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
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi)]
    fps = [GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return sims

# ============================================================
# Dataset summary
# ============================================================

def describe_datasets_with_mw_and_logp(dataset_dict):
    stats = []
    for name, df in dataset_dict.items():
        aptamers = df['sequence'].dropna()
        mols = df['canonical_smiles'].dropna()
        types = df['type'].str.upper()

        if 'molecular_weight' not in df.columns:
            df['molecular_weight'] = compute_molecular_weights(df['canonical_smiles'])
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

# ============================================================
# Distribution plots
# ============================================================

def plot_dataset_distributions(df, figs_dir="figs", filename="dataset_distributions.png"):
    """
    Plots distributions:
      1) Aptamer length
      2) Pairwise Levenshtein distances
      3) Tanimoto similarity of molecules
      4) pKd values
    Saves figure in `figs_dir` at 1200 dpi.
    """
    os.makedirs(figs_dir, exist_ok=True)
    base_color = "#344966"   # Indigo

    sns.set_context("talk")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # --- 1. Aptamer length
    sns.histplot(df['sequence'].dropna().str.len(), bins=30, ax=axes[0],
                 color=base_color, kde=True)
    axes[0].set_title("Aptamer length", fontsize=24)
    axes[0].set_xlabel("Length", fontsize=20)
    axes[0].set_ylabel("Count", fontsize=20)

    # --- 2. Levenshtein diversity
    sequences = df['sequence'].dropna().unique().tolist()
    vals = compute_pairwise_levenshtein(sequences)
    sns.histplot(vals, bins=30, ax=axes[1], color=base_color, kde=True)
    axes[1].set_title("Levenshtein distances", fontsize=24)
    axes[1].set_xlabel("Distance", fontsize=20)
    axes[1].set_ylabel("Count", fontsize=20)

    # --- 3. Tanimoto similarity
    smiles = df['canonical_smiles'].dropna().unique()
    sims = compute_tanimoto_similarity(smiles)
    sns.histplot(sims, bins=30, ax=axes[2], color=base_color, kde=True)
    axes[2].set_title("Tanimoto similarity", fontsize=24)
    axes[2].set_xlabel("Similarity", fontsize=20)
    axes[2].set_ylabel("Count", fontsize=20)

    # --- 4. pKd values
    if df['pKd_value'].notna().sum() > 0:
        sns.histplot(df['pKd_value'].dropna(), bins=30, ax=axes[3],
                     color=base_color, kde=True)
        axes[3].set_title("pKd distribution", fontsize=24)
        axes[3].set_xlabel("pKd", fontsize=20)
        axes[3].set_ylabel("Count", fontsize=20)
    else:
        axes[3].text(0.5, 0.5, "No pKd values", ha="center", va="center", color=base_color, fontsize=16)
        axes[3].set_axis_off()

    for ax in axes:
        ax.spines['top'].set_color("#90A4AE")
        ax.spines['right'].set_color("#90A4AE")
        ax.tick_params(colors=base_color, labelsize=12)
        ax.yaxis.label.set_color(base_color)
        ax.xaxis.label.set_color(base_color)

    plt.tight_layout()
    out_path = os.path.join(figs_dir, filename)
    plt.savefig(out_path, dpi=1200)
    plt.show()

# ============================================================
# Data Splits Graph visualization
# ============================================================

import os
import matplotlib.pyplot as plt
import networkx as nx

def compare_split_strategies(
    df, splits_dict, seq_col="sequence", smi_col="canonical_smiles",
    outdir="figs", filename="split_strategies.png"
):
    """
    Compare different split strategies by plotting interaction graphs side by side.
    Nodes with multiple edges are emphasized by size and border.
    """
    # --- Colors ---
    color_train = "#4DD0E1"   # Cyan accent
    color_val   = "#344966"   # Indigo
    color_apt   = "#90A4AE"   # Gray-Blue
    color_mol   = "#FDFDFD"   # Off-white

    n_strats = len(splits_dict)
    fig, axes = plt.subplots(1, n_strats, figsize=(7 * n_strats, 6))

    if n_strats == 1:
        axes = [axes]

    # Base graph
    G_base = nx.Graph()
    apt_nodes = df[seq_col].unique()
    mol_nodes = df[smi_col].unique()
    G_base.add_nodes_from(apt_nodes, bipartite=0, kind="aptamer")
    G_base.add_nodes_from(mol_nodes, bipartite=1, kind="molecule")
    pos = nx.spring_layout(G_base, seed=42, k=0.5)

    for ax, (name, splits) in zip(axes, splits_dict.items()):
        train_idx, val_idx = splits[0]
        G = G_base.copy()

        # Edges
        edges_train = list(zip(df.iloc[train_idx][seq_col], df.iloc[train_idx][smi_col]))
        edges_val   = list(zip(df.iloc[val_idx][seq_col], df.iloc[val_idx][smi_col]))
        G.add_edges_from(edges_train, split="train")
        G.add_edges_from(edges_val, split="val")

        # --- Node degrees ---
        degrees = dict(G.degree())
        node_sizes = []
        node_lines = []
        for n in G.nodes():
            if degrees[n] > 1:
                node_sizes.append(500)   # bigger
                node_lines.append(2.5)   # thicker border
            else:
                node_sizes.append(250)
                node_lines.append(1.0)

        # Draw aptamers
        nx.draw_networkx_nodes(
            G, pos, nodelist=apt_nodes, node_shape="o",
            node_size=[node_sizes[list(G.nodes()).index(n)] for n in apt_nodes],
            node_color=color_apt, edgecolors=color_val,
            linewidths=[node_lines[list(G.nodes()).index(n)] for n in apt_nodes],
            alpha=0.9, ax=ax, label="Aptamers"
        )
        # Draw molecules
        nx.draw_networkx_nodes(
            G, pos, nodelist=mol_nodes, node_shape="s",
            node_size=[node_sizes[list(G.nodes()).index(n)] for n in mol_nodes],
            node_color=color_mol, edgecolors=color_apt,
            linewidths=[node_lines[list(G.nodes()).index(n)] for n in mol_nodes],
            alpha=0.9, ax=ax, label="Molecules"
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edgelist=edges_train, edge_color=color_train,
            width=2.5, alpha=0.8, ax=ax, label="Train"
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=edges_val, edge_color=color_val,
            width=2.5, alpha=0.8, ax=ax, label="Validation"
        )

        ax.set_title(name, fontsize=20)
        ax.axis("off")

    # Legend
    handles = [
        plt.Line2D([0], [0], color=color_train, lw=3, label="Train edges"),
        plt.Line2D([0], [0], color=color_val, lw=3, label="Validation edges"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_apt,
                   markeredgecolor=color_val, markersize=10, label="Aptamers"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color_mol,
                   markeredgecolor=color_apt, markersize=10, label="Molecules"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=16)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    plt.savefig(outpath, dpi=1200, bbox_inches="tight")
    plt.show()




# ============================================================
# t-SNE visualization
# ============================================================

def _optimize_tsne(X, labels, perplexities=(20, 30, 50), lrs=(10, 100, 500, 1000), random_state=0):
    X_scaled = StandardScaler().fit_transform(X)
    best_score, best_emb = -1, None
    for perp in perplexities:
        for lr in lrs:
            try:
                emb = TSNE(
                    n_components=2, perplexity=perp,
                    learning_rate=lr, random_state=random_state
                ).fit_transform(X_scaled)
                score = silhouette_score(emb, labels)
                if score > best_score:
                    best_score, best_emb = score, emb
            except Exception:
                continue
    return best_emb if best_emb is not None else TSNE(n_components=2, random_state=random_state).fit_transform(X_scaled)


def cluster_and_plot_tsne_from_combined(
    df, aptamer_encoder, molecule_encoder, source_col="source",
    outdir="figs", filename="tsne_sources.png"
):
    """
    Run t-SNE on aptamer and molecule embeddings from a single combined dataset
    with multiple sources.
    """

    # --- Custom palette (max 7 sources) ---
    custom_palette = [
        "#344966", "#4DD0E1", "#90A4AE",
        "#947EE2", "#3B9170", "#4F87FF", "#BA68C8",
    ]

    sources = df[source_col].dropna().unique()
    palette = {src: custom_palette[i % len(custom_palette)] for i, src in enumerate(sources)}

    # --- Aptamers ---
    aptamer_df = df.dropna(subset=["sequence"]).drop_duplicates("sequence")
    aptamers = aptamer_df["sequence"].tolist()
    apt_labels = aptamer_df[source_col].tolist()
    X_apt = aptamer_encoder(aptamers)
    emb_apt = _optimize_tsne(X_apt, apt_labels)

    apt_data = pd.DataFrame({
        "tSNE1": emb_apt[:, 0],
        "tSNE2": emb_apt[:, 1],
        "source": apt_labels
    })

    # --- Molecules ---
    molecule_df = df.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles")
    molecules = molecule_df["canonical_smiles"].tolist()
    mol_labels = molecule_df[source_col].tolist()
    X_mol = molecule_encoder(molecules)
    emb_mol = _optimize_tsne(X_mol, mol_labels)

    mol_data = pd.DataFrame({
        "tSNE1": emb_mol[:, 0],
        "tSNE2": emb_mol[:, 1],
        "source": mol_labels
    })

    # --- Figure layout ---
    sns.set(style="white", font_scale=1.3)
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(
        2, 5,  
        width_ratios=[5, 1, 0.5, 5, 1], 
        height_ratios=[1, 5],
        wspace=0.05, hspace=0.05
    )

    # Aptamers axes
    ax_kde_x1 = fig.add_subplot(gs[0, 0])
    ax_scatter1 = fig.add_subplot(gs[1, 0])
    ax_kde_y1 = fig.add_subplot(gs[1, 1], sharey=ax_scatter1)

    # Molecules axes
    ax_kde_x2 = fig.add_subplot(gs[0, 3])
    ax_scatter2 = fig.add_subplot(gs[1, 3])
    ax_kde_y2 = fig.add_subplot(gs[1, 4], sharey=ax_scatter2)

    # --- Aptamers plotting ---
    sns.scatterplot(
        data=apt_data, x="tSNE1", y="tSNE2", hue="source",
        palette=palette, s=70, alpha=0.85, ax=ax_scatter1,
        edgecolor='black', linewidth=0.4, legend=False
    )
    sns.kdeplot(data=apt_data, x="tSNE1",
                fill=True, color="#90A4AE", alpha=0.3,
                bw_adjust=0.8, ax=ax_kde_x1)
    sns.kdeplot(data=apt_data, y="tSNE2",
                fill=True, color="#90A4AE", alpha=0.3,
                bw_adjust=0.8, ax=ax_kde_y1)

    ax_kde_x1.set_ylabel("") ; ax_kde_x1.set_xlabel("")
    ax_kde_x1.set_xticks([]) ; ax_kde_x1.set_yticks([])
    ax_kde_y1.set_ylabel("") ; ax_kde_y1.set_xlabel("")
    ax_kde_y1.set_xticks([]) ; ax_kde_y1.set_yticks([])
    ax_kde_x1.set_title("Aptamers (one-hot)")

    # --- Molecules plotting ---
    sns.scatterplot(
        data=mol_data, x="tSNE1", y="tSNE2", hue="source",
        palette=palette, s=70, alpha=0.85, ax=ax_scatter2,
        edgecolor='black', linewidth=0.4, legend=False
    )
    sns.kdeplot(data=mol_data, x="tSNE1",
                fill=True, color="#90A4AE", alpha=0.3,
                bw_adjust=0.8, ax=ax_kde_x2)
    sns.kdeplot(data=mol_data, y="tSNE2",
                fill=True, color="#90A4AE", alpha=0.3,
                bw_adjust=0.8, ax=ax_kde_y2)

    ax_kde_x2.set_ylabel("") ; ax_kde_x2.set_xlabel("")
    ax_kde_x2.set_xticks([]) ; ax_kde_x2.set_yticks([])
    ax_kde_y2.set_ylabel("") ; ax_kde_y2.set_xlabel("")
    ax_kde_y2.set_xticks([]) ; ax_kde_y2.set_yticks([])
    ax_kde_x2.set_title("Molecules (Morgan FP)")

    # Legend 
    handles, labels = ax_scatter1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sources),
               frameon=False, fontsize=16)

    # Save & show
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(outpath, dpi=1200, bbox_inches="tight")
    plt.show()

# ============================================================
# Intersections between sources
# ============================================================

def compute_intersections(df, source_col="source"):
    """
    Computes intersections between subsets of a combined dataset, grouped by `source`.

    For each pair of sources, calculates:
      - common (sequence, SMILES) pairs,
      - common aptamer sequences,
      - common molecules (SMILES).

    Returns a symmetric matrix with entries formatted as "pairs/aptamers/molecules".

    Parameters
    ----------
    df : pandas.DataFrame
        Combined dataset with at least columns ['sequence', 'canonical_smiles', source_col].
    source_col : str, default="source"
        Column name defining dataset/source identity.

    Returns
    -------
    pandas.DataFrame
        Intersection matrix (string formatted).
    """
    names = df[source_col].dropna().unique()
    result_df = pd.DataFrame(index=names, columns=names)

    for name1 in names:
        df1 = df[df[source_col] == name1]
        set1_pairs = set(zip(df1['sequence'], df1['canonical_smiles']))
        set1_apt = set(df1['sequence'])
        set1_mol = set(df1['canonical_smiles'])

        for name2 in names:
            df2 = df[df[source_col] == name2]
            set2_pairs = set(zip(df2['sequence'], df2['canonical_smiles']))
            set2_apt = set(df2['sequence'])
            set2_mol = set(df2['canonical_smiles'])

            # Intersections
            common_pairs = len(set1_pairs & set2_pairs)
            common_apt = len(set1_apt & set2_apt)
            common_mol = len(set1_mol & set2_mol)

            result_df.loc[name1, name2] = f"{common_pairs}/{common_apt}/{common_mol}"

    return result_df
