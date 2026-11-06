"""Rebuild ML encoder-pair ROC-AUC heatmaps in the original AptaBench style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENING_DIR = REPO_ROOT / "artifacts" / "ml" / "classification" / "screening"
OUT_DIR = REPO_ROOT / "artifacts" / "figures" / "ml"

SPLIT_FILES = {
    "molecule_disjoint": (
        SCREENING_DIR / "disjoint_molecule" / "results.csv",
        "Molecule-disjoint",
    ),
    "aptamer_disjoint": (
        SCREENING_DIR / "disjoint_aptamer" / "results.csv",
        "Aptamer-disjoint",
    ),
}

APT_ORDER = ["DNABERT2", "GENA", "Kmer3", "Kmer4", "OneHot"]
MOL_ORDER = [
    "CLAMP",
    "ChemBERTa",
    "MACCS",
    "MolFormer",
    "MorganFP",
    "PhysChem",
    "UniMol",
]

VMIN, VMAX = 0.80, 0.90

# Teal sequential palette matched to the original figures:
# dark navy-teal = lower ROC-AUC, light cyan/white = higher.
_TEAL_COLORS = [
    "#0b2a33",
    "#1f5a63",
    "#3a8a8a",
    "#7ebbb8",
    "#c7e6e4",
    "#f4fbfb",
]
CMAP = LinearSegmentedColormap.from_list("apta_teal", _TEAL_COLORS, N=256)


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index="aptamer_encoder",
            columns="molecule_encoder",
            values="ROC-AUC mean",
            aggfunc="mean",
        )
        .reindex(index=APT_ORDER, columns=MOL_ORDER)
    )
    return pivot


def _text_color(value: float) -> str:
    clipped = float(np.clip(value, VMIN, VMAX))
    rgba = CMAP(Normalize(vmin=VMIN, vmax=VMAX)(clipped))
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < 0.48 else "#102028"


def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    sns.set_theme(style="white", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(12.5, 9.5))

    annot = pivot.map(lambda v: f"{v:.3f}")
    heatmap = sns.heatmap(
        pivot,
        ax=ax,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        annot=annot,
        fmt="",
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=0.0,
        linecolor="white",
        cbar_kws={
            "shrink": 0.85,
            "ticks": np.linspace(VMIN, VMAX, 6),
        },
        square=False,
    )

    # Contrast-aware annotation colors + soft outline for readability.
    for text in ax.texts:
        try:
            value = float(text.get_text())
        except ValueError:
            continue
        color = _text_color(value)
        stroke = "black" if color == "white" else "white"
        text.set_color(color)
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.4, foreground=stroke, alpha=0.35),
                path_effects.Normal(),
            ]
        )

    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(axis="both", length=0, labelsize=13)

    # Outline the best encoder combination for this protocol.
    flat_idx = int(np.nanargmax(pivot.to_numpy()))
    best_i, best_j = np.unravel_index(flat_idx, pivot.shape)
    ax.add_patch(
        Rectangle(
            (best_j, best_i),
            1.0,
            1.0,
            fill=False,
            edgecolor="white",
            linewidth=4.5,
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle(
            (best_j, best_i),
            1.0,
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=2.2,
            zorder=4,
        )
    )

    # Keep the original colorbar look: numeric ticks only, no axis label.
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    sns.reset_orig()


def main() -> None:
    for split_key, (csv_path, title) in SPLIT_FILES.items():
        df = pd.read_csv(csv_path)
        pivot = build_pivot(df)
        out_name = f"heatmap_{split_key}_big_rev.png"
        out_path = OUT_DIR / out_name
        plot_heatmap(pivot, title, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
