"""
Processing statistics for the binarise_and_merge step (02), using only the 7
primary-screen assays — one per target pathogen.

Inputs:
    data/processed/02_binarised_assays/*.csv     (smiles, bin) — one per assay
    data/config/primary_assays_manual.csv     selected primary-screen assay per pathogen

Outputs:
    output/03_processing_statistics/
        03_active_counts.png        compounds tested vs active per pathogen (log x-axis)
        03_coactive_counts.png      pairwise co-active compound counts (7×7 heatmap)
        03_breadth_exclusivity.png  spectrum breadth + binary exclusivity (2 panels)

Scientific decisions (human-confirmed):
    Inconclusive (bin = -1): excluded from all overlap calculations.
    Broad spectrum threshold: ≥3 pathogens active.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stylia
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

data_dir = os.path.join(root, "..", "data", "processed")
assays_dir = os.path.join(data_dir, "02_binarised_assays")
config_dir = os.path.join(root, "..", "data", "config")
plot_dir = os.path.join(root, "..", "output", "03_processing_statistics")
os.makedirs(plot_dir, exist_ok=True)

BROAD_SPECTRUM_THRESHOLD = 3  # ≥3 pathogens active = broad spectrum (human-confirmed)

PATHOGEN_CODES = [
    "abaumannii", "calbicans", "ecoli", "efaecalis",
    "kpneumoniae", "paeruginosa", "saureus",
]

PATHOGEN_LABELS = {
    "abaumannii": "A. baumannii",
    "calbicans": "C. albicans",
    "ecoli": "E. coli",
    "efaecalis": "E. faecalis",
    "kpneumoniae": "K. pneumoniae",
    "paeruginosa": "P. aeruginosa",
    "saureus": "S. aureus",
}

# ArticleColors: crimson, turquoise, cobalt, periwinkle, orchid, fuchsia,
#                tangerine, amber, lime, silver, black, white
_PATHOGEN_COLOR_NAMES = {
    "abaumannii": "crimson",
    "calbicans": "turquoise",
    "ecoli": "cobalt",
    "efaecalis": "tangerine",
    "kpneumoniae": "amber",
    "paeruginosa": "orchid",
    "saureus": "periwinkle",
}

# Format: print | Style: article — change with stylia.set_format() / stylia.set_style()
stylia.set_format("print")
stylia.set_style("article")


def _pathogen_color(code):
    nc = stylia.NamedColors()
    return getattr(nc, _PATHOGEN_COLOR_NAMES.get(code, "silver"))




# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_primary_stats():
    """Load 7 primary-screen assay stats from primary_assays_manual.csv."""
    path = os.path.join(config_dir, "primary_assays_manual.csv")
    df = pd.read_csv(path)
    cat_order = {code: i for i, code in enumerate(PATHOGEN_CODES)}
    df["_order"] = df["pathogen_code"].map(cat_order)
    return df.sort_values("_order").drop("_order", axis=1).reset_index(drop=True)


def load_assay_active_sets(stats):
    """Return {pathogen_code: set of active SMILES} — inconclusive excluded."""
    active_sets = {}
    for _, row in stats.iterrows():
        fpath = os.path.join(assays_dir, f"{row['assay_eos_id']}.csv")
        df = pd.read_csv(fpath)
        active_sets[row["pathogen_code"]] = set(df.loc[df["bin"] == 1, "smiles"].tolist())
    return active_sets


def compute_coactive_matrix(active_sets, stats):
    """
    Pairwise co-active compound counts indexed by pathogen name.
    Diagonal = n_active per pathogen (full self-overlap), so it maps correctly
    to the colormap (capped at 100).
    Returns (coact_df, n_active_list, pathogen_codes).
    """
    codes = stats["pathogen_code"].tolist()
    n = len(codes)
    coact_vals = np.zeros((n, n), dtype=float)

    for i in range(n):
        coact_vals[i, i] = float(len(active_sets[codes[i]]))
        for j in range(i + 1, n):
            shared = len(active_sets[codes[i]] & active_sets[codes[j]])
            coact_vals[i, j] = coact_vals[j, i] = float(shared)

    labels = [PATHOGEN_LABELS[c] for c in codes]
    df = pd.DataFrame(coact_vals, index=labels, columns=labels)
    return df, stats["n_active"].tolist(), codes


def load_activity_matrix(stats):
    """Compound × pathogen binary matrix from the 7 primary-screen binarised files."""
    series = []
    for _, row in stats.iterrows():
        fpath = os.path.join(assays_dir, f"{row['assay_eos_id']}.csv")
        df = pd.read_csv(fpath)
        df = df[df["bin"] != -1]
        s = df.set_index("smiles")["bin"].rename(row["pathogen_code"])
        s = s.groupby(level=0).max()  # dedup SMILES: keep highest bin
        series.append(s)
    matrix = pd.concat(series, axis=1, join="outer")
    return matrix.replace(-1, np.nan)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plot 1 — Active counts (log x-axis)
# ---------------------------------------------------------------------------


def plot_active_counts(ax, stats):
    """Horizontal bars: faded = n_tested, solid = n_active. Log x-axis."""
    n = len(stats)
    for i, (_, row) in enumerate(stats.iterrows()):
        code = row["pathogen_code"]
        color = _pathogen_color(code)
        ax.barh(i, row["n_tested"], color=color, alpha=0.2)
        if row["n_active"] > 0:
            ax.barh(i, row["n_active"], color=color)

    ax.set_xscale("log")
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [PATHOGEN_LABELS[row["pathogen_code"]] for _, row in stats.iterrows()],
        fontsize=stylia.FONTSIZE_SMALL,
    )
    ax.set_ylim(-0.5, n - 0.5)

    x_max = int(stats["n_tested"].max())
    ax.set_xlim(1, x_max * 6)
    for i, (_, row) in enumerate(stats.iterrows()):
        ax.text(
            x_max * 1.5, i,
            f"{row['n_active']:,}  ({row['active_rate'] * 100:.3f}%)",
            va="center", fontsize=stylia.FONTSIZE_SMALL,
        )

    stylia.label(
        ax,
        xlabel="Compounds (faded = n tested, solid = n active, log scale)",
        ylabel="",
        title="Active compounds per primary-screen assay",
    )


# ---------------------------------------------------------------------------
# Plot 2 — Co-active counts heatmap
# ---------------------------------------------------------------------------


def plot_coactive_counts(ax, coact_df, n_active_list, pathogen_codes):
    """
    7×7 heatmap. Color scale: coolwarm capped at [0, 100].
    Diagonal: neutral gray, annotated with n_active per pathogen.
    """
    vals = coact_df.values.astype(float)
    n = len(vals)
    labels = coact_df.columns.tolist()

    cmap = plt.cm.coolwarm
    vals_norm = np.clip(vals, 0, 100) / 100.0
    colors = cmap(vals_norm)  # shape (n, n, 4)

    ax.imshow(colors)

    for i in range(n):
        for j in range(n):
            lum = (0.2126 * colors[i, j, 0]
                   + 0.7152 * colors[i, j, 1]
                   + 0.0722 * colors[i, j, 2])
            txt_color = "white" if lum < 0.5 else "black"
            if i == j:
                ax.text(j, i, f"{int(n_active_list[i]):,}",
                        ha="center", va="center",
                        fontsize=stylia.FONTSIZE_SMALL, color=txt_color)
            else:
                ax.text(j, i, str(int(vals[i, j])),
                        ha="center", va="center",
                        fontsize=stylia.FONTSIZE_SMALL, color=txt_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=stylia.FONTSIZE_SMALL)
    ax.set_yticklabels(labels, fontsize=stylia.FONTSIZE_SMALL)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=stylia.FONTSIZE_SMALL)
    cbar.set_label("Co-active compounds (capped at 100)", fontsize=stylia.FONTSIZE_SMALL)

    stylia.label(ax, xlabel="", ylabel="", title="Pairwise co-active compound counts")


# ---------------------------------------------------------------------------
# Plot 3 — Breadth distribution + binary exclusivity (2 panels)
# ---------------------------------------------------------------------------


def _plot_breadth_distribution(ax, matrix):
    nc = stylia.NamedColors()
    active_counts_per_compound = (matrix == 1).sum(axis=1)
    active_counts_per_compound = active_counts_per_compound[active_counts_per_compound > 0]

    dist = active_counts_per_compound.value_counts().sort_index()
    x = dist.index.tolist()
    y = dist.values

    color_exclusive = nc.cobalt
    color_nonexclusive = nc.crimson

    bar_colors = [color_exclusive if xi == 1 else color_nonexclusive for xi in x]
    ax.bar(x, y, color=bar_colors)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_ylim(0.5, float(y.max()) * 4)
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:,}", ha="center", va="bottom",
                fontsize=stylia.FONTSIZE_SMALL)

    n_excl = int((active_counts_per_compound == 1).sum())
    n_nonexcl = int((active_counts_per_compound >= 2).sum())

    ax.legend(
        handles=[
            Patch(color=color_exclusive,    label=f"Exclusive (1): {n_excl:,}"),
            Patch(color=color_nonexclusive, label=f"Non-exclusive (≥2): {n_nonexcl:,}"),
        ],
        fontsize=stylia.FONTSIZE_SMALL,
        loc="upper right",
    )
    stylia.label(
        ax,
        xlabel="Active in N pathogens",
        ylabel="Compounds (log scale)",
        title="Spectrum breadth",
    )


def _plot_binary_exclusivity(ax, matrix):
    nc = stylia.NamedColors()
    active_counts_per_compound = (matrix == 1).sum(axis=1)

    excl_vals, shared_vals, labels = [], [], []
    for code in PATHOGEN_CODES:
        actives_idx = matrix.index[matrix[code] == 1]
        counts = active_counts_per_compound[actives_idx]
        excl_vals.append(int((counts == 1).sum()))
        shared_vals.append(int((counts >= 2).sum()))
        labels.append(PATHOGEN_LABELS[code])

    y = list(range(len(PATHOGEN_CODES)))
    for i, (exc, shr, code) in enumerate(zip(excl_vals, shared_vals, PATHOGEN_CODES)):
        ax.barh(i, exc, color=_pathogen_color(code),
                label="_nolegend_")
        ax.barh(i, shr, left=exc, color=nc.silver,
                label="_nolegend_")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=stylia.FONTSIZE_SMALL)
    ax.legend(
        handles=[
            Patch(color="black", label="Exclusive"),
            Patch(color=nc.silver, label="Non-exclusive (≥2 pathogens)"),
        ],
        fontsize=stylia.FONTSIZE_SMALL,
        loc="lower right",
    )
    stylia.label(ax, xlabel="Active compounds", ylabel="", title="Exclusivity")


def plot_breadth_and_exclusivity(matrix):
    """Two-panel figure: spectrum breadth (left) + binary exclusivity (right)."""
    fig, axs = stylia.create_figure(1, 2, height=0.3)
    _plot_breadth_distribution(axs.next(), matrix)
    _plot_binary_exclusivity(axs.next(), matrix)
    stylia.save_figure(os.path.join(plot_dir, "03_breadth_exclusivity.png"))
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    for f in glob.glob(os.path.join(plot_dir, "*.png")):
        os.remove(f)

    print("Loading data...")
    stats = load_primary_stats()
    active_sets = load_assay_active_sets(stats)
    coact_df, n_active_list, pathogen_codes = compute_coactive_matrix(active_sets, stats)
    matrix = load_activity_matrix(stats)
    print(f"  {len(stats)} assays | compound matrix: {matrix.shape}")

    fig, axs = stylia.create_figure(1, 1, height=0.3)
    plot_active_counts(axs.next(), stats)
    stylia.save_figure(os.path.join(plot_dir, "03_active_counts.png"))
    plt.close()
    print("  03_active_counts.png")

    fig, axs = stylia.create_figure(1, 1, width=0.7, height=0.7)
    plot_coactive_counts(axs.next(), coact_df, n_active_list, pathogen_codes)
    stylia.save_figure(os.path.join(plot_dir, "03_coactive_counts.png"))
    plt.close()
    print("  03_coactive_counts.png")

    plot_breadth_and_exclusivity(matrix)
    print("  03_breadth_exclusivity.png")

    print(f"\nAll plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
