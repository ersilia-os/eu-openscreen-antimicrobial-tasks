"""
Antibiotic similarity enrichment analysis using Ersilia models eos6ojg and eos11sm.

eos6ojg counts known antibiotics with Tanimoto similarity ≥ threshold to each query compound.
eos11sm produces a single continuous abx_score ∈ [0, 1].

Both are tested for enrichment: whether active compounds against each pathogen score higher
than inactive compounds, using AUROC as the comparison metric.

Analyses (three dataset views per model):
    merged  — 7 per-pathogen files combining all assays for that pathogen
    primary — 7 primary-screen assays only, one per pathogen
    assay   — all 41 individual binarised assay files

Inputs:
    output/04_ersilia_models/eos6ojg.csv
    output/04_ersilia_models/eos11sm.csv
    output/04_ersilia_models/eos2xeq.csv
    data/processed/02_merged/02_{pathogen}.csv          ×7
    data/processed/02_binarised_assays/*.csv             ×41
    data/config/primary_assays_manual.csv
    output/00_extract_assays/00_assay_summary.csv
    output/01_fetch_new_assays/01_assay_summary.csv

Outputs:
    output/05_abx_similarity/
        05_eos6ojg_merged_all.png     AUROC vs threshold, all-antibiotics reference (merged)
        05_eos6ojg_primary_all.png    AUROC vs threshold, all-antibiotics reference (primary)
        05_eos6ojg_assay_all.png      AUROC at 0.5, all-antibiotics reference (41 assays)
        05_eos6ojg_merged_subset.png  AUROC vs threshold, antibiotic subset (merged)
        05_eos6ojg_primary_subset.png AUROC vs threshold, antibiotic subset (primary)
        05_eos6ojg_assay_subset.png   AUROC at 0.5, antibiotic subset (41 assays)
        05_eos11sm_merged.png         AUROC per pathogen, abx_score (merged)
        05_eos11sm_primary.png        AUROC per pathogen, abx_score (primary)
        05_eos11sm_assay.png          AUROC per assay, abx_score (41 assays)
        05_eos2xeq_merged.png         AUROC per pathogen, is_sim_known_ab (merged)
        05_eos2xeq_primary.png        AUROC per pathogen, is_sim_known_ab (primary)
        05_eos2xeq_assay.png          AUROC per assay, is_sim_known_ab (41 assays)

Scientific decisions (human-confirmed):
    Inconclusive compounds (bin = -1) excluded from all comparisons.
    Minimum 5 actives required for a reliable AUROC; otherwise NaN (shown as gray bar).

Models:
    eos6ojg — https://github.com/ersilia-os/eos6ojg
    eos11sm — https://github.com/ersilia-os/eos11sm
    eos2xeq — https://github.com/ersilia-os/eos2xeq
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import stylia
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

data_dir = os.path.join(root, "..", "data", "processed")
merged_dir = os.path.join(data_dir, "02_merged")
assays_dir = os.path.join(data_dir, "02_binarised_assays")
config_dir = os.path.join(root, "..", "data", "config")
model_dir = os.path.join(root, "..", "output", "04_ersilia_models")
plot_dir = os.path.join(root, "..", "output", "05_abx_similarity")
os.makedirs(plot_dir, exist_ok=True)

THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 1.0]
SIM_COLS_ALL = [
    "num_sim_0_3_all", "num_sim_0_5_all", "num_sim_0_7_all",
    "num_sim_0_9_all", "num_sim_1_0_all",
]
SIM_COLS_SUBSET = [
    "num_sim_0_3_subset", "num_sim_0_5_subset", "num_sim_0_7_subset",
    "num_sim_0_9_subset", "num_sim_1_0_subset",
]
EOS6OJG_ANALYSES = [
    ("all",    SIM_COLS_ALL,    "num_sim_0_5_all",    "all antibiotics"),
    ("subset", SIM_COLS_SUBSET, "num_sim_0_5_subset", "antibiotic subset"),
]
EOS11SM_COL = "abx_score"
EOS2XEQ_COL = "is_sim_known_ab"
MIN_ACTIVES = 5

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

_PATHOGEN_COLOR_NAMES = {
    "abaumannii": "crimson",
    "calbicans": "turquoise",
    "ecoli": "cobalt",
    "efaecalis": "tangerine",
    "kpneumoniae": "amber",
    "paeruginosa": "orchid",
    "saureus": "periwinkle",
}

# Format: print | Style: article
stylia.set_format("print")
stylia.set_style("article")


def _pathogen_color(code):
    nc = stylia.NamedColors()
    return getattr(nc, _PATHOGEN_COLOR_NAMES.get(code, "silver"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_model_scores(model_id, score_cols):
    """Load Ersilia model output; strip 'B.' prefix; return SMILES-indexed DataFrame."""
    path = os.path.join(model_dir, f"{model_id}.csv")
    df = pd.read_csv(path)
    df["smiles"] = df["input"].str.replace(r"^B\.", "", regex=True)
    return df.set_index("smiles")[score_cols]


def merge_with_sim(pathogen_code, sim_df):
    """Read merged per-pathogen file, left-join sim scores, filter out bin=-1."""
    df = pd.read_csv(os.path.join(merged_dir, f"02_{pathogen_code}.csv"))
    df = df.merge(sim_df, left_on="smiles", right_index=True, how="left")
    return df[df["bin"] != -1]


def load_primary_assay_map():
    """Return {pathogen_code: assay_eos_id} for the 7 primary-screen assays."""
    df = pd.read_csv(os.path.join(config_dir, "primary_assays_manual.csv"))
    return dict(zip(df["pathogen_code"], df["assay_eos_id"]))


def merge_primary_with_sim(assay_eos_id, sim_df):
    """Read a single binarised assay file, left-join sim scores, filter out bin=-1."""
    df = pd.read_csv(os.path.join(assays_dir, f"{assay_eos_id}.csv"))
    df = df.merge(sim_df, left_on="smiles", right_index=True, how="left")
    return df[df["bin"] != -1]


def load_assay_pathogen_mapping():
    """Return {assay_eos_id: pathogen_code} from both assay summary CSVs."""
    dfs = []
    for subdir, fname in (
        ("00_extract_assays", "00_assay_summary.csv"),
        ("01_fetch_new_assays", "01_assay_summary.csv"),
    ):
        path = os.path.join(root, "..", "output", subdir, fname)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, usecols=["pathogen_code", "assay_eos_id"]))
    mapping_df = pd.concat(dfs, ignore_index=True).drop_duplicates("assay_eos_id")
    return dict(zip(mapping_df["assay_eos_id"], mapping_df["pathogen_code"]))


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


def compute_auroc(scores, labels):
    """
    AUROC for predicting activity (bin=1 vs bin=0) from similarity scores.
    Inconclusive labels (bin=-1) and rows with missing scores are excluded.
    Returns np.nan when fewer than MIN_ACTIVES positives or only one class present.
    """
    df = pd.DataFrame({"s": scores, "y": labels})
    df = df[(df["y"] != -1) & df["s"].notna()]
    if int((df["y"] == 1).sum()) < MIN_ACTIVES or df["y"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(df["y"], df["s"]))


# ---------------------------------------------------------------------------
# Plot 1 — eos6ojg: merged AUROC vs threshold (line plot)
# ---------------------------------------------------------------------------


def plot_merged_auroc_lines(ax, merged_aurocs, model_id, col_label, dataset="merged"):
    """
    Line plot: x = Tanimoto threshold, y = AUROC.
    One line per pathogen. Dashed reference at 0.5.
    merged_aurocs: {pathogen_code: [auroc_0.3, ..., auroc_1.0]}
    """
    nc = stylia.NamedColors()

    for code in PATHOGEN_CODES:
        aurocs = merged_aurocs.get(code, [np.nan] * len(THRESHOLDS))
        valid = [(t, a) for t, a in zip(THRESHOLDS, aurocs) if not np.isnan(a)]
        if valid:
            ts, au = zip(*valid)
            ax.plot(ts, au, color=_pathogen_color(code),
                    label=PATHOGEN_LABELS[code], marker="o")

    ax.axhline(0.5, color=nc.silver, linestyle="--")
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels([str(t) for t in THRESHOLDS], fontsize=stylia.FONTSIZE_SMALL)

    all_vals = [a for aurocs in merged_aurocs.values() for a in aurocs if not np.isnan(a)]
    if all_vals:
        y_min = max(0.4, min(all_vals) - 0.05)
        y_max = min(1.0, max(all_vals) + 0.05)
        ax.set_ylim(y_min, y_max)

    ax.legend(
        handles=[Patch(color=_pathogen_color(c), label=PATHOGEN_LABELS[c])
                 for c in PATHOGEN_CODES],
        fontsize=stylia.FONTSIZE_SMALL,
        loc="lower left",
    )
    stylia.label(
        ax,
        xlabel="Tanimoto threshold",
        ylabel="AUROC",
        title=f"Antibiotic similarity enrichment — {col_label} [{model_id}, {dataset}]",
    )


# ---------------------------------------------------------------------------
# Plot 2 — eos11sm/eos2xeq: AUROC per pathogen (horizontal bar chart)
# ---------------------------------------------------------------------------


def plot_merged_auroc_bars(ax, aurocs_dict, model_id, dataset="merged"):
    """
    Horizontal bar chart: one bar per pathogen, coloured by pathogen.
    Reference line at 0.5.
    aurocs_dict: {pathogen_code: auroc}
    """
    nc = stylia.NamedColors()
    n = len(PATHOGEN_CODES)

    for i, code in enumerate(PATHOGEN_CODES):
        auroc = aurocs_dict.get(code, np.nan)
        if np.isnan(auroc):
            ax.barh(i, 0.5, color=nc.silver, alpha=0.3)
        else:
            ax.barh(i, auroc, color=_pathogen_color(code))

    ax.axvline(0.5, color=nc.silver, linestyle="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [PATHOGEN_LABELS[c] for c in PATHOGEN_CODES],
        fontsize=stylia.FONTSIZE_SMALL,
    )
    ax.set_ylim(-0.5, n - 0.5)

    stylia.label(
        ax,
        xlabel="AUROC",
        ylabel="",
        title=f"Antibiotic similarity enrichment per pathogen [{model_id}, {dataset}]",
    )


# ---------------------------------------------------------------------------
# Plot 3 — per-assay AUROC bar chart (both models)
# ---------------------------------------------------------------------------


def plot_assay_auroc(ax, assay_records, model_id, score_label):
    """
    Vertical bar chart: one bar per assay, coloured by pathogen.
    assay_records: sorted list of (pathogen_code, assay_eos_id, n_active, auroc).
    NaN-AUROC bars shown in light gray at height 0.5 (insufficient data).
    Pathogen name labels centred over each group; thin separators between groups.
    """
    nc = stylia.NamedColors()

    for i, (code, _, _, auroc) in enumerate(assay_records):
        if np.isnan(auroc):
            ax.bar(i, 0.5, color=nc.silver, alpha=0.3)
        else:
            ax.bar(i, auroc, color=_pathogen_color(code))

    ax.axhline(0.5, color=nc.silver, linestyle="--")

    group_positions = {}
    for i, (code, _, _, _) in enumerate(assay_records):
        group_positions.setdefault(code, []).append(i)

    ax.set_xticks([])

    y_bottom = ax.get_ylim()[0] if ax.get_ylim()[0] < 0.4 else 0.4
    for code in PATHOGEN_CODES:
        positions = group_positions.get(code, [])
        if not positions:
            continue
        mid = np.mean(positions)
        ax.text(
            mid, y_bottom - 0.015,
            PATHOGEN_LABELS[code],
            ha="center", va="top",
            fontsize=stylia.FONTSIZE_SMALL,
            color=_pathogen_color(code),
            transform=ax.transData,
        )
        if positions[0] > 0:
            ax.axvline(positions[0] - 0.5, color=nc.silver, linewidth=0.5, linestyle=":")

    n_nan = sum(1 for r in assay_records if np.isnan(r[3]))
    ax.legend(
        handles=[
            Patch(color=_pathogen_color(c), label=PATHOGEN_LABELS[c])
            for c in PATHOGEN_CODES
        ] + ([Patch(color=nc.silver, alpha=0.3,
                    label=f"Insufficient data (<{MIN_ACTIVES} actives): {n_nan}")]
              if n_nan > 0 else []),
        fontsize=stylia.FONTSIZE_SMALL,
        loc="upper right",
    )
    stylia.label(
        ax,
        xlabel="",
        ylabel=f"AUROC ({score_label})",
        title=f"Antibiotic similarity enrichment per assay [{model_id}]",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_assay_records(sim_df, mapping, sim_col):
    """Compute per-assay AUROC for one similarity column and return sorted records."""
    records = []
    for fpath in sorted(Path(assays_dir).glob("*.csv")):
        assay_id = fpath.stem
        df = pd.read_csv(fpath).merge(sim_df, left_on="smiles", right_index=True, how="left")
        df = df[df["bin"] != -1]
        n_active = int((df["bin"] == 1).sum())
        auroc = compute_auroc(df[sim_col], df["bin"])
        code = mapping.get(assay_id, "unknown")
        records.append((code, assay_id, n_active, auroc))
    records.sort(key=lambda r: (
        PATHOGEN_CODES.index(r[0]) if r[0] in PATHOGEN_CODES else 99,
        -(r[3] if not np.isnan(r[3]) else 0.0),
    ))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    for f in glob.glob(os.path.join(plot_dir, "05_*.png")):
        os.remove(f)

    mapping = load_assay_pathogen_mapping()
    primary_map = load_primary_assay_map()

    # ─── eos6ojg ────────────────────────────────────────────────────────────
    print("Loading eos6ojg scores...")
    sim6 = load_model_scores("eos6ojg", SIM_COLS_ALL + SIM_COLS_SUBSET)
    print(f"  {len(sim6)} compounds")

    merged6 = {code: merge_with_sim(code, sim6) for code in PATHOGEN_CODES}
    primary6 = {code: merge_primary_with_sim(primary_map[code], sim6) for code in PATHOGEN_CODES}

    for tag, cols, assay_col, col_label in EOS6OJG_ANALYSES:
        print(f"\neos6ojg — {col_label}")

        merged_aurocs = {}
        primary_aurocs = {}
        for code in PATHOGEN_CODES:
            merged_aurocs[code] = [compute_auroc(merged6[code][col], merged6[code]["bin"])
                                   for col in cols]
            primary_aurocs[code] = [compute_auroc(primary6[code][col], primary6[code]["bin"])
                                    for col in cols]
            a05 = merged_aurocs[code][1]
            a05p = primary_aurocs[code][1]
            print(f"  {PATHOGEN_LABELS[code]}: merged={a05:.3f}  primary={a05p:.3f}"
                  if not (np.isnan(a05) or np.isnan(a05p))
                  else f"  {PATHOGEN_LABELS[code]}: merged={a05}  primary={a05p}")

        assay_records = build_assay_records(sim6, mapping, assay_col)
        n_nan = sum(1 for r in assay_records if np.isnan(r[3]))
        print(f"  {len(assay_records)} assays | {n_nan} with insufficient actives (NaN)")

        fig, axs = stylia.create_figure(1, 1)
        plot_merged_auroc_lines(axs.next(), merged_aurocs, "eos6ojg", col_label,
                                dataset="merged")
        stylia.save_figure(os.path.join(plot_dir, f"05_eos6ojg_merged_{tag}.png"))
        print(f"  05_eos6ojg_merged_{tag}.png")

        fig, axs = stylia.create_figure(1, 1)
        plot_merged_auroc_lines(axs.next(), primary_aurocs, "eos6ojg", col_label,
                                dataset="primary")
        stylia.save_figure(os.path.join(plot_dir, f"05_eos6ojg_primary_{tag}.png"))
        print(f"  05_eos6ojg_primary_{tag}.png")

        fig, axs = stylia.create_figure(1, 1)
        plot_assay_auroc(axs.next(), assay_records, "eos6ojg",
                         score_label=f"sim@0.5 ({col_label})")
        stylia.save_figure(os.path.join(plot_dir, f"05_eos6ojg_assay_{tag}.png"))
        print(f"  05_eos6ojg_assay_{tag}.png")

    # ─── eos11sm ────────────────────────────────────────────────────────────
    print("\nLoading eos11sm scores...")
    sim11 = load_model_scores("eos11sm", [EOS11SM_COL])
    print(f"  {len(sim11)} compounds")

    merged11 = {code: merge_with_sim(code, sim11) for code in PATHOGEN_CODES}
    primary11 = {code: merge_primary_with_sim(primary_map[code], sim11) for code in PATHOGEN_CODES}

    print("\neos11sm — abx_score")
    aurocs11_merged = {}
    aurocs11_primary = {}
    for code in PATHOGEN_CODES:
        aurocs11_merged[code] = compute_auroc(merged11[code][EOS11SM_COL], merged11[code]["bin"])
        aurocs11_primary[code] = compute_auroc(primary11[code][EOS11SM_COL], primary11[code]["bin"])
        m, p = aurocs11_merged[code], aurocs11_primary[code]
        print(f"  {PATHOGEN_LABELS[code]}: merged={m:.3f}  primary={p:.3f}"
              if not (np.isnan(m) or np.isnan(p))
              else f"  {PATHOGEN_LABELS[code]}: merged={m}  primary={p}")

    assay_records11 = build_assay_records(sim11, mapping, EOS11SM_COL)
    n_nan11 = sum(1 for r in assay_records11 if np.isnan(r[3]))
    print(f"  {len(assay_records11)} assays | {n_nan11} with insufficient actives (NaN)")

    fig, axs = stylia.create_figure(1, 1, height=0.3)
    plot_merged_auroc_bars(axs.next(), aurocs11_merged, "eos11sm", dataset="merged")
    stylia.save_figure(os.path.join(plot_dir, "05_eos11sm_merged.png"))
    print("  05_eos11sm_merged.png")

    fig, axs = stylia.create_figure(1, 1, height=0.3)
    plot_merged_auroc_bars(axs.next(), aurocs11_primary, "eos11sm", dataset="primary")
    stylia.save_figure(os.path.join(plot_dir, "05_eos11sm_primary.png"))
    print("  05_eos11sm_primary.png")

    fig, axs = stylia.create_figure(1, 1)
    plot_assay_auroc(axs.next(), assay_records11, "eos11sm", score_label="abx_score")
    stylia.save_figure(os.path.join(plot_dir, "05_eos11sm_assay.png"))
    print("  05_eos11sm_assay.png")

    # ─── eos2xeq ────────────────────────────────────────────────────────────
    print("\nLoading eos2xeq scores...")
    sim2xeq = load_model_scores("eos2xeq", [EOS2XEQ_COL])
    print(f"  {len(sim2xeq)} compounds")

    merged2xeq = {code: merge_with_sim(code, sim2xeq) for code in PATHOGEN_CODES}
    primary2xeq = {code: merge_primary_with_sim(primary_map[code], sim2xeq) for code in PATHOGEN_CODES}

    print("\neos2xeq — is_sim_known_ab")
    aurocs2xeq_merged = {}
    aurocs2xeq_primary = {}
    for code in PATHOGEN_CODES:
        aurocs2xeq_merged[code] = compute_auroc(merged2xeq[code][EOS2XEQ_COL], merged2xeq[code]["bin"])
        aurocs2xeq_primary[code] = compute_auroc(primary2xeq[code][EOS2XEQ_COL], primary2xeq[code]["bin"])
        m, p = aurocs2xeq_merged[code], aurocs2xeq_primary[code]
        print(f"  {PATHOGEN_LABELS[code]}: merged={m:.3f}  primary={p:.3f}"
              if not (np.isnan(m) or np.isnan(p))
              else f"  {PATHOGEN_LABELS[code]}: merged={m}  primary={p}")

    assay_records2xeq = build_assay_records(sim2xeq, mapping, EOS2XEQ_COL)
    n_nan2xeq = sum(1 for r in assay_records2xeq if np.isnan(r[3]))
    print(f"  {len(assay_records2xeq)} assays | {n_nan2xeq} with insufficient actives (NaN)")

    fig, axs = stylia.create_figure(1, 1, height=0.3)
    plot_merged_auroc_bars(axs.next(), aurocs2xeq_merged, "eos2xeq", dataset="merged")
    stylia.save_figure(os.path.join(plot_dir, "05_eos2xeq_merged.png"))
    print("  05_eos2xeq_merged.png")

    fig, axs = stylia.create_figure(1, 1, height=0.3)
    plot_merged_auroc_bars(axs.next(), aurocs2xeq_primary, "eos2xeq", dataset="primary")
    stylia.save_figure(os.path.join(plot_dir, "05_eos2xeq_primary.png"))
    print("  05_eos2xeq_primary.png")

    fig, axs = stylia.create_figure(1, 1)
    plot_assay_auroc(axs.next(), assay_records2xeq, "eos2xeq",
                     score_label="is_sim_known_ab")
    stylia.save_figure(os.path.join(plot_dir, "05_eos2xeq_assay.png"))
    print("  05_eos2xeq_assay.png")

    print(f"\nAll plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
