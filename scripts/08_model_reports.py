"""
Aggregate the cross-validation reports from step 07b and render the evaluation figures.

Mirrors the reporting step of ersilia-os/chembl-antimicrobial-models (scripts/10a/10b) but
reports metrics only: no model is filtered out and no composite quality score is computed,
so nothing here encodes a retention threshold.

Outputs, all under output/08_model_reports/:

  08_model_reports.csv
      One row per task (14). Mean and standard deviation across folds for AUROC, AUPRC and
      BEDROC together with their random baselines, the pooled out-of-fold AUROC, the model's
      decision_cutoff_rank, which descriptors LazyQSAR's portfolio kept, and the mean
      out-of-fold AUC of each descriptor.

  08_report_{pathogen}.png   (one per pathogen)
      A — Out-of-fold ROC curves for the primary and secondary tasks. The bold curve pools
          all folds; the faint curves behind it are the individual folds.
      B — Out-of-fold rank-score distributions for actives vs inactives, as a boxplot over
          every point plus a subsampled jittered scatter, with the model's
          decision_cutoff_rank drawn as a dotted line.

  08_roc_primary_secondary.png
      Two panels, primary and secondary, each overlaying the pooled out-of-fold ROC curves
      of all seven pathogens in the per-pathogen colours used elsewhere in this repository.

Pooled vs mean AUROC: the bold ROC curve concatenates the predictions of all folds, so its
AUC (auroc_pooled) is not identical to the mean of the per-fold AUROCs (auroc_mean). The mean
is the headline number, since that is what chembl-antimicrobial-models reports; the pooled
value is given because it is the one the plotted curve actually describes.

Scatter subsampling: panel B caps the plotted points at SCATTER_MAX_POINTS per group with a
RANDOM_SEED-fixed draw, because the primary screens pool ~500K inactive points across folds
and plotting them all is unreadable. The boxplots are always computed on every point.

Usage:
    python scripts/08_model_reports.py
    python scripts/08_model_reports.py --pathogens saureus ecoli
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import stylia
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

from default import (  # noqa: E402
    DESCRIPTORS,
    PATHOGEN_COLORS,
    PATHOGEN_LABELS,
    RANDOM_SEED,
    SCATTER_MAX_POINTS,
    TASKS,
)

repo_root = os.path.abspath(os.path.join(root, ".."))

train_dir = os.path.join(root, "..", "output", "07_train_models")
reports_dir = os.path.join(train_dir, "reports")
models_dir = os.path.join(train_dir, "models")
METADATA_CSV = os.path.join(train_dir, "07_tasks_metadata.csv")

output_dir = os.path.join(root, "..", "output", "08_model_reports")
os.makedirs(output_dir, exist_ok=True)

SUMMARY_CSV = os.path.join(output_dir, "08_model_reports.csv")
ROC_GRID_PNG = os.path.join(output_dir, "08_roc_primary_secondary.png")

# Format: print | Style: article — change with stylia.set_format() / stylia.set_style()
stylia.set_format("print")
stylia.set_style("article")

# Consistent across the seven per-pathogen figures so the reader learns the mapping once.
TASK_COLOR_NAMES = {"primary": "cobalt", "secondary": "tangerine"}


def pathogen_color(code: str):
    return getattr(stylia.NamedColors(), PATHOGEN_COLORS.get(code, "silver"))


def task_color(task: str):
    return getattr(stylia.NamedColors(), TASK_COLOR_NAMES.get(task, "silver"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_folds(pathogen: str, task: str) -> dict | None:
    """Return {fold: (y_true, y_rank)} plus a pooled entry, or None if absent."""
    path = os.path.join(reports_dir, pathogen, f"{task}_folds.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        folds = json.load(f)
    out = {}
    y_true_all, y_rank_all = [], []
    for fold in sorted(folds, key=int):
        yt = np.asarray(folds[fold]["y_true"], dtype=int)
        yr = np.asarray(folds[fold]["y_rank"], dtype=float)
        out[fold] = (yt, yr)
        y_true_all.append(yt)
        y_rank_all.append(yr)
    out["pooled"] = (np.concatenate(y_true_all), np.concatenate(y_rank_all))
    return out


def load_model_metadata(pathogen: str, task: str) -> dict:
    """Return the saved model's metadata.json, or {} when the model is absent."""
    path = os.path.join(models_dir, pathogen, task, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def build_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, t in tasks.iterrows():
        pathogen, task = t["pathogen"], t["task"]
        report_path = os.path.join(reports_dir, pathogen, f"{task}.csv")
        if not os.path.exists(report_path):
            print(f"[WARN] {pathogen}/{task}: no report — skipping")
            continue
        rep = pd.read_csv(report_path)
        folds = load_folds(pathogen, task)
        meta = load_model_metadata(pathogen, task)

        auroc_pooled = np.nan
        if folds is not None:
            yt, yr = folds["pooled"]
            if len(set(yt.tolist())) == 2:
                auroc_pooled = roc_auc_score(yt, yr)

        records.append(
            {
                "pathogen": pathogen,
                "pathogen_label": t["pathogen_label"],
                "task": task,
                "n_folds": len(rep),
                "compounds": t["compounds"],
                "positives": t["positives"],
                "inactives": t["inactives"],
                "ratio": t["ratio"],
                "auroc_mean": round(rep["auroc"].mean(), 4),
                "auroc_std": round(rep["auroc"].std(), 4),
                "auroc_pooled": round(auroc_pooled, 4) if auroc_pooled == auroc_pooled else np.nan,
                "auprc_mean": round(rep["auprc"].mean(), 4),
                "auprc_std": round(rep["auprc"].std(), 4),
                "baseline_auprc": round(rep["baseline_auprc"].mean(), 6),
                "bedroc_mean": round(rep["bedroc"].mean(), 4),
                "bedroc_std": round(rep["bedroc"].std(), 4),
                "baseline_bedroc": round(rep["baseline_bedroc"].mean(), 4),
                "decision_cutoff_rank": round(meta["decision_cutoff_rank"], 4)
                if meta.get("decision_cutoff_rank") is not None
                else np.nan,
                "descriptors_kept": ",".join(meta.get("descriptor_types", [])),
                **{
                    f"oof_auc_{d}": round(rep[f"oof_auc_{d}"].mean(), 4)
                    if f"oof_auc_{d}" in rep and rep[f"oof_auc_{d}"].notna().any()
                    else np.nan
                    for d in DESCRIPTORS
                },
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def draw_chance_line(ax) -> None:
    ax.plot([0, 1], [0, 1], color="k", lw=0.5, linestyle="dashed", zorder=1)


def style_roc_axes(ax) -> None:
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")


def boxplot_stats(arr: np.ndarray) -> dict:
    """Whiskers at the 5th/95th percentiles, matching the reference figure."""
    return dict(
        med=float(np.median(arr)),
        q1=float(np.percentile(arr, 25)),
        q3=float(np.percentile(arr, 75)),
        whislo=float(np.percentile(arr, 5)),
        whishi=float(np.percentile(arr, 95)),
        fliers=[],
    )


def subsample(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cap a point cloud for plotting. Display only — statistics use the full array."""
    if len(arr) <= SCATTER_MAX_POINTS:
        return arr
    idx = rng.choice(len(arr), size=SCATTER_MAX_POINTS, replace=False)
    return arr[idx]


# ---------------------------------------------------------------------------
# Per-pathogen figure
# ---------------------------------------------------------------------------


def plot_pathogen(pathogen: str, summary: pd.DataFrame, rng: np.random.Generator) -> str | None:
    sub = summary[summary["pathogen"] == pathogen]
    if sub.empty:
        print(f"[WARN] {pathogen}: nothing to plot")
        return None
    label = sub["pathogen_label"].iloc[0]
    tasks_present = [t for t in TASKS if t in set(sub["task"])]

    fig, axs = stylia.create_figure(1, 2)
    fig.suptitle(f"{label} — out-of-fold model performance", fontsize=stylia.FONTSIZE)

    # --- Panel A: ROC curves -------------------------------------------------
    ax = axs.next()
    draw_chance_line(ax)
    handles = []
    for task in tasks_present:
        folds = load_folds(pathogen, task)
        if folds is None:
            continue
        color = task_color(task)
        # Individual folds, faint.
        for fold, (yt, yr) in folds.items():
            if fold == "pooled" or len(set(yt.tolist())) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, yr)
            ax.plot(fpr, tpr, color=color, lw=0.4, alpha=0.35, zorder=2)
        # Pooled, bold.
        yt, yr = folds["pooled"]
        if len(set(yt.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yr)
        ax.plot(fpr, tpr, color=color, lw=1.4, zorder=3)

        row = sub[sub["task"] == task].iloc[0]
        handles.append(
            Line2D([0], [0], color=color, lw=1.4,
                   label=f"{task} — pooled {row['auroc_pooled']:.3f}\n"
                         f"    mean {row['auroc_mean']:.3f} ± {row['auroc_std']:.3f}")
        )
    style_roc_axes(ax)
    ax.legend(handles=handles, fontsize=stylia.FONTSIZE_SMALL, loc="lower right",
              frameon=True, framealpha=0.85, handlelength=1.2, borderpad=0.3)
    stylia.label(ax, xlabel="False positive rate", ylabel="True positive rate", abc="A")

    # --- Panel B: rank-score distributions -----------------------------------
    ax = axs.next()
    nc = stylia.NamedColors()
    w = 0.18
    x = list(range(len(tasks_present)))
    for i, task in enumerate(tasks_present):
        folds = load_folds(pathogen, task)
        if folds is None:
            continue
        yt, yr = folds["pooled"]
        actives, inactives = yr[yt == 1], yr[yt == 0]
        if not len(actives) or not len(inactives):
            continue

        a_plot, i_plot = subsample(actives, rng), subsample(inactives, rng)
        ax.scatter(i - w + rng.uniform(-w, w, size=len(i_plot)), i_plot,
                   color=nc.silver, s=1, alpha=0.35, lw=0, zorder=2)
        ax.scatter(i + w + rng.uniform(-w, w, size=len(a_plot)), a_plot,
                   color=nc.crimson, s=1, alpha=0.45, lw=0, zorder=3)

        # Boxplots use every point, not the subsample.
        bp = ax.bxp([boxplot_stats(inactives), boxplot_stats(actives)],
                    positions=[i - w, i + w], widths=w * 1.6,
                    patch_artist=True, showfliers=False, zorder=4)
        for box in bp["boxes"]:
            box.set_facecolor("none")
            box.set_linewidth(0.8)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color("k")
                line.set_linewidth(0.8 if element != "caps" else 0)

        cutoff = sub[sub["task"] == task].iloc[0]["decision_cutoff_rank"]
        if cutoff == cutoff:  # not NaN
            ax.plot([i - 2.4 * w, i + 2.4 * w], [cutoff, cutoff],
                    lw=0.7, color="k", linestyle="dotted", zorder=5)

    # The scatter is capped per group, so both tasks render at similar density however many
    # compounds they hold. Spell the real counts out on the axis so the ~20x difference in
    # prevalence between primary and secondary cannot be misread from the point clouds.
    tick_labels = []
    for task in tasks_present:
        row = sub[sub["task"] == task].iloc[0]
        tick_labels.append(f"{task}\n{row['positives']:,} act / {row['inactives']:,} inact")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim([-0.6, len(tasks_present) - 0.4])
    ax.set_ylim([0, 1])
    ax.legend(
        handles=[
            Patch(facecolor=nc.crimson, edgecolor="none", label="Actives"),
            Patch(facecolor=nc.silver, edgecolor="none", label="Inactives"),
            Line2D([0], [0], color="k", lw=0.7, linestyle="dotted", label="Decision cutoff"),
            Line2D([0], [0], color="none", lw=0,
                   label=f"scatter capped at {SCATTER_MAX_POINTS:,}/group;\nboxplots use all points"),
        ],
        fontsize=stylia.FONTSIZE_SMALL, loc="lower left", frameon=True, framealpha=0.85,
        handlelength=1.2, borderpad=0.3,
    )
    stylia.label(ax, xlabel="", ylabel="Out-of-fold rank score", abc="B")

    out_path = os.path.join(output_dir, f"08_report_{pathogen}.png")
    stylia.save_figure(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Two-panel ROC overlay across pathogens
# ---------------------------------------------------------------------------


def plot_roc_grid(summary: pd.DataFrame) -> str:
    fig, axs = stylia.create_figure(1, 2)
    for panel, task in enumerate(TASKS):
        ax = axs.next()
        draw_chance_line(ax)
        handles = []
        for pathogen in sorted(PATHOGEN_LABELS):
            row = summary[(summary["pathogen"] == pathogen) & (summary["task"] == task)]
            folds = load_folds(pathogen, task)
            if row.empty or folds is None:
                continue
            yt, yr = folds["pooled"]
            if len(set(yt.tolist())) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, yr)
            color = pathogen_color(pathogen)
            ax.plot(fpr, tpr, color=color, lw=1.1, zorder=2)
            handles.append(
                Line2D([0], [0], color=color, lw=1.1,
                       label=f"{row['pathogen_label'].iloc[0]}  {row['auroc_pooled'].iloc[0]:.3f}")
            )
        style_roc_axes(ax)
        ax.legend(handles=handles, fontsize=stylia.FONTSIZE_SMALL, loc="lower right",
                  frameon=True, framealpha=0.85, handlelength=1.2, borderpad=0.3)
        stylia.label(ax, xlabel="False positive rate",
                     ylabel="True positive rate" if panel == 0 else "",
                     title=f"{task.capitalize()} assays",
                     abc="AB"[panel])
    stylia.save_figure(ROC_GRID_PNG)
    return ROC_GRID_PNG


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(pathogens: list[str]) -> None:
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(
            f"{os.path.relpath(METADATA_CSV, repo_root)} not found — run 07a/07b first."
        )
    tasks = pd.read_csv(METADATA_CSV)

    summary = build_summary(tasks)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary ({len(summary)} tasks) -> {os.path.relpath(SUMMARY_CSV, repo_root)}")

    rng = np.random.default_rng(RANDOM_SEED)
    for pathogen in pathogens:
        out = plot_pathogen(pathogen, summary, rng)
        if out:
            print(f"  {os.path.relpath(out, repo_root)}")

    print(f"  {os.path.relpath(plot_roc_grid(summary), repo_root)}")

    capped = summary[summary["inactives"] > SCATTER_MAX_POINTS]
    if len(capped):
        print(f"\nScatter capped at {SCATTER_MAX_POINTS:,} points per group in "
              f"{len(capped)} of {len(summary)} tasks (display only; boxplots use all points).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate step-07 cross-validation reports and render evaluation figures."
    )
    parser.add_argument(
        "--pathogens",
        nargs="+",
        default=sorted(PATHOGEN_LABELS),
        choices=sorted(PATHOGEN_LABELS),
        metavar="CODE",
        help="Restrict the per-pathogen figures to these pathogens (default: all seven)",
    )
    args = parser.parse_args()
    main(args.pathogens)
