"""
Create per-pathogen subsets for ML benchmarking.

Produces three groups of output CSVs under output/06_subset_data/:

  exclusivity/
    {pathogen}_exclusive.csv     — actives in this pathogen's primary assay only
    {pathogen}_nonexclusive.csv  — actives shared with ≥1 other pathogen's primary assay

  secondary/
    {pathogen}_secondary.csv  — full merged dataset (all bin values) across all secondary
                                 assays; deduplicated with active prevailing over inactive

  abx_similarity/
    {pathogen}_abx.csv      — primary assay compounds where eos2xeq is_sim_known_ab == 1
    {pathogen}_non_abx.csv  — primary assay compounds where eos2xeq is_sim_known_ab == 0

Exclusivity files have columns: smiles, inchikey.
Secondary and abx_similarity files have columns: smiles, inchikey, bin.

The primary assay already exists as a canonical binarised file at
data/processed/02_binarised_assays/{assay_eos_id}.csv and is not re-copied here.
"""

import csv
import os

import numpy as np
import stylia

root = os.path.dirname(os.path.abspath(__file__))

binarised_dir = os.path.join(root, "..", "data", "processed", "02_binarised_assays")
merged_dir = os.path.join(root, "..", "data", "processed", "02_merged")
config_dir = os.path.join(root, "..", "data", "config")
model_dir = os.path.join(root, "..", "output", "04_ersilia_models")

output_dir = os.path.join(root, "..", "output", "06_subset_data")
exclusivity_dir = os.path.join(output_dir, "exclusivity")
secondary_dir = os.path.join(output_dir, "secondary")
abx_dir = os.path.join(output_dir, "abx_similarity")

os.makedirs(exclusivity_dir, exist_ok=True)
os.makedirs(secondary_dir, exist_ok=True)
os.makedirs(abx_dir, exist_ok=True)

PRIMARY_ASSAYS_CSV = os.path.join(config_dir, "primary_assays_manual.csv")
ALL_ASSAYS_CSV = os.path.join(config_dir, "assays_annotated_manual.csv")
ALL_SMILES_CSV = os.path.join(merged_dir, "02_all_smiles.csv")
EOS2XEQ_CSV = os.path.join(model_dir, "eos2xeq.csv")

# Format: print | Style: article
stylia.set_format("print")
stylia.set_style("article")

PATHOGEN_LABELS = {
    "abaumannii": "A. baumannii",
    "calbicans": "C. albicans",
    "ecoli": "E. coli",
    "efaecium": "E. faecium",
    "kpneumoniae": "K. pneumoniae",
    "paeruginosa": "P. aeruginosa",
    "saureus": "S. aureus",
}


def load_primary_assay_map() -> dict[str, str]:
    """Return {pathogen_code: assay_eos_id} for the primary-screen assay."""
    mapping = {}
    with open(PRIMARY_ASSAYS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["pathogen_code"]] = row["assay_eos_id"]
    return mapping


def load_all_assays_map() -> dict[str, list[str]]:
    """Return {pathogen_code: [assay_eos_id, ...]} for all assays."""
    mapping: dict[str, list[str]] = {}
    with open(ALL_ASSAYS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            pcode = row["pathogen_code"]
            if pcode not in mapping:
                mapping[pcode] = []
            mapping[pcode].append(row["assay_eos_id"])
    return mapping


def load_inchikey_lookup() -> dict[str, str]:
    """Return {smiles: inchikey} from the global deduplicated SMILES file."""
    lookup = {}
    with open(ALL_SMILES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["smiles"]] = row["inchikey"]
    return lookup


def load_eos2xeq_scores() -> dict[str, int]:
    """Return {smiles: is_sim_known_ab} from eos2xeq output; strips 'B.' prefix."""
    scores = {}
    with open(EOS2XEQ_CSV, newline="") as f:
        for row in csv.DictReader(f):
            raw = row["input"]
            smiles = raw[2:] if raw.startswith("B.") else raw
            scores[smiles] = int(row["is_sim_known_ab"])
    return scores


def load_actives(assay_eos_id: str) -> set[str]:
    """Return set of SMILES with bin == 1 from a binarised assay file."""
    path = os.path.join(binarised_dir, f"{assay_eos_id}.csv")
    if not os.path.exists(path):
        return set()
    actives = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["bin"]) == 1:
                actives.add(row["smiles"])
    return actives


def load_all_compounds(assay_eos_id: str) -> dict[str, int]:
    """Return {smiles: bin} for all compounds in a binarised assay file."""
    path = os.path.join(binarised_dir, f"{assay_eos_id}.csv")
    if not os.path.exists(path):
        return {}
    compounds = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            compounds[row["smiles"]] = int(row["bin"])
    return compounds


def merge_compounds(assay_ids: list[str]) -> dict[str, int]:
    """Merge compounds across multiple assays; active prevails over inactive/inconclusive."""
    merged: dict[str, int] = {}
    for aid in assay_ids:
        for smiles, bin_val in load_all_compounds(aid).items():
            if smiles not in merged or bin_val > merged[smiles]:
                merged[smiles] = bin_val
    return merged


def split_by_abx(
    primary_id: str, scores: dict[str, int]
) -> tuple[dict[str, int], dict[str, int], int]:
    """
    Split primary assay compounds by eos2xeq is_sim_known_ab.
    Returns (abx, non_abx, n_missing) where each dict maps smiles -> bin.
    """
    abx: dict[str, int] = {}
    non_abx: dict[str, int] = {}
    n_missing = 0
    for smiles, bin_val in load_all_compounds(primary_id).items():
        flag = scores.get(smiles)
        if flag is None:
            n_missing += 1
            continue
        if flag == 1:
            abx[smiles] = bin_val
        else:
            non_abx[smiles] = bin_val
    return abx, non_abx, n_missing


def write_subset(smiles_set: set[str], inchikey_lookup: dict[str, str], path: str) -> int:
    """Write sorted smiles + inchikey CSV. Returns number of rows written."""
    rows = sorted(
        ({"smiles": s, "inchikey": inchikey_lookup.get(s, "")} for s in smiles_set),
        key=lambda r: r["smiles"],
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "inchikey"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_labeled(compounds: dict[str, int], inchikey_lookup: dict[str, str], path: str) -> int:
    """Write sorted smiles + inchikey + bin CSV. Returns number of rows written."""
    rows = sorted(
        (
            {"smiles": s, "inchikey": inchikey_lookup.get(s, ""), "bin": b}
            for s, b in compounds.items()
        ),
        key=lambda r: r["smiles"],
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "inchikey", "bin"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def plot_abx_summary(stats: dict, out_path: str) -> None:
    """
    Two-panel figure:
      A — total abx vs non-abx compounds per pathogen (log scale)
      B — active compounds within abx vs non-abx per pathogen
    """
    pcodes = sorted(stats.keys())
    labels = [PATHOGEN_LABELS[p] for p in pcodes]

    abx_totals = [stats[p]["abx"]["total"] for p in pcodes]
    non_abx_totals = [stats[p]["non_abx"]["total"] for p in pcodes]
    abx_actives = [stats[p]["abx"]["active"] for p in pcodes]
    non_abx_actives = [stats[p]["non_abx"]["active"] for p in pcodes]

    nc = stylia.NamedColors()
    x = np.arange(len(pcodes))
    w = 0.35

    fig, axs = stylia.create_figure(1, 2)

    ax = axs.next()
    ax.bar(x - w / 2, abx_totals, w, color=nc.cobalt, label="abx similar")
    ax.bar(x + w / 2, non_abx_totals, w, color=nc.silver, label="not abx similar")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    stylia.label(ax, xlabel="", ylabel="Compounds (log scale)", abc="A")

    ax = axs.next()
    ax.bar(x - w / 2, abx_actives, w, color=nc.cobalt, label="abx similar")
    ax.bar(x + w / 2, non_abx_actives, w, color=nc.silver, label="not abx similar")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    stylia.label(ax, xlabel="", ylabel="Active compounds", abc="B")

    stylia.save_figure(out_path)


def main():
    primary_map = load_primary_assay_map()
    all_assays_map = load_all_assays_map()
    inchikey_lookup = load_inchikey_lookup()

    pathogen_codes = sorted(primary_map.keys())

    # --- Load primary actives for all pathogens ---
    primary_actives: dict[str, set[str]] = {
        pcode: load_actives(primary_map[pcode]) for pcode in pathogen_codes
    }

    # --- Exclusivity subsets ---
    print("Exclusivity subsets (primary assay only):")
    for pcode in pathogen_codes:
        this_actives = primary_actives[pcode]
        other_actives: set[str] = set()
        for other in pathogen_codes:
            if other != pcode:
                other_actives |= primary_actives[other]

        exclusive = this_actives - other_actives
        nonexclusive = this_actives & other_actives

        n_ex = write_subset(
            exclusive,
            inchikey_lookup,
            os.path.join(exclusivity_dir, f"{pcode}_exclusive.csv"),
        )
        n_non = write_subset(
            nonexclusive,
            inchikey_lookup,
            os.path.join(exclusivity_dir, f"{pcode}_nonexclusive.csv"),
        )
        print(f"  {pcode}: {n_ex} exclusive, {n_non} nonexclusive (total {len(this_actives)})")

    # --- Secondary subsets ---
    print("\nSecondary datasets (all compounds, active prevails on duplicate):")
    for pcode in pathogen_codes:
        primary_id = primary_map[pcode]
        all_ids = all_assays_map.get(pcode, [])
        secondary_ids = [aid for aid in all_ids if aid != primary_id]

        merged = merge_compounds(secondary_ids)
        n_active = sum(1 for b in merged.values() if b == 1)
        n_inactive = sum(1 for b in merged.values() if b == 0)
        n_inconclusive = sum(1 for b in merged.values() if b == -1)

        write_labeled(
            merged,
            inchikey_lookup,
            os.path.join(secondary_dir, f"{pcode}_secondary.csv"),
        )
        print(
            f"  {pcode}: {len(merged)} compounds "
            f"({n_active} active, {n_inactive} inactive, {n_inconclusive} inconclusive)"
        )

    # --- ABX similarity subsets ---
    eos2xeq_scores = load_eos2xeq_scores()
    abx_stats: dict = {}
    print("\nABX similarity subsets (primary assay, eos2xeq is_sim_known_ab):")
    for pcode in pathogen_codes:
        primary_id = primary_map[pcode]
        abx, non_abx, n_missing = split_by_abx(primary_id, eos2xeq_scores)

        n_abx_active = sum(1 for b in abx.values() if b == 1)
        n_non_abx_active = sum(1 for b in non_abx.values() if b == 1)

        write_labeled(abx, inchikey_lookup, os.path.join(abx_dir, f"{pcode}_abx.csv"))
        write_labeled(non_abx, inchikey_lookup, os.path.join(abx_dir, f"{pcode}_non_abx.csv"))

        abx_stats[pcode] = {
            "abx": {"total": len(abx), "active": n_abx_active},
            "non_abx": {"total": len(non_abx), "active": n_non_abx_active},
        }
        missing_note = f", {n_missing} missing eos2xeq score" if n_missing else ""
        print(
            f"  {pcode}: {len(abx)} abx ({n_abx_active} active), "
            f"{len(non_abx)} non-abx ({n_non_abx_active} active){missing_note}"
        )

    plot_path = os.path.join(output_dir, "06_abx_similarity.png")
    plot_abx_summary(abx_stats, plot_path)
    print(f"\nPlot → {os.path.relpath(plot_path, os.path.join(root, '..'))}")


if __name__ == "__main__":
    main()
