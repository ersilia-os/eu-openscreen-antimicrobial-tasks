"""
Clean and binarise raw assay data; merge by pathogen.

Reads per-assay CSVs from data/raw/00_extracted_assays/ and
data/raw/01_extracted_assays/, maps activity labels to binary scores,
and produces:

  data/processed/02_binarised_assays/
    00_<assay_id>.csv   — (smiles, bin) for each assay from the SQL dump
    01_<assay_id>.csv   — (smiles, bin) for each assay from the API fetch

  data/processed/
    02_all_smiles.csv   — deduplicated (smiles, inchikey) across all assays
    02_only_smiles.csv  — deduplicated smiles column only
    02_<pathogen>.csv   — merged (smiles, inchikey, bin) per pathogen,
                          deduplicated by SMILES with Active prevailing

Activity mapping:
  active       →  1
  inactive     →  0
  inconclusive → -1
  undefined    → -1  (treated as inconclusive: result not determined)

Rows with no SMILES are dropped before any output is written.
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent  # scripts/

raw_dir = root / ".." / "data" / "raw"
processed_dir = root / ".." / "data" / "processed"
output_dir = root / ".." / "output"
binarised_dir = processed_dir / "02_binarised_assays"

os.makedirs(binarised_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

SUMMARY_00 = output_dir / "00_assay_summary.csv"
SUMMARY_01 = output_dir / "01_assay_summary.csv"

ACTIVITY_MAP = {
    "active": 1,
    "inactive": 0,
    "inconclusive": -1,
    "undefined": -1,
}


def load_assay_pathogen_map(summary_path: Path) -> dict[str, dict]:
    """Return assay_eos_id -> {pathogen, pathogen_code} from a summary CSV."""
    mapping = {}
    if not summary_path.exists():
        return mapping
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["assay_eos_id"]] = {
                "pathogen": row["pathogen"],
                "pathogen_code": row["pathogen_code"],
            }
    return mapping


def read_assay(csv_path: Path) -> list[dict]:
    """Read a raw assay CSV; return rows that have a non-empty SMILES."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                continue
            activity = (row.get("activity") or "").strip().lower()
            bin_val = ACTIVITY_MAP.get(activity)
            if bin_val is None:
                continue
            rows.append({
                "smiles": smiles,
                "inchikey": (row.get("inchikey") or "").strip(),
                "bin": bin_val,
            })
    return rows


def write_binarised(rows: list[dict], out_path: Path) -> int:
    """Write (smiles, bin) CSV. Returns number of rows written."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "bin"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"smiles": row["smiles"], "bin": row["bin"]})
    return len(rows)


def merge_pathogen_rows(all_rows: list[dict]) -> list[dict]:
    """
    Deduplicate by SMILES, keeping the highest bin value (Active prevails).
    Returns rows sorted by bin descending then smiles.
    """
    best: dict[str, dict] = {}
    for row in all_rows:
        smiles = row["smiles"]
        if smiles not in best or row["bin"] > best[smiles]["bin"]:
            best[smiles] = row
    return sorted(best.values(), key=lambda r: (-r["bin"], r["smiles"]))


def process_source(
    raw_subdir: Path,
    assay_map: dict[str, dict],
    prefix: str,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Process all assay CSVs in raw_subdir.
    Returns:
      - all_rows: flat list of all (smiles, inchikey, bin) rows
      - pathogen_rows: pathogen_code -> list of rows
    """
    all_rows: list[dict] = []
    pathogen_rows: dict[str, list[dict]] = defaultdict(list)

    csv_files = sorted(raw_subdir.glob("*.csv"))
    print(f"\n{prefix} source: {len(csv_files)} assay files in {raw_subdir.name}/")

    for csv_path in csv_files:
        assay_id = csv_path.stem  # e.g. EOS300076
        info = assay_map.get(assay_id)
        if info is None:
            print(f"  WARNING: {assay_id} not in summary, skipping.")
            continue

        rows = read_assay(csv_path)
        if not rows:
            print(f"  {assay_id}: no usable rows (empty or all rows lack SMILES).")
            out_path = binarised_dir / f"{assay_id}.csv"
            write_binarised([], out_path)
            continue

        out_path = binarised_dir / f"{assay_id}.csv"
        n = write_binarised(rows, out_path)
        print(f"  {assay_id}: {n} rows → {out_path.name}")

        all_rows.extend(rows)
        pcode = info["pathogen_code"]
        pathogen_rows[pcode].extend(rows)

    return all_rows, pathogen_rows


def main():
    assay_map_00 = load_assay_pathogen_map(SUMMARY_00)
    assay_map_01 = load_assay_pathogen_map(SUMMARY_01)

    # --- Process both sources ---
    all_rows_00, pathogen_rows_00 = process_source(
        raw_dir / "00_extracted_assays", assay_map_00, "00"
    )
    all_rows_01, pathogen_rows_01 = process_source(
        raw_dir / "01_extracted_assays", assay_map_01, "01"
    )

    # --- Deduplicated global SMILES lists ---
    all_rows = all_rows_00 + all_rows_01
    seen_smiles: dict[str, str] = {}  # smiles -> inchikey (first seen)
    for row in all_rows:
        smiles = row["smiles"]
        if smiles not in seen_smiles:
            seen_smiles[smiles] = row["inchikey"]

    all_smiles_path = processed_dir / "02_all_smiles.csv"
    with open(all_smiles_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "inchikey"])
        writer.writeheader()
        for smiles, inchikey in sorted(seen_smiles.items()):
            writer.writerow({"smiles": smiles, "inchikey": inchikey})

    only_smiles_path = processed_dir / "02_only_smiles.csv"
    with open(only_smiles_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles"])
        writer.writeheader()
        for smiles in sorted(seen_smiles):
            writer.writerow({"smiles": smiles})

    print(f"\nGlobal SMILES: {len(seen_smiles)} unique entries")
    print(f"  → {all_smiles_path.name}")
    print(f"  → {only_smiles_path.name}")

    # --- Merged per-pathogen files ---
    # Combine rows from both sources for each pathogen code
    all_pathogen_codes = set(pathogen_rows_00) | set(pathogen_rows_01)
    print(f"\nMerging {len(all_pathogen_codes)} pathogens ...")

    for pcode in sorted(all_pathogen_codes):
        combined = pathogen_rows_00.get(pcode, []) + pathogen_rows_01.get(pcode, [])
        merged = merge_pathogen_rows(combined)

        out_path = processed_dir / f"02_{pcode}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["smiles", "inchikey", "bin"])
            writer.writeheader()
            writer.writerows(merged)

        n_active = sum(1 for r in merged if r["bin"] == 1)
        n_inactive = sum(1 for r in merged if r["bin"] == 0)
        n_inconclusive = sum(1 for r in merged if r["bin"] == -1)
        print(
            f"  {pcode}: {len(merged)} unique SMILES — "
            f"{n_active} active, {n_inactive} inactive, {n_inconclusive} inconclusive"
            f" → {out_path.name}"
        )


if __name__ == "__main__":
    main()
