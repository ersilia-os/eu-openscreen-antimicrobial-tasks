"""
Fetch assays from the ECBD API that are absent from the SQL dump.

For each target EOS ID present in output/00_assay_summary.csv, queries
https://ecbd.eu/api/targets/{id}/ to discover assays not yet in the dump.
Full metadata is retrieved from the assay detail endpoint.

Per-compound data is retrieved via the frontend ajax_data endpoint:
  1. Parse the assay HTML page to extract the endpoint_id for the data viewer.
  2. Call /assays/ajax_data?epid={endpoint_id} to get per-compound IC50/activity.
  3. Fetch SMILES from /api/compounds/EOS{id}/ for each compound.
For embargoed assays (no data rendered on page), a header-only CSV is written.

Outputs:
  output/01_fetch_new_assays/01_assay_summary.csv  — metadata for newly found assays
  data/raw/01_extracted_assays/                    — one CSV per new assay (with data if available)
"""

import csv
import os
import re
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

root = Path(__file__).resolve().parent  # scripts/

raw_dir = root / ".." / "data" / "raw"
output_dir = root / ".." / "output"
script_output_dir = output_dir / "01_fetch_new_assays"
os.makedirs(script_output_dir, exist_ok=True)

SUMMARY_00 = output_dir / "00_extract_assays" / "00_assay_summary.csv"
OUTPUT_CSV = script_output_dir / "01_assay_summary.csv"
ASSAYS_DIR = raw_dir / "01_extracted_assays"
os.makedirs(ASSAYS_DIR, exist_ok=True)

API_BASE = "https://ecbd.eu/api"
WEB_BASE = "https://ecbd.eu"
REQUEST_DELAY = 0.5  # seconds between API calls, to be polite

# Column schema shared with 00_extracted_assays CSVs
COMPOUND_FIELDNAMES = [
    "compound_eos_id", "depositor_compoundid", "activity", "result_comment",
    "value", "operator", "concentration", "endpoint_label", "unit_label",
    "plate", "time", "curve_slope", "curve_bottom", "curve_top",
    "curve_inflection", "smiles", "inchikey", "mw", "formula",
]


def load_known_state(summary_path: Path) -> tuple[set[str], dict[str, dict]]:
    """
    Read 00_assay_summary.csv and return:
    - known_assay_eos_ids: set of assay EOS IDs already in the dump
    - target_to_info: dict mapping target_eos_id -> {pathogen, pathogen_code}
    """
    known_ids: set[str] = set()
    target_info: dict[str, dict] = {}
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            known_ids.add(row["assay_eos_id"])
            tid = row.get("intended_target_eos_id")
            if tid and tid not in target_info:
                target_info[tid] = {
                    "pathogen": row["pathogen"],
                    "pathogen_code": row["pathogen_code"],
                }
    return known_ids, target_info


def api_get(url: str) -> dict | list | None:
    """GET a URL; returns parsed JSON or None on 404/error."""
    try:
        resp = requests.get(url, timeout=30, verify=False)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  WARNING: {url} → {e}")
        return None


def fetch_target_assays(target_eos_id: str) -> list[dict]:
    """Return the list of assay summary dicts from the API for a given target."""
    data = api_get(f"{API_BASE}/targets/{target_eos_id}/")
    if data is None:
        return []
    return data.get("assays", [])


def fetch_assay_detail(assay_eos_id: str) -> dict | None:
    """Return full assay metadata from the assay detail endpoint."""
    return api_get(f"{API_BASE}/assays/{assay_eos_id}/")


def parse_endpoint_ids(assay_eos_id: str) -> list[int]:
    """
    Fetch the assay HTML page and extract all endpoint_id values from
    <endpoint-data> components. Returns an empty list if the page renders
    no data (e.g. embargoed assay).
    """
    try:
        resp = requests.get(f"{WEB_BASE}/assays/{assay_eos_id}", timeout=30, verify=False)
        resp.raise_for_status()
        return [int(m) for m in re.findall(r':endpoint_id="(\d+)"', resp.text)]
    except requests.RequestException as e:
        print(f"  WARNING fetching assay page: {e}")
        return []


def fetch_ajax_data(endpoint_id: int) -> dict:
    """Fetch per-compound result data from the frontend ajax_data endpoint."""
    resp = requests.get(
        f"{WEB_BASE}/assays/ajax_data",
        params={"epid": endpoint_id},
        timeout=30,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_compound_smiles(compound_id: str) -> tuple[str | None, str | None]:
    """Return (smiles, depositor_compoundid) for a compound EOS ID."""
    data = api_get(f"{API_BASE}/compounds/EOS{compound_id}/")
    if data:
        return data.get("_label"), data.get("depositor_compoundid")
    return None, None


def export_assay_csv(
    assay_eos_id: str,
    endpoint_label: str,
    unit_label: str,
    endpoint_ids: list[int],
    out_dir: Path,
) -> int:
    """
    Fetch compound-level data for all endpoint_ids and write the assay CSV.
    Returns the total number of data rows written (0 for header-only).
    """
    out_path = out_dir / f"{assay_eos_id}.csv"

    if not endpoint_ids:
        # Embargoed or no data available: write header-only placeholder
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COMPOUND_FIELDNAMES).writeheader()
        return 0

    all_rows: list[dict] = []

    for epid in endpoint_ids:
        time.sleep(REQUEST_DELAY)
        ajax = fetch_ajax_data(epid)
        ep_label = ajax.get("label") or endpoint_label
        compounds = ajax.get("data", {})
        curves = ajax.get("curves") or {}

        for compound_id, entry in compounds.items():
            inchikey = entry.get("cp")
            activity = entry.get("ac")
            curve = curves.get(compound_id, {}) if isinstance(curves, dict) else {}

            time.sleep(REQUEST_DELAY)
            smiles, depositor_id = fetch_compound_smiles(compound_id)

            for v in entry.get("values", [None]):
                val = v.get("val") if v else None
                conc = v.get("conc") if v else None
                all_rows.append({
                    "compound_eos_id": f"EOS{compound_id}",
                    "depositor_compoundid": depositor_id,
                    "activity": activity,
                    "result_comment": None,
                    "value": val,
                    "operator": None,
                    "concentration": conc,
                    "endpoint_label": ep_label,
                    "unit_label": unit_label,
                    "plate": None,
                    "time": None,
                    "curve_slope": curve.get("slope"),
                    "curve_bottom": curve.get("bottom"),
                    "curve_top": curve.get("top"),
                    "curve_inflection": curve.get("inflection"),
                    "smiles": smiles,
                    "inchikey": inchikey,
                    "mw": None,
                    "formula": None,
                })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPOUND_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    return len(all_rows)


def main():
    known_ids, target_info = load_known_state(SUMMARY_00)
    print(f"Loaded {len(known_ids)} known assay EOS IDs across {len(target_info)} targets.\n")

    summary_rows = []

    for target_eos_id in sorted(target_info.keys()):
        info = target_info[target_eos_id]
        pathogen = info["pathogen"]
        pathogen_code = info["pathogen_code"]
        print(f"{target_eos_id}  ({pathogen})")

        time.sleep(REQUEST_DELAY)
        api_assays = fetch_target_assays(target_eos_id)
        if not api_assays:
            print("  No assays returned by API.")
            continue

        for assay_summary in api_assays:
            eos_ptr = assay_summary.get("eos_ptr_id")
            assay_eos_id = f"EOS{eos_ptr}" if eos_ptr else None
            if not assay_eos_id:
                continue

            if assay_eos_id in known_ids:
                print(f"  {assay_eos_id}: in dump, skipping.")
                continue

            time.sleep(REQUEST_DELAY)
            detail = fetch_assay_detail(assay_eos_id)
            if detail is None:
                print(f"  {assay_eos_id}: NEW but detail endpoint returned nothing, skipping.")
                continue

            activities = assay_summary.get("activities", {}) or {}
            activity_summary = activities.get("summary", {}) or {}
            endpoint = activities.get("endpoint", {}) or {}
            n_compounds = activities.get("compounds", 0)
            n_active = activity_summary.get("active", 0)
            endpoint_label = endpoint.get("endpoint_label", "")
            unit_label = endpoint.get("unit_label", "")

            print(
                f"  {assay_eos_id}: NEW — {detail.get('assay_name', '')} "
                f"({n_compounds} compounds, {n_active} active)"
            )

            # Try to retrieve per-compound data from the frontend
            time.sleep(REQUEST_DELAY)
            endpoint_ids = parse_endpoint_ids(assay_eos_id)
            n_rows = export_assay_csv(assay_eos_id, endpoint_label, unit_label, endpoint_ids, ASSAYS_DIR)

            if n_rows:
                print(f"    → {n_rows} compound rows written to {assay_eos_id}.csv")
            else:
                print(f"    → placeholder CSV written (embargoed or no data available)")

            summary_rows.append({
                "pathogen": pathogen,
                "pathogen_code": pathogen_code,
                "assay_eos_id": assay_eos_id,
                "assay_name": detail.get("assay_name"),
                "n_compounds_tested": n_compounds,
                "n_compounds_active": n_active,
                "embargo_until": detail.get("embargo_until"),
                "assay_description": detail.get("assay_description"),
                "activity_determination": detail.get("activity_determination"),
                "aria_project_id": detail.get("aria_project_id"),
                "intended_target_eos_id": target_eos_id,
                "deprecated": detail.get("deprecated"),
                "assay_stage": assay_summary.get("assay_stage"),
                "endpoint_label": endpoint_label,
                "unit_label": unit_label,
                "target_organism_names": (
                    detail.get("params", {}).get("assay_organism", {}).get("label", "")
                ),
                "compound_data_retrieved": n_rows > 0,
            })

    if not summary_rows:
        print("\nNo new assays found.")
        return

    summary_rows.sort(key=lambda r: r["pathogen"])
    fieldnames = [
        "pathogen", "pathogen_code", "assay_eos_id", "assay_name",
        "n_compounds_tested", "n_compounds_active",
        "embargo_until", "assay_description", "activity_determination",
        "aria_project_id", "intended_target_eos_id", "deprecated",
        "assay_stage", "endpoint_label", "unit_label", "target_organism_names",
        "compound_data_retrieved",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{len(summary_rows)} new assays written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
