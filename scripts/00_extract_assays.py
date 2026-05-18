"""
Extract EU OpenScreen assay summaries for pathogens of interest.

Parses the PostgreSQL dump (ecbd_dump_public.zip) without loading it fully into memory,
extracts relevant tables, and produces a CSV summary of assays matching each pathogen.
"""

import csv
import io
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent  # scripts/

raw_dir = root / ".." / "data" / "raw"
output_dir = root / ".." / "output"
os.makedirs(output_dir, exist_ok=True)

DUMP_ZIP = raw_dir / "ecbd_dump_public.zip"
PATHOGENS_CSV = raw_dir / "pathogens.csv"
OUTPUT_CSV = output_dir / "00_assay_summary.csv"
ASSAYS_DIR = raw_dir / "00_extracted_assays"

# IRI-based matching for organisms present in the EU OpenScreen database.
# For pathogens not listed here, keyword matching on assay name/description/target is used as fallback.
# E. coli: both the species-level (562) and the ATCC 25922 strain (1322345) are included.
# P. aeruginosa: annotated as "group" in the DB (136841) but treated as P. aeruginosa.
# E. faecalis (1351) is used for the "efaecium" entry — it is the closest match available in the DB.
PATHOGEN_IRIS: dict[str, list[str]] = {
    "abaumannii":  ["http://purl.obolibrary.org/obo/NCBITaxon_470"],
    "calbicans":   ["http://purl.obolibrary.org/obo/NCBITaxon_5476"],
    "ecoli":       ["http://purl.obolibrary.org/obo/NCBITaxon_562",
                    "http://purl.obolibrary.org/obo/NCBITaxon_1322345"],
    "efaecium":    ["http://purl.obolibrary.org/obo/NCBITaxon_1351"],
    "kpneumoniae": ["http://purl.obolibrary.org/obo/NCBITaxon_573"],
    "paeruginosa": ["http://purl.obolibrary.org/obo/NCBITaxon_136841"],
    "saureus":     ["http://purl.obolibrary.org/obo/NCBITaxon_1280"],
}

# Tables to extract and their column definitions (order matches COPY statement)
TABLES = {
    "core_assay": [
        "id", "assay_name", "assay_description", "embargo_until",
        "activity_determination", "aria_project_id", "assay_project_id",
        "eos_ptr_id", "intended_target_id", "submission_id", "deprecated",
        "alternative_result",
    ],
    "core_assayparameter": ["id", "name", "ontology", "iri", "label", "assay_id"],
    "core_target": [
        "eos_ptr_id", "target_type", "nucleicacidtype", "celline_type",
        "target_fingerprint", "name", "organism_specification", "sequence",
    ],
    "core_protein": [
        "accession", "name", "gene_name", "organism_name", "organism_tax_id",
        "organism_tax_iri", "domain", "function", "pdb_id", "subunit",
    ],
    "core_target_proteins": ["id", "target_id", "protein_id"],
    "core_result": ["id", "activity", "comment", "control_id", "assay_id", "compound_id"],
    "core_resultvalue": [
        "id", "value", "operator", "concentration", "time", "comment", "plate",
        "curve_slope", "curve_bottom", "curve_top", "endpoint_id", "result_id",
        "curve_inflection",
    ],
    "core_endpoint": [
        "id", "endpoint_ontology", "endpoint_iri", "endpoint_label",
        "unit_ontology", "unit_iri", "unit_label",
    ],
    "core_compound": ["eos_ptr_id", "depositor_compoundid", "structure_id"],
    "core_structure": [
        "id", "name", "smiles", "molblock", "inchi", "inchikey",
        "formula", "mw", "hba", "hbd", "tpsa", "rb", "fp3", "logp", "description",
    ],
}


def load_pathogens(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_copy_block(lines: list[str], columns: list[str]) -> list[dict]:
    """Parse tab-separated rows from a PostgreSQL COPY...FROM stdin block."""
    records = []
    for line in lines:
        if line == r"\.":
            break
        fields = line.split("\t")
        if len(fields) != len(columns):
            # Truncate or pad to avoid index errors on malformed rows
            fields = fields[: len(columns)] + [""] * max(0, len(columns) - len(fields))
        record = {col: (None if val == r"\N" else val) for col, val in zip(columns, fields)}
        records.append(record)
    return records


def extract_tables(zip_path: Path, wanted: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Stream the SQL dump and extract only the requested tables."""
    results = {t: [] for t in wanted}
    copy_pattern = re.compile(r"^COPY public\.(\w+) \(([^)]+)\) FROM stdin;$")

    print(f"Streaming {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        sql_name = zf.namelist()[0]
        with zf.open(sql_name) as raw:
            reader = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            in_copy = None
            buffer = []

            for line in reader:
                line = line.rstrip("\n")

                if in_copy:
                    if line == r"\.":
                        results[in_copy] = parse_copy_block(buffer, wanted[in_copy])
                        print(f"  {in_copy}: {len(results[in_copy])} rows")
                        in_copy = None
                        buffer = []
                    else:
                        buffer.append(line)
                else:
                    m = copy_pattern.match(line)
                    if m and m.group(1) in wanted:
                        in_copy = m.group(1)
                        buffer = []

    return results


def build_pathogen_keywords(pathogens: list[dict]) -> dict[str, list[str]]:
    """
    Build a mapping from pathogen name to search keywords.
    Uses genus + species (or just genus for single-word names).
    """
    keywords: dict[str, list[str]] = {}
    for p in pathogens:
        name = p["pathogen"]
        parts = name.lower().split()
        kws = [name.lower()]
        if len(parts) >= 2:
            kws.append(parts[0])          # genus
            kws.append(" ".join(parts[:2]))  # genus species
        else:
            kws.append(parts[0])
        keywords[name] = list(dict.fromkeys(kws))  # deduplicate preserving order
    return keywords


def text_matches(text: str | None, keywords: list[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def export_assay_files(
    matched_rows: list[dict],
    tables: dict,
    out_dir: Path,
) -> None:
    """Write one CSV per matched assay into out_dir, with per-compound result data."""
    results = tables["core_result"]
    resultvalues = tables["core_resultvalue"]
    endpoints = {r["id"]: r for r in tables["core_endpoint"]}
    compounds = {r["eos_ptr_id"]: r for r in tables["core_compound"]}
    # structure_id (str) -> structure record
    structures = {r["id"]: r for r in tables["core_structure"]}
    # compound eos_ptr_id -> structure record (via structure_id)
    compound_structure: dict[str, dict] = {}
    for comp in compounds.values():
        struct = structures.get(comp["structure_id"])
        if struct:
            compound_structure[comp["eos_ptr_id"]] = struct

    # result_id -> list of resultvalue rows
    result_to_values: dict[str, list[dict]] = defaultdict(list)
    for rv in resultvalues:
        result_to_values[rv["result_id"]].append(rv)

    # Map assay eos_ptr_id -> django id (needed to match core_result.assay_id)
    # Build from matched_rows: we stored eos_ptr in assay_eos_id; we need the django id
    # Re-derive from the assay table passed in tables
    assay_eos_to_django: dict[str, str] = {}
    for a in tables["core_assay"]:
        assay_eos_to_django[f"EOS{a['eos_ptr_id']}"] = a["id"]

    # Collect unique assays to export
    unique_assays: dict[str, dict] = {}  # eos_id -> summary row
    for row in matched_rows:
        eos = row["assay_eos_id"]
        if eos not in unique_assays:
            unique_assays[eos] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "compound_eos_id", "depositor_compoundid", "activity", "result_comment",
        "value", "operator", "concentration", "endpoint_label", "unit_label",
        "plate", "time", "curve_slope", "curve_bottom", "curve_top",
        "curve_inflection", "smiles", "inchikey", "mw", "formula",
    ]

    print(f"\nExporting {len(unique_assays)} assay files to {out_dir} ...")
    for eos_id, summary in sorted(unique_assays.items()):
        django_id = assay_eos_to_django.get(eos_id)
        if django_id is None:
            continue

        # Collect results for this assay (skip controls: compound_id is None)
        assay_results = [r for r in results if r["assay_id"] == django_id and r["compound_id"]]

        out_rows = []
        for res in assay_results:
            compound_id = res["compound_id"]
            comp = compounds.get(compound_id, {})
            struct = compound_structure.get(compound_id, {})

            rvs = result_to_values.get(res["id"])
            if rvs:
                for rv in rvs:
                    ep = endpoints.get(rv["endpoint_id"], {})
                    out_rows.append({
                        "compound_eos_id": f"EOS{compound_id}",
                        "depositor_compoundid": comp.get("depositor_compoundid"),
                        "activity": res["activity"],
                        "result_comment": res["comment"],
                        "value": rv["value"],
                        "operator": rv["operator"],
                        "concentration": rv["concentration"],
                        "endpoint_label": ep.get("endpoint_label"),
                        "unit_label": ep.get("unit_label"),
                        "plate": rv["plate"],
                        "time": rv["time"],
                        "curve_slope": rv["curve_slope"],
                        "curve_bottom": rv["curve_bottom"],
                        "curve_top": rv["curve_top"],
                        "curve_inflection": rv["curve_inflection"],
                        "smiles": struct.get("smiles"),
                        "inchikey": struct.get("inchikey"),
                        "mw": struct.get("mw"),
                        "formula": struct.get("formula"),
                    })
            else:
                # Result exists but no numeric value row (single-point with no endpoint)
                out_rows.append({
                    "compound_eos_id": f"EOS{compound_id}",
                    "depositor_compoundid": comp.get("depositor_compoundid"),
                    "activity": res["activity"],
                    "result_comment": res["comment"],
                    "value": None, "operator": None, "concentration": None,
                    "endpoint_label": None, "unit_label": None,
                    "plate": None, "time": None, "curve_slope": None,
                    "curve_bottom": None, "curve_top": None, "curve_inflection": None,
                    "smiles": struct.get("smiles"),
                    "inchikey": struct.get("inchikey"),
                    "mw": struct.get("mw"),
                    "formula": struct.get("formula"),
                })

        out_path = out_dir / f"{eos_id}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"  {eos_id}: {len(out_rows)} rows → {out_path.name}")


def main():
    pathogens = load_pathogens(PATHOGENS_CSV)
    pathogen_keywords = build_pathogen_keywords(pathogens)

    tables = extract_tables(DUMP_ZIP, TABLES)

    assays = tables["core_assay"]
    assay_params = tables["core_assayparameter"]
    targets = {r["eos_ptr_id"]: r for r in tables["core_target"]}
    proteins = {r["accession"]: r for r in tables["core_protein"]}
    target_proteins = tables["core_target_proteins"]
    results = tables["core_result"]

    # --- Step 1: extract all organisms and write 00_all_organisms.csv ---
    assay_to_target_eos: dict[str, str] = {
        a["id"]: f"EOS{a['intended_target_id']}"
        for a in assays if a.get("intended_target_id")
    }
    all_organisms: dict[str, dict] = {}
    organism_targets: dict[str, set[str]] = defaultdict(set)
    assay_param_iris: dict[str, set[str]] = defaultdict(set)
    assay_param_labels: dict[str, set[str]] = defaultdict(set)

    for p in assay_params:
        if p["name"] == "assay_organism" and p.get("label"):
            iri = p.get("iri") or ""
            key = iri if iri else p["label"]
            if key not in all_organisms:
                all_organisms[key] = {"label": p["label"], "iri": iri, "ontology": p.get("ontology") or ""}
            target_eos = assay_to_target_eos.get(p["assay_id"])
            if target_eos:
                organism_targets[key].add(target_eos)
            if iri:
                assay_param_iris[p["assay_id"]].add(iri)
            assay_param_labels[p["assay_id"]].add(p["label"])

    tax_id_pattern = re.compile(r"NCBITaxon_(\d+)$")
    for key, record in all_organisms.items():
        m = tax_id_pattern.search(record["iri"])
        record["ncbi_tax_id"] = m.group(1) if m else ""
        record["intended_target_eos_ids"] = "; ".join(sorted(organism_targets[key]))

    organisms_path = output_dir / "00_all_organisms.csv"
    with open(organisms_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "ncbi_tax_id", "intended_target_eos_ids", "iri", "ontology"])
        writer.writeheader()
        writer.writerows(sorted(all_organisms.values(), key=lambda r: r["label"].lower()))
    print(f"All organisms: {len(all_organisms)} unique entries → {organisms_path}")

    # --- Step 2: build remaining lookups ---
    # target_id -> protein organism names (used for keyword-fallback text)
    target_to_organisms: dict[str, set[str]] = defaultdict(set)
    for tp in target_proteins:
        protein = proteins.get(tp["protein_id"])
        if protein and protein.get("organism_name"):
            target_to_organisms[tp["target_id"]].add(protein["organism_name"])

    # assay_id -> compound/active counts
    assay_compound_counts: dict[str, int] = defaultdict(int)
    assay_active_counts: dict[str, int] = defaultdict(int)
    for r in results:
        if r["compound_id"] is not None:
            assay_compound_counts[r["assay_id"]] += 1
            if r["activity"] and r["activity"].lower() == "active":
                assay_active_counts[r["assay_id"]] += 1

    # --- Step 3: match assays to pathogens ---
    # Primary: IRI match via core_assayparameter (for pathogens in PATHOGEN_IRIS).
    # Fallback: keyword match on assay name, description, and target organism names.
    print("\nMatching assays to pathogens ...")

    rows = []
    for assay in assays:
        assay_id = assay["id"]
        assay_name = assay.get("assay_name", "")
        assay_desc = assay.get("assay_description", "")
        intended_target_id = assay.get("intended_target_id")
        deprecated = assay.get("deprecated")

        param_iris = assay_param_iris.get(assay_id, set())
        param_labels = assay_param_labels.get(assay_id, set())

        target_organism_names: set[str] = set()
        if intended_target_id:
            target = targets.get(intended_target_id)
            if target:
                org_spec = target.get("organism_specification", "")
                if org_spec:
                    target_organism_names.add(org_spec)
            target_organism_names |= target_to_organisms.get(intended_target_id, set())

        all_organism_names = param_labels | target_organism_names
        combined_text = " ".join([assay_name, assay_desc, " ".join(all_organism_names)])

        for pathogen in pathogens:
            pname = pathogen["pathogen"]
            pcode = pathogen["code"]

            pathogen_iris_set = set(PATHOGEN_IRIS.get(pcode, []))
            if pathogen_iris_set:
                matched = bool(pathogen_iris_set & param_iris)
            else:
                matched = text_matches(combined_text, pathogen_keywords[pname])

            if matched:
                rows.append({
                    "pathogen": pname,
                    "pathogen_code": pcode,
                    "assay_eos_id": f"EOS{assay.get('eos_ptr_id')}",
                    "assay_name": assay_name,
                    "assay_description": assay_desc,
                    "embargo_until": assay.get("embargo_until"),
                    "activity_determination": assay.get("activity_determination"),
                    "aria_project_id": assay.get("aria_project_id"),
                    "intended_target_eos_id": f"EOS{intended_target_id}" if intended_target_id else None,
                    "deprecated": deprecated,
                    "n_compounds_tested": assay_compound_counts.get(assay_id, 0),
                    "n_compounds_active": assay_active_counts.get(assay_id, 0),
                    "target_organism_names": "; ".join(sorted(all_organism_names)),
                })

    # --- Step 4: write summary and export per-assay files ---
    fieldnames = [
        "pathogen", "pathogen_code", "assay_eos_id", "assay_name",
        "n_compounds_tested", "n_compounds_active",
        "embargo_until", "assay_description", "activity_determination",
        "aria_project_id", "intended_target_eos_id", "deprecated", "target_organism_names",
    ]
    rows.sort(key=lambda r: r["pathogen"])
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone. {len(rows)} assay-pathogen matches written to {OUTPUT_CSV}")

    print("\nSummary by pathogen:")
    print(f"{'Pathogen':<35} {'Assays':>7} {'Compounds':>10} {'Active':>8}")
    print("-" * 65)
    by_pathogen: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pathogen[r["pathogen"]].append(r)
    for p in pathogens:
        pname = p["pathogen"]
        prows = by_pathogen[pname]
        print(f"{pname:<35} {len(prows):>7} {sum(r['n_compounds_tested'] for r in prows):>10} {sum(r['n_compounds_active'] for r in prows):>8}")

    export_assay_files(rows, tables, ASSAYS_DIR)


if __name__ == "__main__":
    main()
