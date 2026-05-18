# Scripts

Run in numbered order. Outputs go to `output/` (summaries) and `data/processed/` (analysis-ready files).

## 00_extract_assays.py

Parses the EU OpenScreen PostgreSQL dump (`data/raw/ecbd_dump_public.zip`) without loading it fully into memory. Matches assays to each target pathogen — first by NCBI Taxonomy IRI via `core_assayparameter`, with keyword fallback on assay name, description, and target fields — and writes a summary CSV and one per-compound CSV per assay to `data/raw/00_extracted_assays/`.

**Pathogen mapping:** E. coli includes both the species-level IRI (NCBITaxon_562) and the ATCC 25922 strain (NCBITaxon_1322345). P. aeruginosa is matched via the "group" entry (NCBITaxon_136841). The *E. faecium* entry is matched to *E. faecalis* (NCBITaxon_1351), the closest organism available in the database.

## 01_fetch_new_assays.py

Queries the ECBD REST API for assays linked to the same target EOS IDs as in `00_assay_summary.csv` but absent from the SQL dump. Per-compound IC50/activity data is retrieved via the frontend `ajax_data` endpoint; SMILES are fetched from the compound API. Embargoed assays receive a header-only placeholder CSV.

## 02_binarise_and_merge.py

Reads per-assay CSVs from both `00_extracted_assays/` and `01_extracted_assays/`, maps activity labels to binary scores, and merges by pathogen. Rows without a SMILES string are dropped.

**Activity mapping:** `active` → 1, `inactive` → 0, `inconclusive` → −1, `undefined` → −1.

**Deduplication:** when the same SMILES appears across multiple assays, the highest bin value is kept (Active > Inconclusive/Inactive). 