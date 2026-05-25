# Scripts

Run in numbered order. Each script writes its outputs to a dedicated subfolder under `output/` and/or `data/processed/`.

## Setup

Install dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

Raw data is tracked with [eosvc](https://github.com/ersilia-os/eosvc), not git. Fetch it before running any script:

```bash
eosvc download --path data/raw/
eosvc download --path data/config/
```

The required input files are:

| File | Description |
|---|---|
| `data/config/ecbd_dump_public.zip` | Full PostgreSQL dump of the EU OpenScreen ECBD database |
| `data/config/pathogens.csv` | List of target pathogens with their codes |
| `data/config/assays_annotated_manual.csv` | Hand-annotated reference sheet combining `00_assay_summary` and `01_assay_summary` with assay type (gi/ic50), active cutoff, and single-point concentration. Not used by any script — kept as a human-readable reference. |
| `data/config/primary_assays_manual.csv` | One manually selected primary-screen assay per pathogen (the assay covering the full ~101K-compound library). Used by `03_processing_statistics.py`. |

## 00_extract_assays.py

Parses the EU OpenScreen PostgreSQL dump (`data/config/ecbd_dump_public.zip`) without loading it fully into memory. Matches assays to each target pathogen — first by NCBI Taxonomy IRI via `core_assayparameter`, with keyword fallback on assay name, description, and target fields — and writes a summary CSV to `output/00_extract_assays/` and one per-compound CSV per assay to `data/raw/00_extracted_assays/`.

**Pathogen mapping:** E. coli includes both the species-level IRI (NCBITaxon_562) and the ATCC 25922 strain (NCBITaxon_1322345). P. aeruginosa is matched via the "group" entry (NCBITaxon_136841). The *E. faecium* entry is matched to *E. faecalis* (NCBITaxon_1351), the closest organism available in the database.

## 01_fetch_new_assays.py

Queries the ECBD REST API for assays linked to the same target EOS IDs as in `output/00_extract_assays/00_assay_summary.csv` but absent from the SQL dump. Per-compound IC50/activity data is retrieved via the frontend `ajax_data` endpoint; SMILES are fetched from the compound API. Embargoed assays receive a header-only placeholder CSV. Results go to `output/01_fetch_new_assays/`.

## 02_binarise_and_merge.py

Reads per-assay CSVs from both `data/raw/00_extracted_assays/` and `data/raw/01_extracted_assays/`, maps activity labels to binary scores, and merges by pathogen. Rows without a SMILES string are dropped. Per-pathogen files and the global SMILES list go to `data/processed/02_merged/`; the pathogen summary goes to `output/02_binarise_and_merge/`.

**Activity mapping:** `active` → 1, `inactive` → 0, `inconclusive` → −1, `undefined` → −1.

**Deduplication:** when the same SMILES appears across multiple assays, the highest bin value is kept (Active > Inconclusive/Inactive).

## 03_processing_statistics.py

Produces four visualisations using only the 7 primary-screen assays — one per target pathogen — from `data/config/primary_assays_manual.csv`.

1. **Active counts** (`03_active_counts.png`): horizontal bar chart with log x-axis showing compounds tested (faded) vs active (solid) per pathogen.
2. **Co-active counts** (`03_coactive_counts.png`): 7×7 heatmap of pairwise co-active compound counts. Off-diagonal cells use a crimson gradient; diagonal cells show per-pathogen colour annotated with n_active.
3. **UpSet plot** (`03_upset_plot.png`): multi-pathogen compound intersection chart.
4. **Breadth + exclusivity** (`03_breadth_exclusivity.png`): spectrum-breadth distribution (narrow / medium / broad) alongside binary exclusivity (exclusive vs shared actives) per pathogen.

All plots go to `output/03_processing_statistics/`. Existing PNGs in that folder are deleted before each run.

**Broad spectrum threshold:** ≥3 pathogens active (user-confirmed). Compounds active in exactly 1 pathogen are labelled "narrow", 2 are "medium", ≥3 are "broad spectrum".

**Inconclusive handling:** bin = −1 is excluded from all overlap calculations; only bin = 1 enters cross-pathogen metrics.

## 05_abx_similarity.py

Tests whether active compounds are more similar to known antibiotics than inactive compounds, using three Ersilia models. AUROC is used as the comparison metric — it measures how well a score discriminates actives from inactives across datasets of very different sizes. Each analysis is run on three dataset views: **merged** (all assays combined per pathogen), **primary** (one primary-screen assay per pathogen), and **assay** (all 41 individual binarised assay files). Inconclusive compounds (bin = −1) are excluded. Assays with fewer than 5 actives produce unreliable AUROCs and are shown as gray bars. All plot titles include the source model ID and dataset view in brackets.

**Models** — run via `04_ersilia_models.sh` on `data/processed/02_merged/02_only_smiles.csv`:
- `eos6ojg`: counts antibiotics with Tanimoto similarity ≥ threshold. AUROC computed at 5 thresholds (0.3, 0.5, 0.7, 0.9, 1.0) for merged data; at threshold 0.5 for individual assays. Both `_all` (full reference set) and `_subset` columns are analysed.
- `eos11sm`: single continuous `abx_score` ∈ [0, 1]. Single AUROC per pathogen (merged) and per assay.
- `eos2xeq`: binary flag `is_sim_known_ab` (1 = structurally similar to a known antibiotic). Single AUROC per pathogen (merged) and per assay.

## 06_subset_data.py

Creates per-pathogen subsets of active compounds for ML benchmarking. Reads the primary-assay binarised files and the secondary assay binarised files from `data/processed/02_binarised_assays/`.

**Exclusivity** (`output/06_subset_data/exclusivity/`): for each pathogen, splits primary-assay actives into those exclusive to that pathogen (`{pathogen}_exclusive.csv`) and those shared with at least one other pathogen's primary assay (`{pathogen}_nonexclusive.csv`).

**Secondary** (`output/06_subset_data/secondary/`): for each pathogen, concatenates all secondary assays (academic sub-screens, confirmatory/IC50) into a single labeled dataset (`{pathogen}_secondary.csv`, columns: smiles, inchikey, bin). Compounds appearing in multiple secondary assays are deduplicated with active prevailing. The primary assay is not re-copied here — it already exists at `data/processed/02_binarised_assays/{assay_eos_id}.csv`.

**ABX similarity** (`output/06_subset_data/abx_similarity/`): for each pathogen, splits the primary assay compounds into those structurally similar to known antibiotics (`{pathogen}_abx.csv`) and those that are not (`{pathogen}_non_abx.csv`), using the `is_sim_known_ab` flag from Ersilia model **eos2xeq** (`output/04_ersilia_models/eos2xeq.csv`). Both files include all compounds with their activity label (smiles, inchikey, bin).

A summary plot (`output/06_subset_data/06_abx_similarity.png`) shows for each pathogen: total abx vs non-abx compound counts (log scale, panel A) and the number of actives within each group (panel B).