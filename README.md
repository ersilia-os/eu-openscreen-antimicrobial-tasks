This project has been financed by Project PID2023-148309OA-I00 funded by MICIU/AEI/10.13039/501100011033 and by ERDF, EU.

<img src="https://raw.githubusercontent.com/ersilia-os/ersilia/master/assets/miciu_cofinanciado.jpg" width="300">

# EU OpenScreen Antimicrobial Screening Analysis

Analysis of antimicrobial phenotypic screening data from the [EU OpenScreen](https://www.eu-openscreen.eu/) ECBD database for seven ESKAPE+ pathogens: *A. baumannii*, *C. albicans*, *E. coli*, *E. faecalis*, *K. pneumoniae*, *P. aeruginosa*, and *S. aureus*.

The goal is to produce clean, binarised, per-pathogen compound activity datasets.

## Getting started

```bash
git clone https://github.com/ersilia-os/eu-openscreen-antimicrobial-tasks
cd eu-openscreen-antimicrobial-tasks
pip install -r requirements.txt
```

Data is tracked with [eosvc](https://github.com/ersilia-os/eosvc) (linked to S3), not git. To fetch it:

```bash
eosvc download --path data/
eosvc download --path output/
```

## Running the analysis

Run scripts in order from the `scripts/` directory. See `scripts/README.md` for details on each step.

```bash
python scripts/00_extract_assays.py
python scripts/01_fetch_new_assays.py
python scripts/02_binarise_and_merge.py
```

## Key outputs

| File | Description |
|---|---|
| `output/00_extract_assays/00_assay_summary.csv` | Assay metadata extracted from the SQL dump |
| `output/01_fetch_new_assays/01_assay_summary.csv` | Assay metadata fetched from the ECBD API |
| `output/02_binarise_and_merge/02_pathogens_summary.csv` | Per-pathogen molecule and active counts |
| `data/processed/02_merged/02_<pathogen>.csv` | Binarised, deduplicated activity data per pathogen |
| `data/processed/02_merged/02_all_smiles.csv` | All unique SMILES across pathogens |

## About Ersilia

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit building open-source AI/ML tools for antimicrobial drug discovery in the Global South. Its main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia).

![Ersilia Logo](assets/Ersilia_Brand.png)
