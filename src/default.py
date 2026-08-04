"""
Project-wide constants.

Values are kept aligned with ersilia-os/chembl-antimicrobial-models (src/default.py)
so that training reports produced by the two repositories are directly comparable.
"""

# --- Column names ---
COL_SMILES = "smiles"
COL_INCHIKEY = "inchikey"
COL_BIN = "bin"

# --- Binarised activity encoding (see 02_binarise_and_merge.py) ---
BIN_ACTIVE = 1
BIN_INACTIVE = 0
BIN_INCONCLUSIVE = -1

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Model training (LazyQSAR) ---
N_FOLDS = 5
LAZYQSAR_MODE = "slow"  # "slow" = full descriptor portfolio; "fast" = Morgan only
DESCRIPTORS = ["cddd", "chemeleon", "clamp", "morgan", "rdkit"]

# --- Pathogens ---
# Only these seven of the 15 candidates in data/config/pathogens.csv yielded
# assay data in the EU OpenScreen ECBD database.
PATHOGEN_CODES = [
    "abaumannii",
    "calbicans",
    "ecoli",
    "efaecium",
    "kpneumoniae",
    "paeruginosa",
    "saureus",
]

# NOTE: the "efaecium" datasets are actually E. faecalis (NCBITaxon_1351), the closest
# organism available in ECBD. See 00_extract_assays.py and scripts/README.md.
PATHOGEN_LABELS = {
    "abaumannii": "A. baumannii",
    "calbicans": "C. albicans",
    "ecoli": "E. coli",
    "efaecium": "E. faecium",
    "kpneumoniae": "K. pneumoniae",
    "paeruginosa": "P. aeruginosa",
    "saureus": "S. aureus",
}

# --- Cluster submission (07a/07b) ---
TASKS = ["primary", "secondary"]
# Tasks at or above this compound count are submitted as a separate SLURM array with more
# memory. Splits the 14 tasks into the ~5K secondary sets and the ~101K primary screens.
LARGE_TASK_COMPOUNDS = 30000
