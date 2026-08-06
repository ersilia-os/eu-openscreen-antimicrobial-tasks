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
    "efaecalis",
    "kpneumoniae",
    "paeruginosa",
    "saureus",
]

# "efaecalis" was originally coded as "efaecium": E. faecium was the intended target, but ECBD
# holds no data for it, so every assay behind this code is E. faecalis ATCC 29212
# (NCBITaxon_1351). Code and label were renamed to name the organism actually screened.
# See 00_extract_assays.py and scripts/README.md.
PATHOGEN_LABELS = {
    "abaumannii": "A. baumannii",
    "calbicans": "C. albicans",
    "ecoli": "E. coli",
    "efaecalis": "E. faecalis",
    "kpneumoniae": "K. pneumoniae",
    "paeruginosa": "P. aeruginosa",
    "saureus": "S. aureus",
}

# --- Cluster submission (07a/07b) ---
TASKS = ["primary", "secondary"]
# Tasks at or above this compound count are submitted as a separate SLURM array with more
# memory. Splits the 14 tasks into the ~5K secondary sets and the ~101K primary screens.
LARGE_TASK_COMPOUNDS = 30000

# --- Plotting (08_model_reports) ---
# stylia NamedColors names, one per pathogen. Kept consistent with scripts 03, 05 and 06,
# which each define the same mapping locally.
PATHOGEN_COLORS = {
    "abaumannii": "crimson",
    "calbicans": "turquoise",
    "ecoli": "cobalt",
    "efaecalis": "tangerine",
    "kpneumoniae": "amber",
    "paeruginosa": "orchid",
    "saureus": "periwinkle",
}
# Maximum points drawn per group in the rank-score scatter. Display only: the primary screens
# pool ~500K inactive points across folds, which is unreadable and slow to render. Boxplot
# statistics are always computed on every point.
SCATTER_MAX_POINTS = 5000
