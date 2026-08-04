"""
Step 07a — Build the training sets and task metadata for model training.

Run ONCE on the login node before submitting the SLURM array job (step 07b). Every array
task reads the metadata CSV this produces; if each task rebuilt it they would race on the
same files and corrupt them.

Fourteen tasks are prepared — for each of the seven pathogens:

  primary    the single full-library screen listed in data/config/primary_assays_manual.csv
             (data/processed/02_binarised_assays/{assay_eos_id}.csv, ~101K compounds)
  secondary  all non-primary assays merged per pathogen
             (output/06_subset_data/secondary/{pathogen}_secondary.csv, ~5.3K compounds)

For each task: duplicate SMILES are collapsed keeping the highest bin (the rule already used
by 02_binarise_and_merge.py) and inconclusive compounds (bin = -1) are excluded. The exact
data each model will be trained on is materialised so it can be audited, and the row counts
removed at each step are recorded in the metadata.

Compounds are NOT resampled: the primary screens are 0.01-0.4% active and are passed to
LazyQSAR at their true prevalence, which the reported AUPRC/BEDROC baselines make explicit.

Finally the ready-to-paste sbatch commands are printed, splitting tasks by compound count so
the large ones get more memory (mirrors 08_download_weights.py in chembl-antimicrobial-models).

Usage:
    python scripts/07a_prepare_datasets.py
"""

import os
import sys

import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

from default import (  # noqa: E402
    BIN_ACTIVE,
    BIN_INACTIVE,
    BIN_INCONCLUSIVE,
    COL_BIN,
    COL_INCHIKEY,
    COL_SMILES,
    LARGE_TASK_COMPOUNDS,
    N_FOLDS,
    PATHOGEN_LABELS,
    TASKS,
)

repo_root = os.path.abspath(os.path.join(root, ".."))

config_dir = os.path.join(root, "..", "data", "config")
binarised_dir = os.path.join(root, "..", "data", "processed", "02_binarised_assays")
merged_dir = os.path.join(root, "..", "data", "processed", "02_merged")
secondary_dir = os.path.join(root, "..", "output", "06_subset_data", "secondary")

output_dir = os.path.join(root, "..", "output", "07_train_models")
datasets_dir = os.path.join(output_dir, "datasets")
logs_dir = os.path.join(root, "..", "output", "07_logs")

os.makedirs(datasets_dir, exist_ok=True)
# SLURM will not create the directory for --output/--error and the job fails if it is
# missing, so it must exist before the array is submitted.
os.makedirs(logs_dir, exist_ok=True)

PRIMARY_ASSAYS_CSV = os.path.join(config_dir, "primary_assays_manual.csv")
ALL_SMILES_CSV = os.path.join(merged_dir, "02_all_smiles.csv")
METADATA_CSV = os.path.join(output_dir, "07_tasks_metadata.csv")

DATASET_COLS = [COL_SMILES, COL_INCHIKEY, COL_BIN]

METADATA_COLS = [
    "pathogen",
    "pathogen_label",
    "task",
    "source_file",
    "assay_eos_id",
    "n_rows_source",
    "n_duplicates_collapsed",
    "n_inconclusive_excluded",
    "compounds",
    "positives",
    "inactives",
    "ratio",
    "trainable",
]


def load_primary_assay_map() -> dict[str, str]:
    """Return {pathogen_code: assay_eos_id} for the primary-screen assay."""
    df = pd.read_csv(PRIMARY_ASSAYS_CSV)
    return dict(zip(df["pathogen_code"], df["assay_eos_id"]))


def load_inchikey_lookup() -> dict[str, str]:
    """Return {smiles: inchikey} from the global deduplicated SMILES file."""
    df = pd.read_csv(ALL_SMILES_CSV)
    return dict(zip(df[COL_SMILES], df[COL_INCHIKEY]))


def source_path(pathogen: str, task: str, primary_map: dict[str, str]) -> str:
    if task == "primary":
        return os.path.join(binarised_dir, f"{primary_map[pathogen]}.csv")
    return os.path.join(secondary_dir, f"{pathogen}_secondary.csv")


def build_dataset(path: str, inchikey_lookup: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """
    Load a source assay file and return (training dataframe, counts).

    Duplicate SMILES are collapsed keeping the highest bin, and inconclusive compounds
    (bin = -1) are dropped. Counts report what each step removed.
    """
    df = pd.read_csv(path)
    n_rows_source = len(df)

    invalid = sorted(set(df[COL_BIN].unique()) - {BIN_ACTIVE, BIN_INACTIVE, BIN_INCONCLUSIVE})
    if invalid:
        raise ValueError(f"{os.path.basename(path)} contains unexpected bin values: {invalid}")

    # Collapse duplicate SMILES, active prevailing (same rule as 02_binarise_and_merge.py).
    has_inchikey = COL_INCHIKEY in df.columns
    agg = {COL_BIN: "max"}
    if has_inchikey:
        agg[COL_INCHIKEY] = "first"
    collapsed = df.groupby(COL_SMILES, as_index=False, sort=True).agg(agg)
    if not has_inchikey:
        collapsed[COL_INCHIKEY] = collapsed[COL_SMILES].map(inchikey_lookup).fillna("")

    n_inconclusive = int((collapsed[COL_BIN] == BIN_INCONCLUSIVE).sum())
    dataset = collapsed[collapsed[COL_BIN] != BIN_INCONCLUSIVE][DATASET_COLS]
    dataset = dataset.reset_index(drop=True)

    n_positives = int((dataset[COL_BIN] == BIN_ACTIVE).sum())
    counts = {
        "n_rows_source": n_rows_source,
        "n_duplicates_collapsed": n_rows_source - len(collapsed),
        "n_inconclusive_excluded": n_inconclusive,
        "compounds": len(dataset),
        "positives": n_positives,
        "inactives": len(dataset) - n_positives,
        "ratio": round(n_positives / len(dataset), 6) if len(dataset) else 0.0,
    }
    return dataset, counts


def build_all_datasets() -> pd.DataFrame:
    """Materialise every training set and return the task metadata table."""
    primary_map = load_primary_assay_map()
    inchikey_lookup = load_inchikey_lookup()

    records = []
    for pathogen in sorted(PATHOGEN_LABELS):
        for task in TASKS:
            path = source_path(pathogen, task, primary_map)
            if not os.path.exists(path):
                print(f"[WARN] {pathogen}/{task}: source file not found ({path}) — skipping")
                continue

            dataset, counts = build_dataset(path, inchikey_lookup)

            out_path = os.path.join(datasets_dir, pathogen, f"{task}.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            dataset.to_csv(out_path, index=False)

            trainable = min(counts["positives"], counts["inactives"]) >= N_FOLDS
            records.append(
                {
                    "pathogen": pathogen,
                    "pathogen_label": PATHOGEN_LABELS[pathogen],
                    "task": task,
                    "source_file": os.path.relpath(path, repo_root),
                    "assay_eos_id": primary_map[pathogen] if task == "primary" else "",
                    "trainable": trainable,
                    **counts,
                }
            )
            print(
                f"  {pathogen}/{task}: {counts['compounds']} compounds "
                f"({counts['positives']} active, ratio {counts['ratio']}) "
                f"[{counts['n_duplicates_collapsed']} duplicate rows collapsed, "
                f"{counts['n_inconclusive_excluded']} inconclusive excluded]"
                f"{'' if trainable else '  [NOT TRAINABLE]'}"
            )

    return pd.DataFrame(records, columns=METADATA_COLS)


def _to_array_spec(indices: list[int]) -> str:
    """Compress a sorted index list into SLURM array syntax (0-3,7,9-11)."""
    if not indices:
        return ""
    parts, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def print_sbatch_commands(metadata: pd.DataFrame) -> None:
    """Print the sbatch commands, splitting by compound count so big tasks get more memory."""
    script_path = os.path.join("scripts", "07b_train_models.sh")
    trainable = metadata[metadata["trainable"]]
    small = sorted(trainable.index[trainable["compounds"] < LARGE_TASK_COMPOUNDS].tolist())
    large = sorted(trainable.index[trainable["compounds"] >= LARGE_TASK_COMPOUNDS].tolist())

    skipped = metadata.index[~metadata["trainable"]].tolist()
    if skipped:
        print(f"\n{len(skipped)} task(s) are not trainable and are excluded: {skipped}")

    print("\nSubmit the training array with:\n")
    if small:
        print(f"    sbatch --chdir={repo_root} --job-name=eos-lq-sm "
              f"--array={_to_array_spec(small)}%7 --mem=16G {script_path}")
    if large:
        print(f"    sbatch --chdir={repo_root} --job-name=eos-lq-lg "
              f"--array={_to_array_spec(large)}%4 --mem=64G {script_path}")
    print(f"\n(small = <{LARGE_TASK_COMPOUNDS:,} compounds, large = >=that. The %N suffix caps")
    print(" how many array tasks run at once. Adjust --mem if jobs are OOM-killed.)")


def main() -> None:
    print(f"Building training sets for {len(PATHOGEN_LABELS)} pathogens x {len(TASKS)} tasks")
    metadata = build_all_datasets()
    metadata.to_csv(METADATA_CSV, index=False)
    print(f"\nTask metadata ({len(metadata)} tasks) -> {os.path.relpath(METADATA_CSV, repo_root)}")
    print(f"Training sets            -> {os.path.relpath(datasets_dir, repo_root)}/<pathogen>/<task>.csv")
    print(f"SLURM log directory      -> {os.path.relpath(logs_dir, repo_root)}")
    print_sbatch_commands(metadata)


if __name__ == "__main__":
    main()
