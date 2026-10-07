"""
Step 07b — Train a LazyQSAR model for one task.

Designed to run as a SLURM array job via 07b_train_models.sh, one array task per row in
output/07_train_models/07_tasks_metadata.csv (written by 07a_prepare_datasets.py).

For the dataset identified by <task_id>:
  1. Loads output/07_train_models/datasets/{pathogen}/{task}.csv
  2. Runs N_FOLDS-fold stratified cross-validation, recording per-fold AUROC, AUPRC and
     BEDROC against their random baselines plus the per-descriptor out-of-fold AUCs, in
     output/07_train_models/reports/{pathogen}/{task}.csv (+ _folds.json with raw arrays)
  3. Fits a final model on all the data and saves it to
     output/07_train_models/models/{pathogen}/{task}/

The recipe mirrors ersilia-os/chembl-antimicrobial-models (scripts/09_run_models.py) so the
per-fold reports of the two repositories share a schema and can be compared directly. It
differs deliberately in two places, both to survive preemption on a spot partition:
  - it skips a task whose report and model already exist, so a requeued array task does not
    redo a finished multi-hour fit from scratch;
  - it caps the BLAS/OpenMP/XGBoost thread count to the allocated CPUs, so a task cannot
    oversubscribe the node it lands on.

Tasks whose minority class is smaller than N_FOLDS cannot support stratified CV and are
skipped, producing neither a report nor a model.

Usage:
    python scripts/07b_train_models.py <task_id>       # 0-based row in the metadata CSV
    python scripts/07b_train_models.py --list          # show the task_id of every task
"""

import os

# Thread caps must be set before numpy/xgboost/torch are imported, or the libraries read
# the machine's full core count and oversubscribe the SLURM allocation. setdefault lets
# 07b_train_models.sh override these.
_threads = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _threads)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lazyqsar.ensemble.runner import get_chunk_size, persist_descriptors  # noqa: E402
from lazyqsar.qsar import LazyClassifierQSAR, validate_smiles  # noqa: E402
from lazyqsar.registry import DESCRIPTORS_MODE, get_descriptor_type  # noqa: E402
from lazyqsar.utils.metrics import bedroc_random_baseline, bedroc_score  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))

from default import (  # noqa: E402
    COL_BIN,
    COL_SMILES,
    DESCRIPTORS,
    LAZYQSAR_MODE,
    N_FOLDS,
    RANDOM_SEED,
)

repo_root = os.path.abspath(os.path.join(root, ".."))

output_dir = os.path.join(root, "..", "output", "07_train_models")
datasets_dir = os.path.join(output_dir, "datasets")
reports_dir = os.path.join(output_dir, "reports")
models_dir = os.path.join(output_dir, "models")

METADATA_CSV = os.path.join(output_dir, "07_tasks_metadata.csv")


def load_metadata() -> pd.DataFrame:
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(
            f"{os.path.relpath(METADATA_CSV, repo_root)} not found — "
            "run scripts/07a_prepare_datasets.py first."
        )
    return pd.read_csv(METADATA_CSV)


def stage_descriptors(smiles: list, scratch: str) -> dict:
    """Featurize every compound once into an .npy per descriptor; return their paths.

    LazyClassifierQSAR.fit() otherwise featurizes from scratch on every call and holds
    each candidate descriptor of the portfolio in memory at once. For a task whose
    minority class forces many batches — paeruginosa/primary is 14 actives in 101,022
    compounds, so 73 batches per descriptor — six such fits exceeded the 96 GB of the
    machine and the OS killed the process twice. Staging to disk and handing fit() the
    slice it needs keeps one copy on disk and a memmapped view per fold instead.

    Reuses an existing .npy, so a requeued task does not refeaturize.
    """
    os.makedirs(scratch, exist_ok=True)
    chunk_size = get_chunk_size()
    paths = {}
    for name in DESCRIPTORS_MODE[LAZYQSAR_MODE]:
        path = os.path.join(scratch, f"{name}.npy")
        if not os.path.exists(path):
            print(f"  staging descriptor: {name}", flush=True)
            descriptor = get_descriptor_type(name)()
            persist_descriptors(descriptor, smiles, path, chunk_size)
            del descriptor
        paths[name] = path
    return paths


def precomputed_for(staged: dict, rows: list) -> dict:
    """The staged descriptor rows for one fold, read through a memmap."""
    return {
        name: np.load(path, mmap_mode="r")[rows] for name, path in staged.items()
    }


def num_batches(model) -> float:
    """
    Number of internal sub-models LazyQSAR fitted for the first descriptor, or NaN.

    LazyQSAR (3.4.2 to 3.6.0) exposes this only as a per-descriptor dict written into the saved
    model's metadata.json, with no public attribute on the fitted object, so this reads
    the same private chain chembl-antimicrobial-models uses. Guarded because that chain
    is fragile across LazyQSAR versions, and a missing value must not abort a fold.
    """
    try:
        return len(model.models[0]._model.models) if model.models else np.nan
    except (AttributeError, IndexError, TypeError):
        return np.nan


def run_cv(smiles: list, y: list, pathogen: str, task: str, staged: dict = None) -> None:
    """N_FOLDS-fold stratified CV. Writes the per-fold report CSV and the raw fold arrays."""
    records = []
    fold_data = {}
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, test_idx) in enumerate(kf.split(smiles, y)):
        smiles_train = [smiles[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        smiles_test = [smiles[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        model = LazyClassifierQSAR(mode=LAZYQSAR_MODE)
        if staged is None:
            model.fit(smiles_list=smiles_train, y=y_train)
        else:
            model.fit(smiles_train, y_train,
                      precomputed=precomputed_for(staged, list(train_idx)), validate=False)
        scores_proba = model.predict_proba(smiles_list=smiles_test)[:, 1]
        scores_rank = model.predict_rank(smiles_list=smiles_test)[:, 1]

        # Metrics are computed on the rank scores, as in chembl-antimicrobial-models.
        auroc = roc_auc_score(y_test, scores_rank)
        auprc = average_precision_score(y_test, scores_rank)
        baseline_auroc = 0.5
        baseline_auprc = sum(y_test) / len(y_test)
        bedroc = bedroc_score(np.array(y_test), scores_rank)
        baseline_bedroc = bedroc_random_baseline(np.array(y_test))

        oof_auc_map = dict(zip(model.descriptor_types, model.oof_aucs_))
        oof_per_descriptor = {
            f"oof_auc_{desc}": round(oof_auc_map[desc], 4) if desc in oof_auc_map else np.nan
            for desc in DESCRIPTORS
        }

        records.append(
            {
                "pathogen": pathogen,
                "name": task,
                "model_name": task,
                "fold": fold,
                "compounds_train": len(y_train),
                "compounds_test": len(y_test),
                "positives_train": sum(y_train),
                "positives_test": sum(y_test),
                "auroc": round(auroc, 4),
                "auprc": round(auprc, 4),
                "baseline_auroc": baseline_auroc,
                "baseline_auprc": round(baseline_auprc, 4),
                "bedroc": round(bedroc, 4),
                "baseline_bedroc": round(baseline_bedroc, 4),
                "num_batches": num_batches(model),
                **oof_per_descriptor,
            }
        )

        fold_data[str(fold)] = {
            "y_true": y_test,
            "y_hat": scores_proba.tolist(),
            "y_rank": scores_rank.tolist(),
            "roc_auc": round(auroc, 4),
        }

        print(
            f"  fold {fold}: auroc={auroc:.3f}  auprc={auprc:.3f}  bedroc={bedroc:.3f}  "
            f"(baseline auprc={baseline_auprc:.4f}  baseline bedroc={baseline_bedroc:.3f})",
            flush=True,
        )

    report_dir = os.path.join(reports_dir, pathogen)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{task}.csv")
    folds_path = os.path.join(report_dir, f"{task}_folds.json")

    pd.DataFrame(records).to_csv(report_path, index=False)
    with open(folds_path, "w") as f:
        json.dump(fold_data, f)
    print(f"  Report saved: {os.path.relpath(report_path, repo_root)}")
    print(f"  Folds saved:  {os.path.relpath(folds_path, repo_root)}")


def run(task_id: int, disk_backed: bool = False) -> None:
    metadata = load_metadata()
    if not 0 <= task_id < len(metadata):
        raise IndexError(
            f"task_id {task_id} out of range — the metadata CSV has {len(metadata)} tasks "
            f"(valid ids 0-{len(metadata) - 1})."
        )
    row = metadata.iloc[task_id]
    pathogen, task = row["pathogen"], str(row["task"])

    print(f"[{task_id}] {pathogen}/{task} — {row['compounds']} compounds, "
          f"{row['positives']} active (ratio {row['ratio']})")
    print(f"      mode={LAZYQSAR_MODE!r}  folds={N_FOLDS}  seed={RANDOM_SEED}  "
          f"threads={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    report_path = os.path.join(reports_dir, pathogen, f"{task}.csv")
    model_path = os.path.join(models_dir, pathogen, task)
    report_done = os.path.exists(report_path)
    model_done = os.path.exists(os.path.join(model_path, "metadata.json"))

    # Idempotency guard: the array runs on a preemptible partition with --requeue, so a
    # requeued task must not repeat a fit that already finished.
    if report_done and model_done:
        print("  Report and model already exist — nothing to do")
        return

    if not row["trainable"]:
        print(f"  [SKIP] minority class {min(row['positives'], row['inactives'])} < {N_FOLDS} "
              f"folds ({row['positives']} active, {row['inactives']} inactive) — not trainable")
        return

    dataset = pd.read_csv(os.path.join(datasets_dir, pathogen, f"{task}.csv"))
    smiles = dataset[COL_SMILES].tolist()
    y = dataset[COL_BIN].astype(int).tolist()

    staged = None
    if disk_backed:
        scratch = os.environ.get("LAZYQSAR_FIT_SCRATCH") or os.path.join(
            output_dir, "scratch", pathogen, task)
        print(f"  disk-backed fit, scratch: {scratch}", flush=True)
        validate_smiles(smiles)
        staged = stage_descriptors(smiles, scratch)

    if report_done:
        print("  Report exists — skipping CV")
    else:
        run_cv(smiles, y, pathogen, task, staged=staged)

    if model_done:
        print("  Final model exists — skipping")
    else:
        print("  Training final model on all data", flush=True)
        model = LazyClassifierQSAR(mode=LAZYQSAR_MODE)
        if staged is None:
            model.fit(smiles_list=smiles, y=y)
        else:
            model.fit(smiles, y, precomputed=precomputed_for(staged, list(range(len(smiles)))),
                      validate=False)
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path)
        print(f"  Model saved:  {os.path.relpath(model_path, repo_root)}")

    print("Done.")


def list_tasks() -> None:
    metadata = load_metadata()
    print(f"{'id':>3}  {'pathogen':<12} {'task':<10} {'compounds':>10} {'actives':>8}  trainable")
    for i, r in metadata.iterrows():
        print(f"{i:>3}  {r['pathogen']:<12} {r['task']:<10} {r['compounds']:>10,} "
              f"{r['positives']:>8,}  {bool(r['trainable'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LazyQSAR model for one EU OpenScreen task (SLURM array entry point)."
    )
    parser.add_argument(
        "task_id",
        nargs="?",
        type=int,
        help="0-based row index into output/07_train_models/07_tasks_metadata.csv",
    )
    parser.add_argument(
        "--list", action="store_true", help="List every task with its task_id and exit"
    )
    parser.add_argument(
        "--disk-backed", action="store_true",
        help="Featurize once to .npy and memmap each fold's slice, instead of letting "
             "fit() featurize in memory every call. Much lower peak RAM; needed for "
             "tasks whose imbalance forces many batches.",
    )
    args = parser.parse_args()

    if args.list:
        list_tasks()
    elif args.task_id is None:
        parser.error("a task_id is required (or use --list)")
    else:
        run(args.task_id, disk_backed=args.disk_backed)
