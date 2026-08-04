# Scripts

Run in numbered order. Each script writes its outputs to a dedicated subfolder under `output/` and/or `data/processed/`.

## Setup

Install dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

`lazyqsar` pins `numpy` and `scikit-learn` to versions that conflict with `ersilia`, so keep
`04_ersilia_models.sh` in a separate environment from the rest of the pipeline.

### Cluster setup for model training (step 07b)

Model training runs as a SLURM array on the IRB cluster. Create the environment **inside the
repository** — compute nodes only see the shared filesystem, so it cannot live in conda's
default path. This mirrors `chembl-antimicrobial-models` (`envs/camm`); `envs/` is gitignored.

```bash
conda create --prefix ./envs/eoat python=3.12 -y
conda activate ./envs/eoat
pip install --ignore-installed -r requirements.txt
```

The explicit `pip install` is needed because conda's pip integration does not always resolve
LazyQSAR's transitive dependencies.

Then download the descriptor checkpoints (~610 MB) under the same `HOME` the jobs will use —
see the step 07b notes below for why:

```bash
HOME="$(pwd)/tmp/lazyqsar_home" lazyqsar setup --descriptors
```

XGBoost also needs the OpenMP runtime. It is present on the cluster; on macOS install it with
`brew install libomp`.

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
| `data/config/assays_annotated_manual.csv` | Hand-annotated reference sheet combining `00_assay_summary` and `01_assay_summary` with assay type (gi/ic50), active cutoff, and single-point concentration. Used by `06_subset_data.py` to group assays per pathogen. |
| `data/config/primary_assays_manual.csv` | One manually selected primary-screen assay per pathogen (the assay covering the full ~101K-compound library). Used by `03_processing_statistics.py`, `05_abx_similarity.py`, `06_subset_data.py` and `07a_prepare_datasets.py`. |

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

Produces three visualisations using only the 7 primary-screen assays — one per target pathogen — from `data/config/primary_assays_manual.csv`.

1. **Active counts** (`03_active_counts.png`): horizontal bar chart with log x-axis showing compounds tested (faded) vs active (solid) per pathogen.
2. **Co-active counts** (`03_coactive_counts.png`): 7×7 heatmap of pairwise co-active compound counts. Off-diagonal cells use a crimson gradient; diagonal cells show per-pathogen colour annotated with n_active.
3. **Breadth + exclusivity** (`03_breadth_exclusivity.png`): spectrum-breadth distribution (narrow / medium / broad) alongside binary exclusivity (exclusive vs shared actives) per pathogen.

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

## 07a_prepare_datasets.py

Builds the training set for each of 14 endpoints — for every pathogen, the **primary** full-library screen (`data/processed/02_binarised_assays/{assay_eos_id}.csv`, ~101K compounds) and the **secondary** merged set of all its non-primary assays (`output/06_subset_data/secondary/{pathogen}_secondary.csv`, ~5.3K compounds).

Writes each task's exact training data to `output/07_train_models/datasets/{pathogen}/{task}.csv`, the task table to `output/07_train_models/07_tasks_metadata.csv` (recording how many rows each cleaning step removed), creates `output/07_logs/` for SLURM, and prints the ready-to-paste `sbatch` commands.

**Run this once on the login node before submitting step 07b.** Each array task reads the metadata CSV by row index; if every task rebuilt it they would race on the same files.

## 07b_train_models.py / 07b_train_models.sh

Trains one [LazyQSAR](https://github.com/ersilia-os/lazy-qsar) binary classifier per task, as a SLURM array job — one array index per row of the metadata CSV. Runs cross-validation (`reports/{pathogen}/{task}.csv` plus `_folds.json` with the raw per-fold score arrays) and fits a final model on all data (`models/{pathogen}/{task}/`, an ONNX directory).

The recipe deliberately mirrors [chembl-antimicrobial-models](https://github.com/ersilia-os/chembl-antimicrobial-models) script 09, and the per-fold report schema is identical, so results from the two repositories are directly comparable. It differs in two respects, both deliberate hardening for a preemptible partition:

- **Idempotent.** A task whose report *and* model already exist exits immediately. The reference has no such guard, so with `--requeue` on a spot partition a preempted 20-hour fit restarts from zero.
- **Thread-capped.** `OMP_NUM_THREADS` and the BLAS equivalents are pinned to `SLURM_CPUS_PER_TASK`. Left unset, XGBoost, PyTorch and OpenBLAS each default to the node's full core count, so concurrent array tasks oversubscribe whatever node they land on.

```bash
python scripts/07a_prepare_datasets.py            # once, on the login node
python scripts/07b_train_models.py --list         # task_id of every task
sbatch --chdir=<repo> --array=0-13%7 --mem=16G scripts/07b_train_models.sh
python scripts/07b_train_models.py 11             # or run one task directly
```

The `.sh` is configured for the IRB cluster, matching the reference repo: `--partition=spot_cpu`, `--nodelist=irbccn16,irbccn41,irbccn42`, the `/aloy` Singularity bind paths, and `envs/eoat/bin/python`. `spot_cpu` is preemptible, hence `--requeue` — cheap here because a requeued task skips any fit that already finished.

**Descriptor checkpoints.** LazyQSAR resolves them from `$HOME/.lazyqsar`, a path hardcoded in the library, so `07b_train_models.sh` overrides `HOME` to relocate them into the repository where the compute nodes can see them — the same trick the reference repo uses. It points at `tmp/lazyqsar_home` rather than the reference's `output/08_weights`, because these ~610 MB of checkpoints are regenerable and `output/` is eosvc-tracked. Download them under the same `HOME` the jobs will use:

```bash
HOME="$(pwd)/tmp/lazyqsar_home" lazyqsar setup --descriptors
```

`lazyqsar setup --target-dir DIR` does **not** work for this — it changes where the checkpoints are written but not where the descriptors look for them. Note also that overriding `HOME` redirects every other tool that honours it (matplotlib, torch hub, pip), so that directory accumulates their caches too.

**Inconclusive compounds:** `bin = -1` rows are excluded from training, consistent with scripts 03 and 05. The number excluded per task is recorded in `n_inconclusive_excluded` (0–242 for primary assays, 0–141 for secondary sets) rather than being dropped silently.

**Duplicate SMILES:** collapsed keeping the highest bin, the same rule `02_binarise_and_merge.py` uses. This affects only 0–7 rows per primary assay; the secondary sets are already unique.

**No resampling.** The primary screens are 0.014%–0.375% active and are trained at their true prevalence — negatives are never subsampled and no decoys are added. The reported `baseline_auprc` (fold prevalence) and `baseline_bedroc` make that prevalence explicit, so AUROC/AUPRC/BEDROC must always be read against them.

**Trainability cutoff:** a task is skipped when its minority class holds fewer than `N_FOLDS` (5) compounds, since stratified 5-fold CV is then impossible. All 14 current tasks pass, but *P. aeruginosa* primary is borderline at **14 actives** (~3 per test fold) — its metrics carry very large variance and should not be compared at face value against, say, *S. aureus* primary (378 actives).

**Cross-validation:** 5-fold `StratifiedKFold`, `RANDOM_SEED = 42`. AUROC, AUPRC and BEDROC (alpha 20) are all computed on LazyQSAR's **rank** scores, as in the reference repository; the probability scores are kept in the folds JSON only.

**Reports are not bit-reproducible.** `RANDOM_SEED` fixes the fold split, but LazyQSAR's internal model search (XGBoost / RandomForest / FLAML) is not fully seeded, so two identical runs give slightly different metrics. Measured across three runs of `paeruginosa/secondary` on identical input:

| metric | run-to-run spread |
|---|---|
| `auroc` | ±0.0004 |
| `auprc` | ±0.0002 |
| `bedroc` | ±0.0010 |

The per-descriptor `oof_auc_*` columns were identical across all runs, confirming the variation arises in model fitting and not in featurisation. BEDROC is the most affected because with very few actives per test fold (7 of 1,055 for this task) a single rank swap near the top moves it in the third decimal. **Do not read differences of this magnitude between models as a real effect.** This behaviour is inherited from `chembl-antimicrobial-models`, which seeds identically, so it is not a divergence between the two repositories.

**Model choice:** `LAZYQSAR_MODE = "slow"` — the full descriptor portfolio (cddd, chemeleon, clamp, morgan, rdkit) rather than `"fast"` (Morgan only), matching the reference repository. Constants live in `src/default.py`.

**Cost and memory.** `mode="slow"` computes five descriptor sets, three of them deep-learning encoders, and descriptor computation dominates training time — measured here, one 5.3K-compound secondary task took over an hour of CPU. Each task needs 6 fits (5 folds + 1 final) and each recomputes its descriptors from SMILES, so the ~101K-compound primary tasks are far more expensive again. `07a` therefore prints two `sbatch` commands, splitting at `LARGE_TASK_COMPOUNDS = 30000`: the 7 secondary tasks at `--mem=16G` and the 7 primary tasks at `--mem=64G`. Raise `--mem` if jobs are OOM-killed — a 101K × 2048 float32 descriptor matrix alone is ~830 MB, and several are held at once.