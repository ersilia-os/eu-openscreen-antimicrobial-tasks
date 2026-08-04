#!/bin/bash
# Step 07b — Train LazyQSAR models on the HPC cluster, one array task per dataset.
#
# Prepare once on the login node, then submit with the command 07a prints:
#     python scripts/07a_prepare_datasets.py
#     sbatch --chdir=<repo_root> --array=0-13%7 --mem=16G scripts/07b_train_models.sh
#
# All paths are relative to --chdir (the repository root).
#
# One-off setup, also on the login node — LazyQSAR reads its descriptor checkpoints from
# $HOME/.lazyqsar (the path is hardcoded in the library), so the HOME override below is what
# relocates them into the repo. Download them under the SAME HOME the jobs will use:
#     HOME="$(pwd)/tmp/lazyqsar_home" lazyqsar setup --descriptors
# Note that `lazyqsar setup --target-dir DIR` does NOT work for this: it changes where the
# checkpoints are written but not where the descriptors look for them.

#SBATCH --job-name=eos-lq
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=output/07_logs/%x_%a.out
#SBATCH --error=output/07_logs/_%x_%a.err
#SBATCH --partition=spot_cpu
#SBATCH --nodelist=irbccn16,irbccn41,irbccn42
#SBATCH --requeue

# --- IRB/Aloy cluster configuration, matching chembl-antimicrobial-models ------------
# spot_cpu is preemptible, hence --requeue above. A requeued task is cheap here because
# 07b_train_models.py skips any task whose report and model already exist.
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SINGULARITY_BINDPATH="/home/sbnb:/aloy/home,/data/sbnb/data:/aloy/data,/data/sbnb/scratch:/aloy/scratch"
export LD_LIBRARY_PATH=/apps/manual/software/CUDA/11.6.1/lib64:/apps/manual/software/CUDA/11.6.1/targets/x86_64-linux/lib:/apps/manual/software/CUDA/11.6.1/extras/CUPTI/lib64/:/apps/manual/software/CUDA/11.6.1/nvvm/lib64/:$LD_LIBRARY_PATH
# ------------------------------------------------------------------------------------

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Redirect LazyQSAR's descriptor checkpoint cache into the repository. This works because
# the library resolves the cache as Path.home()/".lazyqsar". Side effect to be aware of:
# every other tool that honours HOME (matplotlib, torch hub, pip) is redirected too, so
# this directory will accumulate their caches as well.
export HOME="$(pwd)/tmp/lazyqsar_home"

# Keep each task inside its CPU allocation. Without this, XGBoost, PyTorch and OpenBLAS each
# default to the node's full core count and several array tasks on one node will thrash.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# The environment lives inside the repository (created with conda --prefix ./envs/eoat), as
# in chembl-antimicrobial-models: on this cluster only the shared filesystem is visible to
# the compute nodes, so it has to sit alongside the code rather than in conda's default path.
envs/eoat/bin/python -u scripts/07b_train_models.py "$SLURM_ARRAY_TASK_ID"
