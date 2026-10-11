#!/bin/bash
# inverse.py master with parallel=True: this CPU job submits one GPU job per
# evaluation (partition etc. from CONFIG["parallel_cfg"]) and waits for it.
# Run it from the repo copy pinned for this campaign (INV_REPO); the GPU jobs
# import the same copy. Resubmitting the same command resumes the run
# (README "Resuming a run"); first scancel GPU jobs left by the old master.
#   sbatch --export=ALL,INV_REPO=/path/to/pinned/DNA_digitwin,INV_CONFIG=/path/to/config.py \
#          $INV_REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch
# Extra inverse.py arguments (smoke tests): INV_ARGS="--out OUT --steps 2000000 --parallel true".
# Partition and walltime below are R3.2's master settings; override with
# sbatch -p / --time. The walltime must end before a maintenance window.
#SBATCH -J inv_par
#SBATCH -A zeelab
#SBATCH -p cpu-g2
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=80:00:00
#SBATCH -o inv_par_%j.out
#SBATCH -e inv_par_%j.out

set -eo pipefail
: "${INV_REPO:?set INV_REPO to the pinned DNA_digitwin checkout}"
: "${INV_CONFIG:?set INV_CONFIG to the campaign config.py}"
export MC_DFM_ROOT=${MC_DFM_ROOT:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM}

set +u
module load cuda/11.8
set -u
source /gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export PYTHONPATH="$INV_REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"

echo "job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-local}, start $(date)"
python3 -u "$INV_REPO/inverse.py" "$INV_CONFIG" ${INV_ARGS:-}
echo "end $(date)"
