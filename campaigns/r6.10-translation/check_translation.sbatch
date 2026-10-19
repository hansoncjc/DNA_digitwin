#!/bin/bash
#SBATCH -J r610_check
#SBATCH -A zeelab
#SBATCH -p gpu-a40
#SBATCH -G 1
#SBATCH --array=0-3
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH -o check_%A_%a.out
set -eo pipefail
: "${R610_REPO:?Set R610_REPO to the pinned repository copy}"
module load cuda/11.8
source /gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM
export PYTHONPATH="$R610_REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
ARGS=()
if [[ -n "${R610_SMOKE_STEPS:-}" ]]; then ARGS=(--steps "$R610_SMOKE_STEPS"); fi
python3 "$R610_REPO/campaigns/r6.10-translation/check_translation.py" --index "$SLURM_ARRAY_TASK_ID" "${ARGS[@]}"
