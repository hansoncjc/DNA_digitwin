#!/bin/bash
# R5.1 prediction: one GPU job per condition (array 0-13 = p0-p13 of predict_config.CONDITIONS).
# Run from the campaign working directory, with R51_REPO set to the pinned checkout:
#   sbatch --array=0-13 --export=ALL,R51_REPO=$REPO $REPO/campaigns/r5.1-predict/predict.sbatch
# Smoke test (2 conditions, short):
#   sbatch --array=0,13 --export=ALL,R51_REPO=$REPO,PRED_OUT=pred_smoke,PRED_STEPS=200000 $REPO/campaigns/r5.1-predict/predict.sbatch
#SBATCH -J r51_pred
#SBATCH -A zeelab
#SBATCH -p gpu-a40
#SBATCH -G 1
#SBATCH --time=03:00:00
#SBATCH --mem=10G
#SBATCH -o pred_%a.out
#SBATCH -e pred_%a.out

set -eo pipefail
REPO=${R51_REPO:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin}  # set R51_REPO to a pinned checkout
VENV=/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
MC_DFM_ROOT=${MC_DFM_ROOT:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM}
PRED_OUT=${PRED_OUT:-predictions}

module load cuda/11.8
source "$VENV"
export PYTHONPATH="$REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv || true

STEPS_ARG=()
if [[ -n "${PRED_STEPS:-}" ]]; then STEPS_ARG=(--steps "$PRED_STEPS"); fi
python3 "$REPO/campaigns/r5.1-predict/make_prediction.py" \
    --index "$SLURM_ARRAY_TASK_ID" --out "$PRED_OUT" "${STEPS_ARG[@]}"
