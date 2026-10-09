#!/bin/bash
# R3.1 ground truth: one GPU job per condition (array 0-23 training, 24-28 held-out).
# Run from the campaign working directory:
#   sbatch --array=0-28 $REPO/campaigns/r3.1-gt/ground_truth.sbatch
# Smoke test (2 conditions, short):
#   sbatch --array=0,23 --export=ALL,GT_OUT=gt_smoke,GT_STEPS=200000 $REPO/campaigns/r3.1-gt/ground_truth.sbatch
#SBATCH -J r32_gt
#SBATCH -A zeelab
#SBATCH -p gpu-a40
#SBATCH -G 1
#SBATCH --time=03:00:00
#SBATCH --mem=10G
#SBATCH -o gt_%a.out
#SBATCH -e gt_%a.out

set -eo pipefail
REPO=${R32_REPO:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin}  # set R32_REPO to a pinned checkout
VENV=/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
MC_DFM_ROOT=${MC_DFM_ROOT:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM}
GT_OUT=${GT_OUT:-ground_truth}

module load cuda/11.8
source "$VENV"
export PYTHONPATH="$REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv || true

STEPS_ARG=()
if [[ -n "${GT_STEPS:-}" ]]; then STEPS_ARG=(--steps "$GT_STEPS"); fi
python3 "$REPO/campaigns/r3.1-gt/make_ground_truth.py" \
    --index "$SLURM_ARRAY_TASK_ID" --out "$GT_OUT" "${STEPS_ARG[@]}"
