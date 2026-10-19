#!/bin/bash
# R6.9 r0 scan: one GPU job per point (array 0-12 = s0-s12 of scan_config.POINTS).
# Run from the campaign working directory, with R69_REPO set to the pinned checkout:
#   sbatch --array=0-12 --export=ALL,R69_REPO=$REPO $REPO/campaigns/r6.9-hs-r0-scan/scan.sbatch
# Smoke test (2 points, short):
#   sbatch --array=0,1 --export=ALL,R69_REPO=$REPO,SCAN_OUT=scan_smoke,SCAN_STEPS=200000 $REPO/campaigns/r6.9-hs-r0-scan/scan.sbatch
#SBATCH -J r69_scan
#SBATCH -A zeelab
#SBATCH -p gpu-a40
#SBATCH -G 1
#SBATCH --time=03:00:00
#SBATCH --mem=10G
#SBATCH -o scan_%a.out
#SBATCH -e scan_%a.out

set -eo pipefail
REPO=${R69_REPO:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin}  # set R69_REPO to a pinned checkout
VENV=/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
MC_DFM_ROOT=${MC_DFM_ROOT:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM}
SCAN_OUT=${SCAN_OUT:-scan}

module load cuda/11.8
source "$VENV"
export PYTHONPATH="$REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
nvidia-smi --query-gpu=name --format=csv || true

STEPS_ARG=()
if [[ -n "${SCAN_STEPS:-}" ]]; then STEPS_ARG=(--steps "$SCAN_STEPS"); fi
python3 "$REPO/campaigns/r6.9-hs-r0-scan/run_point.py" \
    --index "$SLURM_ARRAY_TASK_ID" --out "$SCAN_OUT" "${STEPS_ARG[@]}"
