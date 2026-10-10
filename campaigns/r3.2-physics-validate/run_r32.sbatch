#!/bin/bash
# R3.2 master (CPU): submits one GPU job per training condition per evaluation.
# Run it from a checkout pinned at the commit it was started with (R32_REPO), so
# pulls into the working clone do not change the code of later GPU jobs.
# Walltime must end before the 2026-10-13 09:00 maintenance; resubmit the same
# command from the same run directory to resume (run_bo_resumable marks a cut-off
# evaluation ABANDONED and continues; see run_r32.py).
#   sbatch $REPO/campaigns/r3.2-physics-validate/run_r32.sbatch
# Smoke test: sbatch --time=04:00:00 --export=ALL,R32_ARGS="--smoke --steps 200000",R32_GT=gt_smoke,R32_OUT=out_smoke ...
#SBATCH -J r32_master
#SBATCH -A zeelab
#SBATCH -p cpu-g2
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=80:00:00
#SBATCH -o r32_master_%j.out
#SBATCH -e r32_master_%j.out

set -eo pipefail
REPO=${R32_REPO:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin}  # set R32_REPO to a pinned checkout
export R32_VENV=/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=${MC_DFM_ROOT:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM}
export R32_MODULES=cuda/11.8
R32_GT=${R32_GT:-ground_truth}
R32_OUT=${R32_OUT:-Optimization_Results}

module load cuda/11.8
source "$R32_VENV"
export PYTHONPATH="$REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"

python3 -u "$REPO/campaigns/r3.2-physics-validate/run_r32.py" --gt "$R32_GT" --out "$R32_OUT" ${R32_ARGS:-}
