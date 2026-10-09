#!/bin/bash
# R3.2 master (CPU): submits one GPU job per training condition per evaluation.
# Run it from a checkout pinned at the commit it was started with (R32_REPO), so
# pulls into the working clone do not change the code of later GPU jobs.
# Walltime must end before the 2026-10-13 09:00 maintenance; resubmit after it
# to resume (see run_r32.py about the last eval folder).
#   sbatch $REPO/campaigns/r32_physics/run_r32.sbatch
# Smoke test: sbatch --time=04:00:00 --export=ALL,R32_ARGS="--smoke --steps 200000",R32_GT=gt_smoke,R32_OUT=out_smoke ...
#SBATCH -J r32_master
#SBATCH -A zeelab
#SBATCH -p CHANGE_ME_CPU_PARTITION
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=84:00:00
#SBATCH -o r32_master_%j.out
#SBATCH -e r32_master_%j.out

set -eo pipefail
REPO=${R32_REPO:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin}  # set R32_REPO to a pinned checkout
export R32_VENV=/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=${MC_DFM_ROOT:?set MC_DFM_ROOT to the MC-DFM checkout}
export R32_MODULES=cuda/11.8
R32_GT=${R32_GT:-ground_truth}
R32_OUT=${R32_OUT:-Optimization_Results}

module load cuda/11.8
source "$R32_VENV"
export PYTHONPATH="$REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"

python3 -u "$REPO/campaigns/r32_physics/run_r32.py" --gt "$R32_GT" --out "$R32_OUT" ${R32_ARGS:-}
