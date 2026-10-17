#!/bin/bash
# R4 master (CPU): submits one GPU job per measured curve per evaluation (gpu-a40).
# Run it from a checkout pinned at the commit it was started with (R4_REPO), so
# pulls into the working clone do not change the code of later GPU jobs. The
# measured I(q) files are read from that checkout (campaigns/r4-measured/data/).
# Resubmit the same command from the same run directory to resume
# (run_bo_resumable marks a cut-off evaluation ABANDONED and continues).
#   sbatch $R4_REPO/campaigns/r4-measured/run_r4.sbatch
# Startup checks only: sbatch --time=00:30:00 --export=ALL,R4_ARGS="--check-only" ...
# Smoke test: sbatch --time=04:00:00 --export=ALL,R4_ARGS="--smoke --steps 200000",R4_OUT=out_smoke ...
#SBATCH -J r4_master
#SBATCH -A zeelab
#SBATCH -p cpu-g2
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=80:00:00
#SBATCH -o r4_master_%j.out
#SBATCH -e r4_master_%j.out

set -eo pipefail
REPO=${R4_REPO:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin}  # set R4_REPO to a pinned checkout
export R4_VENV=/gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=${MC_DFM_ROOT:-/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM}
export R4_MODULES=cuda/11.8
R4_OUT=${R4_OUT:-Optimization_Results}

module load cuda/11.8
source "$R4_VENV"
export PYTHONPATH="$REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"

python3 -u "$REPO/campaigns/r4-measured/run_r4.py" --out "$R4_OUT" ${R4_ARGS:-}
