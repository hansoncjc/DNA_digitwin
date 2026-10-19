#!/bin/bash
#SBATCH -J r610_translate
#SBATCH -A zeelab
#SBATCH -p cpu-g2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o translate_%j.out
set -eo pipefail
: "${R610_REPO:?Set R610_REPO to the pinned repository copy}"
module load cuda/11.8
source /gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM
export PYTHONPATH="$R610_REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python3 "$R610_REPO/campaigns/r6.10-translation/translate.py" --validate-only
python3 "$R610_REPO/campaigns/r6.10-translation/translate.py"
