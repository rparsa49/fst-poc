#!/bin/bash

#SBATCH --job-name=fst_dist
#SBATCH --output=logs/fst_dist_%j.out
#SBATCH --error=logs/fst_dist_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f007z0x@dartmouth.edu

set -euo pipefail
export PYTHONUNBUFFERED=1 

echo "Job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"
mkdir -p logs results data

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

ENV_NAME="fst-poc"
ENV_PATH="${HOME}/.conda/envs/${ENV_NAME}"

if [ ! -d "${ENV_PATH}" ]; then
    echo "Creating conda env '${ENV_NAME}' (one-time, ~1 min)"
    conda create -y -n "${ENV_NAME}" python=3.11 pandas numpy matplotlib requests
else
    echo "Reusing existing conda env '${ENV_NAME}'"
fi

conda activate "${ENV_NAME}"

echo "Python: $(which python)"
python --version

python fst_distribution_analysis.py \
    --data-dir "${PWD}/data" \
    --out-dir  "${PWD}/results"

echo "Job finished at $(date)"

