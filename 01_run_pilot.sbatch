#!/bin/bash

#SBATCH --job-name=fst_pilot
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f007z0x@dartmouth.edu

set -uo pipefail
export PYTHONUNBUFFERED=1

echo "[$(date)] Job started on $(hostname)"
mkdir -p logs

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate fst-forecast

if [ ! -f artifacts/features.npz ];  then
    echo "[error] artifacts/features.npz missing -- run 00_setup.sbatch first"; exit 1
fi
if [ ! -f artifacts/scenarios.json ]; then
    echo "[error] artifacts/scenarios.json missing -- run 00_setup.sbatch first"; exit 1
fi

K=$(python -c "import json; print(len(json.load(open('artifacts/scenarios.json'))))")
echo "[$(date)] Found ${K} scenarios"

n_ok=0; n_fail=0
for ((i=0; i<K; i++)); do
    echo ""
    echo "[$(date)] ===================================================="
    echo "[$(date)] === Scenario ${i} / $((K-1))"
    echo "[$(date)] ===================================================="
    if python pipeline.py run_scenario --id ${i}; then
        n_ok=$((n_ok+1))
    else
        echo "[$(date)] !! scenario ${i} failed; continuing"
        n_fail=$((n_fail+1))
    fi
done
echo ""
echo "[$(date)] Scenarios: ok=${n_ok}  fail=${n_fail}"

echo "[$(date)] === Aggregation ==="
python pipeline.py aggregate

echo "[$(date)] Done. Outputs in artifacts/aggregate/"
