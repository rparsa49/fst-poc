#!/bin/bash

#SBATCH --job-name=fst_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=f007z0x@dartmouth.edu


set -euo pipefail
export PYTHONUNBUFFERED=1

echo "[$(date)] Job started on $(hostname)"
mkdir -p logs artifacts

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
ENV_NAME="fst-forecast"
ENV_PATH="${HOME}/.conda/envs/${ENV_NAME}"
if [ ! -d "${ENV_PATH}" ]; then
    echo "[$(date)] Creating conda env (one-time)"
    conda create -y -n "${ENV_NAME}" python=3.11
    conda activate "${ENV_NAME}"
    pip install --quiet pandas numpy matplotlib requests scipy scikit-learn \
                        scikit-image Pillow
    pip install --quiet torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118
else
    conda activate "${ENV_NAME}"
fi
echo "[$(date)] Python: $(which python)"
python --version
python -c "import torch; print('CUDA:', torch.cuda.is_available(),
'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

N_IMG=$(ls data/images/ 2>/dev/null | wc -l)
echo "[$(date)] Found ${N_IMG} downloaded images in data/images/"
if [ "${N_IMG}" -lt 500 ]; then
    echo "[error] Too few images on disk. Run download on the LOGIN node:"
    echo "        python pipeline.py download_images --workers 16"
    exit 1
fi

echo "[$(date)] === Stage: extracting ResNet-50 features ==="
python pipeline.py extract_features

echo "[$(date)] === Stage: generating scenarios ==="
python pipeline.py generate_scenarios --k 20

echo "[$(date)] Setup complete. Next: sbatch 01_run_pilot.sbatch"
