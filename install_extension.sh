#!/bin/bash
#SBATCH --job-name=install_pointm2ae_ext
#SBATCH --output=logs/install_pointm2ae_ext_%j.out
#SBATCH --error=logs/install_pointm2ae_ext_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --time=02:00:00

echo "=========================================="
echo "INSTALL GPU EXTENSIONS (Point-M2AE)"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"

# === Setup CUDA 12.2 manually ===
CUDA_DIR="/usr/local/cuda-12.2"
export PATH="$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
echo "✅ CUDA 12.2 paths exported"

# Vérification de nvcc
echo "Checking nvcc:"
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found — aborting."
    exit 1
else
    nvcc --version
fi

# Activer conda adaptfoundation
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# Vérifier PyTorch + CUDA
python -c "import torch; print('✅ PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Available:', torch.cuda.is_available())"

# Aller dans le dossier extensions
cd ~/adaptfoundation_linearprobing/point_m2ae/Point-M2AE/extensions

echo "🧹 Nettoyage des builds précédents"
rm -rf build/ dist/ *.egg-info */build */dist */*.egg-info

echo "----- Chamfer Distance -----"
cd chamfer_dist
python setup.py install || { echo "❌ Chamfer install FAILED"; exit 1; }
cd ..

echo "----- Earth Mover Distance (EMD) -----"
cd emd
python setup.py install || { echo "❌ EMD install FAILED"; exit 1; }
cd ..

echo "----- KNN CUDA -----"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl || { echo "❌ KNN-CUDA install FAILED"; exit 1; }

echo "✅ GPU extensions successfully installed"
echo "Job ended at: $(date)"
