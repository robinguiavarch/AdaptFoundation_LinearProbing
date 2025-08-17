#!/bin/bash
#SBATCH --job-name=test_knn_cuda
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "========== Test knn_cuda =========="
echo "Node: $(hostname)   Date: $(date)"
echo "==================================="

echo "nvidia-smi:"
nvidia-smi || true
echo "-----------------------------------"

# Activer conda
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# (robuste) s’assurer que les libs torch (libc10, libtorch_cuda, …) sont dans le LD_LIBRARY_PATH
PYVER=$(python - <<'PY'
import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python${PYVER}/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"

python - <<'PY'
import torch, importlib, subprocess

print("torch :", torch.__version__, "| CUDA runtime:", torch.version.cuda)
print("CUDA disponible ? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --- ldd sur le .so de knn_cuda ---
m = importlib.import_module("knn_cuda.knn_cuda")
print("\n[knn_cuda] extension .so :", m.__file__)
print("[ldd sur knn_cuda .so] ↓")
subprocess.run(["ldd", m.__file__], check=False)

# --- mini test KNN sur GPU ---
from knn_cuda import KNN
ref = torch.rand(2, 100, 3, device='cuda')
qry = torch.rand(2, 10,  3, device='cuda')
knn = KNN(k=5, transpose_mode=True)
dist, idx = knn(ref, qry)
print("\nKNN OK ->", dist.shape, idx.shape)
PY

echo "============== DONE ==============="
