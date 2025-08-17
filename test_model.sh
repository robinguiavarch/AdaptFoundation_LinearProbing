#!/bin/bash
#SBATCH --job-name=test_pointm2ae_features
#SBATCH --output=logs/test_pointm2ae_features_%j.out
#SBATCH --error=logs/test_pointm2ae_features_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -eo pipefail

mkdir -p logs

echo "========== POINT-M2AE FEATURE EXTRACTION TEST =========="
echo "Node: $(hostname)   Date: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "========================================================"

# GPU info
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not available"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate adaptfoundation || { echo "ERROR: conda env 'adaptfoundation' not found"; exit 1; }

# PyTorch libs paths
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python${PYVER}/site-packages/torch/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# libtinfo system (silence bash warnings)
for p in /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu; do
  if [ -e "$p/libtinfo.so.6" ]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
    break
  fi
done

# CUDA toolkit
CUDA_HOME=""
for c in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12.3 /usr/local/cuda-12.2 /usr/local/cuda-11.8; do
  [ -d "$c/include" ] && { CUDA_HOME="$c"; break; }
done
[ -z "$CUDA_HOME" ] && { echo "ERROR: No CUDA toolkit found"; exit 2; }

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=4

echo "CUDA_HOME=$CUDA_HOME"
[ -e "$CUDA_HOME/include/cuda_runtime_api.h" ] && echo "CUDA headers OK" || echo "WARNING: CUDA headers missing"

# PyTorch info
python - <<'PY'
import torch, sys
print("• python :", sys.version.split()[0])
print("• torch.__version__      :", torch.__version__)
print("• torch.version.cuda     :", torch.version.cuda)
print("• torch.cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("• device :", torch.cuda.get_device_name(0))
PY

# Purge cache for clean build
rm -rf ~/.cache/torch_extensions/knn* ~/.cache/torch_extensions/*knn* || true

# Test CUDA extensions
python - <<'PY'
import inspect
try:
    import knn_cuda
    print("✅ knn_cuda import OK")
    print("   knn_cuda file:", knn_cuda.__file__)
    print("   knn() params :", len(inspect.signature(knn_cuda.knn).parameters))  # expected: 3 or 4
except Exception as e:
    print("❌ knn_cuda ERROR :", e)

try:
    import pointnet2_ops
    print("✅ pointnet2_ops  : OK")
except Exception as e:
    print("❌ pointnet2_ops ERROR:", e)
PY

# Run Point-M2AE feature extraction tests
cd "$HOME/adaptfoundation_linearprobing"
export PYTHONPATH="$(pwd)/point_m2ae/Point-M2AE:$(pwd):${PYTHONPATH:-}"

echo "Running Point-M2AE feature extraction tests..."
python -u point_m2ae/test_feature_extraction_m2ae.py && echo "🎉 SUCCESS – Point-M2AE tests OK"