#!/bin/bash
#SBATCH --job-name=build_wheels_p2ae
#SBATCH --output=logs/build_wheels_p2ae_%j.out
#SBATCH --error=logs/build_wheels_p2ae_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00

set -euo pipefail

# === Config utilisateur ===
ENV_NAME="adaptfoundation"
CUDA_DIR="/usr/local/cuda-12.2"                 # nvcc attendu ici
WHEEL_DIR="$HOME/wheels/pointm2ae"              # wheels stockés ici
ARCH="8.0"                                      # A100 => sm_80
REPO_ROOT="$HOME/adaptfoundation_linearprobing/point_m2ae"
# ==========================

echo "========== Build wheels (chamfer + pointnet2_ops) =========="
echo "Node: $(hostname)   Date: $(date)"
echo "============================================================"

# 0) Conda + CUDA -------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export PATH="$CUDA_DIR/bin:$PATH"
# ⚠️ LD_LIBRARY_PATH peut être vide ⇒ utiliser une valeur par défaut
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"

echo "nvcc:"
nvcc --version || { echo "❌ nvcc introuvable"; exit 1; }
echo "------------------------------------------------------------"

# PyTorch libs dans LD_LIBRARY_PATH (libc10, libtorch_cuda, etc.)
PYVER=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
TORCH_LIB="$CONDA_PREFIX/lib/python${PYVER}/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$WHEEL_DIR" logs

python - <<'PY'
import torch, sys
print("Python:", sys.executable)
print("torch :", torch.__version__, "  CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "------------------------------------------------------------"
echo "🔧 TORCH_CUDA_ARCH_LIST=$ARCH"
export TORCH_CUDA_ARCH_LIST="$ARCH"

# 1) Build CHAMFER wheel (depuis le repo Point-M2AE) --------------------------
echo "------------------------------------------------------------"
echo "🔨 Build wheel : chamfer"
cd "$REPO_ROOT/Point-M2AE/extensions/chamfer_dist"
pip wheel --no-build-isolation -v -w "$WHEEL_DIR" .
echo "✅ chamfer wheel prêt dans $WHEEL_DIR"

# 2) Build POINTNET2_OPS wheel (patch sm_80) ----------------------------------
echo "------------------------------------------------------------"
echo "🔨 Build wheel : pointnet2_ops (patch sm_80)"
BUILD_DIR="$(mktemp -d /tmp/pn2_XXXXXX)"
git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git "$BUILD_DIR"
cd "$BUILD_DIR/pointnet2_ops_lib"

# Patch setup.py : respecter env + retirer gencode legacy
python - <<'PY'
import re, pathlib
p = pathlib.Path("setup.py")
s = p.read_text()

# Remplace l’affectation dure par "respecte env ou fallback 8.0"
s = re.sub(
    r'os\.environ\[\s*[\'"]TORCH_CUDA_ARCH_LIST[\'"]\s*\]\s*=\s*[^)\n]+',
    'os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0")',
    s
)

# Nettoyage défensif d’éventuels -gencode obsolètes dans extra_compile_args
s = re.sub(r'"-gencode=[^"]+"[, ]*', '', s)

p.write_text(s)
print("✅ setup.py patché (respect env + gencode legacy retirés)")
PY

# Construire le wheel (méthode moderne)
pip wheel --no-build-isolation -v -w "$WHEEL_DIR" .

echo "------------------------------------------------------------"
echo "📦 Wheels disponibles :"
ls -1 "$WHEEL_DIR"
echo "============================================================"
echo "🎉 Terminé. Installe-les ensuite dans tes jobs via :"
echo "   pip install --no-index --find-links $WHEEL_DIR chamfer pointnet2_ops"
