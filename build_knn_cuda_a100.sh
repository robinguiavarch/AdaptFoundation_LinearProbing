#!/bin/bash
#SBATCH --job-name=wheel_knn_cuda
#SBATCH --output=logs/wheel_knn_cuda_%j.out
#SBATCH --error=logs/wheel_knn_cuda_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -e
set +u
set -o pipefail

echo "========== Build wheel: knn_cuda (A100 / CUDA12.x) =========="
echo "Node: $(hostname)   Date: $(date)"
echo "============================================================="

mkdir -p logs
WHEEL_DIR="${HOME}/wheels/pointm2ae"
mkdir -p "$WHEEL_DIR"

# 1) conda env
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda introuvable"; exit 1
fi
eval "$(conda shell.bash hook)"
conda activate adaptfoundation || { echo "❌ échec activation env"; exit 1; }

# 2) nvcc
have_nvcc() { command -v nvcc >/dev/null 2>&1; }
for CUDA_DIR in /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda; do
  if [ -x "${CUDA_DIR}/bin/nvcc" ]; then
    export PATH="${CUDA_DIR}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_DIR}/lib64:${LD_LIBRARY_PATH-}"
    break
  fi
done
if ! have_nvcc && command -v module >/dev/null 2>&1; then
  module --ignore_cache load cuda/12.2 2>/dev/null || true
fi
INSTALLED_NVCC_CONDA=0
if ! have_nvcc && [ "${USE_CONDA_NVCC:-0}" = "1" ]; then
  echo "⚙️  Installation nvcc (conda-forge) dans l'env…"
  conda install -y -c conda-forge cuda-nvcc=12.2 || { echo "❌ install nvcc conda"; exit 1; }
  INSTALLED_NVCC_CONDA=1
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi
if ! have_nvcc; then
  echo "❌ nvcc introuvable. Relance avec USE_CONDA_NVCC=1, ou sur un nœud avec CUDA."; exit 1
fi

echo "nvcc:"; nvcc --version || true
echo "-------------------------------------------------------------"

# 3) Torch libs on LD_LIBRARY_PATH
PYVER=$(python - <<'PY'
import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)
TORCH_LIB="$CONDA_PREFIX/lib/python${PYVER}/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"

python - <<'PY'
import torch, sys
print("Python :", sys.executable)
print("torch  :", torch.__version__, " CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY
echo "-------------------------------------------------------------"

# 4) Arch + gcc
export TORCH_CUDA_ARCH_LIST="8.0"
echo "🔧 TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
KNN_NVCC_CCBIN=""
if [ -x /usr/bin/g++-11 ]; then
  export CC=/usr/bin/gcc-11
  export CXX=/usr/bin/g++-11
  KNN_NVCC_CCBIN="/usr/bin/g++-11"
  echo "🔧 CC=$CC  CXX=$CXX  (nvcc -ccbin=$KNN_NVCC_CCBIN)"
else
  echo "ℹ️ g++-11 non trouvé — on garde le compilateur par défaut"
fi
echo "-------------------------------------------------------------"

# 5) Clone + **rename to avoid knn.o duplicate**
SRC_DIR=$(mktemp -d /tmp/knn_cuda_XXXXXX)
echo "Clonage dans: $SRC_DIR"
git clone --depth 1 https://github.com/unlimblue/KNN_CUDA.git "$SRC_DIR"
cd "$SRC_DIR"

# Le repo met les sources dans knn_cuda/csrc/cuda/{knn.cpp,knn.cu}
if [ -f knn_cuda/csrc/cuda/knn.cpp ]; then
  mv knn_cuda/csrc/cuda/knn.cpp knn_cuda/csrc/cuda/knn_host.cpp
  echo "🩹 Renommé knn.cpp -> knn_host.cpp (évite knn.o dupliqué)"
fi

# 6) setup.py patché (glob, C++17, ABI torch, ccbin)
cat > setup.py <<'EOF'
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path
import os, torch

candidates = []
candidates += [str(p) for p in Path("knn_cuda").rglob("*.cpp")]
candidates += [str(p) for p in Path("knn_cuda").rglob("*.cu")]
if not candidates:
    candidates += [str(p) for p in Path("src").rglob("*.cpp")]
    candidates += [str(p) for p in Path("src").rglob("*.cu")]

print("[setup.py] Sources détectées:")
for s in candidates: print("  -", s)

cxx_args  = ['-O3', '-std=c++17', '-fPIC']
nvcc_args = ['-O3', '-std=c++17', '-Xfatbin', '-compress-all']

try:
    use_abi = torch._C._GLIBCXX_USE_CXX11_ABI
    if not use_abi:
        cxx_args.append('-D_GLIBCXX_USE_CXX11_ABI=0')
        print("[setup.py] ABI: -D_GLIBCXX_USE_CXX11_ABI=0")
except Exception:
    pass

ccbin = os.environ.get("KNN_NVCC_CCBIN", "")
if ccbin:
    nvcc_args.append(f"-ccbin={ccbin}")
    print(f"[setup.py] nvcc ccbin = {ccbin}")

include_dirs = []
if Path("knn_cuda/csrc").exists(): include_dirs.append("knn_cuda/csrc")
if Path("knn_cuda/csrc/cuda").exists(): include_dirs.append("knn_cuda/csrc/cuda")
if Path("src").exists(): include_dirs.append("src")

setup(
    name='knn_cuda',
    version='0.2.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='knn_cuda.knn_cuda',
            sources=candidates,
            extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
            include_dirs=include_dirs,
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
    zip_safe=False,
)
EOF
echo "✅ setup.py prêt"
echo "-------------------------------------------------------------"

# 7) Build wheel
export KNN_NVCC_CCBIN="$KNN_NVCC_CCBIN"
echo "🔨 pip wheel (no deps, no isolation, verbose) → $WHEEL_DIR"
pip wheel . --no-deps --no-build-isolation -v -w "$WHEEL_DIR"

echo "-------------------------------------------------------------"
echo "📦 Wheels dans $WHEEL_DIR :"
ls -lh "$WHEEL_DIR" | sed 's/^/  /'
echo "============================================================="
echo "🎉 Terminé. Installation :"
echo "   pip install --no-index --no-deps --find-links \"$WHEEL_DIR\" knn_cuda"

# 8) Nettoyage nvcc conda si installé ad hoc
if [ "$INSTALLED_NVCC_CONDA" = "1" ]; then
  echo "🧹 Suppression cuda-nvcc (conda) de l'env…"
  conda remove -y cuda-nvcc || true
fi
