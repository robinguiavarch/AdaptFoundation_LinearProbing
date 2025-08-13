#!/bin/bash
#SBATCH --job-name=test_pointm2ae
#SBATCH --output=logs/test_pointm2ae_%j.out
#SBATCH --error=logs/test_pointm2ae_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=01:30:00

echo "========== Point-M2AE GPU TEST =========="
echo "Node: $(hostname)   Date: $(date)"
echo "=========================================="

#################### 1. CUDA 12.2 ##############################################
export CUDA_DIR="/usr/local/cuda-12.2"
export PATH="$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
nvcc --version || { echo "nvcc not found !!!"; exit 1; }
################################################################################

#################### 2. Conda ###################################################
eval "$(conda shell.bash hook)"
conda activate adaptfoundation || { echo "conda env !!!!"; exit 1; }

echo "—— Conda debug ————————————————"
echo "   • which python  : $(which python)"
echo "   • python -V     : $(python -V)"
python - <<'PY'
import torch, sys, os
print(f"   • torch.__version__ : {torch.__version__}")
print(f"   • torch.version.cuda: {torch.version.cuda}")
print(f"   • sys.path[0]       : {sys.path[0]}")
print(f"   • torch.cuda.is_available(): {torch.cuda.is_available()}")
PY
echo "———————————————————————————————"

# — chemin libs PyTorch —
TORCH_LIB="$CONDA_PREFIX/lib/python$(python - <<'PY'
import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)/site-packages/torch/lib"

export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo "🔗 LD_LIBRARY_PATH += $TORCH_LIB"

# Petit contrôle que les .so existent vraiment
echo "—— Contenu de torch/lib —————————"
ls -1 "$TORCH_LIB" | head
echo "———————————————————————————————"
################################################################################

#################### 3. Projet & PYTHONPATH ####################################
cd ~/adaptfoundation_linearprobing/point_m2ae || exit 1
export PYTHONPATH="$(pwd)/Point-M2AE:$(pwd):$PYTHONPATH"
################################################################################

#################### 4. Extensions GPU #########################################
python - <<'PY'
import importlib, subprocess, os, sys, textwrap, shutil, pathlib

def ensure(mod, cmd):
    try:
        importlib.import_module(mod)
        print(f"✅ {mod} OK")
    except ImportError:
        print(f"🚧 build {mod} …")
        subprocess.check_call(cmd, shell=True, executable="/bin/bash")
        importlib.import_module(mod)
        print(f"✅ {mod} compiled")

# 4-1 : chamfer (déjà présent ? sinon compile)
ensure(
    "chamfer",
    "cd Point-M2AE/extensions/chamfer_dist && python setup.py install && cd -"
)

# 4-2 : pointnet2_ops (FPS / ball query) – compile pour A100
ensure(
    "pointnet2_ops",
    "git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/pn2 && "
    "cd /tmp/pn2/pointnet2_ops_lib && python setup.py install && cd -"
)

# 4-3 : knn_cuda – compile depuis la source
ensure(
    "knn_cuda",
    "git clone --depth 1 https://github.com/unlimblue/KNN_CUDA.git /tmp/knn && "
    "cd /tmp/knn && sed -i 's/-std=c++11/-std=c++17/' setup.py && python setup.py install && cd -"
)
PY
################################################################################

#################### 5. Test Point-M2AE ########################################
echo "🚀 running test_model.py"
python test_model.py && echo "🎉  SUCCESS – features OK"
################################################################################
