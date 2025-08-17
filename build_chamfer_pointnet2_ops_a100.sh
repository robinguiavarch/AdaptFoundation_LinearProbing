#!/bin/bash
#SBATCH --job-name=build_chamfer_pn2ops
#SBATCH --output=logs/build_chamfer_pn2ops_%j.out
#SBATCH --error=logs/build_chamfer_pn2ops_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00

set -e

echo "========== Build chamfer + pointnet2_ops (A100) =========="
echo "Node: $(hostname)   Date: $(date)"
echo "=========================================================="

#################### 0) Trouver nvcc (CUDA dev kit) ####################
# Cherche nvcc dans quelques emplacements classiques
for C in /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda /opt/cuda ; do
  if [ -x "$C/bin/nvcc" ]; then
    export CUDA_DIR="$C"
    break
  fi
done
if [ -z "${CUDA_DIR:-}" ]; then
  echo "❌ nvcc introuvable (pas de toolkit CUDA dev sur ce nœud)."
  echo "   Relance sur un nœud A100 qui possède nvcc (/usr/local/cuda-12.x)."
  exit 1
fi
export PATH="$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
echo "✅ nvcc : $CUDA_DIR/bin/nvcc"
nvcc --version
echo "----------------------------------------------------------"

#################### 1) Activer conda & LD_LIBRARY_PATH torch ##########
eval "$(conda shell.bash hook)"
conda activate adaptfoundation || { echo "❌ conda env"; exit 1; }

# Ajouter les .so de PyTorch (libc10.so, libtorch_cuda.so, etc.) dans LD_LIBRARY_PATH
PYVER=$(python - <<'PY'
import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)
TORCH_LIB="${CONDA_PREFIX}/lib/python${PYVER}/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"
echo "🔗 LD_LIBRARY_PATH += $TORCH_LIB"

# Petit rapport
python - <<'PY'
import sys, torch
print("Python:", sys.executable)
print("torch :", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    try:
        print("CUDA arch list:", torch.cuda.get_arch_list())
    except Exception:
        pass
PY
echo "----------------------------------------------------------"

#################### 2) Forcer arch A100 uniquement ####################
# Pour accélérer et éviter les arch obsolètes
export TORCH_CUDA_ARCH_LIST="8.0"
echo "🔧 TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "----------------------------------------------------------"

#################### 3) Chemin du projet ###############################
# On part depuis ton dossier de travail Point-M2AE
cd ~/adaptfoundation_linearprobing/point_m2ae || { echo "❌ chemin projet"; exit 1; }
export PYTHONPATH="$(pwd)/Point-M2AE:$(pwd):$PYTHONPATH"
echo "📁 CWD: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "----------------------------------------------------------"

#################### 4) Build CHAMFER (Point-M2AE) #####################
echo "🔨 Build chamfer (Point-M2AE/extensions/chamfer_dist)…"
pushd Point-M2AE/extensions/chamfer_dist >/dev/null
# Nettoyage
rm -rf build/ dist/ *.egg-info
# Build & install
python setup.py install
popd >/dev/null

# Vérif import chamfer
python - <<'PY'
try:
    import chamfer
    print("🎉 chamfer OK ->", chamfer.__file__)
except Exception as e:
    raise SystemExit(f"💥 Echec import chamfer: {e}")
PY
echo "----------------------------------------------------------"

#################### 5) Build pointnet2_ops (patch A100) ################
echo "🔨 Build pointnet2_ops (patch A100)…"
WORKDIR="/tmp/pn2_${SLURM_JOB_ID:-$$}"
rm -rf "$WORKDIR"
git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git "$WORKDIR"
cd "$WORKDIR/pointnet2_ops_lib"

# Patch: retirer tous les -gencode hardcodés (causent erreurs avec CUDA 12.x)
python - <<'PY'
from pathlib import Path
import re
p = Path("setup.py")
txt = p.read_text()
# retire les occurrences '-gencode=...'
txt = re.sub(r"'-gencode=[^']*',\s*", "", txt)
p.write_text(txt)
print("✅ Patch setup.py : suppression des -gencode obsolètes")
PY

# Nettoyage build éventuel
python setup.py clean || true
rm -rf build/ dist/ *.egg-info

# Build & install dans l'env conda actif
python setup.py install
cd -

# Vérif import pointnet2_ops
python - <<'PY'
try:
    import pointnet2_ops
    print("🎉 pointnet2_ops OK ->", pointnet2_ops.__file__)
except Exception as e:
    raise SystemExit(f"💥 Echec import pointnet2_ops: {e}")
PY
echo "----------------------------------------------------------"

echo "✅ FINI : chamfer + pointnet2_ops installés dans conda: $CONDA_PREFIX"
echo "⚠️  Prochain gros morceau (à part) : KNN_CUDA"
echo "=========================================================="
