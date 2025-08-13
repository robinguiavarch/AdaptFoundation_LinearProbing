#!/bin/bash
#SBATCH --job-name=ctrl_pn2ops
#SBATCH --output=logs/ctrl_pn2ops_%j.out
#SBATCH --error=logs/ctrl_pn2ops_%j.err
#SBATCH --partition=A100          
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

echo "================ GPU / PyTorch sanity-check ================"
echo "Node      : $(hostname)"
echo "Date      : $(date)"
echo "-----------------------------------------------------------"

################ 1)  Runtime CUDA   ################
CUDA_DIR="/usr/local/cuda"   
if [ -d "${CUDA_DIR}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_DIR}/lib64:$LD_LIBRARY_PATH"
    echo "  CUDA runtime libs ajoutées depuis : $CUDA_DIR/lib64"
else
    echo "  Pas de runtime CUDA trouvée dans $CUDA_DIR — on continue quand même"
fi
echo "-----------------------------------------------------------"

################ 2)  Activation de l’environnement Conda ###################
eval "$(conda shell.bash hook)"
conda activate adaptfoundation || { echo " échec activation env"; exit 1; }

# — chemin des bibliothèques PyTorch (.so)  —
PYVER=$(python - <<'PY'
import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)
TORCH_LIB="${CONDA_PREFIX}/lib/python${PYVER}/site-packages/torch/lib"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
echo "🔗 LD_LIBRARY_PATH += ${TORCH_LIB}"
echo "-----------------------------------------------------------"

################ 3)  Rapport Python & import pointnet2_ops ##################
python - <<'PY'
import sys, torch, os, textwrap, importlib

print(f"Python exe            : {sys.executable}")
print(f"PyTorch version       : {torch.__version__}")
print(f"PyTorch CUDA runtime  : {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name              : {torch.cuda.get_device_name(0)}")
    print(f"GPU arch list         : {torch.cuda.get_arch_list()}")

print("-----------------------------------------------------------")
print(" Tentative import pointnet2_ops …")
try:
    import pointnet2_ops
    print(" pointnet2_ops importé avec succès :", pointnet2_ops.__file__)
except ImportError as e:
    print(" ImportError :", e)
    print(textwrap.dedent(\"\"\"\

        L’extension n’est pas (encore) disponible dans cet environnement.
           • Vérifiez qu’elle a bien été compilée avec TORCH_CUDA_ARCH_LIST="8.0"
             et un toolkit CUDA disposant de nvcc (ex. /usr/local/cuda-12.2).
           • Assurez-vous ensuite que le build a été fait **dans** ce même env conda.
           • Si elle vient d’être compilée, relancez un nouveau job pour rafraîchir PYTHONPATH.
    \"\"\"))
PY
echo "======================  Fin contrôle  ======================"
