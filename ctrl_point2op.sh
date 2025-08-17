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

# — chemin des bibliothèques PyTorch (.so) —
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
import sys, torch, textwrap

print(f"Python exe            : {sys.executable}")
print(f"PyTorch version       : {torch.__version__}")
print(f"PyTorch CUDA runtime  : {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        print(f"GPU name              : {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"(info) get_device_name indisponible: {e}")

    # get_arch_list n'existe pas toujours selon versions
    try:
        gal = getattr(torch.cuda, "get_arch_list", None)
        if callable(gal):
            print(f"CUDA arch list        : {gal()}")
    except Exception as e:
        print(f"(info) get_arch_list indisponible: {e}")

print("-----------------------------------------------------------")
print(" Tentative import pointnet2_ops …")
try:
    import pointnet2_ops
    print(" ✅ pointnet2_ops importé avec succès :", getattr(pointnet2_ops, "__file__", "<built-in>"))
except ImportError as e:
    print(" 💥 ImportError :", e)
    msg = """
L’extension pointnet2_ops n’est pas (encore) disponible dans cet environnement.

• Pour A100, compile-la DANS CE MÊME conda env après avoir défini :
    export TORCH_CUDA_ARCH_LIST="8.0"

• Assure-toi d’avoir nvcc (ex. /usr/local/cuda-12.2) dans le PATH pendant la compilation :
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

• Commandes typiques :
    git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/pn2
    cd /tmp/pn2/pointnet2_ops_lib
    python setup.py install

• Après installation, relance un NOUVEAU job pour repartir avec un PYTHONPATH/LD_LIBRARY_PATH propres.
"""
    print(textwrap.dedent(msg))
PY

echo "======================  Fin contrôle  ======================"
