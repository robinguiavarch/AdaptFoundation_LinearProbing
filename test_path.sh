#!/bin/bash
#SBATCH --job-name=test_path
#SBATCH --output=logs/test_pointm2ae_%j.out
#SBATCH --error=logs/test_pointm2ae_%j.err
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=01:30:00

echo "====== Torch shared-libs path checker ======"

# 1. Trouver le chemin torch/lib
TORCH_LIB=$(python - <<'PY'
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("torch")
if spec is None:
    print("")
    sys.exit(0)
torch_root = pathlib.Path(spec.origin).parents[1]           # …/site-packages/torch
lib_path  = torch_root / "lib"
print(lib_path if lib_path.exists() else "")
PY
)

if [ -z "$TORCH_LIB" ]; then
  echo "❌ PyTorch introuvable dans l'environnement actif."
  exit 1
fi

echo "PyTorch shared libs = $TORCH_LIB"

# 2. Vérifier présence dans LD_LIBRARY_PATH
if [[ ":$LD_LIBRARY_PATH:" == *":$TORCH_LIB:"* ]]; then
  echo "✅ $TORCH_LIB déjà présent dans LD_LIBRARY_PATH"
else
  echo "⚠️  $TORCH_LIB ABSENT de LD_LIBRARY_PATH"
  echo "👉  Ajoutez la ligne suivante après 'conda activate':"
  echo ""
  echo "export LD_LIBRARY_PATH=\"$TORCH_LIB:\$LD_LIBRARY_PATH\""
  echo ""
fi

# 3. Test import C++/CUDA (chamfer si dispo, sinon torch)
python - <<PY
try:
    import chamfer
    print("✅ import chamfer  OK (C++ extension)")
except ImportError as e:
    print("ℹ️ chamfer non importable :", e)
    try:
        import torch
        torch.zeros(1)
        print("✅ torch C++ backend OK (libc10.so chargé)")
    except Exception as e2:
        print("❌ torch backend KO :", e2)
PY
