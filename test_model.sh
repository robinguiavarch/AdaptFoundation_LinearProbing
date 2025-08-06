#!/bin/bash
#SBATCH --job-name=test_pointm2ae
#SBATCH --output=logs/test_pointm2ae_%j.out
#SBATCH --error=logs/test_pointm2ae_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:45:00  # Plus de temps pour installations

echo "=========================================="
echo "TEST Point-M2AE Feature Extractor"
echo "=========================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"

# Step 1. Export CUDA 12.2 manually
CUDA_DIR="/usr/local/cuda-12.2"
export PATH="$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="$CUDA_DIR"
echo "✅ CUDA 12.2 paths exported"

# Step 2. Check nvcc and GPU
nvcc --version || { echo "❌ nvcc not found"; exit 1; }
nvidia-smi

# Step 3. Activer conda adaptfoundation
eval "$(conda shell.bash hook)"
conda activate adaptfoundation
echo "✅ Conda environment activé"

# Step 4. Aller dans le dossier racine du test
cd ~/adaptfoundation_linearprobing/point_m2ae/

# Step 5. Installer extensions GPU (dans l'ordre)
echo "🔧 Installation extensions GPU..."

# 5.1. Chamfer Distance
echo "📦 Vérification chamfer..."
python -c "import chamfer" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Installation chamfer_dist..."
    cd ./Point-M2AE/extensions/chamfer_dist
    python setup.py install || { echo "❌ Échec chamfer_dist"; exit 1; }
    cd ../../../
    echo "✅ chamfer_dist installé"
else
    echo "✅ chamfer déjà installé"
fi

# 5.2. PointNet++
echo "📦 Vérification pointnet2_ops..."
python -c "import pointnet2_ops" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Installation PointNet++..."
    pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib" || { echo "❌ Échec PointNet++"; exit 1; }
    echo "✅ PointNet++ installé"
else
    echo "✅ pointnet2_ops déjà installé"
fi

# 5.3. KNN-CUDA  
echo "📦 Vérification knn_cuda..."
python -c "import knn_cuda" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Installation KNN-CUDA..."
    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3.10-linux-x86_64.whl || { echo "❌ Échec KNN-CUDA"; exit 1; }
    echo "✅ KNN-CUDA installé"
else
    echo "✅ knn_cuda déjà installé"
fi

# Step 6. Test final imports
echo "🧪 Test imports Point-M2AE..."
python -c "
try:
    import chamfer
    import pointnet2_ops  
    import knn_cuda
    print('✅ Toutes les extensions GPU disponibles')
except ImportError as e:
    print(f'❌ Import manquant: {e}')
    exit(1)
"

# Step 7. Lancer le test principal
echo "🚀 Lancement de test_model.py"
export PYTHONPATH=$(pwd)/Point-M2AE:$PYTHONPATH
python test_model.py || { echo "❌ test_model.py a échoué"; exit 1; }

echo "✅ Test terminé à $(date)"