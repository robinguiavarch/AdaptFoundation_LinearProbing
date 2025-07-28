#!/bin/bash
#SBATCH --job-name=classif_sam3d
#SBATCH --output=logs/classification_job_sam3d_%j.out
#SBATCH --error=logs/classification_job_sam3d_%j.err
#SBATCH --partition=CPU
#SBATCH --mem=100G     
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=24

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# Run classification for Sam3d
echo "Starting Sam3d Classification"
echo "Configuration: configs/classification_sam3d.yaml"

python scripts/run_classification.py --config-file configs/classification_sam3d.yaml

echo "Sam3d Classification completed"