#!/bin/bash
#SBATCH --job-name=adaptfound_vitg14
#SBATCH --output=logs/classification_vitg14_%j.out
#SBATCH --error=logs/classification_vitg14_%j.err
#SBATCH --partition=CPU
#SBATCH --mem=200G     
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=70

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# Run classification for vitg14 only
echo "Starting AdaptFoundation Classification - vitg14 Giant Model"
echo "Configuration: configs/classification_concat_vitg14.yaml"

python scripts/run_classification.py --config-file configs/classification_concat_vitg14.yaml

echo "vitg14 Classification completed"