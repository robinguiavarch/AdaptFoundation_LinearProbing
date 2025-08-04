#!/bin/bash
#SBATCH --job-name=adaptfound_vitl14
#SBATCH --output=logs/classification_vitl14_%j.out
#SBATCH --error=logs/classification_vitl14_%j.err
#SBATCH --partition=CPU
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=15

# Job info
echo "Job started at: $(date)"
echo "Hostname: $(hostname)" 
echo "CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# Run classification for vitl14 only
echo "Starting AdaptFoundation Classification - vitl14 Large Model"
echo "Configuration: configs/classification_concat_vitl14.yaml"

python scripts/run_classification.py --config-file configs/classification_concat_vitl14.yaml

exit_code=$?
echo "vitl14 Classification completed at: $(date)"
echo "Exit code: $exit_code"
exit $exit_code