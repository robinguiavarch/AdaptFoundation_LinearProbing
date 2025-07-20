#!/bin/bash
#SBATCH --job-name=dinov2_giant_extraction
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=A100              
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=8                               
#SBATCH --time=10:00:00               

# Print job details
echo "===========================================" 
echo "Starting DINOv2 Giant Feature Extraction"
echo "Job started on node: $(hostname)"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "==========================================="

# Navigate to project directory
cd ~/adaptfoundation_linearprobing

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate


# Execute feature extraction
echo "Starting feature extraction with DINOv2 Giant..."
echo "Command: python3 scripts/run_feature_extraction.py"
python3 scripts/run_feature_extraction.py

# Check if extraction completed successfully
if [ $? -eq 0 ]; then
    echo "Feature extraction completed successfully!"
    echo "Checking output directory..."
    ls -la feature_extracted/dinov2_vitg14/
else
    echo "Feature extraction failed with exit code $?"
fi

# Print job completion time
echo "==========================================="
echo "Job finished at: $(date)"
echo "==========================================="