#!/bin/bash
#SBATCH --job-name=classif_sam3d_99
#SBATCH --output=logs/classification_job_sam3d_99_%j.out
#SBATCH --error=logs/classification_job_sam3d_99_%j.err
#SBATCH --partition=CPU
#SBATCH --mem=64G     
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24

# Print job details
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "==============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# Verify environment and dependencies
echo "Environment verification:"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Check required packages
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"

# Verify configuration exists
CONFIG_FILE="configs/classification_sam3d_99.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "Configuration file found: $CONFIG_FILE"

# Verify features directory exists
FEATURES_DIR="feature_extracted_sam3d"
if [ ! -d "$FEATURES_DIR" ]; then
    echo "ERROR: Features directory not found: $FEATURES_DIR"
    exit 1
fi

echo "Features directory found: $FEATURES_DIR"

# Check if PCA 99% files exist
echo "Checking PCA 99% availability:"
for method in avg_pool max_pool sum_pool flatten; do
    PCA_DIR="$FEATURES_DIR/sam_med3d_turbo/$method/PCA_99"
    if [ -d "$PCA_DIR" ]; then
        echo "  ✅ $method/PCA_99 - Available"
        # Count feature files
        FEATURE_COUNT=$(find "$PCA_DIR" -name "*_features.npy" | wc -l)
        echo "     Feature files: $FEATURE_COUNT"
    else
        echo "  ❌ $method/PCA_99 - Missing"
        echo "     Run PCA reduction first: sbatch job_pca_sam3d_99_reduction.sh"
    fi
done

# Run SAM-Med3D Classification with PCA 99%
echo ""
echo "==============================================="
echo "Starting SAM-Med3D Classification with PCA 99%"
echo "Configuration: $CONFIG_FILE"
echo "Target: classification_results_pca_99_sam_med3d_turbo.json"
echo "Expected tasks: 8 (1 model × 4 configs × 2 classifiers × 1 PCA mode)"
echo "==============================================="

# Execute classification
python scripts/run_classification.py --config-file "$CONFIG_FILE"

# Check results
echo ""
echo "==============================================="
echo "Classification Results Summary"
echo "==============================================="

RESULTS_FILE="$FEATURES_DIR/classification_results_pca_99_sam_med3d_turbo.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "✅ Consolidated results file created: $RESULTS_FILE"
    
    # Show file size
    FILE_SIZE=$(du -h "$RESULTS_FILE" | cut -f1)
    echo "   File size: $FILE_SIZE"
    
    # Count configurations processed
    CONFIG_COUNT=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(len(data))" 2>/dev/null || echo "Cannot parse")
    echo "   Configurations processed: $CONFIG_COUNT/4"
    
    # Show individual results
    echo ""
    echo "Individual results files:"
    for method in avg_pool max_pool sum_pool flatten; do
        INDIVIDUAL_FILE="$FEATURES_DIR/sam_med3d_turbo/$method/PCA_99/classification_results.json"
        if [ -f "$INDIVIDUAL_FILE" ]; then
            echo "  ✅ $method/PCA_99/classification_results.json"
        else
            echo "  ❌ $method/PCA_99/classification_results.json - Missing"
        fi
    done
    
else
    echo "❌ Consolidated results file not found: $RESULTS_FILE"
fi

# Performance overview (if results exist)
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Performance Preview (PCA 99%):"
    python -c "
import json, sys
try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)
    
    print('Method        | Classifier | CV Score | Test Score | Overfitting')
    print('------------- | ---------- | -------- | ---------- | -----------')
    
    for config, classifiers in data.items():
        for clf_type, result in classifiers.items():
            if 'best_cv_score' in result:
                cv_score = result['best_cv_score']
                test_score = result['test_metrics']['roc_auc_weighted']
                overfit = result['diagnostics']['overfitting_severity']
                print(f'{config:13} | {clf_type:10} | {cv_score:.4f}   | {test_score:.4f}     | {overfit}')
except Exception as e:
    print(f'Cannot parse results: {e}')
" 2>/dev/null || echo "Cannot display performance preview"
fi

# Disk usage summary
echo ""
echo "Disk Usage Summary:"
echo "PCA 99% directories:"
du -sh "$FEATURES_DIR"/sam_med3d_turbo/*/PCA_99 2>/dev/null || echo "No PCA_99 directories found"

echo ""
echo "Total feature_extracted_sam3d size:"
du -sh "$FEATURES_DIR" 2>/dev/null || echo "Directory not found"

# Final status
echo ""
echo "==============================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

if [ -f "$RESULTS_FILE" ]; then
    echo "✅ SUCCESS: classification_results_pca_99_sam_med3d_turbo.json created"
    echo "🎯 Next step: Compare with other PCA modes (32, 95, 256)"
else
    echo "❌ FAILED: Results file not created"
    echo "Check error logs and PCA 99% availability"
fi
echo "==============================================="