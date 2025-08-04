#!/bin/bash

#SBATCH --job-name=feature_maps_classification
#SBATCH --output=logs/classification_feature_maps_%j.out
#SBATCH --error=logs/classification_feature_maps_%j.err
#SBATCH --partition=CPU
#SBATCH --time=07:30:00
#SBATCH --cpus-per-task=20

# ============================================================================
# Feature Maps & 2.5D Classification Job Script  
# ============================================================================
# Description: Linear probing classification on feature maps and 2.5D variants
# Expected: 10 combinations (PCA 95% failed for concat variants)
# Runtime: ~20-45 minutes for all available combinations
# Memory: <10GB (features already PCA-reduced)
# ============================================================================

# Job info
echo "=========================================="
echo "FEATURE MAPS & 2.5D CLASSIFICATION"
echo "=========================================="
echo "Job started at: $(date)"
echo "Hostname: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

# Create logs directory
mkdir -p logs

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# Check Python environment
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Working directory: $(pwd)"

# Verify required files exist
echo ""
echo "Checking required files..."

if [ ! -f "dinov2_variantes/feature_map_25d/classification.yaml" ]; then
    echo "ERROR: Configuration file not found: dinov2_variantes/feature_map_25d/classification.yaml"
    exit 1
fi

if [ ! -d "feature_extraction_variantes" ]; then
    echo "ERROR: Features directory not found: feature_extraction_variantes"
    exit 1
fi

echo "✓ Configuration file: dinov2_variantes/feature_map_25d/classification.yaml"
echo "✓ Features directory: feature_extraction_variantes"

# List available variants and PCA modes
echo ""
echo "Available feature variants:"
for variant_dir in feature_extraction_variantes/*/; do
    if [ -d "$variant_dir" ]; then
        variant_name=$(basename "$variant_dir")
        echo "  $variant_name:"
        for pca_dir in $variant_dir/PCA_*/; do
            if [ -d "$pca_dir" ]; then
                pca_mode=$(basename "$pca_dir")
                if [ -f "$pca_dir/test_split_features.npy" ]; then
                    echo "    ✓ $pca_mode"
                else
                    echo "    ✗ $pca_mode (missing files)"
                fi
            fi
        done
    fi
done

# Run classification pipeline
echo ""
echo "=========================================="
echo "STARTING CLASSIFICATION PIPELINE"
echo "=========================================="
echo "Configuration: dinov2_variantes/feature_map_25d/classification.yaml"
echo "Features directory: feature_extraction_variantes"

python dinov2_variantes/feature_map_25d/run_pipeline_classification.py \
    --config dinov2_variantes/feature_map_25d/classification.yaml \
    --features-path feature_extraction_variantes

exit_code=$?

echo ""
echo "=========================================="
echo "CLASSIFICATION COMPLETED"
echo "=========================================="
echo "Classification completed at: $(date)"
echo "Exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "✅ Classification pipeline completed successfully!"
    
    # Show results summary
    if [ -f "feature_extraction_variantes/feature_maps_classification_results.json" ]; then
        echo ""
        echo "Results saved to: feature_extraction_variantes/feature_maps_classification_results.json"
        
        # Count successful evaluations
        python -c "
import json
with open('feature_extraction_variantes/feature_maps_classification_results.json', 'r') as f:
    results = json.load(f)
total_combinations = 0
for variant, pca_results in results.get('results', {}).items():
    total_combinations += len(pca_results)
print(f'Total combinations evaluated: {total_combinations}')
"
    fi
    
    # Show output structure
    echo ""
    echo "Output structure:"
    find feature_extraction_variantes -name "classification_results.json" | head -5
    if [ $(find feature_extraction_variantes -name "classification_results.json" | wc -l) -gt 5 ]; then
        echo "... and more"
    fi
    
else
    echo "❌ Classification pipeline failed!"
    echo "Check error logs above for details."
fi

echo ""
echo "=== Job Summary ==="
echo "Expected combinations: 10 (PCA 95% failed for concat variants)"
echo "Configuration: Logistic regression only"
echo "Cross-validation: LeaveOneGroupOut (5 folds)"
echo "Metrics: ROC-AUC weighted + diagnostics"
echo "Next steps: Analyze results and compare with baselines"

exit $exit_code