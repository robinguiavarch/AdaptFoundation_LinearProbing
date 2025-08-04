#!/bin/bash
#SBATCH --job-name=classif_density
#SBATCH --output=logs/classification_density_%j.out
#SBATCH --error=logs/classification_density_%j.err
#SBATCH --partition=CPU
#SBATCH --mem=64G     
#SBATCH --time=08:00:00
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
CONFIG_FILE="sam3d_variantes/turbo_with_density/classification_density.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "Configuration file found: $CONFIG_FILE"

# Verify features directory exists
FEATURES_DIR="feature_extraction_density"
if [ ! -d "$FEATURES_DIR" ]; then
    echo "ERROR: Features directory not found: $FEATURES_DIR"
    exit 1
fi

echo "Features directory found: $FEATURES_DIR"

# Check if PCA files exist for all density approaches
echo "Checking PCA availability for density approaches:"
APPROACHES=("flatten_baseline" "flatten_masking" "flatten_linear_weighting")
PCA_MODES=(32 256 95)

MISSING_COUNT=0
TOTAL_COUNT=0

for approach in "${APPROACHES[@]}"; do
    echo "  Approach: $approach"
    for pca_mode in "${PCA_MODES[@]}"; do
        TOTAL_COUNT=$((TOTAL_COUNT + 1))
        PCA_DIR="$FEATURES_DIR/sam_med3d_turbo_density/$approach/PCA_$pca_mode"
        if [ -d "$PCA_DIR" ]; then
            # Check for required files
            FEATURE_COUNT=$(find "$PCA_DIR" -name "*_features.npy" | wc -l)
            if [ $FEATURE_COUNT -ge 6 ]; then
                echo "    ✅ PCA_$pca_mode - Available ($FEATURE_COUNT feature files)"
            else
                echo "    ⚠️  PCA_$pca_mode - Incomplete ($FEATURE_COUNT/6 feature files)"
                MISSING_COUNT=$((MISSING_COUNT + 1))
            fi
        else
            echo "    ❌ PCA_$pca_mode - Missing"
            MISSING_COUNT=$((MISSING_COUNT + 1))
        fi
    done
done

if [ $MISSING_COUNT -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING_COUNT/$TOTAL_COUNT PCA configurations missing"
    echo "Run PCA reduction first:"
    echo "  python sam3d_variantes/turbo_with_density/run_pipeline_pca_density.py"
    echo ""
    echo "Continuing with available configurations..."
fi

# Run SAM-Med3D Density Classification
echo ""
echo "==============================================="
echo "Starting SAM-Med3D Density Classification"
echo "Configuration: $CONFIG_FILE"
echo "Target: classification_results_density_sam_med3d_turbo_density.json"
echo "Expected tasks: 9 (1 model × 3 approaches × 1 classifier × 3 PCA modes)"
echo "Available tasks: $((TOTAL_COUNT - MISSING_COUNT))"
echo "==============================================="

# Execute classification
PYTHONPATH=. python sam3d_variantes/turbo_with_density/run_pipeline_classification_density.py --config classification_density.yaml

# Check results
echo ""
echo "==============================================="
echo "Classification Results Summary"
echo "==============================================="

RESULTS_FILE="$FEATURES_DIR/classification_results_density_sam_med3d_turbo_density.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "✅ Consolidated results file created: $RESULTS_FILE"
    
    # Show file size
    FILE_SIZE=$(du -h "$RESULTS_FILE" | cut -f1)
    echo "   File size: $FILE_SIZE"
    
    # Count approaches processed
    APPROACH_COUNT=$(python -c "import json; data=json.load(open('$RESULTS_FILE')); print(len(data))" 2>/dev/null || echo "Cannot parse")
    echo "   Approaches processed: $APPROACH_COUNT/3"
    
    # Show individual results
    echo ""
    echo "Individual results files:"
    for approach in "${APPROACHES[@]}"; do
        for pca_mode in "${PCA_MODES[@]}"; do
            INDIVIDUAL_FILE="$FEATURES_DIR/sam_med3d_turbo_density/$approach/PCA_$pca_mode/classification_results.json"
            if [ -f "$INDIVIDUAL_FILE" ]; then
                echo "  ✅ $approach/PCA_$pca_mode/classification_results.json"
            else
                echo "  ❌ $approach/PCA_$pca_mode/classification_results.json - Missing"
            fi
        done
    done
    
else
    echo "❌ Consolidated results file not found: $RESULTS_FILE"
fi

# Performance overview (if results exist)
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Performance Preview - Density Approaches:"
    python -c "
import json, sys
try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)
    
    print('Approach              | PCA Mode | CV Score | Test Score | Overfitting')
    print('--------------------- | -------- | -------- | ---------- | -----------')
    
    # Define display names
    display_names = {
        'flatten_baseline': 'Baseline',
        'flatten_masking': 'Masking',
        'flatten_linear_weighting': 'Linear Weighting'
    }
    
    for approach, pca_modes in data.items():
        approach_display = display_names.get(approach, approach)
        for pca_mode, classifiers in pca_modes.items():
            for clf_type, result in classifiers.items():
                if 'best_cv_score' in result:
                    cv_score = result['best_cv_score']
                    test_score = result['test_metrics']['roc_auc_weighted']
                    overfit = result['diagnostics']['overfitting_severity']
                    print(f'{approach_display:21} | {pca_mode:8} | {cv_score:.4f}   | {test_score:.4f}     | {overfit}')
except Exception as e:
    print(f'Cannot parse results: {e}')
" 2>/dev/null || echo "Cannot display performance preview"
fi

# Density optimization effectiveness analysis
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Density Optimization Effectiveness:"
    python -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)
    
    # Extract best scores per approach
    approach_scores = {}
    for approach, pca_modes in data.items():
        best_test = 0
        best_cv = 0
        for pca_mode, classifiers in pca_modes.items():
            for clf_type, result in classifiers.items():
                if 'test_metrics' in result:
                    test_score = result['test_metrics']['roc_auc_weighted']
                    cv_score = result['best_cv_score']
                    if test_score > best_test:
                        best_test = test_score
                    if cv_score > best_cv:
                        best_cv = cv_score
        approach_scores[approach] = {'test': best_test, 'cv': best_cv}
    
    # Compare with baseline
    if 'flatten_baseline' in approach_scores:
        baseline_test = approach_scores['flatten_baseline']['test']
        print(f'Baseline (Control):      {baseline_test:.4f}')
        
        if 'flatten_masking' in approach_scores:
            masking_test = approach_scores['flatten_masking']['test']
            improvement = masking_test - baseline_test
            print(f'Masking Approach:        {masking_test:.4f} ({improvement:+.4f})')
        
        if 'flatten_linear_weighting' in approach_scores:
            weighting_test = approach_scores['flatten_linear_weighting']['test']
            improvement = weighting_test - baseline_test
            print(f'Linear Weighting:        {weighting_test:.4f} ({improvement:+.4f})')
    
except Exception as e:
    print(f'Cannot analyze effectiveness: {e}')
" 2>/dev/null || echo "Cannot display effectiveness analysis"
fi

# Disk usage summary
echo ""
echo "Disk Usage Summary:"
echo "Classification results directories:"
du -sh "$FEATURES_DIR"/sam_med3d_turbo_density/*/PCA_*/classification_results.json 2>/dev/null | head -10 || echo "No classification results found"

echo ""
echo "Total feature_extraction_density size:"
du -sh "$FEATURES_DIR" 2>/dev/null || echo "Directory not found"

# Final status
echo ""
echo "==============================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"

if [ -f "$RESULTS_FILE" ]; then
    echo "✅ SUCCESS: classification_results_density_sam_med3d_turbo_density.json created"
    echo "🎯 Next steps:"
    echo "   1. Run analysis: python sam3d_variantes/turbo_with_density/run_analysis_density.py"
    echo "   2. Compare density approaches effectiveness"
    echo "   3. Generate comparative tables and visualizations"
else
    echo "❌ FAILED: Results file not created"
    echo "Check error logs and PCA availability for all approaches"
fi
echo "==============================================="