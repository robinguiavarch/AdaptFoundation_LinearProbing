#!/bin/bash
#SBATCH --job-name=classification_vits14_concat
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=CPU                              
#SBATCH --cpus-per-task=65                              
#SBATCH --time=12:00:00               

# Print job details
echo "===========================================" 
echo "Starting Classification Job for dinov2_vits14 with Concatenation Strategy"
echo "Rapid Testing Mode: Single Model Evaluation"
echo "Job started on node: $(hostname)"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "==========================================="


# Navigate to project directory
cd ~/adaptfoundation_linearprobing

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate adaptfoundation

# Verify environment
echo "Using Python: $(which python)"
python --version
python -c "import yaml; print(' yaml available:', yaml.__version__)"

echo "Starting classification with conda environment..."
python scripts/run_classification.py --config-file configs/classification_concat_vits14.yaml

# Check if classification completed successfully
if [ $? -eq 0 ]; then
    echo "==========================================="
    echo "Classification completed successfully!"
    echo "Checking classification results for dinov2_vits14 concatenation configurations..."
    
    # Expected concatenation configurations
    configs=(
        "multi_axes_concatenation"
        "single_axis_axial_concatenation"
        "single_axis_coronal_concatenation"
        "single_axis_sagittal_concatenation"
    )
    
    # Expected PCA modes (including none)
    pca_modes=("none" "PCA_32" "PCA_256" "PCA_95")
    
    # Expected classifiers
    classifiers=("logistic" "knn" "svm_linear")
    
    # Check dinov2_vits14 model directory
    model="dinov2_vits14"
    echo "=== Checking $model (Concatenation Strategy) ==="
    
    if [ -d "feature_extracted_concat/$model" ]; then
        model_results=0
        total_expected=0
        
        # Check each concatenation configuration
        for config in "${configs[@]}"; do
            if [ -d "feature_extracted_concat/$model/$config" ]; then
                config_results=0
                config_expected=0
                
                echo "  Checking configuration: $config"
                
                # Check each PCA mode
                for pca_mode in "${pca_modes[@]}"; do
                    if [ "$pca_mode" = "none" ]; then
                        # Check base configuration directory
                        result_file="feature_extracted_concat/$model/$config/classification_results.json"
                        check_dir="feature_extracted_concat/$model/$config"
                    else
                        # Check PCA subdirectory
                        result_file="feature_extracted_concat/$model/$config/$pca_mode/classification_results.json"
                        check_dir="feature_extracted_concat/$model/$config/$pca_mode"
                    fi
                    
                    if [ -d "$check_dir" ]; then
                        config_expected=$((config_expected + 3))  # 3 classifiers expected
                        total_expected=$((total_expected + 3))
                        
                        if [ -f "$result_file" ]; then
                            # Count classifiers in results file
                            classifier_count=$(grep -o '"logistic"\|"knn"\|"svm_linear"' "$result_file" 2>/dev/null | wc -l)
                            config_results=$((config_results + classifier_count))
                            model_results=$((model_results + classifier_count))
                            
                            echo "    $pca_mode: $classifier_count/3 classifiers completed"
                            
                            # Check for convergence warnings
                            convergence_warnings=$(grep -o '"convergence_warning": true' "$result_file" 2>/dev/null | wc -l)
                            if [ "$convergence_warnings" -gt 0 ]; then
                                echo "      ⚠️  $convergence_warnings convergence warnings found"
                            fi
                        else
                            echo "    $pca_mode: 0/3 classifiers (results file missing)"
                        fi
                    else
                        echo "    $pca_mode: Directory not found - skipping"
                    fi
                done
                
                echo "  $config: $config_results/$config_expected classification results found"
                
                # Check feature dimensions for this config
                sample_feature_file="feature_extracted_concat/$model/$config/PCA_32/test_split_features.npy"
                if [ -f "$sample_feature_file" ]; then
                    echo "     Sample feature file found: $(basename $sample_feature_file)"
                    python3 -c "
import numpy as np
try:
    features = np.load('$sample_feature_file')
    print(f'    Feature shape: {features.shape} (reduced from high-dim concatenation)')
except Exception as e:
    print(f'    Error loading features: {e}')
" 2>/dev/null
                fi
                
            else
                echo "   Configuration $config not found"
            fi
        done
        
        echo ""
        echo "  Total for $model: $model_results/$total_expected classification results"
        
        # Calculate success rate
        if [ "$total_expected" -gt 0 ]; then
            success_rate=$((model_results * 100 / total_expected))
            echo "  Success rate: $success_rate%"
        fi
        
    else
        echo "   Model directory feature_extracted_concat/$model not found"
        echo "  Make sure feature extraction and PCA have been completed first"
    fi
    
    echo ""
    
    # Check consolidated results files
    echo "=== Checking Consolidated Results Files ==="
    consolidated_count=0
    for pca_suffix in "" "_pca_32" "_pca_256" "_pca_95"; do
        consolidated_file="feature_extracted_concat/classification_results${pca_suffix}_${model}.json"
        if [ -f "$consolidated_file" ]; then
            consolidated_count=$((consolidated_count + 1))
            file_size=$(stat -c%s "$consolidated_file" 2>/dev/null || echo "0")
            echo "   Found: classification_results${pca_suffix}_${model}.json (${file_size} bytes)"
            
            # Check number of configurations in consolidated file
            config_count=$(grep -o '"multi_axes_concatenation"\|"single_axis_.*_concatenation"' "$consolidated_file" 2>/dev/null | wc -l)
            echo "     Configurations: $config_count/4"
        else
            echo "   Missing: classification_results${pca_suffix}_${model}.json"
        fi
    done
    
    # Summary
    echo ""
    echo "=== VITS14 CONCATENATION CLASSIFICATION SUMMARY ==="
    echo "Expected: 1 model × 4 configs × 3 classifiers × 4 PCA modes = 48 combinations"
    echo "Individual classification results found: $model_results"
    echo "Consolidated result files found: $consolidated_count/4"
    
    if [ "$model_results" -gt 30 ]; then
        echo " SUCCESS: Most classification results generated!"
        echo "Ready to proceed with full model evaluation (classification_concat_full.yaml)"
    elif [ "$model_results" -gt 10 ]; then
        echo "  PARTIAL SUCCESS: Some classification results generated"
        echo "Check logs for failed combinations"
    else
        echo " ERROR: Very few classification results generated"
        echo "Check configuration and PCA availability"
    fi
    
    # Show sample results structure
    echo ""
    echo "=== Sample Results Analysis ==="
    sample_result=$(find feature_extracted_concat/$model -name "classification_results.json" | head -1)
    if [ -n "$sample_result" ]; then
        echo "Sample result file: $sample_result"
        echo "Classifiers found:"
        grep -o '"logistic"\|"knn"\|"svm_linear"' "$sample_result" 2>/dev/null | sort | uniq
        
        # Check for high-performing results
        echo "Sample performance metrics:"
        python3 -c "
import json
try:
    with open('$sample_result', 'r') as f:
        data = json.load(f)
    for config, classifiers in data.items():
        for clf_name, results in classifiers.items():
            if isinstance(results, dict) and 'test_metrics' in results:
                test_roc = results['test_metrics'].get('roc_auc_weighted', 0)
                cv_roc = results.get('best_cv_score', 0)
                print(f'  {clf_name}: Test ROC-AUC = {test_roc:.4f}, CV ROC-AUC = {cv_roc:.4f}')
except Exception as e:
    print(f'Error reading results: {e}')
" 2>/dev/null
    else
        echo "No classification result files found for analysis"
    fi
    
    # Check disk usage
    echo ""
    echo "=== Disk Usage ==="
    if [ -d "feature_extracted_concat/$model" ]; then
        echo "Total concatenation results disk usage:"
        du -sh feature_extracted_concat/$model/
    fi
    
else
    echo " Classification failed with exit code $?"
    echo "Check the error log for details"
    
    # Show partial results if any
    echo ""
    echo "Checking for any partial results..."
    if [ -d "feature_extracted_concat/$model" ]; then
        partial_results=$(find feature_extracted_concat/$model -name "classification_results.json" | wc -l)
        if [ "$partial_results" -gt 0 ]; then
            echo "Found $partial_results partial result files"
            echo "Partial results locations:"
            find feature_extracted_concat/$model -name "classification_results.json" | head -5
        else
            echo "No classification result files found"
        fi
        
        # Check for common issues
        echo ""
        echo "=== Troubleshooting Information ==="
        echo "1. Check PCA availability:"
        find feature_extracted_concat/$model -name "PCA_*" -type d | head -3
        
        echo "2. Check feature file availability:"
        find feature_extracted_concat/$model -name "*_features.npy" | head -3
        
        echo "3. Check if configurations exist:"
        ls -la feature_extracted_concat/$model/ 2>/dev/null || echo "Model directory not found"
    else
        echo "No partial results found - model directory missing"
    fi
fi

# Print resource usage
echo ""
echo "==========================================="
echo "RESOURCE USAGE SUMMARY:"
echo "Memory usage:"
free -h

echo "CPU usage during job:"
uptime

# Print job completion time
echo "==========================================="
echo "Job finished at: $(date)"
echo "Total job duration: $SECONDS seconds ($((SECONDS/60)) minutes)"
echo "==========================================="
