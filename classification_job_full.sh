#!/bin/bash
#SBATCH --job-name=classification_full_concat
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=CPU                              
#SBATCH --cpus-per-task=65                              
#SBATCH --time=24:00:00               

# Print job details
echo "===========================================" 
echo "Starting Classification Job for Remaining DINOv2 Models with Concatenation Strategy"
echo "Full Evaluation Mode: vitb14, vitl14, vitg14 (vits14 already completed)"
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
python -c "import yaml; print('✅ yaml available:', yaml.__version__)"

# Execute classification for remaining models
echo "Starting classification for remaining DINOv2 models and concatenation configurations..."
echo "Command: python scripts/run_classification.py --config-file configs/classification_concat_full.yaml"
python scripts/run_classification.py --config-file configs/classification_concat_full.yaml

# Check if classification completed successfully
if [ $? -eq 0 ]; then
    echo "==========================================="
    echo "Classification completed successfully!"
    echo "Checking classification results for remaining models concatenation configurations..."
    
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
    
    # Check each remaining model directory
    total_results=0
    expected_results=0
    
    for model in dinov2_vitb14 dinov2_vitl14 dinov2_vitg14; do
        echo "=== Checking $model (Concatenation Strategy) ==="
        
        if [ -d "feature_extracted_concat/$model" ]; then
            model_results=0
            model_expected=0
            
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
                            model_expected=$((model_expected + 3))
                            
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
                        echo "    ✅ Sample feature file found: $(basename $sample_feature_file)"
                        python -c "
import numpy as np
try:
    features = np.load('$sample_feature_file')
    print(f'    Feature shape: {features.shape} (reduced from high-dim concatenation)')
except Exception as e:
    print(f'    Error loading features: {e}')
" 2>/dev/null
                    fi
                    
                else
                    echo "  ❌ Configuration $config not found"
                fi
            done
            
            echo "  Total for $model: $model_results/$model_expected classification results"
            total_results=$((total_results + model_results))
            expected_results=$((expected_results + model_expected))
            
            # Calculate success rate for this model
            if [ "$model_expected" -gt 0 ]; then
                success_rate=$((model_results * 100 / model_expected))
                echo "  Success rate: $success_rate%"
            fi
            
        else
            echo "  ❌ Model directory feature_extracted_concat/$model not found"
            echo "  Make sure feature extraction and PCA have been completed first"
        fi
        echo ""
    done
    
    # Check consolidated results files for remaining models
    echo "=== Checking Consolidated Results Files ==="
    consolidated_count=0
    for model in dinov2_vitb14 dinov2_vitl14 dinov2_vitg14; do
        for pca_suffix in "" "_pca_32" "_pca_256" "_pca_95"; do
            consolidated_file="feature_extracted_concat/classification_results${pca_suffix}_${model}.json"
            if [ -f "$consolidated_file" ]; then
                consolidated_count=$((consolidated_count + 1))
                file_size=$(stat -c%s "$consolidated_file" 2>/dev/null || echo "0")
                echo "  ✅ Found: classification_results${pca_suffix}_${model}.json (${file_size} bytes)"
                
                # Check number of configurations in consolidated file
                config_count=$(grep -o '"multi_axes_concatenation"\|"single_axis_.*_concatenation"' "$consolidated_file" 2>/dev/null | wc -l)
                echo "     Configurations: $config_count/4"
            else
                echo "  ❌ Missing: classification_results${pca_suffix}_${model}.json"
            fi
        done
    done
    
    # Summary
    echo ""
    echo "=== REMAINING MODELS CONCATENATION CLASSIFICATION SUMMARY ==="
    echo "Expected: 3 models × 4 configs × 3 classifiers × 4 PCA modes = 144 combinations"
    echo "Individual classification results found: $total_results/$expected_results"
    echo "Consolidated result files found: $consolidated_count/12"
    
    if [ "$total_results" -gt 100 ]; then
        echo "✅ SUCCESS: Most classification results generated for remaining models!"
        echo "Combined with vits14 results, ready for comparative analysis (pooling vs concatenation)"
    elif [ "$total_results" -gt 50 ]; then
        echo "⚠️  PARTIAL SUCCESS: Some classification results generated"
        echo "Check logs for failed combinations"
    else
        echo "❌ ERROR: Very few classification results generated"
        echo "Check configuration and PCA availability"
    fi
    
    # Show sample results structure
    echo ""
    echo "=== Sample Results Analysis ==="
    sample_result=$(find feature_extracted_concat/ -name "classification_results.json" | head -1)
    if [ -n "$sample_result" ]; then
        echo "Sample result file: $sample_result"
        echo "Classifiers found:"
        grep -o '"logistic"\|"knn"\|"svm_linear"' "$sample_result" 2>/dev/null | sort | uniq
        
        # Check for high-performing results
        echo "Sample performance metrics:"
        python -c "
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
    
    # Check total completion status including vits14
    echo ""
    echo "=== COMPLETE PROJECT STATUS ==="
    total_all_models=0
    if [ -d "feature_extracted_concat/dinov2_vits14" ]; then
        vits14_results=$(find feature_extracted_concat/dinov2_vits14 -name "classification_results.json" 2>/dev/null | wc -l)
        echo "vits14 results (from previous job): $vits14_results"
        total_all_models=$((total_all_models + vits14_results))
    fi
    total_all_models=$((total_all_models + total_results))
    
    echo "Total results across all 4 models: $total_all_models"
    echo "Expected total: 4 models × 4 configs × 3 classifiers × 4 PCA modes = 192 combinations"
    
    if [ "$total_all_models" -gt 150 ]; then
        echo "🎉 PROJECT SUCCESS: All models completed!"
        echo "Ready for final comparative analysis!"
    fi
    
    # Check disk usage
    echo ""
    echo "=== Disk Usage ==="
    if [ -d "feature_extracted_concat" ]; then
        echo "Total concatenation results disk usage:"
        du -sh feature_extracted_concat/
    fi
    
else
    echo "❌ Classification failed with exit code $?"
    echo "Check the error log for details"
    
    # Show partial results if any
    echo ""
    echo "Checking for any partial results..."
    partial_results=$(find feature_extracted_concat/ -name "classification_results.json" 2>/dev/null | wc -l)
    if [ "$partial_results" -gt 0 ]; then
        echo "Found $partial_results partial result files"
        echo "Partial results locations:"
        find feature_extracted_concat/ -name "classification_results.json" | head -5
    else
        echo "No classification result files found"
    fi
    
    # Check for common issues
    echo ""
    echo "=== Troubleshooting Information ==="
    echo "1. Check PCA availability:"
    find feature_extracted_concat/ -name "PCA_*" -type d | head -5
    
    echo "2. Check feature file availability:"
    find feature_extracted_concat/ -name "*_features.npy" | head -5
    
    echo "3. Check if model directories exist:"
    ls -la feature_extracted_concat/ 2>/dev/null || echo "feature_extracted_concat directory not found"
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

# Next steps reminder
echo ""
echo "NEXT STEPS AFTER SUCCESSFUL COMPLETION:"
echo "1. Analyze concatenation results across all 4 models (including vits14)"
echo "2. Compare concatenation vs pooling strategies:"
echo "   python scripts/run_analysis.py --features-path feature_extracted_concat"
echo "3. Identify optimal configuration for Phase 2 roadmap"
echo "4. Proceed to comparative analysis and final recommendations"
echo "=========================================="