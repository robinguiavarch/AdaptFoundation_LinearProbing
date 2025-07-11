"""
Classification pipeline script for AdaptFoundation project.

This script runs linear probing classification on extracted features
with comprehensive evaluation and hyperparameter optimization.
"""

import argparse
import json
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from classification.linear_probing import LinearProber


def run_classification_pipeline(features_path, config_name=None, 
                               classifier_type=None, use_pca=False, pca_mode=95):
    """
    Run classification pipeline on extracted features.
    
    Args:
        features_path (str): Path to feature_extracted directory
        config_name (str, optional): Specific configuration to evaluate
        classifier_type (str, optional): Specific classifier to train
        use_pca (bool): Whether to use PCA-reduced features
        pca_mode (int): PCA mode (95, 256, 32)
    """
    print("=== AdaptFoundation Classification Pipeline ===")
    print(f"Features path: {features_path}")
    print(f"Use PCA: {use_pca}")
    if use_pca:
        print(f"PCA mode: {pca_mode}")
    
    # Initialize linear prober
    prober = LinearProber(features_path)
    
    # Get available configurations
    available_configs = prober.get_available_configurations()
    print(f"Available configurations: {available_configs}")
    
    # Determine configurations to evaluate
    if config_name:
        if config_name not in available_configs:
            print(f"Configuration '{config_name}' not found")
            return
        configs_to_evaluate = [config_name]
    else:
        configs_to_evaluate = available_configs
    
    # Determine classifiers to evaluate
    if classifier_type:
        classifiers_to_evaluate = [classifier_type]
    else:
        classifiers_to_evaluate = ['logistic', 'knn', 'svm_linear']
    
    # Track all results
    all_results = {}
    
    # Evaluate each configuration
    for config in configs_to_evaluate:
        print(f"\n=== Evaluating Configuration: {config} ===")
        
        # Check PCA availability
        pca_available = prober.check_pca_availability(config, pca_mode)
        if use_pca and not pca_available:
            print(f"PCA_{pca_mode} not available for {config}, skipping PCA evaluation")
            continue
        
        config_results = {}
        
        # Evaluate each classifier
        for clf_type in classifiers_to_evaluate:
            print(f"\n--- Training {clf_type} ---")
            start_time = time.time()
            
            try:
                result = prober.train_classifier(config, clf_type, use_pca, pca_mode)
                elapsed_time = time.time() - start_time
                result['training_time'] = elapsed_time
                
                config_results[clf_type] = result
                
                print(f" {clf_type} completed in {elapsed_time:.2f}s")
                print(f"   Best CV score: {result['best_cv_score']:.4f}")
                print(f"   ROC-AUC: {result['roc_auc_score']:.4f}")
                
            except Exception as e:
                print(f" Error training {clf_type}: {str(e)}")
                config_results[clf_type] = {'error': str(e)}
        
        all_results[config] = config_results
    
    # Save results with PCA mode info
    if use_pca:
        suffix = f"_pca_{pca_mode}"
    else:
        suffix = ""
    results_file = Path(features_path) / f"classification_results{suffix}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n=== Classification Pipeline Completed ===")
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print(f"\n=== Summary ===")
    for config, config_results in all_results.items():
        print(f"\nConfiguration: {config}")
        for clf_type, result in config_results.items():
            if 'error' not in result:
                print(f"  {clf_type}: ROC-AUC = {result['roc_auc_score']:.4f}")
            else:
                print(f"  {clf_type}: ERROR")


def main():
    """
    Main entry point for classification script.
    """
    parser = argparse.ArgumentParser(
        description="Run linear probing classification on extracted features"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="feature_extracted",
        help="Path to feature_extracted directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Specific configuration to evaluate (e.g., 'multi_axes_average')"
    )
    
    parser.add_argument(
        "--classifier",
        type=str,
        choices=['logistic', 'knn', 'svm_linear'],
        help="Specific classifier to train"
    )
    
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Use PCA-reduced features"
    )
    
    parser.add_argument(
        "--pca-mode",
        type=int,
        choices=[32, 95, 256],
        default=95,
        help="PCA mode: 32 (Champollion V1), 95 (95%% variance), 256 (Champollion V0)"
    )
    
    args = parser.parse_args()
    
    # Validate features path
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"Features directory does not exist: {features_path}")
        sys.exit(1)
    
    # Run classification pipeline
    run_classification_pipeline(
        str(features_path),
        config_name=args.config,
        classifier_type=args.classifier,
        use_pca=args.use_pca,
        pca_mode=args.pca_mode
    )


if __name__ == "__main__":
    main()