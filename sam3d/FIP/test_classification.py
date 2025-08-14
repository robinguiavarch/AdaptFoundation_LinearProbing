"""
Test script for F.I.P. binary classification pipeline.

This script validates the classification pipeline setup and data loading
without running the full GridSearchCV to save time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from run_pipeline_classification import FIPLinearProber, load_yaml_config


def test_data_loading():
    """
    Test data loading and validate F.I.P. dataset structure.
    """
    features_path = "feature_extraction_sam3d_fip"
    config_path = "sam3d/FIP/classification_fip.yaml"
    
    if not Path(features_path).exists():
        print(f"Features path not found: {features_path}")
        return False
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        return False
    
    config = load_yaml_config(config_path)
    prober = FIPLinearProber(features_path, config.get('investigation', {}))
    
    combinations = prober.get_available_combinations()
    print(f"Available combinations: {len(combinations)}")
    
    if not combinations:
        print("No valid combinations found")
        return False
    
    # Test first combination
    variant_name, pca_mode = combinations[0]
    print(f"Testing: {variant_name}/PCA_{pca_mode}")
    
    variant_path = Path(features_path) / variant_name / f"PCA_{pca_mode}"
    
    try:
        X_train_val, y_train_val, groups = prober._load_cv_data_with_groups(variant_path)
        X_test, y_test = prober._load_test_data(variant_path)
        
        print(f"Train/val shape: {X_train_val.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Classes: {np.unique(y_train_val)}")
        print(f"Groups (folds): {np.unique(groups)}")
        print(f"Class distribution: {np.bincount(y_train_val)}")
        
        # Validate binary classification
        assert len(np.unique(y_train_val)) == 2, "Should be binary classification"
        assert len(np.unique(groups)) == 5, "Should have 5 folds"
        assert X_train_val.shape[0] == len(y_train_val), "Feature-label mismatch"
        
        print("Data loading test: PASSED")
        return True
        
    except Exception as e:
        print(f"Data loading test: FAILED - {e}")
        return False


def test_classifier_config():
    """
    Test classifier configuration and model setup.
    """
    config = load_yaml_config("sam3d/FIP/classification_fip.yaml")
    prober = FIPLinearProber("feature_extraction_sam3d_fip")
    
    try:
        model, param_grid = prober._get_logistic_regression_config(config['classifier_config'])
        
        print(f"Model: {model}")
        print(f"Param grid keys: {list(param_grid.keys())}")
        print(f"L1 ratio values: {len(param_grid['l1_ratio'])}")
        print(f"C values: {len(param_grid['C'])}")
        
        total_combinations = len(param_grid['l1_ratio']) * len(param_grid['C'])
        print(f"Total hyperparameter combinations: {total_combinations}")
        
        print("Classifier config test: PASSED")
        return True
        
    except Exception as e:
        print(f"Classifier config test: FAILED - {e}")
        return False


def main():
    """
    Run all validation tests.
    """
    print("F.I.P. Classification Pipeline Test")
    print("=" * 40)
    
    tests = [
        test_data_loading,
        test_classifier_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    if all(results):
        print("All tests PASSED - Pipeline ready for execution")
    else:
        print("Some tests FAILED - Check configuration")


if __name__ == "__main__":
    main()