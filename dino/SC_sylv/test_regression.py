"""
Test script for SC_sylv 6D regression pipeline.

This script validates the regression pipeline setup and data loading
without running the full GridSearchCV to save time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from run_pipeline_regression import SCLinearProber, load_yaml_config


def test_data_loading():
    """
    Test data loading and validate SC_sylv dataset structure.
    """
    features_path = "feature_extracted_sc_dinov2"
    config_path = "dino/SC_sylv/regression_sc.yaml"
    
    if not Path(features_path).exists():
        print(f"Features path not found: {features_path}")
        return False
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        return False
    
    config = load_yaml_config(config_path)
    prober = SCLinearProber(features_path, config.get('investigation', {}))
    
    combinations = prober.get_available_combinations()
    print(f"Available combinations: {len(combinations)}")
    
    if not combinations:
        print("No valid combinations found")
        return False
    
    variant_name, pca_mode = combinations[0]
    print(f"Testing: {variant_name}/PCA_{pca_mode}")
    
    variant_path = Path(features_path) / variant_name / f"PCA_{pca_mode}"
    
    try:
        X_train_val, y_train_val, groups = prober._load_cv_data_with_groups(variant_path)
        X_test, y_test = prober._load_test_data(variant_path)
        
        print(f"Train/val shape: {X_train_val.shape}")
        print(f"Train/val labels shape: {y_train_val.shape}")
        print(f"Test shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Groups (folds): {np.unique(groups)}")
        print(f"Label dimensions: {y_train_val.shape[1]}")
        print(f"Sample labels (first subject): {y_train_val[0]}")
        
        assert y_train_val.shape[1] == 6, "Should have 6D labels"
        assert len(np.unique(groups)) == 5, "Should have 5 folds"
        assert X_train_val.shape[0] == len(y_train_val), "Feature-label mismatch"
        assert y_test.shape[1] == 6, "Test labels should be 6D"
        
        print("Data loading test: PASSED")
        return True
        
    except Exception as e:
        print(f"Data loading test: FAILED - {e}")
        return False


def test_regressor_config():
    """
    Test regressor configuration and model setup.
    """
    config = load_yaml_config("dino/SC_sylv/regression_sc.yaml")
    prober = SCLinearProber("feature_extracted_sc_dinov2")
    
    try:
        model, param_grid = prober._get_elasticnet_regression_config(config['regressor_config'])
        
        print(f"Model: {model}")
        print(f"Param grid keys: {list(param_grid.keys())}")
        print(f"L1 ratio values: {len(param_grid['l1_ratio'])}")
        print(f"Alpha values: {len(param_grid['alpha'])}")
        
        total_combinations = len(param_grid['l1_ratio']) * len(param_grid['alpha'])
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Total per variant/PCA: {total_combinations} × 6 dimensions = {total_combinations * 6}")
        
        print("Regressor config test: PASSED")
        return True
        
    except Exception as e:
        print(f"Regressor config test: FAILED - {e}")
        return False


def main():
    """
    Run all validation tests.
    """
    print("SC_sylv 6D Regression Pipeline Test")
    print("=" * 40)
    
    tests = [
        test_data_loading,
        test_regressor_config
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