"""
Test script for Point-M2AE multiclass classification pipeline.

This script validates the classification pipeline setup and data loading
without running the full GridSearchCV to save time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from point_m2ae.run_pipeline_classification_m2ae import PointM2AELinearProberWithInvestigation, load_yaml_config


def test_data_loading():
    """
    Test data loading and validate Point-M2AE dataset structure.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    features_path = "feature_extraction_point_m2ae"
    config_path = "point_m2ae/classification_m2ae.yaml"
    
    if not Path(features_path).exists():
        print(f"Features path not found: {features_path}")
        return False
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        return False
    
    config = load_yaml_config(config_path)
    prober = PointM2AELinearProberWithInvestigation(
        input_base_path=features_path,
        investigation_config=config.get('investigation', {})
    )
    
    approaches = prober.get_available_approaches()
    print(f"Available approaches: {approaches}")
    
    if not approaches:
        print("No valid approaches found")
        return False
    
    # Test first approach with multiple modes
    approach_name = approaches[0]
    modes = ['PCA_32', 'PCA_256', 'raw_features']
    
    for mode in modes:
        if not prober.check_mode_availability(approach_name, mode):
            print(f"Mode {mode} not available for {approach_name}")
            continue
            
        print(f"Testing: {approach_name}/{mode}")
        
        try:
            X_train_val, y_train_val, groups = prober._load_all_cv_data(approach_name, mode)
            X_test, y_test = prober._load_test_data(approach_name, mode)
            
            print(f"  Train/val shape: {X_train_val.shape}")
            print(f"  Test shape: {X_test.shape}")
            print(f"  Classes: {np.unique(y_train_val)}")
            print(f"  Groups (folds): {np.unique(groups)}")
            print(f"  Class distribution: {np.bincount(y_train_val)}")
            
            # Validate multiclass classification
            assert len(np.unique(y_train_val)) == 4, "Should be 4-class classification"
            assert len(np.unique(groups)) == 5, "Should have 5 folds"
            assert X_train_val.shape[0] == len(y_train_val), "Feature-label mismatch"
            
            # Validate expected dimensions
            if mode == 'raw_features':
                expected_dim = 384 if approach_name == 'feat_mean' else 768
            else:
                expected_dim = int(mode.split('_')[1])
            
            assert X_train_val.shape[1] == expected_dim, f"Expected {expected_dim}D, got {X_train_val.shape[1]}D"
            
            print(f"  {mode} test: PASSED")
            
        except Exception as e:
            print(f"  {mode} test: FAILED - {e}")
            return False
    
    print("Data loading test: PASSED")
    return True


def test_mode_detection():
    """
    Test mode detection and path resolution for Point-M2AE features.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    features_path = "feature_extraction_point_m2ae"
    prober = PointM2AELinearProberWithInvestigation(input_base_path=features_path)
    
    test_cases = [
        ('feat_mean', 'PCA_32', 32),
        ('feat_mean', 'PCA_256', 256),
        ('feat_mean', 'raw_features', 384),
        ('feat_mean_max', 'PCA_32', 32),
        ('feat_mean_max', 'PCA_256', 256),
        ('feat_mean_max', 'raw_features', 768)
    ]
    
    try:
        for approach, mode, expected_dim in test_cases:
            features_path_str, detected_dim = prober._load_features_with_mode_detection(approach, mode)
            
            print(f"  {approach}/{mode}: {features_path_str} -> {detected_dim}D")
            
            assert detected_dim == expected_dim, f"Expected {expected_dim}D, got {detected_dim}D"
            
            # Validate path structure
            expected_path = Path(features_path)
            if mode == 'raw_features':
                expected_path = expected_path / approach
            else:
                expected_path = expected_path / approach / mode
            
            assert Path(features_path_str) == expected_path, f"Path mismatch: {features_path_str}"
        
        print("Mode detection test: PASSED")
        return True
        
    except Exception as e:
        print(f"Mode detection test: FAILED - {e}")
        return False


def test_classifier_config():
    """
    Test classifier configuration and model setup.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    config = load_yaml_config("point_m2ae/classification_m2ae.yaml")
    prober = PointM2AELinearProberWithInvestigation(
        classifier_params=config.get('classifier_params', {})
    )
    
    try:
        model, param_grid = prober._get_logistic_regression_config()
        
        print(f"Model: {model}")
        print(f"Param grid keys: {list(param_grid.keys())}")
        print(f"L1 ratio values: {len(param_grid['l1_ratio'])}")
        print(f"C values: {len(param_grid['C'])}")
        
        total_combinations = len(param_grid['l1_ratio']) * len(param_grid['C'])
        print(f"Total hyperparameter combinations: {total_combinations}")
        
        # Validate model configuration
        assert hasattr(model, 'fit'), "Model should have fit method"
        assert 'C' in param_grid, "Parameter grid should contain C values"
        assert 'l1_ratio' in param_grid, "Parameter grid should contain l1_ratio values"
        
        print("Classifier config test: PASSED")
        return True
        
    except Exception as e:
        print(f"Classifier config test: FAILED - {e}")
        return False


def test_investigation_config():
    """
    Test investigation configuration and CV analysis setup.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    config = load_yaml_config("point_m2ae/classification_m2ae.yaml")
    investigation_config = config.get('investigation', {})
    
    try:
        # Validate investigation configuration
        required_keys = ['enabled', 'save_cv_results', 'detailed_fold_analysis']
        for key in required_keys:
            assert key in investigation_config, f"Missing investigation config key: {key}"
        
        print(f"Investigation enabled: {investigation_config['enabled']}")
        print(f"Save CV results: {investigation_config['save_cv_results']}")
        print(f"Detailed fold analysis: {investigation_config['detailed_fold_analysis']}")
        print(f"Outlier threshold: {investigation_config.get('outlier_detection_threshold', 0.05)}")
        
        prober = PointM2AELinearProberWithInvestigation(
            investigation_config=investigation_config
        )
        
        assert prober.investigation_config['enabled'] == investigation_config['enabled']
        
        print("Investigation config test: PASSED")
        return True
        
    except Exception as e:
        print(f"Investigation config test: FAILED - {e}")
        return False


def test_yaml_config_structure():
    """
    Test YAML configuration file structure and required sections.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    config_path = "point_m2ae/classification_m2ae.yaml"
    
    try:
        config = load_yaml_config(config_path)
        
        required_sections = [
            'models', 'configurations', 'classifiers', 'classification_modes',
            'paths', 'classifier_params', 'investigation'
        ]
        
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"
        
        # Validate specific configurations
        assert len(config['configurations']) == 2, "Should have 2 feature approaches"
        assert len(config['classification_modes']) == 3, "Should have 3 classification modes"
        assert 'feat_mean' in config['configurations'], "Should include feat_mean approach"
        assert 'feat_mean_max' in config['configurations'], "Should include feat_mean_max approach"
        assert 'raw_features' in config['classification_modes'], "Should include raw_features mode"
        
        print(f"Models: {config['models']}")
        print(f"Approaches: {config['configurations']}")
        print(f"Modes: {config['classification_modes']}")
        print(f"Input path: {config['paths']['input_base_path']}")
        print(f"Output path: {config['paths']['output_base_path']}")
        
        print("YAML config structure test: PASSED")
        return True
        
    except Exception as e:
        print(f"YAML config structure test: FAILED - {e}")
        return False


def main():
    """
    Run all validation tests for Point-M2AE classification pipeline.
    """
    print("Point-M2AE Classification Pipeline Test")
    print("=" * 50)
    
    tests = [
        test_yaml_config_structure,
        test_mode_detection,
        test_data_loading,
        test_classifier_config,
        test_investigation_config
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("All tests PASSED - Pipeline ready for execution")
        print("Expected combinations: 6 (2 approaches × 3 modes)")
        print("Ready for: python run_pipeline_classification_m2ae.py")
    else:
        print("Some tests FAILED - Check configuration and data availability")
        failed_tests = [test.__name__ for test, result in zip(tests, results) if not result]
        print(f"Failed tests: {failed_tests}")


if __name__ == "__main__":
    main()