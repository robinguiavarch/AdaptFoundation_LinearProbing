#!/usr/bin/env python3
"""
Classification pipeline for Point-M2AE 45 configurations.

This script runs linear probing classification on Point-M2AE features
from 45 configurations (C1A1-C9A5) with raw features only.
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Standard library imports
import argparse
import json
import time
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

# Project imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    import sklearn.metrics
    print("Classification pipeline initialized for Point-M2AE 45 configurations")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class PointM2AEConfigsLinearProber:
    """
    Linear probing classifier for Point-M2AE 45 configurations.
    
    Handles classification on features from 45 configurations (C1A1-C9A5)
    with raw features only (no PCA).
    
    Attributes:
        input_base_path (Path): Base path to feature_extraction_point_m2ae_cfgs directory
        output_base_path (Path): Base path to point_m2ae_cfgs directory
        random_state (int): Random state for reproducibility
        n_jobs (int): Number of parallel jobs
        classifier_params (Dict): Classifier parameters from YAML
        investigation_config (Dict): Investigation configuration parameters
    """
    
    def __init__(self, input_base_path: str = 'feature_extraction_point_m2ae_cfgs', 
                 output_base_path: str = 'point_m2ae_cfgs',
                 random_state: int = 42, n_jobs: int = -1, 
                 classifier_params: Dict = None, investigation_config: Dict = None):
        """
        Initialize the Point-M2AE configurations linear probing classifier.
        
        Args:
            input_base_path (str): Base path to read features from. Defaults to 'feature_extraction_point_m2ae_cfgs'.
            output_base_path (str): Base path to save results to. Defaults to 'point_m2ae_cfgs'.
            random_state (int): Random state for reproducibility. Defaults to 42.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
            classifier_params (Dict, optional): Custom classifier parameters from YAML.
            investigation_config (Dict, optional): Investigation configuration parameters.
        """
        self.input_base_path = Path(input_base_path)
        self.output_base_path = Path(output_base_path)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.classifier_params = classifier_params or {}
        self.investigation_config = investigation_config or {}
        
        # Validate input path
        if not self.input_base_path.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_base_path}")
        
        # Create output directory if needed
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        print(f"PointM2AEConfigsLinearProber initialized:")
        print(f"  Input path: {self.input_base_path}")
        print(f"  Output path: {self.output_base_path}")
    
    def _get_expected_dim(self, config_name: str) -> int:
        """
        Get expected feature dimension based on configuration aggregation method.
        
        Args:
            config_name (str): Configuration name (e.g., 'C1A1', 'C8A2')
            
        Returns:
            int: Expected feature dimension
        """
        if len(config_name) != 4 or not config_name.startswith('C') or 'A' not in config_name:
            raise ValueError(f"Invalid configuration name: {config_name}")
        
        aggregation_key = config_name[2:]  # Extract 'A1', 'A2', etc.
        
        dimension_map = {
            'A1': 384,   # mean
            'A2': 1536,  # mean+std+min+max
            'A3': 576,   # multi-level
            'A4': 384,   # adaptive
            'A5': 384    # attention
        }
        
        if aggregation_key not in dimension_map:
            raise ValueError(f"Unknown aggregation method: {aggregation_key}")
        
        return dimension_map[aggregation_key]
    
    def _load_all_cv_data(self, config_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all CV splits with group labels for LeaveOneGroupOut.
        
        Args:
            config_name (str): Configuration name (e.g., 'C1A1')
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features (X), labels (y), groups
        """
        config_path = self.input_base_path / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_path}")
        
        expected_dim = self._get_expected_dim(config_name)
        
        all_features = []
        all_labels = []
        all_groups = []
        
        # Load all 5 CV splits
        for fold_id in range(5):
            split_name = f"train_val_split_{fold_id}"
            
            features_file = config_path / f"{split_name}_features.npy"
            metadata_file = config_path / f"{split_name}_metadata.csv"
            
            if not features_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(f"Missing files for {config_name}/{split_name}")
            
            features = np.load(features_file)
            metadata = pd.read_csv(metadata_file)
            labels = metadata['Label'].values
            
            # Validate dimensionality
            if features.shape[1] != expected_dim:
                print(f"Warning: Expected {expected_dim}D, got {features.shape[1]}D for {config_name}")
            
            # Group ID = fold ID (for LeaveOneGroupOut)
            groups = np.full(len(features), fold_id)
            
            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups)
            
            print(f"    Loaded {split_name}: {features.shape}")
        
        # Concatenate all data
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        groups = np.concatenate(all_groups, axis=0)
        
        print(f"    Total CV data: {X.shape}")
        return X, y, groups
    
    def _load_test_data(self, config_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test split data separately from training data.
        
        Args:
            config_name (str): Configuration name (e.g., 'C1A1')
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Test features and test labels
        """
        config_path = self.input_base_path / config_name
        expected_dim = self._get_expected_dim(config_name)
        
        test_features_file = config_path / "test_split_features.npy"
        test_metadata_file = config_path / "test_split_metadata.csv"
        
        if not test_features_file.exists() or not test_metadata_file.exists():
            raise FileNotFoundError(f"Missing test files for {config_name}")
        
        X_test = np.load(test_features_file)
        test_metadata = pd.read_csv(test_metadata_file)
        y_test = test_metadata['Label'].values
        
        # Validate dimensionality
        if X_test.shape[1] != expected_dim:
            print(f"Warning: Expected {expected_dim}D, got {X_test.shape[1]}D for {config_name} test")
        
        print(f"    Test data: {X_test.shape}")
        return X_test, y_test
    
    def _get_logistic_regression_config(self) -> Tuple[LogisticRegression, Dict]:
        """
        Get logistic regression model and parameter grid with YAML-configurable parameters.
        
        Returns:
            Tuple[LogisticRegression, Dict]: Model and parameter grid
        """
        # Get parameters from YAML config or use defaults
        logistic_params = self.classifier_params.get('logistic', {})
        
        max_iter = logistic_params.get('max_iter', 20000)
        solver = logistic_params.get('solver', 'saga')
        penalty = logistic_params.get('penalty', 'elasticnet')
        n_jobs = logistic_params.get('n_jobs', self.n_jobs)
        random_state = logistic_params.get('random_state', self.random_state)
        
        print(f"    Logistic config: max_iter={max_iter}, solver={solver}, penalty={penalty}")
        
        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        # Parameter grid from YAML or defaults
        c_values = logistic_params.get('C_values', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        l1_ratio_values = logistic_params.get('l1_ratio_values', np.linspace(0, 1, 11).tolist())
        
        parameters = {
            'C': c_values,
            'l1_ratio': l1_ratio_values
        }
        
        return model, parameters
    
    def _save_cv_results_detailed(self, config_name: str, cv_results_df: pd.DataFrame, clf: GridSearchCV) -> None:
        """
        Save complete cv_results_ for detailed investigation.
        
        Args:
            config_name (str): Name of the configuration
            cv_results_df (pd.DataFrame): Complete cv_results_ DataFrame
            clf (GridSearchCV): Fitted GridSearchCV object
        """
        if not self.investigation_config.get('save_cv_results', False):
            return
        
        output_dir = self.output_base_path / config_name / "raw_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cv_results_ complete as CSV
        cv_results_file = output_dir / "cv_results_complete.csv"
        cv_results_df.to_csv(cv_results_file, index=False)
        
        # Save cv_results_ complete as JSON
        cv_results_json_file = output_dir / "cv_results_complete.json"
        cv_results_dict = cv_results_df.to_dict('records')
        with open(cv_results_json_file, 'w') as f:
            json.dump(cv_results_dict, f, indent=2, default=str)
    
    def train_classifier(self, config_name: str) -> Dict:
        """
        Train logistic regression classifier for a specific configuration.
        
        Args:
            config_name (str): Configuration name (e.g., 'C1A1')
        
        Returns:
            Dict: Complete training results including test metrics and diagnostics
        """
        print(f"Training logistic regression on {config_name}")
        
        # Load training and validation data
        print("  Loading train/val data...")
        start_load = time.time()
        X_train_val, y_train_val, groups = self._load_all_cv_data(config_name)
        
        # Load test data separately
        print("  Loading test data...")
        X_test, y_test = self._load_test_data(config_name)
        load_time = time.time() - start_load
        
        print(f"  Data loaded in {load_time:.2f}s")
        print(f"  Feature dimensionality: {X_train_val.shape[1]}D")
        print(f"  Classes: {len(np.unique(y_train_val))}")
        
        # Get model configuration
        model, parameters = self._get_logistic_regression_config()
        
        # Setup cross-validation
        logo = LeaveOneGroupOut()
        cv_splits = list(logo.split(X_train_val, y_train_val, groups=groups))
        
        print(f"  Starting GridSearchCV with {len(cv_splits)} CV splits...")
        start_gridsearch = time.time()
        
        # Grid search with comprehensive diagnostics
        clf = GridSearchCV(
            model, parameters,
            cv=cv_splits,
            scoring='roc_auc_ovr_weighted',
            refit=True,
            n_jobs=self.n_jobs,
            return_train_score=True,
            verbose=1
        )
        
        clf.fit(X_train_val, y_train_val)
        gridsearch_time = time.time() - start_gridsearch
        
        print(f"  GridSearchCV completed in {gridsearch_time:.2f}s")
        
        # Extract cv_results_ for investigation
        cv_results_df = pd.DataFrame(clf.cv_results_)
        best_idx = clf.best_index_
        
        # Save detailed cv_results_ if investigation enabled
        self._save_cv_results_detailed(config_name, cv_results_df, clf)
        
        # Final evaluation on test set
        print("  Evaluating on test set...")
        start_test = time.time()
        
        best_model = clf.best_estimator_
        best_model.fit(X_train_val, y_train_val)
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)
        
        # Calculate test metrics
        scorer = sklearn.metrics.get_scorer('roc_auc_ovr_weighted')
        test_roc_auc_weighted = scorer(best_model, X_test, y_test)
        test_accuracy = np.mean(y_test == y_test_pred)
        test_time = time.time() - start_test
        
        # Extract CV diagnostics
        print("  Computing diagnostics...")
        
        # CV metrics and overfitting analysis
        best_train_score = cv_results_df.iloc[best_idx]['mean_train_score']
        best_val_score = cv_results_df.iloc[best_idx]['mean_test_score']
        overfitting_gap = best_train_score - best_val_score
        cv_stability = cv_results_df.iloc[best_idx]['std_test_score']
        
        # Convergence check
        convergence_warning = False
        if hasattr(best_model, 'n_iter_'):
            max_iter = best_model.max_iter
            actual_iter = best_model.n_iter_[0] if len(best_model.n_iter_) > 0 else 0
            convergence_warning = actual_iter >= max_iter
        
        total_time = load_time + gridsearch_time + test_time
        
        # Compile results
        results = {
            'config_name': config_name,
            'classifier_type': 'logistic',
            'best_params': clf.best_params_,
            'best_cv_score': clf.best_score_,
            'test_metrics': {
                'roc_auc_weighted': test_roc_auc_weighted,
                'accuracy': test_accuracy,
                'n_test_samples': len(y_test)
            },
            'cv_metrics': {
                'roc_auc_weighted': clf.best_score_,
                'mean_train_score': best_train_score,
                'mean_val_score': best_val_score,
                'overfitting_gap': overfitting_gap,
                'cv_stability': cv_stability
            },
            'diagnostics': {
                'convergence_warning': convergence_warning,
                'overfitting_gap': overfitting_gap,
                'overfitting_severity': 'high' if overfitting_gap > 0.1 else 'medium' if overfitting_gap > 0.05 else 'low',
                'cv_stability': cv_stability
            },
            'data_info': {
                'train_val_shape': X_train_val.shape,
                'test_shape': X_test.shape,
                'n_classes': len(np.unique(y_train_val)),
                'n_cv_splits': len(cv_splits),
                'feature_dimensionality': X_train_val.shape[1]
            },
            'timing': {
                'load_time': load_time,
                'gridsearch_time': gridsearch_time,
                'test_eval_time': test_time,
                'total_time': total_time
            },
            'cv_results_summary': {
                'best_index': best_idx,
                'n_combinations_tested': len(cv_results_df)
            }
        }
        
        print(f"  Best CV score: {clf.best_score_:.4f}")
        print(f"  Test ROC-AUC: {test_roc_auc_weighted:.4f}")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Overfitting: {results['diagnostics']['overfitting_severity']}")
        
        return results
    
    def get_available_configurations(self) -> List[str]:
        """
        Get list of available configurations for classification.
        
        Returns:
            List[str]: List of available configuration names
        """
        configurations = []
        for path in self.input_base_path.iterdir():
            if path.is_dir() and len(path.name) == 4 and path.name.startswith('C') and 'A' in path.name:
                configurations.append(path.name)
        return sorted(configurations)
    
    def check_configuration_availability(self, config_name: str) -> bool:
        """
        Check if features are available for a configuration.
        
        Args:
            config_name (str): Configuration name
        
        Returns:
            bool: True if feature files are available
        """
        config_path = self.input_base_path / config_name
        
        required_files = [
            "test_split_features.npy",
            "train_val_split_0_features.npy"
        ]
        return config_path.exists() and all((config_path / f).exists() for f in required_files)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_m2ae_configs_classification_plan(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate classification execution plan from YAML configuration for 45 configurations.
    
    Args:
        config (Dict[str, Any]): Loaded YAML configuration
        
    Returns:
        List[Dict[str, Any]]: List of classification tasks
    """
    tasks = []
    models = config['models']
    configurations = config['configurations']
    classifiers = config['classifiers']
    classification_modes = config['classification_modes']
    
    for model in models:
        for configuration in configurations:
            for classifier in classifiers:
                for mode in classification_modes:
                    task = {
                        'model': model,
                        'configuration': configuration,
                        'classifier': classifier,
                        'mode': mode
                    }
                    tasks.append(task)
    
    print(f"Generated {len(tasks)} Point-M2AE configurations classification tasks")
    print(f"Models: {len(models)}, Configurations: {len(configurations)}, Classifiers: {len(classifiers)}, Modes: {len(classification_modes)}")
    
    return tasks


def validate_m2ae_configs_task(task: Dict[str, Any], input_base_path: str, 
                              validation_config: Dict[str, Any]) -> bool:
    """
    Validate that a Point-M2AE configurations classification task can be executed.
    
    Args:
        task (Dict[str, Any]): Task to validate
        input_base_path (str): Path to feature_extraction_point_m2ae_cfgs directory
        validation_config (Dict[str, Any]): Validation configuration
        
    Returns:
        bool: True if task is valid
    """
    if not validation_config.get('check_feature_files', True):
        return True
    
    prober = PointM2AEConfigsLinearProber(input_base_path=input_base_path)
    
    try:
        return prober.check_configuration_availability(task['configuration'])
    except Exception as e:
        print(f"Skipping {task['configuration']}: {e}")
        return False


def execute_m2ae_configs_task(task: Dict[str, Any], input_base_path: str, 
                             output_base_path: str, reporting_config: Dict[str, Any], 
                             yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single Point-M2AE configurations classification task.
    
    Args:
        task (Dict[str, Any]): Task to execute
        input_base_path (str): Path to feature_extraction_point_m2ae_cfgs directory
        output_base_path (str): Path to point_m2ae_cfgs directory
        reporting_config (Dict[str, Any]): Reporting configuration
        yaml_config (Dict[str, Any]): Full YAML configuration
        
    Returns:
        Dict[str, Any]: Task results
    """
    model = task['model']
    configuration = task['configuration']
    classifier = task['classifier']
    mode = task['mode']
    
    if reporting_config.get('verbose', True):
        print(f"Executing: {configuration} | {classifier} | {mode}")
    
    try:
        # Extract classifier and investigation parameters from YAML config
        classifier_params = yaml_config.get('classifier_params', {})
        investigation_config = yaml_config.get('investigation', {})
        
        prober = PointM2AEConfigsLinearProber(
            input_base_path=input_base_path,
            output_base_path=output_base_path,
            classifier_params=classifier_params,
            investigation_config=investigation_config
        )
        
        start_time = time.time()
        result = prober.train_classifier(configuration)
        total_time = time.time() - start_time
        
        result['task_metadata'] = {
            'model': model,
            'configuration': configuration,
            'classifier': classifier,
            'mode': mode,
            'total_pipeline_time': total_time
        }
        
        if reporting_config.get('verbose', True):
            cv_score = result['best_cv_score']
            test_score = result['test_metrics']['roc_auc_weighted']
            overfitting = result['diagnostics']['overfitting_severity']
            convergence = "OK" if not result['diagnostics']['convergence_warning'] else "WARNING"
            print(f"  Completed: CV={cv_score:.4f} | Test={test_score:.4f} | Overfit={overfitting} | Conv={convergence}")
        
        return {'status': 'success', 'result': result}
        
    except Exception as e:
        error_msg = str(e)
        print(f"  Error: {error_msg}")
        return {'status': 'error', 'error': error_msg, 'task': task}


def save_m2ae_configs_results(task: Dict[str, Any], result_data: Dict[str, Any], 
                             output_base_path: str, output_config: Dict[str, Any]) -> None:
    """
    Save Point-M2AE configurations classification results.
    
    Args:
        task (Dict[str, Any]): Executed task
        result_data (Dict[str, Any]): Task results
        output_base_path (str): Path to point_m2ae_cfgs directory
        output_config (Dict[str, Any]): Output configuration
    """
    if result_data['status'] != 'success':
        return
    
    result = result_data['result']
    configuration = task['configuration']
    classifier = task['classifier']
    mode = task['mode']
    
    # Save individual results
    if output_config.get('save_individual', True):
        config_dir = Path(output_base_path) / configuration / mode
        config_dir.mkdir(parents=True, exist_ok=True)
        results_file = config_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({configuration: {classifier: result}}, f, indent=2, default=str)
    
    # Save consolidated results
    if output_config.get('save_consolidated', True):
        consolidated_file = Path(output_base_path) / "classification_results_point_m2ae_cfgs.json"
        
        if consolidated_file.exists():
            with open(consolidated_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {
                'experiment_info': {
                    'pipeline': 'Point-M2AE 45 Configurations Classification',
                    'model': 'point_m2ae_encoder',
                    'total_configurations': 45,
                    'total_runtime': 0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'results': {}
            }
        
        # Structure: {configuration: {mode: {classifier: result}}}
        if configuration not in all_results['results']:
            all_results['results'][configuration] = {}
        if mode not in all_results['results'][configuration]:
            all_results['results'][configuration][mode] = {}
        all_results['results'][configuration][mode][classifier] = result
        
        with open(consolidated_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)


def run_yaml_m2ae_configs_classification(config_file: str) -> None:
    """
    Run Point-M2AE configurations classification pipeline using YAML configuration.
    
    Args:
        config_file (str): Path to YAML configuration file
    """
    print("=" * 80)
    print("POINT-M2AE 45 CONFIGURATIONS CLASSIFICATION")
    print("=" * 80)
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    input_base_path = config['paths']['input_base_path']
    output_base_path = config['paths']['output_base_path']
    investigation_enabled = config.get('investigation', {}).get('enabled', False)
    
    print(f"Input base path (reading): {input_base_path}")
    print(f"Output base path (writing): {output_base_path}")
    print(f"Investigation enabled: {investigation_enabled}")
    print(f"Configurations: 45 (C1A1-C9A5, excluding C10A1-C10A5)")
    
    tasks = get_m2ae_configs_classification_plan(config)
    
    # Validate tasks (check input path)
    valid_tasks = []
    for task in tasks:
        if validate_m2ae_configs_task(task, input_base_path, config['validation']):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    # Execute tasks
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"\n[{i}/{len(valid_tasks)}] {task['configuration']} - {task['mode']}")
        
        result_data = execute_m2ae_configs_task(task, input_base_path, output_base_path, config['reporting'], config)
        results.append({'task': task, 'result': result_data})
        
        save_m2ae_configs_results(task, result_data, output_base_path, config['output'])
    
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r['result']['status'] == 'success')
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print("POINT-M2AE 45 CONFIGURATIONS CLASSIFICATION COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Tasks: {successful} successful, {failed} failed")
    print(f"Investigation enabled: {investigation_enabled}")
    
    if successful > 0:
        print(f"\nResults structure:")
        print(f"point_m2ae_cfgs/")
        print(f"  ├── C1A1/raw_features/classification_results.json")
        print(f"  ├── C1A2/raw_features/classification_results.json")
        print(f"  ├── ...")
        print(f"  ├── C9A5/raw_features/classification_results.json")
        print(f"  └── classification_results_point_m2ae_cfgs.json")
        
        # Generate performance summary
        print(f"\nPerformance Summary:")
        for r in results:
            if r['result']['status'] == 'success':
                task = r['task']
                result = r['result']['result']
                cv_score = result['best_cv_score']
                test_score = result['test_metrics']['roc_auc_weighted']
                dimensionality = result['data_info']['feature_dimensionality']
                print(f"  {task['configuration']}: CV={cv_score:.4f} | Test={test_score:.4f} | Dim={dimensionality}")
        
        print(f"\nConsolidated results:")
        print(f"point_m2ae_cfgs/classification_results_point_m2ae_cfgs.json")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if r['result']['status'] == 'error':
                task = r['task']
                print(f"  {task['configuration']}: {r['result']['error']}")


def run_single_m2ae_config(input_base_path: str, output_base_path: str, 
                          config_name: str, classifier_params: Dict = None, 
                          investigation_config: Dict = None) -> None:
    """
    Run classification for a single Point-M2AE configuration.
    
    Args:
        input_base_path (str): Path to feature_extraction_point_m2ae_cfgs directory
        output_base_path (str): Path to point_m2ae_cfgs directory
        config_name (str): Configuration name
        classifier_params (Dict, optional): Custom classifier parameters
        investigation_config (Dict, optional): Investigation configuration
    """
    print("=" * 80)
    print("POINT-M2AE CONFIGURATIONS CLASSIFICATION - SINGLE CONFIGURATION")
    print("=" * 80)
    print(f"Input base path (reading): {input_base_path}")
    print(f"Output base path (writing): {output_base_path}")
    print(f"Configuration: {config_name}")
    print(f"Investigation enabled: {investigation_config.get('enabled', False) if investigation_config else False}")
    
    try:
        # Initialize prober
        prober = PointM2AEConfigsLinearProber(
            input_base_path=input_base_path,
            output_base_path=output_base_path,
            classifier_params=classifier_params or {},
            investigation_config=investigation_config or {}
        )
        
        available_configurations = prober.get_available_configurations()
        print(f"Available configurations: {available_configurations}")
        
        if config_name not in available_configurations:
            print(f"Error: Configuration '{config_name}' not found.")
            print(f"Available configurations: {available_configurations}")
            return
        
        # Check configuration availability
        if not prober.check_configuration_availability(config_name):
            print(f"Error: Features not available for {config_name}")
            return
        
        # Train classifier
        start_time = time.time()
        result = prober.train_classifier(config_name)
        elapsed_time = time.time() - start_time
        
        print(f"\nSingle configuration classification completed in {elapsed_time:.2f}s")
        print(f"Successfully processed {config_name}")
        print(f"  Best CV score: {result['best_cv_score']:.4f}")
        print(f"  Test ROC-AUC: {result['test_metrics']['roc_auc_weighted']:.4f}")
        print(f"  Test accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"  Overfitting: {result['diagnostics']['overfitting_severity']}")
        print(f"  Dimensionality: {result['data_info']['feature_dimensionality']}D")
        
        # Save results
        config_dir = Path(output_base_path) / config_name / "raw_features"
        config_dir.mkdir(parents=True, exist_ok=True)
        results_file = config_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({config_name: {'logistic': result}}, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        
        if investigation_config and investigation_config.get('save_cv_results', False):
            print(f"Investigation files saved:")
            print(f"  - cv_results_complete.csv")
            print(f"  - cv_results_complete.json")
        
    except Exception as e:
        print(f"Error processing {config_name}: {str(e)}")
        return


def main():
    """
    Main entry point for Point-M2AE configurations classification script.
    """
    parser = argparse.ArgumentParser(
        description="Run linear probing classification on Point-M2AE 45 configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # YAML Configuration Mode (RECOMMENDED)
    parser.add_argument(
        "--config",
        type=str,
        default="classification_m2ae_cfgs.yaml",
        help="Path to YAML configuration file"
    )
    
    # Single configuration mode
    parser.add_argument(
        "--configuration",
        type=str,
        help="Process only a specific configuration (e.g., C1A1, C8A2)"
    )
    
    # Manual Mode parameters
    parser.add_argument(
        "--input-base-path",
        type=str,
        default="feature_extraction_point_m2ae_cfgs",
        help="Path to feature_extraction_point_m2ae_cfgs directory (for reading features)"
    )
    
    parser.add_argument(
        "--output-base-path",
        type=str,
        default="point_m2ae_cfgs",
        help="Path to point_m2ae_cfgs directory (for saving results)"
    )
    
    args = parser.parse_args()
    
    # Construct config path relative to script location
    script_dir = Path(__file__).parent
    
    if args.configuration:
        # Single configuration mode
        print("=== Single Point-M2AE Configuration Classification ===")
        
        input_path = Path(args.input_base_path)
        if not input_path.exists():
            print(f"Input directory does not exist: {input_path}")
            sys.exit(1)
        
        output_path = Path(args.output_base_path)
        
        # Default investigation config for single configuration
        investigation_config = {
            'enabled': True,
            'save_cv_results': True,
            'detailed_fold_analysis': True,
            'outlier_detection_threshold': 0.05
        }
        
        run_single_m2ae_config(
            str(input_path),
            str(output_path),
            args.configuration,
            investigation_config=investigation_config
        )
    else:
        # YAML Configuration Mode
        config_path = Path(__file__).parent / args.config
        if not config_path.exists():
            print(f"Configuration file does not exist: {config_path}")
            sys.exit(1)
        
        try:
            run_yaml_m2ae_configs_classification(str(config_path))
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()