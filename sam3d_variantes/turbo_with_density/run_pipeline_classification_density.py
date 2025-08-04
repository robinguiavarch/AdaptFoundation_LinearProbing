#!/usr/bin/env python3
"""
Classification pipeline for SAM-Med3D density optimization approaches.

This script runs linear probing classification on density-optimized features
with comprehensive evaluation and hyperparameter optimization.

Usage:
python sam3d_variantes/turbo_with_density/run_pipeline_classification_density.py
python sam3d_variantes/turbo_with_density/run_pipeline_classification_density.py --config classification_density.yaml
python sam3d_variantes/turbo_with_density/run_pipeline_classification_density.py --approach flatten_baseline --pca-mode 32
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
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Project imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    import sklearn.metrics
    print("Classification pipeline initialized for SAM-Med3D density approaches")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class DensityLinearProber:
    """
    Linear probing classifier for SAM-Med3D density approach evaluation.
    
    Specialized for density-optimized features with support for variable dimensions
    and consistent evaluation across baseline, masking, and linear_weighting approaches.
    """
    
    def __init__(self, features_base_path: str, model_name: str = 'sam_med3d_turbo_density',
                 random_state: int = 42, n_jobs: int = -1, 
                 classifier_params: Dict = None):
        """
        Initialize the density linear probing classifier.
        
        Args:
            features_base_path (str): Base path to feature_extraction_density directory
            model_name (str): Model name. Defaults to 'sam_med3d_turbo_density'.
            random_state (int): Random state for reproducibility. Defaults to 42.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
            classifier_params (Dict, optional): Custom classifier parameters from YAML.
        """
        self.features_base_path = Path(features_base_path)
        self.model_name = model_name
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.classifier_params = classifier_params or {}
        
        # Validate paths
        self.model_path = self.features_base_path / model_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        print(f"DensityLinearProber initialized:")
        print(f"  Features path: {self.model_path}")
        print(f"  Random state: {self.random_state}")
        print(f"  Parallel jobs: {self.n_jobs}")
    
    def _load_all_cv_data(self, approach_name: str, pca_mode: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all CV splits with group labels for LeaveOneGroupOut.
        
        Args:
            approach_name (str): Approach name (e.g., 'flatten_baseline')
            pca_mode (int): PCA mode (32, 256, 95)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features (X), labels (y), groups
        """
        approach_path = self.model_path / approach_name / f"PCA_{pca_mode}"
        
        if not approach_path.exists():
            raise FileNotFoundError(f"PCA directory not found: {approach_path}")
        
        all_features = []
        all_labels = []
        all_groups = []
        
        # Load all 5 CV splits
        for fold_id in range(5):
            split_name = f"train_val_split_{fold_id}"
            
            features_file = approach_path / f"{split_name}_features.npy"
            metadata_file = approach_path / f"{split_name}_metadata.csv"
            
            if not features_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(f"Missing files for {approach_name}/PCA_{pca_mode}/{split_name}")
            
            features = np.load(features_file)
            metadata = pd.read_csv(metadata_file)
            labels = metadata['Label'].values
            
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
    
    def _load_test_data(self, approach_name: str, pca_mode: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test split data separately from training data.
        
        Args:
            approach_name (str): Approach name (e.g., 'flatten_baseline')
            pca_mode (int): PCA mode (32, 256, 95)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Test features and test labels
        """
        approach_path = self.model_path / approach_name / f"PCA_{pca_mode}"
        
        test_features_file = approach_path / "test_split_features.npy"
        test_metadata_file = approach_path / "test_split_metadata.csv"
        
        if not test_features_file.exists() or not test_metadata_file.exists():
            raise FileNotFoundError(f"Missing test files for {approach_name}/PCA_{pca_mode}")
        
        X_test = np.load(test_features_file)
        test_metadata = pd.read_csv(test_metadata_file)
        y_test = test_metadata['Label'].values
        
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
    
    def train_classifier(self, approach_name: str, pca_mode: int) -> Dict:
        """
        Train logistic regression classifier with comprehensive evaluation.
        
        Args:
            approach_name (str): Density approach name
            pca_mode (int): PCA mode (32, 256, 95)
        
        Returns:
            Dict: Complete training results including test metrics and diagnostics
        """
        print(f"Training logistic regression on {approach_name}/PCA_{pca_mode}")
        
        # Load training and validation data
        print("  Loading train/val data...")
        start_load = time.time()
        X_train_val, y_train_val, groups = self._load_all_cv_data(approach_name, pca_mode)
        
        # Load test data separately
        print("  Loading test data...")
        X_test, y_test = self._load_test_data(approach_name, pca_mode)
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
        cv_results_df = pd.DataFrame(clf.cv_results_)
        best_idx = clf.best_index_
        
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
            'approach_name': approach_name,
            'pca_mode': pca_mode,
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
    
    def get_available_approaches(self) -> List[str]:
        """
        Get list of available density approaches for classification.
        
        Returns:
            List[str]: List of available approach names
        """
        approaches = []
        for path in self.model_path.iterdir():
            if path.is_dir() and path.name.startswith('flatten_'):
                approaches.append(path.name)
        return approaches
    
    def check_pca_availability(self, approach_name: str, pca_mode: int) -> bool:
        """
        Check if PCA-reduced features are available for an approach.
        
        Args:
            approach_name (str): Approach name
            pca_mode (int): PCA mode
        
        Returns:
            bool: True if PCA files are available
        """
        pca_path = self.model_path / approach_name / f"PCA_{pca_mode}"
        required_files = [
            "test_split_features.npy",
            "train_val_split_0_features.npy"
        ]
        return pca_path.exists() and all((pca_path / f).exists() for f in required_files)


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


def get_density_classification_plan(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate classification execution plan from YAML configuration.
    
    Args:
        config (Dict[str, Any]): Loaded YAML configuration
        
    Returns:
        List[Dict[str, Any]]: List of classification tasks
    """
    tasks = []
    models = config['models']
    configurations = config['configurations']  # density approaches
    classifiers = config['classifiers']
    pca_modes = config['pca_modes']
    
    for model in models:
        for approach in configurations:
            for classifier in classifiers:
                for pca_mode in pca_modes:
                    task = {
                        'model': model,
                        'approach': approach,
                        'classifier': classifier,
                        'pca_mode': pca_mode
                    }
                    tasks.append(task)
    
    print(f"Generated {len(tasks)} density classification tasks")
    print(f"Models: {len(models)}, Approaches: {len(configurations)}, Classifiers: {len(classifiers)}, PCA modes: {len(pca_modes)}")
    
    return tasks


def validate_density_task(task: Dict[str, Any], features_base_path: str, 
                         validation_config: Dict[str, Any]) -> bool:
    """
    Validate that a density classification task can be executed.
    
    Args:
        task (Dict[str, Any]): Task to validate
        features_base_path (str): Path to feature_extraction_density directory
        validation_config (Dict[str, Any]): Validation configuration
        
    Returns:
        bool: True if task is valid
    """
    if not validation_config.get('check_feature_files', True):
        return True
    
    approach_path = Path(features_base_path) / task['model'] / task['approach']
    pca_path = approach_path / f"PCA_{task['pca_mode']}"
    
    if not pca_path.exists():
        print(f"Skipping {task['approach']}/PCA_{task['pca_mode']}: Directory not found")
        return False
    
    # Check required files
    required_files = [
        "test_split_features.npy",
        "train_val_split_0_features.npy"
    ]
    
    for file_name in required_files:
        if not (pca_path / file_name).exists():
            print(f"Skipping {task['approach']}/PCA_{task['pca_mode']}: Missing {file_name}")
            return False
    
    return True


def execute_density_task(task: Dict[str, Any], features_base_path: str, 
                        reporting_config: Dict[str, Any], 
                        yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single density classification task.
    
    Args:
        task (Dict[str, Any]): Task to execute  
        features_base_path (str): Path to feature_extraction_density directory
        reporting_config (Dict[str, Any]): Reporting configuration
        yaml_config (Dict[str, Any]): Full YAML configuration
        
    Returns:
        Dict[str, Any]: Task results
    """
    model = task['model']
    approach = task['approach']
    classifier = task['classifier']
    pca_mode = task['pca_mode']
    
    if reporting_config.get('verbose', True):
        print(f"Executing: {model} | {approach} | {classifier} | PCA_{pca_mode}")
    
    try:
        # Extract classifier parameters from YAML config
        classifier_params = yaml_config.get('classifier_params', {})
        
        prober = DensityLinearProber(
            features_base_path=features_base_path,
            model_name=model,
            classifier_params=classifier_params
        )
        
        start_time = time.time()
        result = prober.train_classifier(approach, pca_mode)
        total_time = time.time() - start_time
        
        result['task_metadata'] = {
            'model': model,
            'approach': approach,
            'classifier': classifier,
            'pca_mode': pca_mode,
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


def save_density_results(task: Dict[str, Any], result_data: Dict[str, Any], 
                        features_base_path: str, output_config: Dict[str, Any]) -> None:
    """
    Save density classification results.
    
    Args:
        task (Dict[str, Any]): Executed task
        result_data (Dict[str, Any]): Task results
        features_base_path (str): Path to feature_extraction_density directory
        output_config (Dict[str, Any]): Output configuration
    """
    if result_data['status'] != 'success':
        return
    
    result = result_data['result']
    model = task['model']
    approach = task['approach']
    classifier = task['classifier']
    pca_mode = task['pca_mode']
    
    # Save individual results
    if output_config.get('save_individual', True):
        approach_dir = Path(features_base_path) / model / approach / f"PCA_{pca_mode}"
        results_file = approach_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({approach: {classifier: result}}, f, indent=2, default=str)
    
    # Save consolidated results
    if output_config.get('save_consolidated', True):
        consolidated_file = Path(features_base_path) / f"classification_results_density_{model}.json"
        
        if consolidated_file.exists():
            with open(consolidated_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Structure: {approach: {pca_mode: {classifier: result}}}
        if approach not in all_results:
            all_results[approach] = {}
        if f"PCA_{pca_mode}" not in all_results[approach]:
            all_results[approach][f"PCA_{pca_mode}"] = {}
        all_results[approach][f"PCA_{pca_mode}"][classifier] = result
        
        with open(consolidated_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)


def run_yaml_density_classification(config_file: str) -> None:
    """
    Run density classification pipeline using YAML configuration.
    
    Args:
        config_file (str): Path to YAML configuration file
    """
    print("=" * 80)
    print("SAM-MED3D DENSITY CLASSIFICATION PIPELINE")
    print("=" * 80)
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    features_base_path = config['output']['base_path']
    
    print(f"Features base path: {features_base_path}")
    
    tasks = get_density_classification_plan(config)
    
    # Validate tasks
    valid_tasks = []
    for task in tasks:
        if validate_density_task(task, features_base_path, config['validation']):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    # Execute tasks
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"\n[{i}/{len(valid_tasks)}] {task['approach']} - PCA_{task['pca_mode']}")
        
        result_data = execute_density_task(task, features_base_path, config['reporting'], config)
        results.append({'task': task, 'result': result_data})
        
        save_density_results(task, result_data, features_base_path, config['output'])
    
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r['result']['status'] == 'success')
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print("DENSITY CLASSIFICATION COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Tasks: {successful} successful, {failed} failed")
    
    if successful > 0:
        print(f"\nResults structure:")
        print(f"feature_extraction_density/sam_med3d_turbo_density/")
        for approach in ['flatten_baseline', 'flatten_masking', 'flatten_linear_weighting']:
            print(f"  ├── {approach}/")
            for pca_mode in [32, 256, 95, 99]:
                print(f"  │   └── PCA_{pca_mode}/classification_results.json")
        print(f"\nConsolidated results:")
        print(f"feature_extraction_density/classification_results_density_sam_med3d_turbo_density.json")
        
        # Generate performance summary
        print(f"\nPerformance Summary:")
        for r in results:
            if r['result']['status'] == 'success':
                task = r['task']
                result = r['result']['result']
                cv_score = result['best_cv_score']
                test_score = result['test_metrics']['roc_auc_weighted']
                print(f"  {task['approach']}/PCA_{task['pca_mode']}: CV={cv_score:.4f} | Test={test_score:.4f}")
        
        print(f"\nNext steps:")
        print(f"1. Analyze density optimization effectiveness")
        print(f"2. Compare baseline vs masking vs linear_weighting")
        print(f"3. Evaluate statistical significance of improvements")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if r['result']['status'] == 'error':
                task = r['task']
                print(f"  {task['approach']}/PCA_{task['pca_mode']}: {r['result']['error']}")


def run_single_density_approach(features_base_path: str, model_name: str, 
                               approach_name: str, pca_mode: int,
                               classifier_params: Dict = None) -> None:
    """
    Run classification for a single density approach.
    
    Args:
        features_base_path (str): Path to feature_extraction_density directory
        model_name (str): Model name
        approach_name (str): Density approach name
        pca_mode (int): PCA mode
        classifier_params (Dict, optional): Custom classifier parameters
    """
    print("=" * 80)
    print("SAM-MED3D DENSITY CLASSIFICATION - SINGLE APPROACH")
    print("=" * 80)
    print(f"Features base path: {features_base_path}")
    print(f"Model: {model_name}")
    print(f"Approach: {approach_name}")
    print(f"PCA mode: {pca_mode}")
    
    try:
        # Initialize prober
        prober = DensityLinearProber(
            features_base_path=features_base_path,
            model_name=model_name,
            classifier_params=classifier_params or {}
        )
        
        available_approaches = prober.get_available_approaches()
        print(f"Available approaches: {available_approaches}")
        
        if approach_name not in available_approaches:
            print(f"Error: Approach '{approach_name}' not found.")
            print(f"Available approaches: {available_approaches}")
            return
        
        # Check PCA availability
        if not prober.check_pca_availability(approach_name, pca_mode):
            print(f"Error: PCA_{pca_mode} not available for {approach_name}")
            return
        
        # Train classifier
        start_time = time.time()
        result = prober.train_classifier(approach_name, pca_mode)
        elapsed_time = time.time() - start_time
        
        print(f"\nSingle approach classification completed in {elapsed_time:.2f}s")
        print(f"Successfully processed {approach_name}/PCA_{pca_mode}")
        print(f"  Best CV score: {result['best_cv_score']:.4f}")
        print(f"  Test ROC-AUC: {result['test_metrics']['roc_auc_weighted']:.4f}")
        print(f"  Test accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"  Overfitting: {result['diagnostics']['overfitting_severity']}")
        
        # Save results
        approach_dir = Path(features_base_path) / model_name / approach_name / f"PCA_{pca_mode}"
        results_file = approach_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({approach_name: {'logistic': result}}, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error processing {approach_name}/PCA_{pca_mode}: {str(e)}")
        return


def main():
    """
    Main entry point for density classification script.
    """
    parser = argparse.ArgumentParser(
        description="Run linear probing classification on SAM-Med3D density-optimized features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # YAML Configuration Mode (RECOMMENDED)
    parser.add_argument(
        "--config",
        type=str,
        default="classification_density.yaml",
        help="Path to YAML configuration file"
    )
    
    # Single approach mode
    parser.add_argument(
        "--approach",
        type=str,
        choices=['flatten_baseline', 'flatten_masking', 'flatten_linear_weighting'],
        help="Process only a specific density approach"
    )
    
    # Manual Mode parameters
    parser.add_argument(
        "--features-base-path",
        type=str,
        default="feature_extraction_density",
        help="Path to feature_extraction_density directory"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="sam_med3d_turbo_density",
        help="Density model name"
    )
    
    parser.add_argument(
        "--pca-mode",
        type=int,
        choices=[32, 95, 99, 256],
        help="PCA mode: 32, 95, 99, or 256"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.approach and args.pca_mode is None:
        print("Error: When using --approach, must specify --pca-mode")
        sys.exit(1)
    
    # Construct config path relative to script location
    script_dir = Path(__file__).parent
    
    if args.approach:
        # Single approach mode
        print("=== Single Density Approach Classification ===")
        
        features_path = Path(args.features_base_path)
        if not features_path.exists():
            print(f"Features directory does not exist: {features_path}")
            sys.exit(1)
        
        run_single_density_approach(
            str(features_path),
            args.model_name,
            args.approach,
            args.pca_mode
        )
    else:
        # YAML Configuration Mode
        config_path = script_dir / args.config
        if not config_path.exists():
            print(f"Configuration file does not exist: {config_path}")
            sys.exit(1)
        
        try:
            run_yaml_density_classification(str(config_path))
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()