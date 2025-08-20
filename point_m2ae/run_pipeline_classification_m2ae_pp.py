#!/usr/bin/env python3
"""
Classification pipeline for Point-M2AE feature approaches with preprocessing and CV investigation.

This script runs linear probing classification on Point-M2AE features with preprocessing v1
with detailed cross-validation analysis to investigate Test vs CV score discrepancy.

Usage:
python point_m2ae/run_pipeline_classification_m2ae_pp.py
python point_m2ae/run_pipeline_classification_m2ae_pp.py --config classification_m2ae_pp.yaml
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
    print("Classification pipeline with CV investigation initialized for Point-M2AE features with preprocessing")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class PointM2AEPreprocessingLinearProberWithInvestigation:
    """
    Linear probing classifier for Point-M2AE feature approaches with preprocessing and CV investigation.
    
    Specialized for Point-M2AE features with preprocessing v1 and detailed fold-level analysis
    to investigate Test vs CV score discrepancy phenomenon and preprocessing effectiveness.
    
    Attributes:
        input_base_path (Path): Base path to feature_extraction_point_m2ae_pp directory
        output_base_path (Path): Base path to point_m2ae directory
        random_state (int): Random state for reproducibility
        n_jobs (int): Number of parallel jobs
        classifier_params (Dict): Classifier parameters from YAML
        investigation_config (Dict): Investigation configuration parameters
    """
    
    def __init__(self, input_base_path: str = 'feature_extraction_point_m2ae_pp', 
                 output_base_path: str = 'point_m2ae',
                 random_state: int = 42, n_jobs: int = -1, 
                 classifier_params: Dict = None, investigation_config: Dict = None):
        """
        Initialize the Point-M2AE linear probing classifier with preprocessing and investigation capabilities.
        
        Args:
            input_base_path (str): Base path to read features from. Defaults to 'feature_extraction_point_m2ae_pp'.
            output_base_path (str): Base path to save results to. Defaults to 'point_m2ae'.
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
        
        print(f"PointM2AEPreprocessingLinearProberWithInvestigation initialized:")
        print(f"  Input path: {self.input_base_path}")
        print(f"  Output path: {self.output_base_path}")
        print(f"  Investigation enabled: {self.investigation_config.get('enabled', False)}")
    
    def _load_features_with_mode_detection(self, approach_name: str, mode: str) -> Tuple[str, int]:
        """
        Detect feature path and dimensionality based on preprocessing approach and mode.
        
        Args:
            approach_name (str): Feature approach ('feat_mean_v1')
            mode (str): Classification mode ('PCA_32', 'PCA_256', or 'raw_features')
        
        Returns:
            Tuple[str, int]: Feature path and expected dimensionality
        """
        if mode == 'raw_features':
            # Load from feat_mean_v1/ directly
            features_path = self.input_base_path / approach_name
            expected_dim = 384  # Always 384D for feat_mean_v1
        else:
            # Load from feat_mean_v1/PCA_32/ or feat_mean_v1/PCA_256/
            features_path = self.input_base_path / approach_name / mode
            expected_dim = int(mode.split('_')[1])  # Extract 32 or 256
        
        return str(features_path), expected_dim
    
    def _load_all_cv_data(self, approach_name: str, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all CV splits with group labels for LeaveOneGroupOut.
        
        Args:
            approach_name (str): Feature approach name ('feat_mean_v1')
            mode (str): Classification mode ('PCA_32', 'PCA_256', or 'raw_features')
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features (X), labels (y), groups
        """
        features_path, expected_dim = self._load_features_with_mode_detection(approach_name, mode)
        approach_path = Path(features_path)
        
        if not approach_path.exists():
            raise FileNotFoundError(f"Feature directory not found: {approach_path}")
        
        all_features = []
        all_labels = []
        all_groups = []
        
        # Load all 5 CV splits
        for fold_id in range(5):
            split_name = f"train_val_split_{fold_id}"
            
            features_file = approach_path / f"{split_name}_features.npy"
            metadata_file = approach_path / f"{split_name}_metadata.csv"
            
            if not features_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(f"Missing files for {approach_name}/{mode}/{split_name}")
            
            features = np.load(features_file)
            metadata = pd.read_csv(metadata_file)
            labels = metadata['Label'].values
            
            # Validate dimensionality
            if features.shape[1] != expected_dim:
                print(f"Warning: Expected {expected_dim}D, got {features.shape[1]}D for {approach_name}/{mode}")
            
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
    
    def _load_test_data(self, approach_name: str, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test split data separately from training data.
        
        Args:
            approach_name (str): Feature approach name ('feat_mean_v1')
            mode (str): Classification mode ('PCA_32', 'PCA_256', or 'raw_features')
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Test features and test labels
        """
        features_path, expected_dim = self._load_features_with_mode_detection(approach_name, mode)
        approach_path = Path(features_path)
        
        test_features_file = approach_path / "test_split_features.npy"
        test_metadata_file = approach_path / "test_split_metadata.csv"
        
        if not test_features_file.exists() or not test_metadata_file.exists():
            raise FileNotFoundError(f"Missing test files for {approach_name}/{mode}")
        
        X_test = np.load(test_features_file)
        test_metadata = pd.read_csv(test_metadata_file)
        y_test = test_metadata['Label'].values
        
        # Validate dimensionality
        if X_test.shape[1] != expected_dim:
            print(f"Warning: Expected {expected_dim}D, got {X_test.shape[1]}D for {approach_name}/{mode} test")
        
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
    
    def _extract_cv_detailed_analysis(self, cv_results_df: pd.DataFrame, 
                                     best_idx: int, test_roc_auc: float, 
                                     cv_score: float, approach_name: str, mode: str) -> Dict:
        """
        Extract detailed cross-validation analysis from GridSearchCV results with preprocessing context.
        
        Args:
            cv_results_df (pd.DataFrame): Complete cv_results_ from GridSearchCV
            best_idx (int): Index of best parameter combination
            test_roc_auc (float): Test set ROC-AUC score
            cv_score (float): Cross-validation ROC-AUC score
            approach_name (str): Feature approach name
            mode (str): Classification mode
        
        Returns:
            Dict: Detailed CV analysis including fold scores and preprocessing-specific diagnostics
        """
        fold_test_scores = []
        fold_train_scores = []
        
        # Extract scores for each fold
        for fold_id in range(5):
            test_key = f'split{fold_id}_test_score'
            train_key = f'split{fold_id}_train_score'
            
            if test_key in cv_results_df.columns:
                fold_test_scores.append(float(cv_results_df.iloc[best_idx][test_key]))
            if train_key in cv_results_df.columns:
                fold_train_scores.append(float(cv_results_df.iloc[best_idx][train_key]))
        
        # Compute analysis metrics
        fold_overfitting_gaps = [train - test for train, test in zip(fold_train_scores, fold_test_scores)]
        mean_cv_score = np.mean(fold_test_scores)
        problematic_threshold = self.investigation_config.get('outlier_detection_threshold', 0.05)
        
        # Preprocessing-specific analysis
        feature_stability_score = 1.0 - np.std(fold_test_scores)  # Higher = more stable
        is_preprocessing_v1 = 'v1' in approach_name
        is_raw_features = mode == 'raw_features'
        
        # Get expected dimensionality
        _, feature_dim = self._load_features_with_mode_detection(approach_name, mode)
        
        cv_detailed_analysis = {
            'fold_test_scores': fold_test_scores,
            'fold_train_scores': fold_train_scores,
            'fold_overfitting_gaps': fold_overfitting_gaps,
            'worst_fold_id': int(np.argmin(fold_test_scores)),
            'best_fold_id': int(np.argmax(fold_test_scores)),
            'fold_score_range': float(max(fold_test_scores) - min(fold_test_scores)),
            'fold_score_std': float(np.std(fold_test_scores)),
            'fold_variance': float(np.var(fold_test_scores)),
            'test_vs_cv_gap': float(test_roc_auc - cv_score),
            'problematic_folds': [i for i, score in enumerate(fold_test_scores) 
                                 if score < (mean_cv_score - problematic_threshold)],
            'mean_overfitting_gap': float(np.mean(fold_overfitting_gaps)),
            'max_overfitting_gap': float(max(fold_overfitting_gaps)),
            'preprocessing_analysis': {
                'approach_name': approach_name,
                'mode': mode,
                'dimensionality': feature_dim,
                'stability_score': feature_stability_score,
                'is_preprocessing_v1': is_preprocessing_v1,
                'is_raw_features': is_raw_features,
                'preprocessing_effectiveness': 'improved' if feature_stability_score > 0.95 else 'neutral' if feature_stability_score > 0.90 else 'degraded',
                'dimensionality_efficiency': 'high' if is_raw_features else 'pca_reduced',
                'anatomical_preservation': 'enabled' if is_preprocessing_v1 else 'disabled'
            }
        }
        
        return cv_detailed_analysis
    
    def _save_cv_results_detailed(self, approach_name: str, mode: str, 
                                 cv_results_df: pd.DataFrame, clf: GridSearchCV) -> None:
        """
        Save complete cv_results_ for detailed investigation.
        
        Args:
            approach_name (str): Name of the feature approach
            mode (str): Classification mode identifier
            cv_results_df (pd.DataFrame): Complete cv_results_ DataFrame
            clf (GridSearchCV): Fitted GridSearchCV object
        """
        if not self.investigation_config.get('save_cv_results', False):
            return
        
        output_dir = self.output_base_path / approach_name / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cv_results_ complete as CSV
        cv_results_file = output_dir / "cv_results_complete.csv"
        cv_results_df.to_csv(cv_results_file, index=False)
        
        # Save cv_results_ complete as JSON
        cv_results_json_file = output_dir / "cv_results_complete.json"
        cv_results_dict = cv_results_df.to_dict('records')
        with open(cv_results_json_file, 'w') as f:
            json.dump(cv_results_dict, f, indent=2, default=str)
        
        # Investigation metadata with preprocessing-specific information
        _, feature_dim = self._load_features_with_mode_detection(approach_name, mode)
        
        investigation_metadata = {
            'total_parameter_combinations': len(cv_results_df),
            'best_params_index': int(clf.best_index_),
            'best_score': float(clf.best_score_),
            'cv_methodology': 'LeaveOneGroupOut (5 folds)',
            'scoring_metric': 'roc_auc_ovr_weighted',
            'hyperparameter_grid_size': {
                'l1_ratio_values': len(clf.param_grid['l1_ratio']),
                'C_values': len(clf.param_grid['C'])
            },
            'preprocessing_approach_info': {
                'approach_name': approach_name,
                'mode': mode,
                'feature_dimensionality': feature_dim,
                'is_preprocessing_v1': 'v1' in approach_name,
                'preprocessing_type': 'fixed_normalization_anatomical_preservation',
                'is_raw_features': mode == 'raw_features',
                'is_pca_reduced': mode != 'raw_features'
            }
        }
        
        investigation_file = output_dir / "cv_investigation_metadata.json"
        with open(investigation_file, 'w') as f:
            json.dump(investigation_metadata, f, indent=2, default=str)
    
    def train_classifier(self, approach_name: str, mode: str) -> Dict:
        """
        Train logistic regression classifier with comprehensive evaluation and CV investigation for preprocessing.
        
        Args:
            approach_name (str): Feature approach name ('feat_mean_v1')
            mode (str): Classification mode ('PCA_32', 'PCA_256', or 'raw_features')
        
        Returns:
            Dict: Complete training results including test metrics, diagnostics, and preprocessing investigation
        """
        print(f"Training logistic regression on {approach_name}/{mode}")
        
        # Load training and validation data
        print("  Loading train/val data...")
        start_load = time.time()
        X_train_val, y_train_val, groups = self._load_all_cv_data(approach_name, mode)
        
        # Load test data separately
        print("  Loading test data...")
        X_test, y_test = self._load_test_data(approach_name, mode)
        load_time = time.time() - start_load
        
        print(f"  Data loaded in {load_time:.2f}s")
        print(f"  Feature dimensionality: {X_train_val.shape[1]}D")
        print(f"  Classes: {len(np.unique(y_train_val))}")
        print(f"  Preprocessing: {'v1 (anatomical preservation)' if 'v1' in approach_name else 'standard'}")
        
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
        self._save_cv_results_detailed(approach_name, mode, cv_results_df, clf)
        
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
        
        # Generate detailed CV investigation analysis with preprocessing context
        cv_detailed_analysis = self._extract_cv_detailed_analysis(
            cv_results_df, best_idx, test_roc_auc_weighted, clf.best_score_, approach_name, mode
        )
        
        total_time = load_time + gridsearch_time + test_time
        
        # Compile results with preprocessing investigation
        results = {
            'approach_name': approach_name,
            'mode': mode,
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
            'cv_detailed_analysis': cv_detailed_analysis,
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
        print(f"  Test vs CV gap: {cv_detailed_analysis['test_vs_cv_gap']:.4f}")
        print(f"  Problematic folds: {len(cv_detailed_analysis['problematic_folds'])}")
        print(f"  Overfitting: {results['diagnostics']['overfitting_severity']}")
        print(f"  Preprocessing effectiveness: {cv_detailed_analysis['preprocessing_analysis']['preprocessing_effectiveness']}")
        
        return results
    
    def get_available_approaches(self) -> List[str]:
        """
        Get list of available feature approaches for classification with preprocessing.
        
        Returns:
            List[str]: List of available approach names
        """
        approaches = []
        for path in self.input_base_path.iterdir():
            if path.is_dir() and path.name.startswith('feat_'):
                approaches.append(path.name)
        return approaches
    
    def check_mode_availability(self, approach_name: str, mode: str) -> bool:
        """
        Check if features are available for a preprocessing approach and mode.
        
        Args:
            approach_name (str): Approach name
            mode (str): Classification mode
        
        Returns:
            bool: True if feature files are available
        """
        features_path, _ = self._load_features_with_mode_detection(approach_name, mode)
        features_dir = Path(features_path)
        
        required_files = [
            "test_split_features.npy",
            "train_val_split_0_features.npy"
        ]
        return features_dir.exists() and all((features_dir / f).exists() for f in required_files)


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


def get_m2ae_preprocessing_classification_plan(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate classification execution plan from YAML configuration for preprocessing approaches.
    
    Args:
        config (Dict[str, Any]): Loaded YAML configuration
        
    Returns:
        List[Dict[str, Any]]: List of classification tasks
    """
    tasks = []
    models = config['models']
    configurations = config['configurations']  # feature approaches with preprocessing
    classifiers = config['classifiers']
    classification_modes = config['classification_modes']
    
    for model in models:
        for approach in configurations:
            for classifier in classifiers:
                for mode in classification_modes:
                    task = {
                        'model': model,
                        'approach': approach,
                        'classifier': classifier,
                        'mode': mode
                    }
                    tasks.append(task)
    
    print(f"Generated {len(tasks)} Point-M2AE preprocessing classification tasks with investigation")
    print(f"Models: {len(models)}, Approaches: {len(configurations)}, Classifiers: {len(classifiers)}, Modes: {len(classification_modes)}")
    
    return tasks


def validate_m2ae_preprocessing_task(task: Dict[str, Any], input_base_path: str, 
                                    validation_config: Dict[str, Any]) -> bool:
    """
    Validate that a Point-M2AE preprocessing classification task can be executed.
    
    Args:
        task (Dict[str, Any]): Task to validate
        input_base_path (str): Path to feature_extraction_point_m2ae_pp directory
        validation_config (Dict[str, Any]): Validation configuration
        
    Returns:
        bool: True if task is valid
    """
    if not validation_config.get('check_feature_files', True):
        return True
    
    # Use mode detection to get correct path
    prober = PointM2AEPreprocessingLinearProberWithInvestigation(input_base_path=input_base_path)
    
    try:
        return prober.check_mode_availability(task['approach'], task['mode'])
    except Exception as e:
        print(f"Skipping {task['approach']}/{task['mode']}: {e}")
        return False


def execute_m2ae_preprocessing_task_with_investigation(task: Dict[str, Any], input_base_path: str, 
                                                      output_base_path: str, reporting_config: Dict[str, Any], 
                                                      yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single Point-M2AE preprocessing classification task with investigation.
    
    Args:
        task (Dict[str, Any]): Task to execute  
        input_base_path (str): Path to feature_extraction_point_m2ae_pp directory
        output_base_path (str): Path to point_m2ae directory
        reporting_config (Dict[str, Any]): Reporting configuration
        yaml_config (Dict[str, Any]): Full YAML configuration
        
    Returns:
        Dict[str, Any]: Task results with investigation
    """
    model = task['model']
    approach = task['approach']
    classifier = task['classifier']
    mode = task['mode']
    
    if reporting_config.get('verbose', True):
        print(f"Executing: {approach} | {classifier} | {mode}")
    
    try:
        # Extract classifier and investigation parameters from YAML config
        classifier_params = yaml_config.get('classifier_params', {})
        investigation_config = yaml_config.get('investigation', {})
        
        prober = PointM2AEPreprocessingLinearProberWithInvestigation(
            input_base_path=input_base_path,
            output_base_path=output_base_path,
            classifier_params=classifier_params,
            investigation_config=investigation_config
        )
        
        start_time = time.time()
        result = prober.train_classifier(approach, mode)
        total_time = time.time() - start_time
        
        result['task_metadata'] = {
            'model': model,
            'approach': approach,
            'classifier': classifier,
            'mode': mode,
            'total_pipeline_time': total_time
        }
        
        if reporting_config.get('verbose', True):
            cv_score = result['best_cv_score']
            test_score = result['test_metrics']['roc_auc_weighted']
            gap = result['cv_detailed_analysis']['test_vs_cv_gap']
            problematic_folds = len(result['cv_detailed_analysis']['problematic_folds'])
            overfitting = result['diagnostics']['overfitting_severity']
            preprocessing_eff = result['cv_detailed_analysis']['preprocessing_analysis']['preprocessing_effectiveness']
            convergence = "OK" if not result['diagnostics']['convergence_warning'] else "WARNING"
            print(f"  Completed: CV={cv_score:.4f} | Test={test_score:.4f} | Gap={gap:.4f} | ProbFolds={problematic_folds} | Overfit={overfitting} | Prep={preprocessing_eff} | Conv={convergence}")
        
        return {'status': 'success', 'result': result}
        
    except Exception as e:
        error_msg = str(e)
        print(f"  Error: {error_msg}")
        return {'status': 'error', 'error': error_msg, 'task': task}


def save_m2ae_preprocessing_results_with_investigation(task: Dict[str, Any], result_data: Dict[str, Any], 
                                                      output_base_path: str, output_config: Dict[str, Any]) -> None:
    """
    Save Point-M2AE preprocessing classification results with investigation analysis.
    
    Args:
        task (Dict[str, Any]): Executed task
        result_data (Dict[str, Any]): Task results
        output_base_path (str): Path to point_m2ae directory
        output_config (Dict[str, Any]): Output configuration
    """
    if result_data['status'] != 'success':
        return
    
    result = result_data['result']
    approach = task['approach']
    classifier = task['classifier']
    mode = task['mode']
    
    # Save individual results
    if output_config.get('save_individual', True):
        approach_dir = Path(output_base_path) / approach / mode
        approach_dir.mkdir(parents=True, exist_ok=True)
        results_file = approach_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({approach: {classifier: result}}, f, indent=2, default=str)
    
    # Save consolidated results with preprocessing investigation
    if output_config.get('save_consolidated', True):
        consolidated_file = Path(output_base_path) / "classification_results_point_m2ae_pp.json"
        
        if consolidated_file.exists():
            with open(consolidated_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {
                'experiment_info': {
                    'pipeline': 'Point-M2AE Classification with Preprocessing and CV Investigation',
                    'model': 'point_m2ae_encoder',
                    'preprocessing_version': 'v1_fixed_normalization',
                    'investigation_enabled': True,
                    'total_runtime': 0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'preprocessing_investigation_analysis': {},
                'results': {}
            }
        
        # Structure: {approach: {mode: {classifier: result}}}
        if approach not in all_results['results']:
            all_results['results'][approach] = {}
        if mode not in all_results['results'][approach]:
            all_results['results'][approach][mode] = {}
        all_results['results'][approach][mode][classifier] = result
        
        # Update preprocessing investigation analysis
        _update_preprocessing_investigation_summary(all_results, approach, mode, result)
        
        with open(consolidated_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)


def _update_preprocessing_investigation_summary(all_results: Dict, approach: str, mode: str, result: Dict) -> None:
    """
    Update preprocessing investigation summary in consolidated results.
    
    Args:
        all_results (Dict): Consolidated results dictionary
        approach (str): Feature approach name
        mode (str): Classification mode
        result (Dict): Classification result with preprocessing investigation
    """
    if 'preprocessing_investigation_analysis' not in all_results:
        all_results['preprocessing_investigation_analysis'] = {}
    
    config_key = f"{approach}_{mode}"
    
    if 'cv_detailed_analysis' in result:
        cv_analysis = result['cv_detailed_analysis']
        preprocessing_analysis = cv_analysis['preprocessing_analysis']
        test_score = result['test_metrics']['roc_auc_weighted']
        cv_score = result['best_cv_score']
        
        all_results['preprocessing_investigation_analysis'][config_key] = {
            'approach': approach,
            'mode': mode,
            'test_score': test_score,
            'cv_score': cv_score,
            'test_vs_cv_gap': cv_analysis['test_vs_cv_gap'],
            'problematic_folds': cv_analysis['problematic_folds'],
            'fold_variance': cv_analysis['fold_variance'],
            'preprocessing_stability_score': preprocessing_analysis['stability_score'],
            'preprocessing_effectiveness': preprocessing_analysis['preprocessing_effectiveness'],
            'anatomical_preservation': preprocessing_analysis['anatomical_preservation'],
            'dimensionality': preprocessing_analysis['dimensionality'],
            'is_raw_features': preprocessing_analysis['is_raw_features']
        }


def run_yaml_m2ae_preprocessing_classification_with_investigation(config_file: str) -> None:
    """
    Run Point-M2AE preprocessing classification pipeline with CV investigation using YAML configuration.
    
    Args:
        config_file (str): Path to YAML configuration file
    """
    print("=" * 80)
    print("POINT-M2AE PREPROCESSING CLASSIFICATION WITH CV INVESTIGATION")
    print("=" * 80)
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    input_base_path = config['paths']['input_base_path']
    output_base_path = config['paths']['output_base_path']
    investigation_enabled = config.get('investigation', {}).get('enabled', False)
    
    print(f"Input base path (reading): {input_base_path}")
    print(f"Output base path (writing): {output_base_path}")
    print(f"Investigation enabled: {investigation_enabled}")
    print(f"Preprocessing focus: v1 fixed normalization with anatomical preservation")
    
    tasks = get_m2ae_preprocessing_classification_plan(config)
    
    # Validate tasks (check input path)
    valid_tasks = []
    for task in tasks:
        if validate_m2ae_preprocessing_task(task, input_base_path, config['validation']):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    print(f"Expected phenomenon: Test ROC-AUC > CV ROC-AUC - investigating Point-M2AE with preprocessing")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    # Execute tasks with preprocessing investigation
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"\n[{i}/{len(valid_tasks)}] {task['approach']} - {task['mode']}")
        
        result_data = execute_m2ae_preprocessing_task_with_investigation(task, input_base_path, output_base_path, config['reporting'], config)
        results.append({'task': task, 'result': result_data})
        
        save_m2ae_preprocessing_results_with_investigation(task, result_data, output_base_path, config['output'])
    
    total_time = time.time() - start_time
    
    # Summary with preprocessing investigation analysis
    successful = sum(1 for r in results if r['result']['status'] == 'success')
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print("POINT-M2AE PREPROCESSING CLASSIFICATION WITH INVESTIGATION COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Tasks: {successful} successful, {failed} failed")
    print(f"Investigation enabled: {investigation_enabled}")
    print(f"Preprocessing version: v1 (fixed normalization)")
    
    if successful > 0:
        print(f"\nResults structure:")
        print(f"point_m2ae/")
        print(f"  ├── feat_mean_v1/")
        for mode in ['PCA_32', 'PCA_256', 'raw_features']:
            print(f"  │   └── {mode}/")
            print(f"  │       ├── classification_results.json")
            if investigation_enabled:
                print(f"  │       ├── cv_results_complete.csv")
                print(f"  │       └── cv_investigation_metadata.json")
        
        print(f"\nConsolidated results:")
        print(f"point_m2ae/classification_results_point_m2ae_pp.json")
        
        # Generate performance and preprocessing investigation summary
        print(f"\nPerformance Summary with Preprocessing Investigation:")
        gaps = []
        preprocessing_effectiveness = []
        for r in results:
            if r['result']['status'] == 'success':
                task = r['task']
                result = r['result']['result']
                cv_score = result['best_cv_score']
                test_score = result['test_metrics']['roc_auc_weighted']
                gap = result['cv_detailed_analysis']['test_vs_cv_gap']
                prob_folds = len(result['cv_detailed_analysis']['problematic_folds'])
                dimensionality = result['cv_detailed_analysis']['preprocessing_analysis']['dimensionality']
                prep_eff = result['cv_detailed_analysis']['preprocessing_analysis']['preprocessing_effectiveness']
                gaps.append(gap)
                preprocessing_effectiveness.append(prep_eff)
                print(f"  {task['approach']}/{task['mode']}: CV={cv_score:.4f} | Test={test_score:.4f} | Gap={gap:.4f} | ProbFolds={prob_folds} | Dim={dimensionality} | Prep={prep_eff}")
        
        if gaps:
            print(f"\nPreprocessing Investigation Analysis:")
            print(f"  Mean Test-CV gap: {np.mean(gaps):.4f}")
            print(f"  Max Test-CV gap: {max(gaps):.4f}")
            print(f"  Gap std deviation: {np.std(gaps):.4f}")
            print(f"  Configurations with large gaps (>0.05): {sum(1 for g in gaps if g > 0.05)}/{len(gaps)}")
            print(f"  Preprocessing effectiveness distribution: {dict(zip(*np.unique(preprocessing_effectiveness, return_counts=True)))}")
        
        print(f"\nNext steps:")
        print(f"1. Analyze fold-level patterns with preprocessing v1")
        print(f"2. Evaluate anatomical position preservation effectiveness")
        print(f"3. Compare raw vs PCA performance with preprocessing")
        print(f"4. Assess optimal dimensionality for Point-M2AE features with preprocessing")
        print(f"5. Compare against standard Point-M2AE preprocessing if available")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if r['result']['status'] == 'error':
                task = r['task']
                print(f"  {task['approach']}/{task['mode']}: {r['result']['error']}")


def run_single_m2ae_preprocessing_approach_with_investigation(input_base_path: str, output_base_path: str, 
                                                             approach_name: str, mode: str,
                                                             classifier_params: Dict = None, 
                                                             investigation_config: Dict = None) -> None:
    """
    Run classification for a single Point-M2AE preprocessing approach with investigation.
    
    Args:
        input_base_path (str): Path to feature_extraction_point_m2ae_pp directory
        output_base_path (str): Path to point_m2ae directory
        approach_name (str): Feature approach name
        mode (str): Classification mode
        classifier_params (Dict, optional): Custom classifier parameters
        investigation_config (Dict, optional): Investigation configuration
    """
    print("=" * 80)
    print("POINT-M2AE PREPROCESSING CLASSIFICATION - SINGLE APPROACH WITH INVESTIGATION")
    print("=" * 80)
    print(f"Input base path (reading): {input_base_path}")
    print(f"Output base path (writing): {output_base_path}")
    print(f"Approach: {approach_name}")
    print(f"Mode: {mode}")
    print(f"Investigation enabled: {investigation_config.get('enabled', False) if investigation_config else False}")
    print(f"Preprocessing: {'v1 (anatomical preservation)' if 'v1' in approach_name else 'unknown'}")
    
    try:
        # Initialize prober with preprocessing investigation
        prober = PointM2AEPreprocessingLinearProberWithInvestigation(
            input_base_path=input_base_path,
            output_base_path=output_base_path,
            classifier_params=classifier_params or {},
            investigation_config=investigation_config or {}
        )
        
        available_approaches = prober.get_available_approaches()
        print(f"Available approaches: {available_approaches}")
        
        if approach_name not in available_approaches:
            print(f"Error: Approach '{approach_name}' not found.")
            print(f"Available approaches: {available_approaches}")
            return
        
        # Check mode availability
        if not prober.check_mode_availability(approach_name, mode):
            print(f"Error: Mode '{mode}' not available for {approach_name}")
            return
        
        # Train classifier with preprocessing investigation
        start_time = time.time()
        result = prober.train_classifier(approach_name, mode)
        elapsed_time = time.time() - start_time
        
        print(f"\nSingle approach preprocessing classification with investigation completed in {elapsed_time:.2f}s")
        print(f"Successfully processed {approach_name}/{mode}")
        print(f"  Best CV score: {result['best_cv_score']:.4f}")
        print(f"  Test ROC-AUC: {result['test_metrics']['roc_auc_weighted']:.4f}")
        print(f"  Test accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"  Test vs CV gap: {result['cv_detailed_analysis']['test_vs_cv_gap']:.4f}")
        print(f"  Problematic folds: {len(result['cv_detailed_analysis']['problematic_folds'])}")
        print(f"  Overfitting: {result['diagnostics']['overfitting_severity']}")
        print(f"  Preprocessing effectiveness: {result['cv_detailed_analysis']['preprocessing_analysis']['preprocessing_effectiveness']}")
        print(f"  Dimensionality: {result['cv_detailed_analysis']['preprocessing_analysis']['dimensionality']}D")
        
        # Save results
        approach_dir = Path(output_base_path) / approach_name / mode
        approach_dir.mkdir(parents=True, exist_ok=True)
        results_file = approach_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({approach_name: {'logistic': result}}, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        
        if investigation_config and investigation_config.get('save_cv_results', False):
            print(f"Investigation files saved:")
            print(f"  - cv_results_complete.csv")
            print(f"  - cv_results_complete.json") 
            print(f"  - cv_investigation_metadata.json")
        
    except Exception as e:
        print(f"Error processing {approach_name}/{mode}: {str(e)}")
        return


def main():
    """
    Main entry point for Point-M2AE preprocessing classification with investigation script.
    """
    parser = argparse.ArgumentParser(
        description="Run linear probing classification on Point-M2AE features with preprocessing and CV investigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # YAML Configuration Mode (RECOMMENDED)
    parser.add_argument(
        "--config",
        type=str,
        default="classification_m2ae_pp.yaml",
        help="Path to YAML configuration file"
    )
    
    # Single approach mode
    parser.add_argument(
        "--approach",
        type=str,
        choices=['feat_mean_v1'],
        help="Process only a specific feature approach with preprocessing"
    )
    
    # Manual Mode parameters
    parser.add_argument(
        "--input-base-path",
        type=str,
        default="feature_extraction_point_m2ae_pp",
        help="Path to feature_extraction_point_m2ae_pp directory (for reading features)"
    )
    
    parser.add_argument(
        "--output-base-path",
        type=str,
        default="point_m2ae",
        help="Path to point_m2ae directory (for saving results)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['PCA_32', 'PCA_256', 'raw_features'],
        help="Classification mode: PCA_32, PCA_256, or raw_features"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.approach and args.mode is None:
        print("Error: When using --approach, must specify --mode")
        sys.exit(1)
    
    # Construct config path relative to script location
    script_dir = Path(__file__).parent
    
    if args.approach:
        # Single approach mode with preprocessing investigation
        print("=== Single Point-M2AE Preprocessing Approach Classification with Investigation ===")
        
        input_path = Path(args.input_base_path)
        if not input_path.exists():
            print(f"Input directory does not exist: {input_path}")
            sys.exit(1)
        
        output_path = Path(args.output_base_path)
        
        # Default investigation config for single approach
        investigation_config = {
            'enabled': True,
            'save_cv_results': True,
            'detailed_fold_analysis': True,
            'outlier_detection_threshold': 0.05,
            'preprocessing_effectiveness': True
        }
        
        run_single_m2ae_preprocessing_approach_with_investigation(
            str(input_path),
            str(output_path),
            args.approach,
            args.mode,
            investigation_config=investigation_config
        )
    else:
        # YAML Configuration Mode with Preprocessing Investigation
        config_path = script_dir / args.config
        if not config_path.exists():
            print(f"Configuration file does not exist: {config_path}")
            sys.exit(1)
        
        try:
            run_yaml_m2ae_preprocessing_classification_with_investigation(str(config_path))
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()