"""
Linear probing classification module for foundation model evaluation.

This module implements linear classification methods for evaluating 
pre-extracted features using proper cross-validation with pre-stratified splits.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import sklearn.metrics


class LinearProber:
    """
    Linear probing classifier for foundation model feature evaluation.
    
    Uses LeaveOneGroupOut CV with pre-stratified splits to respect data structure.
    """
    
    def __init__(self, features_base_path: str, model_name: str = 'dinov2_vits14',
                 random_state: int = 0, n_jobs: int = -1):
        """
        Initialize the linear probing classifier.
        
        Args:
            features_base_path (str): Base path to feature_extracted directory
            model_name (str): Foundation model name. Defaults to 'dinov2_vits14'.
            random_state (int): Random state for reproducibility. Defaults to 0.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
        """
        self.features_base_path = Path(features_base_path)
        self.model_name = model_name
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Validate paths
        self.model_path = self.features_base_path / model_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
    
    def _load_all_cv_data(self, config_name: str, use_pca: bool = False, 
                         pca_mode: int = 95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all CV splits with group labels for LeaveOneGroupOut.
        
        Args:
            config_name (str): Configuration name
            use_pca (bool): Whether to use PCA-reduced features
            pca_mode (int): PCA mode (95, 256, 32)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features (X), labels (y), groups
        """
        config_path = self.model_path / config_name
        
        if use_pca:
            pca_dir = f"PCA_{pca_mode}"
            config_path = config_path / pca_dir
            if not config_path.exists():
                raise FileNotFoundError(f"PCA directory not found: {config_path}")
        
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
            
            # Group ID = fold ID (for LeaveOneGroupOut)
            groups = np.full(len(features), fold_id)
            
            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups)
        
        # Concatenate all data
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        groups = np.concatenate(all_groups, axis=0)
        
        return X, y, groups
    
    def _load_test_data(self, config_name: str, use_pca: bool = False, 
                       pca_mode: int = 95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test split data separately from training data.
        
        Args:
            config_name (str): Configuration name
            use_pca (bool): Whether to use PCA-reduced features
            pca_mode (int): PCA mode (95, 256, 32)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Test features and test labels
        """
        config_path = self.model_path / config_name
        
        if use_pca:
            pca_dir = f"PCA_{pca_mode}"
            config_path = config_path / pca_dir
        
        test_features_file = config_path / "test_split_features.npy"
        test_metadata_file = config_path / "test_split_metadata.csv"
        
        if not test_features_file.exists() or not test_metadata_file.exists():
            raise FileNotFoundError(f"Missing test files for {config_name}")
        
        X_test = np.load(test_features_file)
        test_metadata = pd.read_csv(test_metadata_file)
        y_test = test_metadata['Label'].values
        
        return X_test, y_test
    
    def _get_logistic_regression_config(self) -> Tuple[LogisticRegression, Dict]:
        """Get logistic regression model and parameter grid."""
        model = LogisticRegression(
            solver='saga',
            penalty='elasticnet',
            max_iter=5000,  
            random_state=self.random_state
        )
        
        parameters = {
            'l1_ratio': np.linspace(0, 1, 11),
            'C': [10**k for k in range(-3, 4)]
        }
        
        return model, parameters
    
    def _get_knn_config(self) -> Tuple[KNeighborsClassifier, Dict]:
        """Get KNN model and parameter grid."""
        model = KNeighborsClassifier(n_jobs=self.n_jobs)
        
        parameters = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
            'leaf_size': [1, 5, 10, 20, 30],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'manhattan', 'cosine']
        }
        
        return model, parameters
    
    def _get_linear_svm_config(self) -> Tuple[LinearSVC, Dict]:
        """Get Linear SVM model and parameter grid."""
        model = LinearSVC(random_state=self.random_state, max_iter=2000)
        
        parameters = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'class_weight': [None, 'balanced'],
            'loss': ['hinge', 'squared_hinge']
        }
        
        return model, parameters
    

    
    def train_classifier(self, config_name: str, classifier_type: str,
                        use_pca: bool = False, pca_mode: int = 95) -> Dict:
        """
        Train classifier with test set evaluation and comprehensive diagnostics.
        
        Args:
            config_name (str): Configuration name to train on
            classifier_type (str): Type of classifier ('logistic', 'knn', 'svm_linear')
            use_pca (bool): Whether to use PCA-reduced features
            pca_mode (int): PCA mode (95, 256, 32)
        
        Returns:
            Dict: Complete training results including test metrics and diagnostics
        """
        pca_info = f"PCA_{pca_mode}" if use_pca else "No PCA"
        print(f"Training {classifier_type} on {config_name} ({pca_info})")
        
        # Load training and validation data
        print("Loading train/val data...")
        start_load = time.time()
        X_train_val, y_train_val, groups = self._load_all_cv_data(config_name, use_pca, pca_mode)
        
        # Load test data separately
        print("Loading test data...")
        X_test, y_test = self._load_test_data(config_name, use_pca, pca_mode)
        load_time = time.time() - start_load
        
        print(f"Data loaded in {load_time:.2f}s")
        print(f"Train/val shape: {X_train_val.shape}, Test shape: {X_test.shape}")
        
        # Get model configuration
        if classifier_type == 'logistic':
            model, parameters = self._get_logistic_regression_config()
            scoring = 'roc_auc_ovr_weighted'
        elif classifier_type == 'knn':
            model, parameters = self._get_knn_config()
            scoring = 'roc_auc_ovr_weighted'
        elif classifier_type == 'svm_linear':
            model, parameters = self._get_linear_svm_config()
            scoring = 'roc_auc_ovr_weighted'
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Setup cross-validation
        logo = LeaveOneGroupOut()
        cv_splits = list(logo.split(X_train_val, y_train_val, groups=groups))
        
        print("Starting GridSearchCV with detailed diagnostics...")
        start_gridsearch = time.time()
        
        # Grid search with comprehensive diagnostics
        clf = GridSearchCV(
            model, parameters,
            cv=cv_splits,
            scoring=scoring,
            refit=True,
            n_jobs=self.n_jobs,
            return_train_score=True,
            verbose=1
        )
        
        clf.fit(X_train_val, y_train_val)
        gridsearch_time = time.time() - start_gridsearch
        
        print(f"GridSearchCV completed in {gridsearch_time:.2f}s")
        
        # Final evaluation on test set
        print("Final evaluation on test set...")
        start_test = time.time()
        
        best_model = clf.best_estimator_
        best_model.fit(X_train_val, y_train_val)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate test metrics
        if hasattr(best_model, 'predict_proba'):
            y_test_proba = best_model.predict_proba(X_test)
            scorer = sklearn.metrics.get_scorer('roc_auc_ovr_weighted')
            test_roc_auc_weighted = scorer(best_model, X_test, y_test)
        else:
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)
            y_test_pred_bin = lb.transform(y_test_pred)
            test_roc_auc_weighted = roc_auc_score(y_test_bin, y_test_pred_bin, average='weighted')
        
        test_accuracy = np.mean(y_test == y_test_pred)
        test_time = time.time() - start_test
        
        # Extract CV diagnostics from GridSearchCV results
        print("Computing CV diagnostics from GridSearchCV...")
        cv_results_df = pd.DataFrame(clf.cv_results_)
        best_idx = clf.best_index_
        
        # CV metrics and overfitting analysis
        best_train_score = cv_results_df.iloc[best_idx]['mean_train_score']
        best_val_score = cv_results_df.iloc[best_idx]['mean_test_score']
        overfitting_gap = best_train_score - best_val_score
        cv_stability = cv_results_df.iloc[best_idx]['std_test_score']
        
        # Convergence check for logistic regression
        convergence_warning = False
        if classifier_type == 'logistic' and hasattr(best_model, 'n_iter_'):
            max_iter = best_model.max_iter
            actual_iter = best_model.n_iter_[0] if len(best_model.n_iter_) > 0 else 0
            convergence_warning = actual_iter >= max_iter
        
        total_time = load_time + gridsearch_time + test_time
        
        # Compile complete results
        results = {
            'classifier_type': classifier_type,
            'config_name': config_name,
            'use_pca': use_pca,
            'pca_mode': pca_mode if use_pca else None,
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
                'n_cv_splits': len(cv_splits)
            },
            'timing': {
                'load_time': load_time,
                'gridsearch_time': gridsearch_time,
                'test_eval_time': test_time,
                'total_time': total_time
            },
            'cv_results_summary': {
                'best_index': best_idx,
                'n_combinations_tested': len(cv_results_df),
                'param_grid_size': len(parameters)
            }
        }
        
        print(f"Best CV score: {clf.best_score_:.4f}")
        print(f"Test ROC-AUC (weighted): {test_roc_auc_weighted:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Overfitting gap: {overfitting_gap:.4f} ({results['diagnostics']['overfitting_severity']})")
        if convergence_warning:
            print("Convergence warning: max_iter reached")
        
        return results
    
    def get_available_configurations(self) -> List[str]:
        """Get list of available configurations for classification."""
        configs = []
        for path in self.model_path.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                configs.append(path.name)
        return configs
    
    def check_pca_availability(self, config_name: str, pca_mode: int = 95) -> bool:
        """Check if PCA-reduced features are available for a configuration."""
        pca_dir = f"PCA_{pca_mode}"
        pca_path = self.model_path / config_name / pca_dir
        return pca_path.exists() and (pca_path / "pca_metadata.json").exists()