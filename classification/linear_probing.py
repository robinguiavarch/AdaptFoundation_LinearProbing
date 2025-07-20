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
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


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
    
    def _calculate_roc_auc_ovr(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate ROC-AUC One-vs-Rest for multi-class classification."""
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    
    def train_classifier(self, config_name: str, classifier_type: str,
                        use_pca: bool = False, pca_mode: int = 95) -> Dict:
        """
        Train classifier with LeaveOneGroupOut cross-validation.
        
        Args:
            config_name (str): Configuration name to train on
            classifier_type (str): Type of classifier ('logistic', 'knn', 'svm_linear')
            use_pca (bool): Whether to use PCA-reduced features
            pca_mode (int): PCA mode (95, 256, 32)
        
        Returns:
            Dict: Training results including best parameters and CV scores
        """
        pca_info = f"PCA_{pca_mode}" if use_pca else "No PCA"
        print(f"Training {classifier_type} on {config_name} ({pca_info})")
        
        # Load data with group information
        print("Loading data...")
        start_load = time.time()
        X, y, groups = self._load_all_cv_data(config_name, use_pca, pca_mode)
        load_time = time.time() - start_load
        print(f"Data loaded in {load_time:.2f}s")
        print(f"Data shape: {X.shape}, Labels: {len(np.unique(y))} classes")
        print(f"Groups (folds): {np.unique(groups)}")
        
        # Get model configuration
        if classifier_type == 'logistic':
            model, parameters = self._get_logistic_regression_config()
            scoring = 'roc_auc_ovr_weighted'
        elif classifier_type == 'knn':
            model, parameters = self._get_knn_config()
            scoring = 'roc_auc_ovr_weighted'
        elif classifier_type == 'svm_linear':
            model, parameters = self._get_linear_svm_config()
            scoring = 'accuracy'  # LinearSVC compatibility
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Calculate total combinations for progress tracking
        total_combinations = 1
        for param_values in parameters.values():
            total_combinations *= len(param_values)
        
        print(f"GridSearch: {total_combinations} hyperparameter combinations × 5 folds = {total_combinations * 5} total fits")
        
        # Setup LeaveOneGroupOut cross-validation
        logo = LeaveOneGroupOut()
        cv_splits = list(logo.split(X, y, groups=groups))
        
        print(f"Using LeaveOneGroupOut CV: {len(cv_splits)} splits")
        print("Starting GridSearchCV...")
        
        # Grid search with LeaveOneGroupOut
        start_gridsearch = time.time()
        clf = GridSearchCV(
            model, parameters, 
            cv=cv_splits,  # Use pre-computed splits
            scoring=scoring, 
            refit=True, 
            n_jobs=self.n_jobs,
            verbose=1  # Add sklearn verbose output
        )
        
        clf.fit(X, y)
        gridsearch_time = time.time() - start_gridsearch
        
        print(f"GridSearchCV completed in {gridsearch_time:.2f}s")
        print(f"Best parameters found: {clf.best_params_}")
        print(f"Best cross-validation score: {clf.best_score_:.4f}")
        
        # Get best model and calculate ROC-AUC with cross_val_predict
        print("Computing final ROC-AUC with cross_val_predict...")
        start_final = time.time()
        best_model = clf.best_estimator_
        
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = cross_val_predict(
                best_model, X, y, cv=cv_splits, method='predict_proba'
            )
            roc_auc = self._calculate_roc_auc_ovr(y, y_pred_proba)
        else:
            # For LinearSVC without probability
            y_pred = cross_val_predict(best_model, X, y, cv=cv_splits)
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y)
            y_pred_bin = lb.transform(y_pred)
            roc_auc = roc_auc_score(y_bin, y_pred_bin, average='weighted')
        
        final_time = time.time() - start_final
        total_time = load_time + gridsearch_time + final_time
        
        print(f"Final evaluation completed in {final_time:.2f}s")
        print(f"Total training time: {total_time:.2f}s")
        
        results = {
            'classifier_type': classifier_type,
            'config_name': config_name,
            'use_pca': use_pca,
            'pca_mode': pca_mode if use_pca else None,
            'best_params': clf.best_params_,
            'best_cv_score': clf.best_score_,
            'roc_auc_score': roc_auc,
            'data_shape': X.shape,
            'n_classes': len(np.unique(y)),
            'n_cv_splits': len(cv_splits),
            'total_combinations': total_combinations,
            'timing': {
                'load_time': load_time,
                'gridsearch_time': gridsearch_time,
                'final_eval_time': final_time,
                'total_time': total_time
            },
            'cv_results': clf.cv_results_
        }
        
        print(f"Best CV score: {clf.best_score_:.4f}")
        print(f"Cross-val ROC-AUC: {roc_auc:.4f}")
        
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