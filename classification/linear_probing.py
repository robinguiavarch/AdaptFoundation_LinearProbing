"""
Linear probing classification module for foundation model evaluation.

This module implements linear classification methods for evaluating 
pre-extracted features using cross-validation and grid search optimization.

# Without PCA (1152D)
python scripts/run_classification.py --config multi_axes_average

# PCA 95% variance 
python scripts/run_classification.py --config multi_axes_average --use-pca --pca-mode 95

# PCA 256D (Champollion V0)
python scripts/run_classification.py --config multi_axes_average --use-pca --pca-mode 256

# PCA 32D (Champollion V1)
python scripts/run_classification.py --config multi_axes_average --use-pca --pca-mode 32
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


class LinearProber:
    """
    Linear probing classifier for foundation model feature evaluation.
    
    Implements logistic regression, linear SVM, and KNN classification with 
    hyperparameter optimization and cross-validation for multi-class problems.
    
    Attributes:
        features_base_path (Path): Base path to extracted features directory
        model_name (str): Name of the foundation model
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        n_jobs (int): Number of parallel jobs for grid search
    """
    
    def __init__(self, features_base_path: str, model_name: str = 'dinov2_vits14',
                 cv_folds: int = 5, random_state: int = 0, n_jobs: int = -1):
        """
        Initialize the linear probing classifier.
        
        Args:
            features_base_path (str): Base path to feature_extracted directory
            model_name (str): Foundation model name. Defaults to 'dinov2_vits14'.
            cv_folds (int): Number of CV folds. Defaults to 5.
            random_state (int): Random state for reproducibility. Defaults to 0.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
        """
        self.features_base_path = Path(features_base_path)
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Validate paths
        self.model_path = self.features_base_path / model_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
    
    def _load_features_and_labels(self, config_name: str, 
                                 use_pca: bool = False, pca_mode: int = 95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels for all training splits.
        
        Args:
            config_name (str): Configuration name (e.g., 'multi_axes_average')
            use_pca (bool): Whether to use PCA-reduced features. Defaults to False.
            pca_mode (int): PCA mode (95, 256, 32). Defaults to 95.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels arrays
        """
        config_path = self.model_path / config_name
        
        if use_pca:
            pca_dir = f"PCA_{pca_mode}"
            config_path = config_path / pca_dir
            if not config_path.exists():
                raise FileNotFoundError(f"PCA directory not found: {config_path}")
        
        all_features = []
        all_labels = []
        
        # Load all training splits
        for i in range(5):
            split_name = f"train_val_split_{i}"
            
            features_file = config_path / f"{split_name}_features.npy"
            metadata_file = config_path / f"{split_name}_metadata.csv"
            
            if not features_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(f"Missing files for {config_name}/{split_name}")
            
            features = np.load(features_file)
            metadata = pd.read_csv(metadata_file)
            labels = metadata['Label'].values
            
            all_features.append(features)
            all_labels.append(labels)
        
        # Concatenate all data
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        return X, y
    
    def _get_logistic_regression_config(self) -> Tuple[LogisticRegression, Dict]:
        """
        Get logistic regression model and parameter grid.
        
        Returns:
            Tuple[LogisticRegression, Dict]: Model and parameter grid
        """
        model = LogisticRegression(
            solver='saga',
            penalty='elasticnet',
            max_iter=20000,
            random_state=self.random_state
        )
        
        parameters = {
            'l1_ratio': np.linspace(0, 1, 11),
            'C': [10**k for k in range(-3, 4)]
        }
        
        return model, parameters
    
    def _get_knn_config(self) -> Tuple[KNeighborsClassifier, Dict]:
        """
        Get KNN model and parameter grid.
        
        Returns:
            Tuple[KNeighborsClassifier, Dict]: Model and parameter grid
        """
        model = KNeighborsClassifier(n_jobs=self.n_jobs)
        
        parameters = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
            'leaf_size': [1, 5, 10, 20, 30],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'manhattan', 'cosine']
        }
        
        return model, parameters
    
    def _get_linear_svm_config(self) -> Tuple[LinearSVC, Dict]:
        """
        Get Linear SVM model and parameter grid.
        
        Returns:
            Tuple[LinearSVC, Dict]: Model and parameter grid
        """
        model = LinearSVC(random_state=self.random_state, max_iter=20000)
        
        parameters = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'class_weight': [None, 'balanced'],
            'loss': ['hinge', 'squared_hinge']
        }
        
        return model, parameters
    
    def _calculate_roc_auc_ovr(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Calculate ROC-AUC One-vs-Rest for multi-class classification.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
        
        Returns:
            float: ROC-AUC One-vs-Rest weighted score
        """
        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    
    def train_classifier(self, config_name: str, classifier_type: str,
                        use_pca: bool = False, pca_mode: int = 95) -> Dict:
        """
        Train classifier with hyperparameter optimization.
        
        Args:
            config_name (str): Configuration name to train on
            classifier_type (str): Type of classifier ('logistic', 'knn', 'svm_linear')
            use_pca (bool): Whether to use PCA-reduced features. Defaults to False.
            pca_mode (int): PCA mode (95, 256, 32). Defaults to 95.
        
        Returns:
            Dict: Training results including best parameters and CV scores
        """
        pca_info = f"PCA_{pca_mode}" if use_pca else "No PCA"
        print(f"Training {classifier_type} on {config_name} ({pca_info})")
        
        # Load data
        X, y = self._load_features_and_labels(config_name, use_pca, pca_mode)
        print(f"Data shape: {X.shape}, Labels: {len(np.unique(y))} classes")
        
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
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                           random_state=self.random_state)
        
        # Grid search
        clf = GridSearchCV(model, parameters, cv=cv, scoring=scoring, 
                         refit=True, n_jobs=self.n_jobs)
        
        clf.fit(X, y)
        
        # Calculate ROC-AUC for all models (unified metric)
        if hasattr(clf.best_estimator_, 'predict_proba'):
            y_pred_proba = clf.best_estimator_.predict_proba(X)
            roc_auc = self._calculate_roc_auc_ovr(y, y_pred_proba)
        else:
            # For LinearSVC without probability
            y_pred = clf.best_estimator_.predict(X)
            lb = LabelBinarizer()
            y_bin = lb.fit_transform(y)
            y_pred_bin = lb.transform(y_pred)
            roc_auc = roc_auc_score(y_bin, y_pred_bin, average='weighted')
        
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
            'cv_results': clf.cv_results_
        }
        
        print(f"Best CV score: {clf.best_score_:.4f}")
        print(f"ROC-AUC score: {roc_auc:.4f}")
        
        return results
    
    def evaluate_all_classifiers(self, config_name: str, 
                                use_pca: bool = False, pca_mode: int = 95) -> Dict[str, Dict]:
        """
        Evaluate all classifier types on a configuration.
        
        Args:
            config_name (str): Configuration name to evaluate
            use_pca (bool): Whether to use PCA-reduced features. Defaults to False.
            pca_mode (int): PCA mode (95, 256, 32). Defaults to 95.
        
        Returns:
            Dict[str, Dict]: Results for each classifier type
        """
        classifiers = ['logistic', 'knn', 'svm_linear']
        results = {}
        
        for classifier_type in classifiers:
            try:
                results[classifier_type] = self.train_classifier(
                    config_name, classifier_type, use_pca, pca_mode
                )
            except Exception as e:
                print(f"Error training {classifier_type}: {str(e)}")
                results[classifier_type] = {'error': str(e)}
        
        return results
    
    def get_available_configurations(self) -> List[str]:
        """
        Get list of available configurations for classification.
        
        Returns:
            List[str]: List of available configuration names
        """
        configs = []
        for path in self.model_path.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                configs.append(path.name)
        return configs
    
    def check_pca_availability(self, config_name: str, pca_mode: int = 95) -> bool:
        """
        Check if PCA-reduced features are available for a configuration.
        
        Args:
            config_name (str): Configuration name to check
            pca_mode (int): PCA mode (95, 256, 32). Defaults to 95.
        
        Returns:
            bool: True if PCA features are available
        """
        pca_dir = f"PCA_{pca_mode}"
        pca_path = self.model_path / config_name / pca_dir
        return pca_path.exists() and (pca_path / "pca_metadata.json").exists()