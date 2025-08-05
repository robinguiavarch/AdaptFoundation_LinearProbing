"""
Classification pipeline for Method 2.5D CLS Token variants.

This script runs linear probing classification on 2.5D CLS token extracted features
using logistic regression with proper cross-validation on pre-stratified splits.
"""

import argparse
import json
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import sklearn.metrics


class Method25DCLSLinearProber:
    """
    Linear probing classifier for Method 2.5D CLS Token variants evaluation.
    
    Evaluates logistic regression performance on all variant/PCA combinations
    from the method_25d_cls_features directory structure.
    
    Attributes:
        features_base_path (Path): Path to method_25d_cls_features directory
        random_state (int): Random state for reproducibility
        n_jobs (int): Number of parallel jobs for computation
    """
    
    def __init__(self, features_base_path: str, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the Method 2.5D CLS linear probing classifier.
        
        Args:
            features_base_path (str): Path to method_25d_cls_features directory
            random_state (int): Random state for reproducibility. Defaults to 42.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
        """
        self.features_base_path = Path(features_base_path)
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        if not self.features_base_path.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_base_path}")
    
    def _load_cv_data_with_groups(self, variant_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all cross-validation splits with group labels for LeaveOneGroupOut.
        
        Args:
            variant_path (Path): Path to variant/PCA directory containing split files
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels, and group identifiers
        """
        all_features = []
        all_labels = []
        all_groups = []
        
        for fold_id in range(5):
            features_file = variant_path / f"train_val_split_{fold_id}_features.npy"
            metadata_file = variant_path / f"train_val_split_{fold_id}_metadata.csv"
            
            if not features_file.exists() or not metadata_file.exists():
                raise FileNotFoundError(f"Missing files in {variant_path}: fold {fold_id}")
            
            features = np.load(features_file)
            metadata = pd.read_csv(metadata_file)
            labels = metadata['Label'].values
            groups = np.full(len(features), fold_id)
            
            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups)
        
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        groups = np.concatenate(all_groups, axis=0)
        
        return X, y, groups
    
    def _load_test_data(self, variant_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test split data for final evaluation.
        
        Args:
            variant_path (Path): Path to variant/PCA directory containing test files
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Test features and test labels
        """
        test_features_file = variant_path / "test_split_features.npy"
        test_metadata_file = variant_path / "test_split_metadata.csv"
        
        if not test_features_file.exists() or not test_metadata_file.exists():
            raise FileNotFoundError(f"Missing test files in {variant_path}")
        
        X_test = np.load(test_features_file)
        test_metadata = pd.read_csv(test_metadata_file)
        y_test = test_metadata['Label'].values
        
        return X_test, y_test
    
    def _get_logistic_regression_config(self, config: Dict) -> Tuple[LogisticRegression, Dict]:
        """
        Get logistic regression model and parameter grid from configuration.
        
        Args:
            config (Dict): Logistic regression configuration parameters
        
        Returns:
            Tuple[LogisticRegression, Dict]: Configured model and parameter grid
        """
        logistic_params = config.get('logistic', {})
        
        max_iter = logistic_params.get('max_iter', 20000)
        solver = logistic_params.get('solver', 'saga')
        penalty = logistic_params.get('penalty', 'elasticnet')
        
        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            max_iter=max_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        param_grid = {
            'l1_ratio': logistic_params.get('l1_ratio', np.linspace(0, 1, 11)),
            'C': logistic_params.get('C', [10**k for k in range(-3, 4)])
        }
        
        return model, param_grid
    
    def train_variant_classifier(self, variant_name: str, pca_mode: str, 
                                classifier_config: Dict) -> Dict:
        """
        Train logistic regression classifier on a specific variant/PCA combination.
        
        Args:
            variant_name (str): Variant name (e.g., 'concat_baseline')
            pca_mode (str): PCA mode ('32', '256', '99')
            classifier_config (Dict): Classifier configuration parameters
        
        Returns:
            Dict: Complete training results with metrics and diagnostics
        """
        variant_path = self.features_base_path / variant_name / f"PCA_{pca_mode}"
        
        if not variant_path.exists():
            raise FileNotFoundError(f"Variant path not found: {variant_path}")
        
        start_load = time.time()
        X_train_val, y_train_val, groups = self._load_cv_data_with_groups(variant_path)
        X_test, y_test = self._load_test_data(variant_path)
        load_time = time.time() - start_load
        
        model, param_grid = self._get_logistic_regression_config(classifier_config)
        
        logo = LeaveOneGroupOut()
        cv_splits = list(logo.split(X_train_val, y_train_val, groups=groups))
        
        start_gridsearch = time.time()
        clf = GridSearchCV(
            model, param_grid,
            cv=cv_splits,
            scoring='roc_auc_ovr_weighted',
            refit=True,
            n_jobs=self.n_jobs,
            return_train_score=True,
            verbose=0
        )
        
        clf.fit(X_train_val, y_train_val)
        gridsearch_time = time.time() - start_gridsearch
        
        start_test = time.time()
        best_model = clf.best_estimator_
        best_model.fit(X_train_val, y_train_val)
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)
        
        scorer = sklearn.metrics.get_scorer('roc_auc_ovr_weighted')
        test_roc_auc_weighted = scorer(best_model, X_test, y_test)
        test_accuracy = np.mean(y_test == y_test_pred)
        test_time = time.time() - start_test
        
        cv_results_df = pd.DataFrame(clf.cv_results_)
        best_idx = clf.best_index_
        
        best_train_score = cv_results_df.iloc[best_idx]['mean_train_score']
        best_val_score = cv_results_df.iloc[best_idx]['mean_test_score']
        overfitting_gap = best_train_score - best_val_score
        cv_stability = cv_results_df.iloc[best_idx]['std_test_score']
        
        convergence_warning = False
        if hasattr(best_model, 'n_iter_'):
            max_iter = best_model.max_iter
            actual_iter = best_model.n_iter_[0] if len(best_model.n_iter_) > 0 else 0
            convergence_warning = actual_iter >= max_iter
        
        results = {
            'variant_name': variant_name,
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
                'total_time': load_time + gridsearch_time + test_time
            }
        }
        
        return results
    
    def get_available_combinations(self) -> List[Tuple[str, str]]:
        """
        Get list of available variant/PCA combinations for evaluation.
        
        Returns:
            List[Tuple[str, str]]: List of (variant_name, pca_mode) combinations
        """
        combinations = []
        
        for variant_dir in self.features_base_path.iterdir():
            if variant_dir.is_dir():
                variant_name = variant_dir.name
                
                for pca_dir in variant_dir.iterdir():
                    if pca_dir.is_dir() and pca_dir.name.startswith('PCA_'):
                        pca_mode = pca_dir.name.replace('PCA_', '')
                        
                        metadata_file = pca_dir / "metadata.json"
                        test_file = pca_dir / "test_split_features.npy"
                        
                        if metadata_file.exists() and test_file.exists():
                            combinations.append((variant_name, pca_mode))
        
        return combinations
    
    def run_all_combinations(self, classifier_config: Dict, output_config: Dict) -> Dict:
        """
        Run classification on all available variant/PCA combinations.
        
        Args:
            classifier_config (Dict): Classifier configuration parameters
            output_config (Dict): Output configuration for saving results
        
        Returns:
            Dict: Results for all evaluated combinations
        """
        combinations = self.get_available_combinations()
        
        if not combinations:
            return {}
        
        all_results = {}
        start_time = time.time()
        
        for i, (variant_name, pca_mode) in enumerate(combinations, 1):
            print(f"[{i}/{len(combinations)}] Processing {variant_name}/PCA_{pca_mode}")
            
            try:
                result = self.train_variant_classifier(variant_name, pca_mode, classifier_config)
                
                if variant_name not in all_results:
                    all_results[variant_name] = {}
                all_results[variant_name][f"PCA_{pca_mode}"] = result
                
                if output_config.get('save_individual', True):
                    self._save_individual_result(variant_name, pca_mode, result)
                
                cv_score = result['best_cv_score']
                test_score = result['test_metrics']['roc_auc_weighted']
                gap = result['cv_metrics']['overfitting_gap']
                print(f"  CV: {cv_score:.4f}, Test: {test_score:.4f}, Gap: {gap:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        total_time = time.time() - start_time
        
        if output_config.get('save_consolidated', True):
            self._save_consolidated_results(all_results, total_time)
        
        print(f"Classification completed in {total_time:.1f}s")
        return all_results
    
    def _save_individual_result(self, variant_name: str, pca_mode: str, result: Dict) -> None:
        """
        Save individual result to variant/PCA directory.
        
        Args:
            variant_name (str): Name of the variant
            pca_mode (str): PCA mode identifier
            result (Dict): Classification result to save
        """
        output_dir = self.features_base_path / variant_name / f"PCA_{pca_mode}"
        results_file = output_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _save_consolidated_results(self, all_results: Dict, total_time: float) -> None:
        """
        Save consolidated results to main directory.
        
        Args:
            all_results (Dict): All classification results
            total_time (float): Total execution time
        """
        consolidated_file = self.features_base_path / "method_25d_cls_classification_results.json"
        
        summary = {
            'experiment_info': {
                'pipeline': 'Method 2.5D CLS Token Classification',
                'model': 'DINOv2 Giant (vitg14)',
                'feature_type': 'CLS tokens (1536D)',
                'classifier': 'Logistic Regression',
                'total_runtime': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': all_results
        }
        
        with open(consolidated_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML configuration file
    
    Returns:
        Dict[str, Any]: Loaded configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """
    Main entry point for Method 2.5D CLS Token classification pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Method 2.5D CLS Token Classification Pipeline"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="dinov2_variantes/method_25d_cls/classification.yaml",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="dinov2_variantes/method_25d_cls/method_25d_cls_features",
        help="Path to method_25d_cls_features directory"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_yaml_config(str(config_path))
    
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"Features directory not found: {features_path}")
        sys.exit(1)
    
    print("Method 2.5D CLS Token Classification Pipeline")
    print(f"Model: DINOv2 Giant (vitg14)")
    print(f"Feature type: CLS tokens (1536D)")
    print(f"Features path: {features_path}")
    print(f"Configuration: {config_path}")
    
    try:
        prober = Method25DCLSLinearProber(
            str(features_path),
            random_state=config.get('random_state', 42),
            n_jobs=config.get('n_jobs', -1)
        )
        
        results = prober.run_all_combinations(
            config['classifier_config'],
            config['output']
        )
        
        if results:
            print("Results Summary:")
            for variant_name, variant_results in results.items():
                print(f"{variant_name}:")
                for pca_mode, result in variant_results.items():
                    cv_score = result['best_cv_score']
                    test_score = result['test_metrics']['roc_auc_weighted']
                    print(f"  {pca_mode}: CV={cv_score:.4f}, Test={test_score:.4f}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()