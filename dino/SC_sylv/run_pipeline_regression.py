"""
Regression pipeline for SC_sylv 6D regression with CV investigation.

This script runs linear probing regression on DINOv2 Giant pooling features
with detailed cross-validation analysis to investigate Test vs CV score discrepancy.
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
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
import sklearn.metrics


class SCLinearProber:
    """
    Linear probing regressor for SC_sylv 6D regression with CV investigation.
    
    Evaluates ElasticNet regression performance on DINOv2 Giant pooling features
    for Isomap 6D targets with detailed fold-level analysis.
    
    Attributes:
        features_base_path (Path): Path to feature_extracted_sc_dinov2 directory
        random_state (int): Random state for reproducibility
        n_jobs (int): Number of parallel jobs for computation
        investigation_config (Dict): Investigation configuration parameters
    """
    
    def __init__(self, features_base_path: str, investigation_config: Dict = None, 
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the SC_sylv linear probing regressor.
        
        Args:
            features_base_path (str): Path to feature_extracted_sc_dinov2 directory
            investigation_config (Dict, optional): Investigation configuration. Defaults to None.
            random_state (int): Random state for reproducibility. Defaults to 42.
            n_jobs (int): Number of parallel jobs. Defaults to -1.
        """
        self.features_base_path = Path(features_base_path)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.investigation_config = investigation_config or {}
        
        if not self.features_base_path.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_base_path}")
    
    def _load_cv_data_with_groups(self, variant_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all cross-validation splits with group labels for LeaveOneGroupOut.
        
        Critical: Preserves pre-stratified SC_sylv folds by assigning group = fold_id.
        
        Args:
            variant_path (Path): Path to variant/PCA directory containing split files
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features, labels (6D), and group identifiers
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
            
            # Parse 6D labels from string representation
            labels_6d = []
            for label_str in metadata['Labels']:
                label_array = np.array(eval(label_str))  # Convert string to array
                labels_6d.append(label_array)
            labels_6d = np.array(labels_6d)
            
            groups = np.full(len(features), fold_id)
            
            all_features.append(features)
            all_labels.append(labels_6d)
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
            Tuple[np.ndarray, np.ndarray]: Test features and test labels (6D)
        """
        test_features_file = variant_path / "test_split_features.npy"
        test_metadata_file = variant_path / "test_split_metadata.csv"
        
        if not test_features_file.exists() or not test_metadata_file.exists():
            raise FileNotFoundError(f"Missing test files in {variant_path}")
        
        X_test = np.load(test_features_file)
        test_metadata = pd.read_csv(test_metadata_file)
        
        # Parse 6D labels from string representation
        labels_6d = []
        for label_str in test_metadata['Labels']:
            label_array = np.array(eval(label_str))
            labels_6d.append(label_array)
        y_test = np.array(labels_6d)
        
        return X_test, y_test
    
    def _get_elasticnet_regression_config(self, config: Dict) -> Tuple[ElasticNet, Dict]:
        """
        Get ElasticNet regression model and parameter grid from configuration.
        
        Args:
            config (Dict): ElasticNet regression configuration parameters
        
        Returns:
            Tuple[ElasticNet, Dict]: Configured model and parameter grid
        """
        elasticnet_params = config.get('elasticnet', {})
        
        max_iter = elasticnet_params.get('max_iter', 10000)
        
        model = ElasticNet(
            max_iter=max_iter,
            random_state=self.random_state
        )
        
        param_grid = {
            'l1_ratio': elasticnet_params.get('l1_ratio', np.linspace(0, 1, 11)),
            'alpha': elasticnet_params.get('alpha', [10**k for k in range(-3, 4)])
        }
        
        return model, param_grid
    
    def _extract_cv_detailed_analysis(self, cv_results_df: pd.DataFrame, 
                                     best_idx: int, test_r2: float, 
                                     cv_score: float) -> Dict:
        """
        Extract detailed cross-validation analysis from GridSearchCV results.
        
        Args:
            cv_results_df (pd.DataFrame): Complete cv_results_ from GridSearchCV
            best_idx (int): Index of best parameter combination
            test_r2 (float): Test set R² score
            cv_score (float): Cross-validation R² score
        
        Returns:
            Dict: Detailed CV analysis including fold scores and diagnostics
        """
        fold_test_scores = []
        fold_train_scores = []
        
        for fold_id in range(5):
            test_key = f'split{fold_id}_test_score'
            train_key = f'split{fold_id}_train_score'
            
            if test_key in cv_results_df.columns:
                fold_test_scores.append(float(cv_results_df.iloc[best_idx][test_key]))
            if train_key in cv_results_df.columns:
                fold_train_scores.append(float(cv_results_df.iloc[best_idx][train_key]))
        
        fold_overfitting_gaps = [train - test for train, test in zip(fold_train_scores, fold_test_scores)]
        mean_cv_score = np.mean(fold_test_scores)
        problematic_threshold = self.investigation_config.get('outlier_detection_threshold', 0.05)
        
        cv_detailed_analysis = {
            'fold_test_scores': fold_test_scores,
            'fold_train_scores': fold_train_scores,
            'fold_overfitting_gaps': fold_overfitting_gaps,
            'worst_fold_id': int(np.argmin(fold_test_scores)),
            'best_fold_id': int(np.argmax(fold_test_scores)),
            'fold_score_range': float(max(fold_test_scores) - min(fold_test_scores)),
            'fold_score_std': float(np.std(fold_test_scores)),
            'fold_variance': float(np.var(fold_test_scores)),
            'test_vs_cv_gap': float(test_r2 - cv_score),
            'problematic_folds': [i for i, score in enumerate(fold_test_scores) 
                                 if score < (mean_cv_score - problematic_threshold)],
            'mean_overfitting_gap': float(np.mean(fold_overfitting_gaps)),
            'max_overfitting_gap': float(max(fold_overfitting_gaps))
        }
        
        return cv_detailed_analysis
    
    def _save_cv_results_detailed(self, variant_name: str, pca_mode: str, target_dim: int,
                                 cv_results_df: pd.DataFrame, clf: GridSearchCV) -> None:
        """
        Save complete cv_results_ for detailed investigation.
        
        Args:
            variant_name (str): Name of the variant
            pca_mode (str): PCA mode identifier
            target_dim (int): Target dimension (0-5)
            cv_results_df (pd.DataFrame): Complete cv_results_ DataFrame
            clf (GridSearchCV): Fitted GridSearchCV object
        """
        if not self.investigation_config.get('save_cv_results', False):
            return
        
        output_dir = self.features_base_path / variant_name / f"PCA_{pca_mode}"
        
        cv_results_file = output_dir / f"cv_results_complete_dim{target_dim}.csv"
        cv_results_df.to_csv(cv_results_file, index=False)
        
        cv_results_json_file = output_dir / f"cv_results_complete_dim{target_dim}.json"
        cv_results_dict = cv_results_df.to_dict('records')
        with open(cv_results_json_file, 'w') as f:
            json.dump(cv_results_dict, f, indent=2, default=str)
        
        investigation_metadata = {
            'total_parameter_combinations': len(cv_results_df),
            'best_params_index': int(clf.best_index_),
            'best_score': float(clf.best_score_),
            'cv_methodology': 'LeaveOneGroupOut (5 folds)',
            'scoring_metric': 'r2',
            'target_dimension': target_dim,
            'hyperparameter_grid_size': {
                'l1_ratio_values': len(clf.param_grid['l1_ratio']),
                'alpha_values': len(clf.param_grid['alpha'])
            }
        }
        
        investigation_file = output_dir / f"cv_investigation_metadata_dim{target_dim}.json"
        with open(investigation_file, 'w') as f:
            json.dump(investigation_metadata, f, indent=2, default=str)
    
    def train_variant_regressor(self, variant_name: str, pca_mode: str, 
                               regressor_config: Dict) -> Dict:
        """
        Train ElasticNet regressor on a specific variant/PCA combination for 6D targets.
        
        Args:
            variant_name (str): Variant name (e.g., 'pooling_spatial_without_25d')
            pca_mode (str): PCA mode ('32', '256', '95', '99')
            regressor_config (Dict): Regressor configuration parameters
        
        Returns:
            Dict: Complete training results with metrics, diagnostics, and investigation
        """
        variant_path = self.features_base_path / variant_name / f"PCA_{pca_mode}"
        
        if not variant_path.exists():
            raise FileNotFoundError(f"Variant path not found: {variant_path}")
        
        start_load = time.time()
        X_train_val, y_train_val, groups = self._load_cv_data_with_groups(variant_path)
        X_test, y_test = self._load_test_data(variant_path)
        load_time = time.time() - start_load
        
        model, param_grid = self._get_elasticnet_regression_config(regressor_config)
        
        logo = LeaveOneGroupOut()
        cv_splits = list(logo.split(X_train_val, y_train_val, groups=groups))
        
        # Train separate regressor for each of the 6 dimensions
        results_per_dimension = {}
        total_gridsearch_time = 0
        total_test_time = 0
        
        for target_dim in range(6):
            print(f"    Training dimension {target_dim + 1}/6...")
            
            # Extract 1D target for this dimension
            y_train_val_1d = y_train_val[:, target_dim]
            y_test_1d = y_test[:, target_dim]
            
            start_gridsearch = time.time()
            clf = GridSearchCV(
                model, param_grid,
                cv=cv_splits,
                scoring='r2',
                refit=True,
                n_jobs=self.n_jobs,
                return_train_score=True,
                verbose=0
            )
            
            clf.fit(X_train_val, y_train_val_1d)
            gridsearch_time = time.time() - start_gridsearch
            total_gridsearch_time += gridsearch_time
            
            cv_results_df = pd.DataFrame(clf.cv_results_)
            best_idx = clf.best_index_
            
            self._save_cv_results_detailed(variant_name, pca_mode, target_dim, cv_results_df, clf)
            
            start_test = time.time()
            best_model = clf.best_estimator_
            test_r2 = best_model.score(X_test, y_test_1d)
            test_time = time.time() - start_test
            total_test_time += test_time
            
            best_train_score = cv_results_df.iloc[best_idx]['mean_train_score']
            best_val_score = cv_results_df.iloc[best_idx]['mean_test_score']
            overfitting_gap = best_train_score - best_val_score
            cv_stability = cv_results_df.iloc[best_idx]['std_test_score']
            
            cv_detailed_analysis = self._extract_cv_detailed_analysis(
                cv_results_df, best_idx, test_r2, clf.best_score_
            )
            
            results_per_dimension[f'dim_{target_dim}'] = {
                'best_params': clf.best_params_,
                'best_cv_r2': clf.best_score_,
                'test_r2': test_r2,
                'cv_metrics': {
                    'r2': clf.best_score_,
                    'mean_train_score': best_train_score,
                    'mean_val_score': best_val_score,
                    'overfitting_gap': overfitting_gap,
                    'cv_stability': cv_stability
                },
                'cv_detailed_analysis': cv_detailed_analysis,
                'diagnostics': {
                    'overfitting_gap': overfitting_gap,
                    'overfitting_severity': 'high' if overfitting_gap > 0.1 else 'medium' if overfitting_gap > 0.05 else 'low',
                    'cv_stability': cv_stability
                }
            }
        
        # Aggregate metrics across all dimensions
        all_cv_r2 = [results_per_dimension[f'dim_{i}']['best_cv_r2'] for i in range(6)]
        all_test_r2 = [results_per_dimension[f'dim_{i}']['test_r2'] for i in range(6)]
        all_gaps = [results_per_dimension[f'dim_{i}']['cv_detailed_analysis']['test_vs_cv_gap'] for i in range(6)]
        
        aggregated_metrics = {
            'mean_cv_r2': float(np.mean(all_cv_r2)),
            'mean_test_r2': float(np.mean(all_test_r2)),
            'mean_test_cv_gap': float(np.mean(all_gaps)),
            'std_cv_r2': float(np.std(all_cv_r2)),
            'std_test_r2': float(np.std(all_test_r2)),
            'min_cv_r2': float(np.min(all_cv_r2)),
            'max_cv_r2': float(np.max(all_cv_r2)),
            'min_test_r2': float(np.min(all_test_r2)),
            'max_test_r2': float(np.max(all_test_r2))
        }
        
        results = {
            'variant_name': variant_name,
            'pca_mode': pca_mode,
            'regressor_type': 'elasticnet',
            'target_dimensions': 6,
            'results_per_dimension': results_per_dimension,
            'aggregated_metrics': aggregated_metrics,
            'data_info': {
                'train_val_shape': X_train_val.shape,
                'test_shape': X_test.shape,
                'target_shape': y_train_val.shape,
                'n_cv_splits': len(cv_splits),
                'feature_dimensionality': X_train_val.shape[1]
            },
            'timing': {
                'load_time': load_time,
                'gridsearch_time': total_gridsearch_time,
                'test_eval_time': total_test_time,
                'total_time': load_time + total_gridsearch_time + total_test_time
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
    
    def _generate_investigation_summary(self, all_results: Dict) -> Dict:
        """
        Generate comprehensive investigation summary from all results.
        
        Args:
            all_results (Dict): Complete results from all variant/PCA combinations
        
        Returns:
            Dict: Investigation summary with global patterns and analysis
        """
        investigation_summary = {
            'test_vs_cv_analysis': {},
            'dimension_analysis': {},
            'global_patterns': {}
        }
        
        gaps = []
        fold_variances = []
        dimension_performances = {f'dim_{i}': [] for i in range(6)}
        
        for variant_name, variant_results in all_results.items():
            for pca_mode, result in variant_results.items():
                config_key = f"{variant_name}_{pca_mode.replace('PCA_', '')}"
                
                aggregated = result['aggregated_metrics']
                gap = aggregated['mean_test_cv_gap']
                
                investigation_summary['test_vs_cv_analysis'][config_key] = {
                    'mean_test_r2': aggregated['mean_test_r2'],
                    'mean_cv_r2': aggregated['mean_cv_r2'],
                    'gap': gap,
                    'std_test_r2': aggregated['std_test_r2'],
                    'dimension_range': aggregated['max_test_r2'] - aggregated['min_test_r2']
                }
                
                gaps.append(gap)
                
                # Collect dimension-wise performance
                for dim_key, dim_result in result['results_per_dimension'].items():
                    dimension_performances[dim_key].append(dim_result['test_r2'])
                    if 'cv_detailed_analysis' in dim_result:
                        fold_variances.append(dim_result['cv_detailed_analysis']['fold_variance'])
        
        # Dimension analysis
        for dim_key, performances in dimension_performances.items():
            if performances:
                investigation_summary['dimension_analysis'][dim_key] = {
                    'mean_performance': float(np.mean(performances)),
                    'std_performance': float(np.std(performances)),
                    'best_performance': float(np.max(performances)),
                    'worst_performance': float(np.min(performances))
                }
        
        if gaps:
            investigation_summary['global_patterns'] = {
                'mean_test_cv_gap': float(np.mean(gaps)),
                'std_test_cv_gap': float(np.std(gaps)),
                'max_test_cv_gap': float(max(gaps)),
                'min_test_cv_gap': float(min(gaps)),
                'mean_fold_variance': float(np.mean(fold_variances)) if fold_variances else 0.0,
                'best_performing_dimension': max(investigation_summary['dimension_analysis'].items(), 
                                                key=lambda x: x[1]['mean_performance'])[0] if investigation_summary['dimension_analysis'] else None,
                'most_stable_dimension': min(investigation_summary['dimension_analysis'].items(),
                                           key=lambda x: x[1]['std_performance'])[0] if investigation_summary['dimension_analysis'] else None
            }
        
        return investigation_summary
    
    def run_all_combinations(self, regressor_config: Dict, output_config: Dict) -> Dict:
        """
        Run regression on all available variant/PCA combinations.
        
        Args:
            regressor_config (Dict): Regressor configuration parameters
            output_config (Dict): Output configuration for saving results
        
        Returns:
            Dict: Results for all evaluated combinations with investigation analysis
        """
        combinations = self.get_available_combinations()
        
        if not combinations:
            return {}
        
        all_results = {}
        start_time = time.time()
        
        for i, (variant_name, pca_mode) in enumerate(combinations, 1):
            print(f"[{i}/{len(combinations)}] Processing {variant_name}/PCA_{pca_mode}")
            
            try:
                result = self.train_variant_regressor(variant_name, pca_mode, regressor_config)
                
                if variant_name not in all_results:
                    all_results[variant_name] = {}
                all_results[variant_name][f"PCA_{pca_mode}"] = result
                
                if output_config.get('save_individual', True):
                    self._save_individual_result(variant_name, pca_mode, result)
                
                mean_cv_r2 = result['aggregated_metrics']['mean_cv_r2']
                mean_test_r2 = result['aggregated_metrics']['mean_test_r2']
                gap = result['aggregated_metrics']['mean_test_cv_gap']
                
                print(f"  Mean CV R²: {mean_cv_r2:.4f}, Mean Test R²: {mean_test_r2:.4f}, Gap: {gap:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        total_time = time.time() - start_time
        
        if output_config.get('save_consolidated', True):
            self._save_consolidated_results(all_results, total_time)
        
        print(f"Regression completed in {total_time:.1f}s")
        return all_results
    
    def _save_individual_result(self, variant_name: str, pca_mode: str, result: Dict) -> None:
        """
        Save individual result to variant/PCA directory.
        
        Args:
            variant_name (str): Name of the variant
            pca_mode (str): PCA mode identifier
            result (Dict): Regression result to save
        """
        output_dir = self.features_base_path / variant_name / f"PCA_{pca_mode}"
        results_file = output_dir / "regression_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _save_consolidated_results(self, all_results: Dict, total_time: float) -> None:
        """
        Save consolidated results with investigation analysis to main directory.
        
        Args:
            all_results (Dict): All regression results
            total_time (float): Total execution time
        """
        consolidated_file = self.features_base_path / "sc_regression_results.json"
        
        investigation_summary = self._generate_investigation_summary(all_results)
        
        summary = {
            'experiment_info': {
                'pipeline': 'SC_sylv 6D Regression with CV Investigation',
                'model': 'DINOv2 Giant (vitg14)',
                'regressor': 'ElasticNet',
                'task': 'Isomap 6D regression',
                'total_runtime': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'investigation_enabled': self.investigation_config.get('enabled', False)
            },
            'results': all_results,
            'investigation_analysis': investigation_summary
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
    Main entry point for SC_sylv 6D regression pipeline with investigation.
    """
    parser = argparse.ArgumentParser(
        description="SC_sylv 6D Regression Pipeline with CV Investigation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="regression_sc.yaml",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="../../feature_extracted_sc_dinov2",
        help="Path to feature_extracted_sc_dinov2 directory"
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
    
    print("SC_sylv 6D Regression Pipeline with CV Investigation")
    print(f"Features path: {features_path}")
    print(f"Configuration: {config_path}")
    print(f"Investigation enabled: {config.get('investigation', {}).get('enabled', False)}")
    
    try:
        prober = SCLinearProber(
            str(features_path),
            investigation_config=config.get('investigation', {}),
            random_state=config.get('random_state', 42),
            n_jobs=config.get('n_jobs', -1)
        )
        
        results = prober.run_all_combinations(
            config['regressor_config'],
            config['output']
        )
        
        if results:
            print("\nResults Summary:")
            for variant_name, variant_results in results.items():
                print(f"{variant_name}:")
                for pca_mode, result in variant_results.items():
                    mean_cv_r2 = result['aggregated_metrics']['mean_cv_r2']
                    mean_test_r2 = result['aggregated_metrics']['mean_test_r2']
                    gap = result['aggregated_metrics']['mean_test_cv_gap']
                    print(f"  {pca_mode}: CV R²={mean_cv_r2:.4f}, Test R²={mean_test_r2:.4f}, Gap={gap:.4f}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()