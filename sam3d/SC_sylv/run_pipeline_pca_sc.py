#!/usr/bin/env python3
"""
PCA dimensionality reduction pipeline for S.C.-sylv dataset features.

This script applies PCA reduction to SAM-Med3D features extracted from S.C.-sylv dataset
with support for fixed components and variance threshold modes.

Usage:
python sam3d/SC_sylv/run_pipeline_pca_sc.py
python sam3d/SC_sylv/run_pipeline_pca_sc.py --config pca_reduction_sc.yaml
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import time 
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sklearn.decomposition import PCA

try:
    print("Initializing PCA pipeline for S.C.-sylv features")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class SCDimensionalityReducer:
    """
    Handles dimensionality reduction of S.C.-sylv features using PCA.
    
    Supports both fixed components and variance threshold modes for S.C.-sylv dataset.
    
    Attributes:
        features_base_path (Path): Base path to feature_extraction_sam3d_sc directory
        variance_threshold (float): Cumulative variance threshold mode
        fixed_n_components (int): Fixed number of components mode
        pca_model (PCA): Fitted PCA model
        n_components (int): Actual number of components used
        reduction_mode (str): PCA reduction mode
    """
    
    def __init__(self, features_base_path: str, 
                 variance_threshold: float = None, n_components: int = None):
        """
        Initialize S.C.-sylv dimensionality reducer.
        
        Args:
            features_base_path (str): Base path to feature_extraction_sam3d_sc directory
            variance_threshold (float, optional): Cumulative variance threshold mode
            n_components (int, optional): Fixed number of components mode
        """
        self.features_base_path = Path(features_base_path)
        
        if variance_threshold is not None and n_components is not None:
            raise ValueError("Cannot specify both variance_threshold and n_components")
        if variance_threshold is None and n_components is None:
            raise ValueError("Must specify either variance_threshold or n_components")
        
        self.variance_threshold = variance_threshold
        self.fixed_n_components = n_components
        self.pca_model = None
        self.n_components = None
        self.reduction_mode = "variance" if variance_threshold is not None else "fixed_components"
        
        self.flatten_path = self.features_base_path / "flatten"
        if not self.flatten_path.exists():
            raise FileNotFoundError(f"Flatten directory not found: {self.flatten_path}")
        
        print(f"S.C.-sylv Dimensionality Reducer initialized")
        print(f"Features path: {self.flatten_path}")
        print(f"Reduction mode: {self.reduction_mode}")
        if self.reduction_mode == "variance":
            print(f"Variance threshold: {self.variance_threshold}")
        else:
            print(f"Target components: {self.fixed_n_components}")
    
    def _load_split_features(self, split_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load features and metadata for a specific split.
        
        Args:
            split_name (str): Split name (e.g., 'train_val_split_0')
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Features array and metadata dataframe
        """
        features_file = self.flatten_path / f"{split_name}_features.npy"
        metadata_file = self.flatten_path / f"{split_name}_metadata.csv"
        
        if not features_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing files for {split_name}")
        
        features = np.load(features_file)
        metadata = pd.read_csv(metadata_file)
        
        return features, metadata
    
    def _get_training_data(self) -> np.ndarray:
        """
        Concatenate all training splits for PCA fitting.
        
        Returns:
            np.ndarray: Concatenated training features
        """
        training_features = []
        
        for i in range(5):
            split_name = f"train_val_split_{i}"
            try:
                features, _ = self._load_split_features(split_name)
                training_features.append(features)
                print(f"    Loaded {split_name}: {features.shape}")
            except FileNotFoundError as e:
                print(f"    Warning: {e}")
                continue
        
        if not training_features:
            raise ValueError("No training data found")
        
        concatenated = np.concatenate(training_features, axis=0)
        print(f"    Total training data: {concatenated.shape}")
        return concatenated
    
    def fit_pca(self) -> Dict:
        """
        Fit PCA model on training data with variance threshold or fixed components.
        
        Returns:
            Dict: PCA fitting information including variance analysis
        """
        print("Fitting PCA on S.C.-sylv training data")
        print(f"Mode: {self.reduction_mode}")
        
        training_features = self._get_training_data()
        original_dim = training_features.shape[1]
        
        print(f"Training data shape: {training_features.shape}")
        print(f"Original dimensionality: {original_dim}")
        
        if self.reduction_mode == "variance":
            print(f"Target variance: {self.variance_threshold}")
            
            pca_full = PCA()
            pca_full.fit(training_features)
            
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
        else:
            self.n_components = min(self.fixed_n_components, original_dim)
            print(f"Target components: {self.n_components}")
            
            if self.n_components >= original_dim:
                print(f"Warning: Requested {self.fixed_n_components} components but only {original_dim} available")
        
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(training_features)
        
        final_variance = np.sum(self.pca_model.explained_variance_ratio_)
        
        pca_full = PCA()
        pca_full.fit(training_features)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        print(f"Selected {self.n_components} components ({self.n_components}/{original_dim})")
        print(f"Variance explained: {final_variance:.4f}")
        print(f"Dimensionality reduction: {original_dim} -> {self.n_components}")
        
        return {
            'reduction_mode': self.reduction_mode,
            'original_dim': int(original_dim),
            'reduced_dim': int(self.n_components),
            'variance_threshold': float(self.variance_threshold) if self.variance_threshold else None,
            'fixed_n_components': int(self.fixed_n_components) if self.fixed_n_components else None,
            'actual_variance': float(final_variance),
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumulative_variance.tolist()
        }
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA model.
        
        Args:
            features (np.ndarray): Input features to transform
        
        Returns:
            np.ndarray: PCA-transformed features
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")
        
        return self.pca_model.transform(features)
    
    def process_pca_reduction(self) -> Dict:
        """
        Complete PCA processing: fit on training data and transform all splits.
        
        Returns:
            Dict: Processing results and PCA information
        """
        print(f"\n{'='*60}")
        print("Processing S.C.-sylv PCA reduction")
        print(f"{'='*60}")
        
        pca_info = self.fit_pca()
        
        if self.reduction_mode == "variance":
            if self.variance_threshold == 0.995:
                output_dir_name = "PCA_995"
            else:
                output_dir_name = f"PCA_{int(self.variance_threshold*100)}"
        else:
            output_dir_name = f"PCA_{self.n_components}"
        
        output_dir = self.flatten_path / output_dir_name
        output_dir.mkdir(exist_ok=True)
        
        print(f"Saving PCA-transformed features to: {output_dir}")
        
        split_info = {}
        
        for i in range(5):
            split_name = f"train_val_split_{i}"
            try:
                features, metadata = self._load_split_features(split_name)
                
                transformed_features = self.transform_features(features)
                
                np.save(output_dir / f"{split_name}_features.npy", transformed_features)
                metadata.to_csv(output_dir / f"{split_name}_metadata.csv", index=False)
                
                split_info[split_name] = {
                    'original_shape': list(features.shape),
                    'transformed_shape': list(transformed_features.shape)
                }
                
                print(f"  {split_name}: {features.shape} -> {transformed_features.shape}")
                
            except FileNotFoundError as e:
                print(f"  Warning: Skipping {split_name} - {e}")
                continue
        
        try:
            test_features, test_metadata = self._load_split_features("test_split")
            transformed_test = self.transform_features(test_features)
            
            np.save(output_dir / "test_split_features.npy", transformed_test)
            test_metadata.to_csv(output_dir / "test_split_metadata.csv", index=False)
            
            split_info["test_split"] = {
                'original_shape': list(test_features.shape),
                'transformed_shape': list(transformed_test.shape)
            }
            
            print(f"  test_split: {test_features.shape} -> {transformed_test.shape}")
            
        except FileNotFoundError as e:
            print(f"  Warning: Skipping test_split - {e}")
        
        metadata_info = {
            'creation_timestamp': datetime.now().isoformat(),
            'pca_info': pca_info,
            'split_info': split_info,
            'processing_config': {
                'reduction_mode': self.reduction_mode,
                'variance_threshold': float(self.variance_threshold) if self.variance_threshold else None,
                'fixed_n_components': int(self.fixed_n_components) if self.fixed_n_components else None,
                'dataset': 'S.C.-sylv'
            }
        }
        
        with open(output_dir / "pca_metadata.json", 'w') as f:
            json.dump(metadata_info, f, indent=2)
        
        total_samples = sum(info['original_shape'][0] for info in split_info.values())
        print(f"PCA processing completed for S.C.-sylv dataset")
        print(f"  Total samples processed: {total_samples}")
        print(f"  Reduction: {pca_info['original_dim']}D -> {pca_info['reduced_dim']}D")
        print(f"  Variance explained: {pca_info['actual_variance']:.4f}")
        print(f"  Saved to: {output_dir}")
        
        return metadata_info


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


def get_pca_execution_plan(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate PCA execution plan from YAML configuration for S.C.-sylv dataset.
    
    Args:
        config (Dict[str, Any]): Loaded YAML configuration
        
    Returns:
        List[Dict[str, Any]]: List of PCA execution tasks
    """
    tasks = []
    configurations = config['configurations']
    pca_modes = config['pca_modes']
    
    for configuration in configurations:
        for pca_mode in pca_modes:
            if pca_mode['mode'] == 'fixed':
                task = {
                    'configuration': configuration,
                    'mode': 'fixed',
                    'n_components': pca_mode['n_components'],
                    'description': pca_mode['description']
                }
            else:
                task = {
                    'configuration': configuration,
                    'mode': 'variance',
                    'variance_threshold': pca_mode['variance_threshold'],
                    'description': pca_mode['description']
                }
            tasks.append(task)
    
    print(f"Generated {len(tasks)} S.C.-sylv PCA tasks")
    print(f"Configurations: {len(configurations)}, PCA modes: {len(pca_modes)}")
    
    return tasks


def validate_pca_task(task: Dict[str, Any], features_base_path: str, 
                     validation_config: Dict[str, Any]) -> bool:
    """
    Validate that a PCA task can be executed.
    
    Args:
        task (Dict[str, Any]): Task to validate
        features_base_path (str): Path to feature_extraction_sam3d_sc directory
        validation_config (Dict[str, Any]): Validation configuration
        
    Returns:
        bool: True if task is valid
    """
    if not validation_config.get('check_feature_files', True):
        return True
    
    flatten_path = Path(features_base_path) / "flatten"
    
    if not flatten_path.exists():
        print(f"Skipping {task['configuration']}: Directory not found")
        return False
    
    required_files = [
        "test_split_features.npy",
        "train_val_split_0_features.npy"
    ]
    
    for file_name in required_files:
        if not (flatten_path / file_name).exists():
            print(f"Skipping {task['configuration']}: Missing {file_name}")
            return False
    
    return True


def execute_pca_task(task: Dict[str, Any], features_base_path: str, 
                    processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single PCA task.
    
    Args:
        task (Dict[str, Any]): Task to execute
        features_base_path (str): Path to feature_extraction_sam3d_sc directory
        processing_config (Dict[str, Any]): Processing configuration
        
    Returns:
        Dict[str, Any]: Task results
    """
    configuration = task['configuration']
    
    if processing_config.get('detailed_logging', True):
        print(f"Executing S.C.-sylv PCA: {configuration} | {task['description']}")
    
    try:
        if task['mode'] == 'fixed':
            reducer = SCDimensionalityReducer(
                features_base_path=features_base_path,
                n_components=task['n_components']
            )
        else:
            reducer = SCDimensionalityReducer(
                features_base_path=features_base_path,
                variance_threshold=task['variance_threshold']
            )
        
        result = reducer.process_pca_reduction()
        
        if processing_config.get('detailed_logging', True):
            original_dim = result['pca_info']['original_dim']
            reduced_dim = result['pca_info']['reduced_dim']
            variance = result['pca_info']['actual_variance']
            print(f"Completed: {original_dim}D → {reduced_dim}D | Variance: {variance:.4f}")
        
        return {'status': 'success', 'result': result, 'task': task}
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        return {'status': 'error', 'error': error_msg, 'task': task}


def run_yaml_pca_reduction(config_file: str) -> None:
    """
    Run PCA reduction pipeline using YAML configuration.
    
    Args:
        config_file (str): Path to YAML configuration file
    """
    print("=" * 80)
    print("S.C.-sylv PCA DIMENSIONALITY REDUCTION")
    print("=" * 80)
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    features_base_path = config['data']['features_base_path']
    
    print(f"Features base path: {features_base_path}")
    
    tasks = get_pca_execution_plan(config)
    
    valid_tasks = []
    for task in tasks:
        if validate_pca_task(task, features_base_path, config['validation']):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"\n[{i}/{len(valid_tasks)}] {task['configuration']} - {task['description']}")
        
        result_data = execute_pca_task(task, features_base_path, config['processing'])
        results.append(result_data)
    
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print("S.C.-sylv PCA PROCESSING COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Tasks: {successful} successful, {failed} failed")
    
    if successful > 0:
        print(f"\nStructure created:")
        print(f"feature_extraction_sam3d_sc/flatten/")
        for mode in ['PCA_32', 'PCA_256', 'PCA_95', 'PCA_995']:
            print(f"  └── {mode}/")
        print(f"\nNext steps:")
        print(f"1. Run linear probing regression on PCA-reduced features")
        print(f"2. Evaluate S.C.-sylv regression performance")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if r['status'] == 'error':
                task = r['task']
                print(f"  {task['configuration']}: {r['error']}")


def main():
    """
    Main entry point for PCA reduction script.
    """
    parser = argparse.ArgumentParser(
        description="Apply PCA dimensionality reduction to S.C.-sylv SAM-Med3D features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="pca_reduction_sc.yaml",
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"Configuration file does not exist: {config_path}")
        sys.exit(1)
    
    try:
        run_yaml_pca_reduction(str(config_path))
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()