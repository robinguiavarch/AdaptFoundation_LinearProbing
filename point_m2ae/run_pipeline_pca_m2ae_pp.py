#!/usr/bin/env python3
"""
PCA dimensionality reduction pipeline for Point-M2AE features with preprocessing.

This script applies PCA reduction to Point-M2AE features with v1 preprocessing approach:
feat_mean_v1 (384D) to target dimensions 32D and 256D.
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
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import PCA


class PointM2AEPreprocessingDimensionalityReducer:
    """
    Handles dimensionality reduction of Point-M2AE features with preprocessing using PCA.
    
    Supports feat_mean_v1 (384D) approach with fixed component reduction to 32D and 256D 
    for downstream classification.
    
    Attributes:
        features_base_path (Path): Base path to feature_extraction_point_m2ae_pp directory
        n_components (int): Number of PCA components to retain
        pca_model (PCA): Fitted PCA model instance
    """
    
    def __init__(self, features_base_path: str, n_components: int):
        """
        Initialize the Point-M2AE dimensionality reducer with preprocessing support.
        
        Args:
            features_base_path (str): Base path to feature_extraction_point_m2ae_pp directory
            n_components (int): Number of PCA components (32 or 256)
        """
        self.features_base_path = Path(features_base_path)
        self.n_components = n_components
        self.pca_model = None
        
        if not self.features_base_path.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_base_path}")
        
        print(f"PointM2AEPreprocessingDimensionalityReducer initialized:")
        print(f"  Features path: {self.features_base_path}")
        print(f"  Target components: {self.n_components}")
    
    def _load_split_features(self, approach_name: str, split_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load features and metadata for a specific preprocessing approach and split.
        
        Args:
            approach_name (str): Approach name ('feat_mean_v1')
            split_name (str): Split name (e.g., 'train_val_split_0')
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Features array and metadata dataframe
        """
        approach_path = self.features_base_path / approach_name
        
        features_file = approach_path / f"{split_name}_features.npy"
        metadata_file = approach_path / f"{split_name}_metadata.csv"
        
        if not features_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing files for {approach_name}/{split_name}")
        
        features = np.load(features_file)
        metadata = pd.read_csv(metadata_file)
        
        return features, metadata
    
    def _get_training_data(self, approach_name: str) -> np.ndarray:
        """
        Concatenate all training splits for PCA fitting.
        
        Args:
            approach_name (str): Approach name ('feat_mean_v1')
        
        Returns:
            np.ndarray: Concatenated training features
        """
        training_features = []
        
        for i in range(5):
            split_name = f"train_val_split_{i}"
            try:
                features, _ = self._load_split_features(approach_name, split_name)
                training_features.append(features)
                print(f"    Loaded {split_name}: {features.shape}")
            except FileNotFoundError as e:
                print(f"    Warning: {e}")
                continue
        
        if not training_features:
            raise ValueError(f"No training data found for {approach_name}")
        
        concatenated = np.concatenate(training_features, axis=0)
        print(f"    Total training data: {concatenated.shape}")
        return concatenated
    
    def fit_pca(self, approach_name: str) -> Dict:
        """
        Fit PCA model on training data for specific preprocessing approach.
        
        Args:
            approach_name (str): Approach name to fit PCA on
        
        Returns:
            Dict: PCA fitting information and statistics
        """
        print(f"Fitting PCA on approach: {approach_name}")
        
        training_features = self._get_training_data(approach_name)
        original_dim = training_features.shape[1]
        
        print(f"Training data shape: {training_features.shape}")
        print(f"Original dimensionality: {original_dim}")
        print(f"Target components: {self.n_components}")
        
        if self.n_components >= original_dim:
            print(f"Warning: Requested {self.n_components} components but only {original_dim} available")
            self.n_components = original_dim
        
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(training_features)
        
        final_variance = np.sum(self.pca_model.explained_variance_ratio_)
        
        print(f"Selected {self.n_components} components ({self.n_components}/{original_dim})")
        print(f"Variance explained: {final_variance:.4f}")
        print(f"Dimensionality reduction: {original_dim} -> {self.n_components}")
        
        return {
            'approach_name': approach_name,
            'original_dim': int(original_dim),
            'reduced_dim': int(self.n_components),
            'actual_variance': float(final_variance),
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
            'training_samples': int(training_features.shape[0])
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
    
    def process_approach(self, approach_name: str) -> Dict:
        """
        Complete PCA processing for a preprocessing approach: fit on training, transform all splits.
        
        Args:
            approach_name (str): Preprocessing approach name to process
        
        Returns:
            Dict: Processing results and PCA information
        """
        print(f"\n{'='*60}")
        print(f"Processing preprocessing approach: {approach_name}")
        print(f"{'='*60}")
        
        pca_info = self.fit_pca(approach_name)
        
        output_dir_name = f"PCA_{self.n_components}"
        approach_path = self.features_base_path / approach_name
        output_dir = approach_path / output_dir_name
        output_dir.mkdir(exist_ok=True)
        
        print(f"Saving PCA-transformed features to: {output_dir}")
        
        split_info = {}
        
        # Process training/validation splits
        for i in range(5):
            split_name = f"train_val_split_{i}"
            try:
                features, metadata = self._load_split_features(approach_name, split_name)
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
        
        # Process test split
        try:
            test_features, test_metadata = self._load_split_features(approach_name, "test_split")
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
        
        # Save metadata
        metadata_info = {
            'creation_timestamp': datetime.now().isoformat(),
            'pca_info': pca_info,
            'split_info': split_info,
            'processing_config': {
                'n_components': int(self.n_components),
                'approach_name': approach_name
            }
        }
        
        with open(output_dir / "pca_metadata.json", 'w') as f:
            json.dump(metadata_info, f, indent=2)
        
        total_samples = sum(info['original_shape'][0] for info in split_info.values())
        print(f"PCA processing completed for {approach_name}")
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
    Generate PCA execution plan from YAML configuration.
    
    Args:
        config (Dict[str, Any]): Loaded YAML configuration
        
    Returns:
        List[Dict[str, Any]]: List of PCA execution tasks
    """
    tasks = []
    approaches = config['approaches']
    pca_modes = config['pca_modes']
    
    for approach in approaches:
        for pca_mode in pca_modes:
            task = {
                'approach': approach,
                'n_components': pca_mode['n_components'],
                'description': pca_mode['description']
            }
            tasks.append(task)
    
    print(f"Generated {len(tasks)} PCA tasks")
    print(f"Approaches: {len(approaches)}, PCA modes: {len(pca_modes)}")
    
    return tasks


def validate_pca_task(task: Dict[str, Any], features_base_path: str) -> bool:
    """
    Validate that a PCA task can be executed.
    
    Args:
        task (Dict[str, Any]): Task to validate
        features_base_path (str): Path to feature_extraction_point_m2ae_pp directory
        
    Returns:
        bool: True if task is valid
    """
    approach_path = Path(features_base_path) / task['approach']
    
    if not approach_path.exists():
        print(f"Skipping {task['approach']}: Directory not found")
        return False
    
    required_files = [
        "test_split_features.npy",
        "train_val_split_0_features.npy"
    ]
    
    for file_name in required_files:
        if not (approach_path / file_name).exists():
            print(f"Skipping {task['approach']}: Missing {file_name}")
            return False
    
    return True


def execute_pca_task(task: Dict[str, Any], features_base_path: str) -> Dict[str, Any]:
    """
    Execute a single PCA task.
    
    Args:
        task (Dict[str, Any]): Task to execute
        features_base_path (str): Path to feature_extraction_point_m2ae_pp directory
        
    Returns:
        Dict[str, Any]: Task results
    """
    approach = task['approach']
    n_components = task['n_components']
    
    print(f"Executing PCA: {approach} | {task['description']}")
    
    try:
        reducer = PointM2AEPreprocessingDimensionalityReducer(
            features_base_path=features_base_path,
            n_components=n_components
        )
        
        result = reducer.process_approach(approach)
        
        original_dim = result['pca_info']['original_dim']
        reduced_dim = result['pca_info']['reduced_dim']
        variance = result['pca_info']['actual_variance']
        print(f"Completed: {original_dim}D -> {reduced_dim}D | Variance: {variance:.4f}")
        
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
    print("=" * 60)
    print("POINT-M2AE PCA DIMENSIONALITY REDUCTION WITH PREPROCESSING")
    print("=" * 60)
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    features_base_path = config['data']['features_base_path']
    
    print(f"Features base path: {features_base_path}")
    
    tasks = get_pca_execution_plan(config)
    
    valid_tasks = []
    for task in tasks:
        if validate_pca_task(task, features_base_path):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"\n[{i}/{len(valid_tasks)}] {task['approach']} - {task['description']}")
        
        result_data = execute_pca_task(task, features_base_path)
        results.append(result_data)
    
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print("\n" + "=" * 60)
    print("POINT-M2AE PCA PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Tasks: {successful} successful, {failed} failed")
    
    if successful > 0:
        print(f"\nStructure created:")
        print(f"feature_extraction_point_m2ae_pp/")
        print(f"  feat_mean_v1/")
        for mode in ['PCA_32', 'PCA_256']:
            print(f"    {mode}/")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if r['status'] == 'error':
                task = r['task']
                print(f"  {task['approach']}: {r['error']}")


def run_single_approach(features_base_path: str, approach_name: str, n_components: int) -> None:
    """
    Run PCA reduction for a single preprocessing approach.
    
    Args:
        features_base_path (str): Path to feature_extraction_point_m2ae_pp directory
        approach_name (str): Preprocessing approach name
        n_components (int): Number of PCA components
    """
    print("=" * 60)
    print("POINT-M2AE PCA - SINGLE PREPROCESSING APPROACH")
    print("=" * 60)
    print(f"Features base path: {features_base_path}")
    print(f"Approach: {approach_name}")
    print(f"Components: {n_components}")
    
    try:
        reducer = PointM2AEPreprocessingDimensionalityReducer(
            features_base_path=features_base_path,
            n_components=n_components
        )
        
        start_time = time.time()
        result = reducer.process_approach(approach_name)
        elapsed_time = time.time() - start_time
        
        print(f"\nSingle approach PCA processing completed in {elapsed_time:.2f}s")
        print(f"Successfully processed {approach_name}")
        print(f"  Original dimension: {result['pca_info']['original_dim']}")
        print(f"  Reduced dimension: {result['pca_info']['reduced_dim']}")
        print(f"  Variance explained: {result['pca_info']['actual_variance']:.4f}")
        
    except Exception as e:
        print(f"Error processing {approach_name}: {str(e)}")
        return


def main():
    """Main entry point for PCA reduction script."""
    parser = argparse.ArgumentParser(
        description="Apply PCA dimensionality reduction to Point-M2AE features with preprocessing"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="pca_reduction_m2ae_pp.yaml",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--approach",
        type=str,
        choices=['feat_mean_v1'],
        help="Process only a specific preprocessing approach"
    )
    
    parser.add_argument(
        "--features-base-path",
        type=str,
        default="feature_extraction_point_m2ae_pp",
        help="Path to feature_extraction_point_m2ae_pp directory"
    )
    
    parser.add_argument(
        "--n-components",
        type=int,
        help="Number of components for PCA (32 or 256)"
    )
    
    args = parser.parse_args()
    
    if args.approach and args.n_components is None:
        print("Error: When using --approach, must specify --n-components")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    
    if args.approach:
        print("=== Single Preprocessing Approach PCA Processing ===")
        
        features_path = Path(args.features_base_path)
        if not features_path.exists():
            print(f"Features directory does not exist: {features_path}")
            sys.exit(1)
        
        run_single_approach(
            str(features_path),
            args.approach,
            args.n_components
        )
    else:
        config_path = script_dir / args.config
        if not config_path.exists():
            print(f"Configuration file does not exist: {config_path}")
            sys.exit(1)
        
        try:
            run_yaml_pca_reduction(str(config_path))
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()