#!/usr/bin/env python3
"""
PCA dimensionality reduction pipeline for SAM-Med3D density optimization approaches.

This script applies PCA reduction to density-optimized features with support for
variable dimensions (masking approach) and consistent processing across all approaches.

Usage:
python sam3d_variantes/turbo_with_density/run_pipeline_pca_density.py
python sam3d_variantes/turbo_with_density/run_pipeline_pca_density.py --config pca_reduction_density.yaml

For single approach:
python sam3d_variantes/turbo_with_density/run_pipeline_pca_density.py --approach flatten_baseline --mode fixed --n-components 32
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Standard library imports
import argparse
import yaml
import time 
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sklearn.decomposition import PCA

# Project imports
try:
    print("Initializing PCA pipeline for SAM-Med3D density approaches...")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class DensityDimensionalityReducer:
    """
    Handles dimensionality reduction of density-optimized features using PCA.
    
    Specialized for SAM-Med3D density approaches with support for variable dimensions
    (masking approach) and consistent processing across baseline, masking, and linear_weighting.
    """
    
    def __init__(self, features_base_path: str, model_name: str = 'sam_med3d_turbo_density',
                 variance_threshold: float = None, n_components: int = None):
        """
        Initialize the density dimensionality reducer.
        
        Args:
            features_base_path (str): Base path to feature_extraction_density directory
            model_name (str): Model name. Defaults to 'sam_med3d_turbo_density'.
            variance_threshold (float, optional): Cumulative variance threshold mode
            n_components (int, optional): Fixed number of components mode
        """
        self.features_base_path = Path(features_base_path)
        self.model_name = model_name
        
        # Validate reduction mode
        if variance_threshold is not None and n_components is not None:
            raise ValueError("Cannot specify both variance_threshold and n_components")
        if variance_threshold is None and n_components is None:
            raise ValueError("Must specify either variance_threshold or n_components")
        
        self.variance_threshold = variance_threshold
        self.fixed_n_components = n_components
        self.pca_model = None
        self.n_components = None
        self.reduction_mode = "variance" if variance_threshold is not None else "fixed_components"
        
        # Validate paths
        self.model_path = self.features_base_path / model_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        print(f"DensityDimensionalityReducer initialized:")
        print(f"  Features path: {self.model_path}")
        print(f"  Reduction mode: {self.reduction_mode}")
        if self.reduction_mode == "variance":
            print(f"  Variance threshold: {self.variance_threshold}")
        else:
            print(f"  Target components: {self.fixed_n_components}")
    
    def _load_split_features(self, approach_name: str, split_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load features and metadata for a specific approach and split.
        
        Args:
            approach_name (str): Approach name (e.g., 'flatten_baseline')
            split_name (str): Split name (e.g., 'train_val_split_0')
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Features array and metadata dataframe
        """
        approach_path = self.model_path / approach_name
        
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
            approach_name (str): Approach name (e.g., 'flatten_baseline')
        
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
        Fit PCA model on training data with variance threshold or fixed components.
        
        Args:
            approach_name (str): Approach name to fit PCA on
        
        Returns:
            Dict: PCA fitting information including variance analysis
        """
        print(f"Fitting PCA on approach: {approach_name}")
        print(f"Mode: {self.reduction_mode}")
        
        # Load all training data
        training_features = self._get_training_data(approach_name)
        original_dim = training_features.shape[1]
        
        print(f"Training data shape: {training_features.shape}")
        print(f"Original dimensionality: {original_dim}")
        
        if self.reduction_mode == "variance":
            # Variance threshold mode
            print(f"Target variance: {self.variance_threshold}")
            
            # Fit PCA with all components first to analyze variance
            pca_full = PCA()
            pca_full.fit(training_features)
            
            # Calculate cumulative variance
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Find number of components for desired variance threshold
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
        else:
            # Fixed components mode
            self.n_components = min(self.fixed_n_components, original_dim)
            print(f"Target components: {self.n_components}")
            
            if self.n_components >= original_dim:
                print(f"Warning: Requested {self.fixed_n_components} components but only {original_dim} available")
        
        # Fit final PCA with selected number of components
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(training_features)
        
        # Calculate final variance explained
        final_variance = np.sum(self.pca_model.explained_variance_ratio_)
        
        # Calculate cumulative variance for all components (for metadata)
        pca_full = PCA()
        pca_full.fit(training_features)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        print(f"Selected {self.n_components} components ({self.n_components}/{original_dim})")
        print(f"Variance explained: {final_variance:.4f}")
        print(f"Dimensionality reduction: {original_dim} -> {self.n_components}")
        
        return {
            'approach_name': approach_name,
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
    
    def process_approach(self, approach_name: str) -> Dict:
        """
        Complete PCA processing for a density approach: fit on training, transform all splits.
        
        Args:
            approach_name (str): Approach name to process (e.g., 'flatten_baseline')
        
        Returns:
            Dict: Processing results and PCA information
        """
        print(f"\n{'='*60}")
        print(f"Processing density approach: {approach_name}")
        print(f"{'='*60}")
        
        # Fit PCA on training data
        pca_info = self.fit_pca(approach_name)
        
        # Create output directory with appropriate name
        if self.reduction_mode == "variance":
            output_dir_name = f"PCA_{int(self.variance_threshold*100)}"
        else:
            output_dir_name = f"PCA_{self.n_components}"
        
        approach_path = self.model_path / approach_name
        output_dir = approach_path / output_dir_name
        output_dir.mkdir(exist_ok=True)
        
        print(f"Saving PCA-transformed features to: {output_dir}")
        
        # Process all splits
        split_info = {}
        
        # Process training/validation splits
        for i in range(5):
            split_name = f"train_val_split_{i}"
            try:
                features, metadata = self._load_split_features(approach_name, split_name)
                
                # Transform features
                transformed_features = self.transform_features(features)
                
                # Save transformed features and metadata
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
                'reduction_mode': self.reduction_mode,
                'variance_threshold': float(self.variance_threshold) if self.variance_threshold else None,
                'fixed_n_components': int(self.fixed_n_components) if self.fixed_n_components else None,
                'approach_name': approach_name,
                'model_name': self.model_name
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
    
    def get_available_approaches(self) -> List[str]:
        """
        Get list of available density approaches for PCA processing.
        
        Returns:
            List[str]: List of available approach names
        """
        approaches = []
        for path in self.model_path.iterdir():
            if path.is_dir() and path.name.startswith('flatten_'):
                # Check if it has the required feature files
                if (path / "test_split_features.npy").exists():
                    approaches.append(path.name)
        return approaches


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


def get_density_pca_execution_plan(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate PCA execution plan from YAML configuration for density approaches.
    
    Args:
        config (Dict[str, Any]): Loaded YAML configuration
        
    Returns:
        List[Dict[str, Any]]: List of PCA execution tasks
    """
    tasks = []
    models = config['models']
    configurations = config['configurations']  # density approaches
    pca_modes = config['pca_modes']
    
    for model in models:
        for approach in configurations:
            for pca_mode in pca_modes:
                if pca_mode['mode'] == 'fixed':
                    task = {
                        'model': model,
                        'approach': approach,
                        'mode': 'fixed',
                        'n_components': pca_mode['n_components'],
                        'description': pca_mode['description']
                    }
                else:  # variance mode
                    task = {
                        'model': model,
                        'approach': approach,
                        'mode': 'variance',
                        'variance_threshold': pca_mode['variance_threshold'],
                        'description': pca_mode['description']
                    }
                tasks.append(task)
    
    print(f"Generated {len(tasks)} density PCA tasks")
    print(f"Models: {len(models)}, Approaches: {len(configurations)}, PCA modes: {len(pca_modes)}")
    
    return tasks


def validate_density_pca_task(task: Dict[str, Any], features_base_path: str, 
                             validation_config: Dict[str, Any]) -> bool:
    """
    Validate that a density PCA task can be executed.
    
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
    
    if not approach_path.exists():
        print(f"Skipping {task['model']}/{task['approach']}: Directory not found")
        return False
    
    # Check if required feature files exist
    required_files = [
        "test_split_features.npy",
        "train_val_split_0_features.npy"
    ]
    
    for file_name in required_files:
        if not (approach_path / file_name).exists():
            print(f"Skipping {task['model']}/{task['approach']}: Missing {file_name}")
            return False
    
    return True


def execute_density_pca_task(task: Dict[str, Any], features_base_path: str, 
                           processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single density PCA task.
    
    Args:
        task (Dict[str, Any]): Task to execute
        features_base_path (str): Path to feature_extraction_density directory
        processing_config (Dict[str, Any]): Processing configuration
        
    Returns:
        Dict[str, Any]: Task results
    """
    model = task['model']
    approach = task['approach']
    
    if processing_config.get('detailed_logging', True):
        print(f"Executing density PCA: {model} | {approach} | {task['description']}")
    
    try:
        # Initialize dimensionality reducer
        if task['mode'] == 'fixed':
            reducer = DensityDimensionalityReducer(
                features_base_path=features_base_path,
                model_name=model,
                n_components=task['n_components']
            )
        else:  # variance mode
            reducer = DensityDimensionalityReducer(
                features_base_path=features_base_path,
                model_name=model,
                variance_threshold=task['variance_threshold']
            )
        
        # Process approach
        result = reducer.process_approach(approach)
        
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


def run_yaml_density_pca_reduction(config_file: str) -> None:
    """
    Run density PCA reduction pipeline using YAML configuration.
    
    Args:
        config_file (str): Path to YAML configuration file
    """
    print("=" * 80)
    print("SAM-MED3D DENSITY PCA DIMENSIONALITY REDUCTION")
    print("=" * 80)
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    features_base_path = config['data']['features_base_path']
    
    print(f"Features base path: {features_base_path}")
    
    tasks = get_density_pca_execution_plan(config)
    
    # Validate tasks
    valid_tasks = []
    for task in tasks:
        if validate_density_pca_task(task, features_base_path, config['validation']):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    # Execute tasks
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"\n[{i}/{len(valid_tasks)}] {task['approach']} - {task['description']}")
        
        result_data = execute_density_pca_task(task, features_base_path, config['processing'])
        results.append(result_data)
    
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print("\n" + "=" * 80)
    print("DENSITY PCA PROCESSING COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Tasks: {successful} successful, {failed} failed")
    
    if successful > 0:
        print(f"\nStructure created:")
        print(f"feature_extraction_density/sam_med3d_turbo_density/")
        for approach in ['flatten_baseline', 'flatten_masking', 'flatten_linear_weighting']:
            print(f"  ├── {approach}/")
            for mode in ['PCA_32', 'PCA_256', 'PCA_95']:
                print(f"  │   └── {mode}/")
        print(f"\nNext steps:")
        print(f"1. Run classification pipeline on PCA-reduced features")
        print(f"2. Compare density optimization results")
        print(f"3. Analyze density approach effectiveness")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if r['status'] == 'error':
                task = r['task']
                print(f"  {task['model']}/{task['approach']}: {r['error']}")


def run_single_density_approach(features_base_path: str, model_name: str, approach_name: str,
                               variance_threshold: float = None, n_components: int = None) -> None:
    """
    Run PCA reduction for a single density approach.
    
    Args:
        features_base_path (str): Path to feature_extraction_density directory
        model_name (str): Model name
        approach_name (str): Density approach name
        variance_threshold (float, optional): Variance threshold for PCA
        n_components (int, optional): Fixed number of components
    """
    print("=" * 80)
    print("SAM-MED3D DENSITY PCA - SINGLE APPROACH")
    print("=" * 80)
    print(f"Features base path: {features_base_path}")
    print(f"Model: {model_name}")
    print(f"Approach: {approach_name}")
    
    # Determine reduction parameters
    if n_components is not None:
        print(f"Mode: Fixed components ({n_components})")
        reduction_params = {'n_components': n_components}
    elif variance_threshold is not None:
        print(f"Mode: Variance threshold ({variance_threshold})")
        reduction_params = {'variance_threshold': variance_threshold}
    else:
        print("Error: Must specify either --variance-threshold or --n-components")
        return
    
    try:
        # Initialize dimensionality reducer
        reducer = DensityDimensionalityReducer(
            features_base_path=features_base_path,
            model_name=model_name,
            **reduction_params
        )
        
        available_approaches = reducer.get_available_approaches()
        print(f"Available approaches: {available_approaches}")
        
        if approach_name not in available_approaches:
            print(f"Error: Approach '{approach_name}' not found.")
            print(f"Available approaches: {available_approaches}")
            return
        
        # Process the approach
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
    """
    Main entry point for density PCA reduction script.
    """
    parser = argparse.ArgumentParser(
        description="Apply PCA dimensionality reduction to SAM-Med3D density-optimized features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # YAML Configuration Mode (RECOMMENDED)
    parser.add_argument(
        "--config",
        type=str,
        default="pca_reduction_density.yaml",
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
        "--variance-threshold",
        type=float,
        help="Variance threshold for PCA (e.g., 0.95)"
    )
    
    parser.add_argument(
        "--n-components",
        type=int,
        help="Fixed number of components for PCA (e.g., 32, 256)"
    )
    
    # PCA mode shortcuts
    parser.add_argument(
        "--mode",
        type=str,
        choices=['fixed', 'variance'],
        help="PCA mode: fixed (specify --n-components) or variance (specify --variance-threshold)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.approach and (args.variance_threshold is None and args.n_components is None):
        print("Error: When using --approach, must specify either --variance-threshold or --n-components")
        sys.exit(1)
    
    if args.variance_threshold and args.n_components:
        print("Error: Cannot specify both --variance-threshold and --n-components")
        sys.exit(1)
    
    # Construct config path relative to script location
    script_dir = Path(__file__).parent
    
    if args.approach:
        # Single approach mode
        print("=== Single Density Approach PCA Processing ===")
        
        features_path = Path(args.features_base_path)
        if not features_path.exists():
            print(f"Features directory does not exist: {features_path}")
            sys.exit(1)
        
        run_single_density_approach(
            str(features_path),
            args.model_name,
            args.approach,
            args.variance_threshold,
            args.n_components
        )
    else:
        # YAML Configuration Mode
        config_path = script_dir / args.config
        if not config_path.exists():
            print(f"Configuration file does not exist: {config_path}")
            sys.exit(1)
        
        try:
            run_yaml_density_pca_reduction(str(config_path))
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()