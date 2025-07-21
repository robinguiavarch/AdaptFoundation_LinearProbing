"""
PCA dimensionality reduction script for AdaptFoundation project.

This script applies PCA reduction to extracted features with configurable
variance threshold and processes specified configurations.

YAML Configuration Mode (RECOMMENDED):
python scripts/run_pca_reduction.py --config-file configs/pca_reduction.yaml

Manual Mode (Legacy):
python scripts/run_pca_reduction.py --n-components 256 --model-name dinov2_vits14
python scripts/run_pca_reduction.py --variance-threshold 0.95 --model-name dinov2_vitg14
"""

import argparse
import yaml
from pathlib import Path
import sys
import time 
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.pca_processing import DimensionalityReducer


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
    models = config['models']
    configurations = config['configurations']
    pca_modes = config['pca_modes']
    
    for model in models:
        for cfg in configurations:
            for pca_mode in pca_modes:
                if pca_mode['mode'] == 'fixed':
                    task = {
                        'model': model,
                        'config': cfg,
                        'mode': 'fixed',
                        'n_components': pca_mode['n_components'],
                        'description': pca_mode['description']
                    }
                else:  # variance mode
                    task = {
                        'model': model,
                        'config': cfg,
                        'mode': 'variance',
                        'variance_threshold': pca_mode['variance_threshold'],
                        'description': pca_mode['description']
                    }
                tasks.append(task)
    
    print(f"Generated {len(tasks)} PCA tasks")
    print(f"Models: {len(models)}, Configs: {len(configurations)}, PCA modes: {len(pca_modes)}")
    
    return tasks


def validate_pca_task(task: Dict[str, Any], features_base_path: str, 
                     validation_config: Dict[str, Any]) -> bool:
    """
    Validate that a PCA task can be executed.
    
    Args:
        task (Dict[str, Any]): Task to validate
        features_base_path (str): Path to feature_extracted directory
        validation_config (Dict[str, Any]): Validation configuration
        
    Returns:
        bool: True if task is valid
    """
    if not validation_config.get('check_feature_files', True):
        return True
    
    config_path = Path(features_base_path) / task['model'] / task['config']
    
    if not config_path.exists():
        print(f"Skipping {task['model']}/{task['config']}: Directory not found")
        return False
    
    # Check if required feature files exist
    required_files = [
        "test_split_features.npy",
        "train_val_split_0_features.npy"
    ]
    
    for file_name in required_files:
        if not (config_path / file_name).exists():
            print(f"Skipping {task['model']}/{task['config']}: Missing {file_name}")
            return False
    
    return True


def execute_pca_task(task: Dict[str, Any], features_base_path: str, 
                    processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single PCA task.
    
    Args:
        task (Dict[str, Any]): Task to execute
        features_base_path (str): Path to feature_extracted directory
        processing_config (Dict[str, Any]): Processing configuration
        
    Returns:
        Dict[str, Any]: Task results
    """
    model = task['model']
    config = task['config']
    
    if processing_config.get('detailed_logging', True):
        print(f"Executing PCA: {model} | {config} | {task['description']}")
    
    try:
        # Initialize dimensionality reducer
        if task['mode'] == 'fixed':
            reducer = DimensionalityReducer(
                features_base_path=features_base_path,
                model_name=model,
                n_components=task['n_components']
            )
        else:  # variance mode
            reducer = DimensionalityReducer(
                features_base_path=features_base_path,
                model_name=model,
                variance_threshold=task['variance_threshold']
            )
        
        # Process configuration
        result = reducer.process_configuration(config)
        
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
    print("AdaptFoundation PCA Dimensionality Reduction (YAML Mode)")
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    features_base_path = config['data']['features_base_path']
    
    print(f"Features base path: {features_base_path}")
    
    tasks = get_pca_execution_plan(config)
    
    # Validate tasks
    valid_tasks = []
    for task in tasks:
        if validate_pca_task(task, features_base_path, config['validation']):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    # Execute tasks
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"[{i}/{len(valid_tasks)}]")
        
        result_data = execute_pca_task(task, features_base_path, config['processing'])
        results.append(result_data)
    
    total_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print("PCA Processing Completed")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tasks: {successful} successful, {failed} failed")
    
    if failed > 0:
        print("Failed tasks:")
        for r in results:
            if r['status'] == 'error':
                task = r['task']
                print(f"  {task['model']}/{task['config']}: {r['error']}")


def run_manual_pca_reduction(features_base_path: str, model_name: str, 
                            variance_threshold: float = None, n_components: int = None,
                            apply_to_configs: List[str] = None) -> None:
    """
    Run PCA reduction in manual mode.
    
    Args:
        features_base_path (str): Path to feature_extracted directory
        model_name (str): Foundation model name
        variance_threshold (float, optional): Variance threshold for PCA
        n_components (int, optional): Fixed number of components
        apply_to_configs (List[str], optional): Configurations to process
    """
    print("AdaptFoundation PCA Dimensionality Reduction (Manual Mode)")
    print(f"Features base path: {features_base_path}")
    print(f"Model: {model_name}")
    
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
    
    if apply_to_configs is None:
        apply_to_configs = ['multi_axes_average']
    
    print(f"Configurations to process: {apply_to_configs}")
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer(
        features_base_path=features_base_path,
        model_name=model_name,
        **reduction_params
    )
    
    print(f"Available configurations: {reducer.get_available_configurations()}")
    
    # Process each specified configuration
    for config_name in apply_to_configs:
        if config_name not in reducer.get_available_configurations():
            print(f"Warning: Configuration '{config_name}' not found. Skipping.")
            continue
        
        print(f"Processing Configuration: {config_name}")
        
        try:
            result = reducer.process_configuration(config_name)
            
            print(f"Successfully processed {config_name}")
            print(f"  Original dimension: {result['pca_info']['original_dim']}")
            print(f"  Reduced dimension: {result['pca_info']['reduced_dim']}")
            print(f"  Variance explained: {result['pca_info']['actual_variance']:.4f}")
            
        except Exception as e:
            print(f"Error processing {config_name}: {str(e)}")
            continue
    
    print("PCA Processing Completed")


def main():
    """
    Main entry point for PCA reduction script.
    """
    parser = argparse.ArgumentParser(
        description="Apply PCA dimensionality reduction to extracted features"
    )
    
    # YAML Configuration Mode
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Manual Mode (Legacy)
    parser.add_argument(
        "--features-base-path",
        type=str,
        default="feature_extracted",
        help="Path to feature_extracted directory"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vits14",
        help="Foundation model name"
    )
    
    parser.add_argument(
        "--variance-threshold",
        type=float,
        help="Variance threshold for PCA"
    )
    
    parser.add_argument(
        "--n-components",
        type=int,
        help="Fixed number of components for PCA"
    )
    
    parser.add_argument(
        "--configs",
        nargs='+',
        default=['multi_axes_average'],
        help="Configurations to process"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.config_file and args.variance_threshold and args.n_components:
        print("Error: Cannot specify both --variance-threshold and --n-components")
        sys.exit(1)
    
    if args.config_file:
        # YAML Configuration Mode
        config_file = Path(args.config_file)
        if not config_file.exists():
            print(f"Configuration file does not exist: {config_file}")
            sys.exit(1)
        
        run_yaml_pca_reduction(str(config_file))
    else:
        # Manual Mode (Legacy)
        features_path = Path(args.features_base_path)
        if not features_path.exists():
            print(f"Features directory does not exist: {features_path}")
            sys.exit(1)
        
        run_manual_pca_reduction(
            str(features_path),
            args.model_name,
            args.variance_threshold,
            args.n_components,
            args.configs
        )


if __name__ == "__main__":
    main()