"""
PCA dimensionality reduction script for AdaptFoundation project.

This script applies PCA reduction to extracted features with configurable
variance threshold and processes specified configurations.

# PCA 256 dim (Champollion V0)
python3 scripts/run_pca_reduction.py --n-components 256

# PCA 32 dim (Champollion V1)  
python3 scripts/run_pca_reduction.py --n-components 32

# Mode variance threshold 0.95
python3 scripts/run_pca_reduction.py --variance-threshold 0.95

"""

"""
PCA dimensionality reduction script for AdaptFoundation project.

This script applies PCA reduction to extracted features with configurable
variance threshold and processes specified configurations.
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.pca_processing import DimensionalityReducer


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration YAML file
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pca_reduction(config_path, variance_threshold=None, n_components=None):
    """
    Run PCA dimensionality reduction using YAML configuration or direct parameters.
    
    Args:
        config_path (str): Path to YAML configuration file
        variance_threshold (float, optional): Override variance threshold
        n_components (int, optional): Override with fixed number of components
    """
    # Load configuration
    config = load_config(config_path)
    
    print("=== AdaptFoundation PCA Dimensionality Reduction ===")
    print(f"Configuration: {config_path}")
    
    # Determine reduction parameters - Fixed logic order
    if n_components is not None:
        # Fixed components mode - CLI override
        print(f"Mode: Fixed components ({n_components})")
        reduction_params = {'n_components': n_components}
        apply_to_configs = config.get('pca', {}).get('apply_to_configs', ['multi_axes_average'])
    elif variance_threshold is not None:
        # Variance threshold mode - CLI override
        print(f"Mode: Variance threshold ({variance_threshold})")
        reduction_params = {'variance_threshold': variance_threshold}
        apply_to_configs = config.get('pca', {}).get('apply_to_configs', ['multi_axes_average'])
    else:
        # Use config file settings - fallback only
        pca_config = config.get('pca', {})
        if not pca_config.get('enabled', False):
            print("PCA is disabled in configuration. Exiting.")
            return
        
        config_variance = pca_config.get('variance_threshold', 0.95)
        print(f"Mode: Variance threshold ({config_variance}) - from config")
        reduction_params = {'variance_threshold': config_variance}
        apply_to_configs = pca_config.get('apply_to_configs', ['multi_axes_average'])
    
    print(f"Configurations to process: {apply_to_configs}")
    
    # Initialize dimensionality reducer
    reducer = DimensionalityReducer(
        features_base_path=config['output']['base_path'],
        model_name=config['model']['name'],
        **reduction_params
    )
    
    print(f"Available configurations: {reducer.get_available_configurations()}")
    
    # Process each specified configuration
    for config_name in apply_to_configs:
        if config_name not in reducer.get_available_configurations():
            print(f"Warning: Configuration '{config_name}' not found. Skipping.")
            continue
        
        print(f"\n=== Processing Configuration: {config_name} ===")
        
        try:
            result = reducer.process_configuration(config_name)
            
            print(f" Successfully processed {config_name}")
            print(f"   Original dimension: {result['pca_info']['original_dim']}")
            print(f"   Reduced dimension: {result['pca_info']['reduced_dim']}")
            print(f"   Variance explained: {result['pca_info']['actual_variance']:.4f}")
            
        except Exception as e:
            print(f" Error processing {config_name}: {str(e)}")
            continue
    
    print("\n=== PCA Processing Completed ===")


def main():
    """
    Main entry point for PCA reduction script.
    """
    parser = argparse.ArgumentParser(
        description="Apply PCA dimensionality reduction to extracted features"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature_extraction.yaml",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--variance-threshold",
        type=float,
        help="Override variance threshold from config (e.g., 0.90, 0.95, 0.99)"
    )
    
    parser.add_argument(
        "--n-components",
        type=int,
        help="Use fixed number of components instead of variance threshold (e.g., 32, 256)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.variance_threshold and args.n_components:
        print("Error: Cannot specify both --variance-threshold and --n-components")
        sys.exit(1)
    
    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file does not exist: {config_path}")
        sys.exit(1)
    
    # Run PCA reduction with correct parameters
    run_pca_reduction(
        str(config_path),
        variance_threshold=args.variance_threshold,
        n_components=args.n_components
    )


if __name__ == "__main__":
    main()