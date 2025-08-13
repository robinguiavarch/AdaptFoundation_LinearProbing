#!/usr/bin/env python3
"""
Feature extraction script with concatenation strategy for AdaptFoundation project.

This script orchestrates batch extraction of features from 3D skeletal volumes
using concatenation-based aggregation across all DINOv2 models and configurations.
"""

import yaml
import argparse
import time
from pathlib import Path
import sys
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.feature_saver_concat import FeatureDatasetSaver


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration YAML file
    
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict) -> None:
    """
    Validate configuration structure and required fields.
    
    Args:
        config (Dict): Configuration dictionary to validate
    """
    required_sections = ['dataset', 'models', 'configurations', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    if not config['models']:
        raise ValueError("No models specified in configuration")
    
    if not config['configurations']:
        raise ValueError("No configurations specified in configuration")
    
    print("Configuration validation passed")


def print_batch_summary(config: Dict) -> None:
    """
    Print summary of batch processing plan.
    
    Args:
        config (Dict): Configuration dictionary
    """
    models = config['models']
    configurations = config['configurations']
    
    print("=" * 80)
    print("BATCH FEATURE EXTRACTION WITH CONCATENATION STRATEGY")
    print("=" * 80)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Data path: {config['dataset']['data_path']}")
    print(f"Output base: {config['output']['base_path']}")
    print()
    print(f"Models to process ({len(models)}):")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']} ({model['feature_dimension']}D per slice)")
    print()
    print(f"Configurations per model ({len(configurations)}):")
    for i, cfg in enumerate(configurations, 1):
        print(f"  {i}. {cfg['name']}")
    print()
    print(f"Total extractions: {len(models)} models × {len(configurations)} configs = {len(models) * len(configurations)}")
    print("=" * 80)


def process_model_batch(model_config: Dict, configurations: List[Dict], 
                       data_path: str, output_base_path: str) -> None:
    """
    Process all configurations for a single model.
    
    Args:
        model_config (Dict): Model configuration dictionary
        configurations (List[Dict]): List of configuration dictionaries
        data_path (str): Path to HCP OFC dataset
        output_base_path (str): Base output directory
    """
    model_name = model_config['name']
    feature_dim = model_config['feature_dimension']
    
    print(f"\nProcessing model: {model_name} ({feature_dim}D per slice)")
    print("-" * 60)
    
    # Initialize feature saver for this model
    saver = FeatureDatasetSaver(
        data_path=data_path,
        output_base_path=output_base_path,
        model_name=model_name
    )
    
    # Process each configuration
    model_start_time = time.time()
    
    for i, config in enumerate(configurations, 1):
        config_name = config['name']
        pooling_strategy = config.get('pooling_strategy', 'average')
        required_axes = config.get('required_axes', None)
        
        print(f"\n  Configuration {i}/{len(configurations)}: {config_name}")
        
        # Calculate expected dimensions for this model
        if required_axes is None:
            # Multi-axes: (22+38+30) × feature_dim
            expected_dim = 90 * feature_dim
        elif len(required_axes) == 1:
            # Single axis dimensions
            axis_slices = {'axial': 22, 'coronal': 38, 'sagittal': 30}
            expected_dim = axis_slices.get(required_axes[0], 0) * feature_dim
        else:
            # Custom axes
            expected_dim = "Variable"
        
        print(f"    Expected output dimension: {expected_dim}D")
        
        config_start_time = time.time()
        
        try:
            # Save configuration using concatenation strategy
            saver.save_configuration(
                pooling_strategy=pooling_strategy,
                required_axes=required_axes
            )
            
            config_time = time.time() - config_start_time
            print(f"    Configuration completed in {config_time:.2f}s")
            
        except Exception as e:
            print(f"    ERROR in configuration {config_name}: {e}")
            continue
    
    model_time = time.time() - model_start_time
    print(f"\nModel {model_name} completed in {model_time:.2f}s")


def run_batch_extraction(config_path: str) -> None:
    """
    Run complete batch feature extraction for all models and configurations.
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    # Load and validate configuration
    print("Loading configuration...")
    config = load_config(config_path)
    validate_config(config)
    
    # Print batch summary
    print_batch_summary(config)
    
    # Extract configuration components
    models = config['models']
    configurations = config['configurations']
    data_path = config['dataset']['data_path']
    output_base_path = config['output']['base_path']
    
    # Start batch processing
    batch_start_time = time.time()
    
    print(f"\nStarting batch processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Sequential processing enabled to avoid memory issues")
    
    # Process each model sequentially
    for i, model_config in enumerate(models, 1):
        print(f"\n{'='*20} MODEL {i}/{len(models)} {'='*20}")
        
        try:
            process_model_batch(
                model_config=model_config,
                configurations=configurations,
                data_path=data_path,
                output_base_path=output_base_path
            )
            
        except Exception as e:
            print(f"ERROR processing model {model_config['name']}: {e}")
            continue
    
    # Print final summary
    batch_time = time.time() - batch_start_time
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {batch_time:.2f}s ({batch_time/60:.1f} minutes)")
    print(f"Models processed: {len(models)}")
    print(f"Configurations per model: {len(configurations)}")
    print(f"Total extractions: {len(models) * len(configurations)}")
    print(f"Output directory: {output_base_path}")
    print()
    print("Next steps:")
    print("1. Apply PCA dimensionality reduction (MANDATORY for high dimensions)")
    print("2. Run linear probing classification")
    print("3. Compare results with pooling strategy")
    print(f"{'='*80}")


def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Batch feature extraction with concatenation strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/feature_extraction_concat.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Validate configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Run batch extraction
        run_batch_extraction(str(config_path))
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()