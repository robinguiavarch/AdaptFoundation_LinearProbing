#!/usr/bin/env python3
"""
Feature extraction script with SAM-Med3D strategy for AdaptFoundation project.

This script orchestrates batch extraction of features from 3D skeletal volumes
using SAM-Med3D native 3D approach with configurable spatial aggregation.

Usage:
python scripts/run_feature_extraction_sam3d.py
python scripts/run_feature_extraction_sam3d.py --config configs/feature_extraction_sam3d.yaml

In the cluster gpu-gw:
PYTHONPATH=. python3 scripts/run_feature_extraction_sam3d.py
"""

import yaml
import argparse
import time
from pathlib import Path
import sys
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.feature_saver_sam3d import FeatureDatasetSaverSAM3D


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
    required_sections = ['model', 'processing', 'data', 'aggregation_configs']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    if not config['aggregation_configs']:
        raise ValueError("No aggregation configurations specified")
    
    print("Configuration validation passed")


def print_batch_summary(config: Dict) -> None:
    """
    Print summary of batch processing plan.
    
    Args:
        config (Dict): Configuration dictionary
    """
    model_info = config['model']
    aggregation_configs = config['aggregation_configs']
    data_info = config['data']
    
    print("=" * 80)
    print("BATCH FEATURE EXTRACTION WITH SAM-MED3D STRATEGY")
    print("=" * 80)
    print(f"Model: {model_info['name']} ({model_info['type']})")
    print(f"Checkpoint: {model_info['checkpoint_path']}")
    print(f"Data path: {data_info['dataset_path']}")
    print(f"Output base: {data_info['output_base_path']}")
    print()
    print(f"Spatial aggregation methods ({len(aggregation_configs)}):")
    for i, (method_name, method_config) in enumerate(aggregation_configs.items(), 1):
        output_dim = method_config['output_dim']
        pca_required = method_config['pca_required']
        print(f"  {i}. {method_name}: {output_dim}D {'(PCA required)' if pca_required else ''}")
        print(f"     {method_config['description']}")
    print()
    print(f"Total extractions: {len(aggregation_configs)} aggregation methods")
    print(f"Input size: {config['processing']['input_size']}")
    print(f"Batch size: {config['processing']['batch_size']}")
    print("=" * 80)


def process_aggregation_method(method_name: str, method_config: Dict, 
                              saver: FeatureDatasetSaverSAM3D) -> None:
    """
    Process a single aggregation method configuration.
    
    Args:
        method_name (str): Name of the aggregation method
        method_config (Dict): Method configuration dictionary
        saver (FeatureDatasetSaverSAM3D): Feature dataset saver instance
    """
    output_dim = method_config['output_dim']
    pca_required = method_config['pca_required']
    description = method_config['description']
    
    print(f"\nProcessing aggregation method: {method_name}")
    print(f"  Description: {description}")
    print(f"  Output dimension: {output_dim}D")
    print(f"  PCA required: {pca_required}")
    print("-" * 60)
    
    method_start_time = time.time()
    
    try:
        # Save configuration using SAM-Med3D strategy
        saver.save_configuration(aggregation_method=method_name)
        
        method_time = time.time() - method_start_time
        print(f"  Method {method_name} completed in {method_time:.2f}s")
        
    except Exception as e:
        print(f"  ERROR in method {method_name}: {e}")
        raise


def validate_extracted_features(output_path: Path, aggregation_methods: List[str]) -> bool:
    """
    Validate integrity of extracted SAM-Med3D feature datasets.
    
    Args:
        output_path (Path): Base output directory
        aggregation_methods (List[str]): List of aggregation methods to validate
        
    Returns:
        bool: True if all validations pass
    """
    model_path = output_path / "sam_med3d_turbo"
    
    if not model_path.exists():
        print(f"ERROR: Output directory not found: {model_path}")
        return False
    
    validation_passed = True
    
    for method_name in aggregation_methods:
        method_dir = model_path / method_name
        
        if not method_dir.exists():
            print(f"Method directory missing: {method_name}")
            validation_passed = False
            continue
        
        # Check required files
        required_files = [
            "metadata.json",
            "test_split_features.npy",
            "test_split_metadata.csv"
        ]
        
        # Add train/val split files
        for i in range(5):
            required_files.extend([
                f"train_val_split_{i}_features.npy",
                f"train_val_split_{i}_metadata.csv"
            ])
        
        missing_files = []
        for file_name in required_files:
            if not (method_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Missing files in {method_name}: {missing_files}")
            validation_passed = False
        else:
            print(f"{method_name}: All files present")
    
    return validation_passed


def run_batch_extraction(config_path: str) -> None:
    """
    Run complete batch feature extraction for all SAM-Med3D aggregation methods.
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    # Load and validate configuration
    print("Loading SAM-Med3D configuration...")
    config = load_config(config_path)
    validate_config(config)
    
    # Print batch summary
    print_batch_summary(config)
    
    # Extract configuration components
    model_name = config['model']['name']
    data_path = config['data']['dataset_path']
    output_base_path = config['data']['output_base_path']
    aggregation_configs = config['aggregation_configs']
    
    # Initialize feature dataset saver
    print(f"\nInitializing SAM-Med3D feature saver...")
    saver = FeatureDatasetSaverSAM3D(
        data_path=data_path,
        output_base_path=output_base_path,
        model_name=model_name,
        config_path=config_path
    )
    
    # Start batch processing
    batch_start_time = time.time()
    
    print(f"\nStarting batch processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Sequential processing enabled for optimal memory usage")
    
    # Process each aggregation method sequentially
    aggregation_methods = list(aggregation_configs.keys())
    
    for i, method_name in enumerate(aggregation_methods, 1):
        print(f"\n{'='*20} METHOD {i}/{len(aggregation_methods)} {'='*20}")
        
        method_config = aggregation_configs[method_name]
        
        try:
            process_aggregation_method(
                method_name=method_name,
                method_config=method_config,
                saver=saver
            )
            
        except Exception as e:
            print(f"ERROR processing method {method_name}: {e}")
            continue
    
    # Print final summary
    batch_time = time.time() - batch_start_time
    
    print(f"\n{'='*80}")
    print("SAM-MED3D BATCH PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {batch_time:.2f}s ({batch_time/60:.1f} minutes)")
    print(f"Aggregation methods processed: {len(aggregation_methods)}")
    print(f"Output directory: {output_base_path}")
    print()
    print("Next steps:")
    print("1. Apply PCA dimensionality reduction (MANDATORY for flatten method)")
    print("2. Run linear probing classification")
    print("3. Compare SAM-Med3D results with DINOv2 strategies")
    print(f"{'='*80}")
    
    # Validate results
    print(f"\n=== Validation ===")
    validation_passed = validate_extracted_features(
        Path(output_base_path), 
        aggregation_methods
    )
    
    if validation_passed:
        print("All SAM-Med3D configurations validated successfully")
        print(f"Features saved to: {Path(output_base_path) / model_name}")
    else:
        print("Validation failed - check error messages above")
        sys.exit(1)


def run_single_method(config_path: str, method_name: str) -> None:
    """
    Run feature extraction for a single aggregation method.
    
    Args:
        config_path (str): Path to configuration YAML file
        method_name (str): Name of the aggregation method to process
    """
    # Load configuration
    config = load_config(config_path)
    validate_config(config)
    
    # Verify method exists
    if method_name not in config['aggregation_configs']:
        available_methods = list(config['aggregation_configs'].keys())
        print(f"ERROR: Method '{method_name}' not found in configuration")
        print(f"Available methods: {available_methods}")
        sys.exit(1)
    
    print("=== SAM-Med3D Single Method Extraction ===")
    print(f"Configuration: {config_path}")
    print(f"Method: {method_name}")
    
    # Initialize saver
    saver = FeatureDatasetSaverSAM3D(
        data_path=config['data']['dataset_path'],
        output_base_path=config['data']['output_base_path'],
        model_name=config['model']['name'],
        config_path=config_path
    )
    
    # Process single method
    start_time = time.time()
    method_config = config['aggregation_configs'][method_name]
    
    try:
        process_aggregation_method(method_name, method_config, saver)
        
        elapsed_time = time.time() - start_time
        print(f"\nSingle method extraction completed in {elapsed_time:.2f}s")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Batch feature extraction with SAM-Med3D strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/feature_extraction_sam3d.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['avg_pool', 'max_pool', 'sum_pool', 'flatten'],
        help='Process only a specific aggregation method'
    )
    
    args = parser.parse_args()
    
    # Validate configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        if args.method:
            # Process single method
            run_single_method(str(config_path), args.method)
        else:
            # Process all methods
            run_batch_extraction(str(config_path))
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()