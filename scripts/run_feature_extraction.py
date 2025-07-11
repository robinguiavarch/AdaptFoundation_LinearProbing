"""
Feature extraction orchestration script for AdaptFoundation project.

This script runs the complete feature extraction pipeline for all standard
configurations and saves the results in an organized structure.

Default config: 
python scripts/run_feature_extraction.py

Personnalized config:
python scripts/run_feature_extraction.py --config configs/feature_extraction.yaml

In the cluster gpu-gw:
PYTHONPATH=. python3 scripts/run_feature_extraction.py
"""

import argparse
import time
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.feature_saver import FeatureDatasetSaver


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


def get_configurations_from_config(config):
    """
    Extract configurations list from loaded config.
    
    Args:
        config (Dict): Loaded configuration dictionary
        
    Returns:
        List[Dict]: List of configuration dictionaries
    """
    configurations = []
    
    for cfg in config['configurations']:
        configurations.append({
            'pooling_strategy': cfg['pooling_strategy'],
            'required_axes': cfg['required_axes'],
            'name': cfg['name']
        })
    
    return configurations


def validate_extracted_features(output_path, configurations):
    """
    Validate integrity of extracted feature datasets.
    
    Args:
        output_path (Path): Base output directory
        configurations (List[Dict]): List of configurations to validate
        
    Returns:
        bool: True if all validations pass
    """
    model_path = output_path / "dinov2_vits14"
    
    if not model_path.exists():
        print(f"ERROR: Output directory not found: {model_path}")
        return False
    
    validation_passed = True
    
    for config in configurations:
        config_dir = model_path / config['name']
        
        if not config_dir.exists():
            print(f"Configuration directory missing: {config['name']}")
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
            if not (config_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Missing files in {config['name']}: {missing_files}")
            validation_passed = False
        else:
            print(f"{config['name']}: All files present")
    
    return validation_passed


def run_feature_extraction(config_path):
    """
    Run complete feature extraction pipeline using YAML configuration.
    
    Args:
        config_path (str): Path to YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    print("=== AdaptFoundation Feature Extraction Pipeline ===")
    print(f"Configuration: {config_path}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Model: {config['model']['name']}")
    
    # Initialize feature dataset saver
    saver = FeatureDatasetSaver(
        data_path=config['dataset']['data_path'],
        output_base_path=config['output']['base_path'],
        model_name=config['model']['name']
    )
    
    # Get configurations from YAML
    configurations = get_configurations_from_config(config)
    print(f"\nProcessing {len(configurations)} configurations...")
    
    # Track timing
    start_time = time.time()
    
    # Process each configuration
    for i, cfg in enumerate(configurations, 1):
        print(f"\n=== Configuration {i}/{len(configurations)}: {cfg['name']} ===")
        
        config_start_time = time.time()
        
        try:
            saver.save_configuration(
                pooling_strategy=cfg['pooling_strategy'],
                required_axes=cfg['required_axes']
            )
            
            config_elapsed = time.time() - config_start_time
            print(f"{cfg['name']} completed in {config_elapsed:.1f}s")
            
        except Exception as e:
            print(f"Error processing {cfg['name']}: {str(e)}")
            continue
    
    total_elapsed = time.time() - start_time
    print(f"\n=== Pipeline Completed ===")
    print(f"Total processing time: {total_elapsed:.1f}s")
    print(f"Average per configuration: {total_elapsed/len(configurations):.1f}s")
    
    # Validate results
    print(f"\n=== Validation ===")
    validation_passed = validate_extracted_features(
        Path(config['output']['base_path']), 
        configurations
    )
    
    if validation_passed:
        print("All configurations validated successfully")
        print(f"Features saved to: {Path(config['output']['base_path']) / config['model']['name']}")
    else:
        print("Validation failed - check error messages above")
        sys.exit(1)


def main():
    """
    Main entry point for feature extraction script.
    """
    parser = argparse.ArgumentParser(
        description="Extract features from HCP OFC dataset using foundation models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature_extraction.yaml",
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file does not exist: {config_path}")
        sys.exit(1)
    
    # Run feature extraction
    run_feature_extraction(str(config_path))


if __name__ == "__main__":
    main()