#!/usr/bin/env python3
"""
Feature extraction pipeline with SAM-Med3D density optimization for AdaptFoundation project.

This script orchestrates batch extraction of features from 3D skeletal volumes
using SAM-Med3D turbo with density-based spatial optimization approaches.

Usage:
python sam3d_variantes/turbo_with_density/run_pipeline_feature_extraction.py
python sam3d_variantes/turbo_with_density/run_pipeline_feature_extraction.py --config feature_extraction.yaml

In the cluster gpu-gw:
PYTHONPATH=. python3 sam3d_variantes/turbo_with_density/run_pipeline_feature_extraction.py
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Standard library imports
import yaml
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List

# Project imports - AFTER path setup
try:
    from data.loaders import HCPOFCDataLoader
    from models.feature_extraction_sam3d import SAMMed3DFeatureExtractor
    from sam3d_variantes.turbo_with_density.feature_extraction_core import SAMMed3DTurboDensityExtractor
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class DensityFeatureDatasetSaver:
    """
    Specialized saver for SAM-Med3D density optimization features.
    
    Creates structure: feature_extraction_density/sam_med3d_turbo_density/flatten_{approach}/
    """
    
    def __init__(self, data_path: str, output_base_path: str, model_name: str):
        """
        Initialize density feature saver.
        
        Args:
            data_path (str): Path to HCP OFC dataset
            output_base_path (str): Base output directory
            model_name (str): Model name for directory structure
        """
        self.data_path = data_path
        self.output_base_path = Path(output_base_path)
        self.model_name = model_name
        
        # Create model output directory
        self.model_output_path = self.output_base_path / model_name
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = HCPOFCDataLoader(data_path)
        
        print(f"DensityFeatureDatasetSaver initialized:")
        print(f"  Source data: {data_path}")
        print(f"  Output base: {self.model_output_path}")
        print(f"  Model: {model_name}")
    
    def save_density_approach(self, approach_name: str, base_extractor) -> None:
        """
        Extract and save features for a specific density approach.
        
        Args:
            approach_name (str): Density approach name ('baseline', 'masking', 'linear_weighting')
            base_extractor: Base SAM-Med3D extractor
        """
        print(f"Processing density approach: {approach_name}")
        
        # Create approach directory
        approach_dir = self.model_output_path / f"flatten_{approach_name}"
        approach_dir.mkdir(exist_ok=True)
        
        # Initialize density extractor
        density_extractor = SAMMed3DTurboDensityExtractor(
            approach=approach_name,
            base_extractor=base_extractor
        )
        
        # Process all splits
        results = {}
        split_names = [
            "train_val_split_0.csv",
            "train_val_split_1.csv", 
            "train_val_split_2.csv",
            "train_val_split_3.csv",
            "train_val_split_4.csv",
            "test_split.csv"
        ]
        
        for split_name in split_names:
            print(f"  Processing {split_name}...")
            try:
                # Load split data
                volumes, labels, subject_ids = self.data_loader.load_split_as_tensor(split_name)
                
                # Extract features using density approach
                features = density_extractor.extract_features_batch(volumes)
                
                # Store results
                results[split_name] = {
                    'features': features,
                    'labels': torch.tensor(labels),
                    'subject_ids': subject_ids,
                    'n_subjects': len(subject_ids),
                    'feature_dim': features.shape[1]
                }
                
                # Save split data
                self._save_split_data(features, labels, subject_ids, split_name, approach_dir)
                
                print(f"    Saved {len(subject_ids)} subjects, features shape: {features.shape}")
                
            except Exception as e:
                print(f"    Error processing {split_name}: {e}")
                continue
        
        # Print summary (no metadata.json creation)
        if results:
            total_samples = sum(r['n_subjects'] for r in results.values())
            feature_dim = next(iter(results.values()))['feature_dim']
            
            print(f"  Approach {approach_name} completed:")
            print(f"    Total samples: {total_samples}")
            print(f"    Feature dimension: {feature_dim}")
            print(f"    Saved to: {approach_dir}")
        else:
            print(f"  Warning: No splits processed for {approach_name}")
    
    def _save_split_data(self, features: torch.Tensor, labels: List, subject_ids: List[str], 
                        split_name: str, approach_dir: Path) -> None:
        """
        Save features and metadata for a single split.
        """
        base_name = split_name.replace('.csv', '')
        
        # Convert to numpy
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        labels_np = np.array(labels)
        
        # Save features
        features_file = approach_dir / f"{base_name}_features.npy"
        np.save(features_file, features_np)
        
        # Save metadata
        metadata_df = pd.DataFrame({
            'Subject': subject_ids,
            'Label': labels_np
        })
        metadata_file = approach_dir / f"{base_name}_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)


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
    required_sections = ['model', 'processing', 'data', 'density_approaches', 'aggregation_configs']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    if not config['density_approaches']:
        raise ValueError("No density approaches specified")
    
    print("Configuration validation passed")


def print_batch_summary(config: Dict) -> None:
    """
    Print summary of batch processing plan.
    
    Args:
        config (Dict): Configuration dictionary
    """
    model_info = config['model']
    density_approaches = config['density_approaches']
    data_info = config['data']
    
    print("=" * 80)
    print("BATCH FEATURE EXTRACTION WITH SAM-MED3D DENSITY OPTIMIZATION")
    print("=" * 80)
    print(f"Model: {model_info['name']} ({model_info['type']})")
    print(f"Checkpoint: {model_info['checkpoint_path']}")
    print(f"Data path: {data_info['dataset_path']}")
    print(f"Output base: {data_info['output_base_path']}")
    print()
    print(f"Density optimization approaches ({len(density_approaches)}):")
    for i, (approach_name, approach_config) in enumerate(density_approaches.items(), 1):
        output_dim = approach_config['expected_output_dim']
        description = approach_config['description']
        print(f"  {i}. {approach_name}: {output_dim}D")
        print(f"     {description}")
    print()
    print(f"Total extractions: {len(density_approaches)} density approaches")
    print(f"Input size: {config['processing']['input_size']}")
    print(f"Batch size: {config['processing']['batch_size']}")
    print("=" * 80)


def process_density_approach(approach_name: str, approach_config: Dict, 
                           saver: DensityFeatureDatasetSaver, 
                           base_extractor) -> None:
    """
    Process a single density optimization approach.
    
    Args:
        approach_name (str): Name of the density approach
        approach_config (Dict): Approach configuration dictionary
        saver (DensityFeatureDatasetSaver): Feature dataset saver instance
        base_extractor: Base SAM-Med3D extractor
    """
    description = approach_config['description']
    expected_dim = approach_config['expected_output_dim']
    
    print(f"\nProcessing density approach: {approach_name}")
    print(f"  Description: {description}")
    print(f"  Expected dimension: {expected_dim}D")
    print("-" * 60)
    
    approach_start_time = time.time()
    
    try:
        # Process approach using saver
        saver.save_density_approach(approach_name, base_extractor)
        
        approach_time = time.time() - approach_start_time
        print(f"  Approach {approach_name} completed in {approach_time:.2f}s")
        print(f"  Using unified pipeline: consistent processing for all approaches")
        
    except Exception as e:
        print(f"  ERROR in approach {approach_name}: {e}")
        import traceback
        traceback.print_exc()
        raise


def validate_extracted_features(output_path: Path, density_approaches: List[str]) -> bool:
    """
    Validate integrity of extracted density-optimized feature datasets.
    
    Args:
        output_path (Path): Base output directory
        density_approaches (List[str]): List of density approaches to validate
        
    Returns:
        bool: True if all validations pass
    """
    model_path = output_path / "sam_med3d_turbo_density"
    
    if not model_path.exists():
        print(f"ERROR: Output directory not found: {model_path}")
        return False
    
    validation_passed = True
    
    for approach_name in density_approaches:
        approach_dir = model_path / f"flatten_{approach_name}"
        
        if not approach_dir.exists():
            print(f"Approach directory missing: flatten_{approach_name}")
            validation_passed = False
            continue
        
        # Check required files (NO MORE metadata.json)
        required_files = [
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
            if not (approach_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Missing files in flatten_{approach_name}: {missing_files}")
            validation_passed = False
        else:
            print(f"flatten_{approach_name}: All files present ✅")
    
    return validation_passed


def run_batch_extraction(config_path: str) -> None:
    """
    Run complete batch feature extraction for all density optimization approaches.
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    # Load and validate configuration
    print("Loading SAM-Med3D density configuration...")
    config = load_config(config_path)
    validate_config(config)
    
    # Print batch summary
    print_batch_summary(config)
    
    # Extract configuration components
    model_name = config['model']['name']
    data_path = config['data']['dataset_path']
    output_base_path = config['data']['output_base_path']
    density_approaches = config['density_approaches']
    
    # Initialize feature dataset saver
    print(f"\nInitializing SAM-Med3D density feature saver...")
    saver = DensityFeatureDatasetSaver(
        data_path=data_path,
        output_base_path=output_base_path,
        model_name=f"{model_name}_density"
    )
    
    # Initialize base extractor (reused for all approaches)
    print(f"Initializing base SAM-Med3D extractor...")
    try:
        base_extractor = SAMMed3DFeatureExtractor(
            config_path=config_path,
            aggregation_method='flatten'
        )
        print(f"✅ Base extractor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing base extractor: {e}")
        print(f"Config aggregation_configs keys: {list(config.get('aggregation_configs', {}).keys())}")
        raise
    
    # Start batch processing
    batch_start_time = time.time()
    
    print(f"\nStarting batch processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Sequential processing enabled for optimal memory usage")
    
    # Process each density approach sequentially
    approach_names = list(density_approaches.keys())
    
    for i, approach_name in enumerate(approach_names, 1):
        print(f"\n{'='*20} APPROACH {i}/{len(approach_names)} {'='*20}")
        
        approach_config = density_approaches[approach_name]
        
        try:
            process_density_approach(
                approach_name=approach_name,
                approach_config=approach_config,
                saver=saver,
                base_extractor=base_extractor
            )
            
        except Exception as e:
            print(f"ERROR processing approach {approach_name}: {e}")
            continue
    
    # Print final summary
    batch_time = time.time() - batch_start_time
    
    print(f"\n{'='*80}")
    print("SAM-MED3D DENSITY BATCH PROCESSING COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {batch_time:.2f}s ({batch_time/60:.1f} minutes)")
    print(f"Density approaches processed: {len(approach_names)}")
    print(f"Output directory: {output_base_path}")
    print()
    print("Next steps:")
    print("1. Apply PCA dimensionality reduction (MANDATORY for all approaches)")
    print("2. Run linear probing classification")
    print("3. Compare density optimization results with baseline SAM-Med3D")
    print(f"{'='*80}")
    
    # Validate results
    print(f"\n=== Validation ===")
    validation_passed = validate_extracted_features(
        Path(output_base_path), 
        approach_names
    )
    
    if validation_passed:
        print("All SAM-Med3D density configurations validated successfully")
        print(f"Features saved to: {Path(output_base_path) / f'{model_name}_density'}")
    else:
        print("Validation failed - check error messages above")
        sys.exit(1)


def run_single_approach(config_path: str, approach_name: str) -> None:
    """
    Run feature extraction for a single density approach.
    
    Args:
        config_path (str): Path to configuration YAML file
        approach_name (str): Name of the density approach to process
    """
    # Load configuration
    config = load_config(config_path)
    validate_config(config)
    
    # Verify approach exists
    if approach_name not in config['density_approaches']:
        available_approaches = list(config['density_approaches'].keys())
        print(f"ERROR: Approach '{approach_name}' not found in configuration")
        print(f"Available approaches: {available_approaches}")
        sys.exit(1)
    
    print("=== SAM-Med3D Density Single Approach Extraction ===")
    print(f"Configuration: {config_path}")
    print(f"Approach: {approach_name}")
    
    # Initialize saver
    saver = DensityFeatureDatasetSaver(
        data_path=config['data']['dataset_path'],
        output_base_path=config['data']['output_base_path'],
        model_name=f"{config['model']['name']}_density"
    )
    
    # Initialize base extractor
    base_extractor = SAMMed3DFeatureExtractor(
        config_path=config_path,
        aggregation_method='flatten'
    )
    
    # Process single approach
    start_time = time.time()
    approach_config = config['density_approaches'][approach_name]
    
    try:
        process_density_approach(approach_name, approach_config, saver, base_extractor)
        
        elapsed_time = time.time() - start_time
        print(f"\nSingle approach extraction completed in {elapsed_time:.2f}s")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Batch feature extraction with SAM-Med3D density optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='feature_extraction.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--approach',
        type=str,
        choices=['baseline', 'masking', 'linear_weighting'],
        help='Process only a specific density approach'
    )
    
    args = parser.parse_args()
    
    # Construct full path relative to script location
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        if args.approach:
            # Process single approach
            run_single_approach(str(config_path), args.approach)
        else:
            # Process all approaches
            run_batch_extraction(str(config_path))
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()