#!/usr/bin/env python3
"""
Point-M2AE feature extraction pipeline with optimized preprocessing for AdaptFoundation project.

Orchestrates batch extraction of features from 3D skeletal volumes using
Point-M2AE hierarchical encoder with two preprocessing approaches.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import yaml
import argparse
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List

from data.loaders import HCPOFCDataLoader
from point_m2ae.feature_extraction_core_m2ae_pp import PointM2AEFeatureExtractor


class PointM2AEFeatureDatasetSaver:
    """
    Feature dataset saver for Point-M2AE extraction pipeline with preprocessing versions.
    
    Creates structure: feature_extraction_point_m2ae_pp/feat_mean_{version}/
    
    Attributes:
        data_path (str): Path to HCP OFC dataset
        output_base_path (Path): Base output directory  
        preprocessing_versions (List[str]): List of preprocessing versions to process
        data_loader: HCP data loader instance
    """
    
    def __init__(self, data_path: str, output_base_path: str, preprocessing_versions: List[str] = ['v1', 'v2']):
        """
        Initialize Point-M2AE feature saver with preprocessing versions.
        
        Args:
            data_path (str): Path to HCP OFC dataset
            output_base_path (str): Base output directory
            preprocessing_versions (List[str]): Preprocessing versions to process. Defaults to ['v1', 'v2'].
        """
        self.data_path = data_path
        self.output_base_path = Path(output_base_path)
        self.preprocessing_versions = preprocessing_versions
        
        # Create base output directory
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = HCPOFCDataLoader(data_path)
    
    def save_feature_approach(self, approach_name: str, checkpoint_path: Path, config: Dict) -> None:
        """
        Extract and save features for specific approach with all preprocessing versions.
        
        Args:
            approach_name (str): Feature approach ('feat_mean')
            checkpoint_path (Path): Path to Point-M2AE checkpoint
            config (Dict): Full configuration dictionary
        """
        print(f"Processing approach: {approach_name}")
        
        # Process each preprocessing version
        for prep_version in self.preprocessing_versions:
            print(f"  Processing preprocessing version: {prep_version}")
            
            # Create versioned approach directory
            versioned_approach_name = f"{approach_name}_{prep_version}"
            approach_dir = self.output_base_path / versioned_approach_name
            approach_dir.mkdir(exist_ok=True)
            
            # Initialize extractor with preprocessing version
            extractor = PointM2AEFeatureExtractor(approach_name, checkpoint_path, config, prep_version)
            
            # Process all splits for this version
            self._process_all_splits(extractor, approach_dir, prep_version)
    
    def _process_all_splits(self, extractor: PointM2AEFeatureExtractor, approach_dir: Path, prep_version: str) -> None:
        """
        Process all splits for a given preprocessing version.
        
        Args:
            extractor (PointM2AEFeatureExtractor): Feature extractor instance
            approach_dir (Path): Output directory for this approach version
            prep_version (str): Preprocessing version identifier
        """
        split_names = [
            "train_val_split_0.csv",
            "train_val_split_1.csv", 
            "train_val_split_2.csv",
            "train_val_split_3.csv",
            "train_val_split_4.csv",
            "test_split.csv"
        ]
        
        for split_name in split_names:
            print(f"    Processing {split_name}...")
            try:
                volumes, labels, subject_ids = self.data_loader.load_split_as_tensor(split_name)
                features = extractor.extract_features_batch(volumes)
                
                # Validate preprocessing output
                self._validate_preprocessing_output(features, prep_version)
                
                self._save_split_data(features, labels, subject_ids, split_name, approach_dir)
                print(f"      Saved {len(subject_ids)} subjects, features shape: {features.shape}")
                
            except Exception as e:
                print(f"      Error processing {split_name}: {e}")
                continue
    
    def _validate_preprocessing_output(self, features: torch.Tensor, prep_version: str) -> None:
        """
        Validate preprocessing output dimensions and ranges.
        
        Args:
            features (torch.Tensor): Extracted features
            prep_version (str): Preprocessing version
        """
        expected_final_dim = 384  # Always 384D after feat_mean aggregation
        
        if features.shape[1] != expected_final_dim:
            raise ValueError(f"Expected {expected_final_dim}D features, got {features.shape[1]}D")
    
    def _save_split_data(self, features: torch.Tensor, labels, subject_ids, 
                        split_name: str, approach_dir: Path) -> None:
        """
        Save features and metadata for single split.
        
        Args:
            features (torch.Tensor): Extracted features
            labels: Subject labels
            subject_ids: Subject identifiers
            split_name (str): Split filename
            approach_dir (Path): Approach output directory
        """
        base_name = split_name.replace('.csv', '')
        
        # Save features
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        features_file = approach_dir / f"{base_name}_features.npy"
        np.save(features_file, features_np)
        
        # Save metadata
        metadata_df = pd.DataFrame({
            'Subject': subject_ids,
            'Label': np.array(labels)
        })
        metadata_file = approach_dir / f"{base_name}_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict) -> None:
    """
    Validate configuration structure.
    
    Args:
        config (Dict): Configuration dictionary
    """
    required_sections = ['model', 'processing', 'data', 'feature_approaches', 'preprocessing']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    
    if not config['feature_approaches']:
        raise ValueError("No feature approaches specified")


def print_batch_summary(config: Dict, preprocessing_versions: List[str]) -> None:
    """
    Print batch processing summary.
    
    Args:
        config (Dict): Configuration dictionary
        preprocessing_versions (List[str]): Preprocessing versions to process
    """
    model_info = config['model']
    approaches = config['feature_approaches']
    data_info = config['data']
    
    print("=" * 60)
    print("POINT-M2AE FEATURE EXTRACTION WITH OPTIMIZED PREPROCESSING")
    print("=" * 60)
    print(f"Model: {model_info['name']} ({model_info['type']})")
    print(f"Checkpoint: {model_info['checkpoint_path']}")
    print(f"Data path: {data_info['dataset_path']}")
    print(f"Output: {data_info['output_base_path']}")
    print()
    print(f"Approaches ({len(approaches)}):")
    for i, (name, conf) in enumerate(approaches.items(), 1):
        print(f"  {i}. {name}: {conf['expected_output_dim']}D")
    print()
    print(f"Preprocessing versions ({len(preprocessing_versions)}):")
    for i, version in enumerate(preprocessing_versions, 1):
        version_info = config['preprocessing']['versions'][version]
        print(f"  {i}. {version}: {version_info['description']}")
    print("=" * 60)


def run_batch_extraction(config_path: str, preprocessing_versions: List[str] = None) -> None:
    """
    Run complete batch feature extraction with preprocessing versions.
    
    Args:
        config_path (str): Path to configuration file
        preprocessing_versions (List[str], optional): Preprocessing versions to process. 
            Defaults to None (uses config default).
    """
    config = load_config(config_path)
    validate_config(config)
    
    if preprocessing_versions is None:
        preprocessing_versions = ['v1', 'v2']
    
    print_batch_summary(config, preprocessing_versions)
    
    # Initialize saver
    saver = PointM2AEFeatureDatasetSaver(
        data_path=config['data']['dataset_path'],
        output_base_path=config['data']['output_base_path'],
        preprocessing_versions=preprocessing_versions
    )
    
    # Process approaches
    checkpoint_path = Path(config['model']['checkpoint_path'])
    approaches = config['feature_approaches']
    
    start_time = time.time()
    
    for i, approach_name in enumerate(approaches.keys(), 1):
        print(f"\n--- APPROACH {i}/{len(approaches)} ---")
        
        try:
            saver.save_feature_approach(approach_name, checkpoint_path, config)
        except Exception as e:
            print(f"ERROR processing {approach_name}: {e}")
            continue
    
    # Summary
    total_time = time.time() - start_time
    total_combinations = len(approaches) * len(preprocessing_versions)
    
    print(f"\n{'='*60}")
    print("POINT-M2AE EXTRACTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Combinations processed: {total_combinations}")
    print(f"Output: {config['data']['output_base_path']}")


def run_single_approach(config_path: str, approach_name: str, preprocessing_version: str = None) -> None:
    """
    Run extraction for single approach and preprocessing version.
    
    Args:
        config_path (str): Path to configuration file
        approach_name (str): Approach to process
        preprocessing_version (str, optional): Specific preprocessing version. Defaults to None (both).
    """
    config = load_config(config_path)
    validate_config(config)
    
    if approach_name not in config['feature_approaches']:
        available = list(config['feature_approaches'].keys())
        print(f"ERROR: Approach '{approach_name}' not found")
        print(f"Available: {available}")
        sys.exit(1)
    
    preprocessing_versions = [preprocessing_version] if preprocessing_version else ['v1', 'v2']
    
    print(f"Processing single approach: {approach_name}")
    print(f"Preprocessing versions: {preprocessing_versions}")
    
    saver = PointM2AEFeatureDatasetSaver(
        data_path=config['data']['dataset_path'],
        output_base_path=config['data']['output_base_path'],
        preprocessing_versions=preprocessing_versions
    )
    
    checkpoint_path = Path(config['model']['checkpoint_path'])
    
    try:
        saver.save_feature_approach(approach_name, checkpoint_path, config)
        print(f"Single approach extraction completed")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Point-M2AE feature extraction pipeline with optimized preprocessing"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='feature_extraction_m2ae_pp.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--approach',
        type=str,
        choices=['feat_mean'],
        help='Process specific approach only'
    )
    
    parser.add_argument(
        '--preprocessing',
        type=str,
        choices=['v1', 'v2', 'both'],
        default='both',
        help='Preprocessing version: v1 (fixed), v2 (topological), both (default)'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Parse preprocessing argument
    if args.preprocessing == 'both':
        preprocessing_versions = ['v1', 'v2']
    else:
        preprocessing_versions = [args.preprocessing]
    
    try:
        if args.approach:
            preprocessing_version = None if args.preprocessing == 'both' else args.preprocessing
            run_single_approach(str(config_path), args.approach, preprocessing_version)
        else:
            run_batch_extraction(str(config_path), preprocessing_versions)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()