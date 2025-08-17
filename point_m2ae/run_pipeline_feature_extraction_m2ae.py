#!/usr/bin/env python3
"""
Point-M2AE feature extraction pipeline for AdaptFoundation project.

Orchestrates batch extraction of features from 3D skeletal volumes using
Point-M2AE hierarchical encoder with two aggregation approaches.
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
from typing import Dict

from data.loaders import HCPOFCDataLoader
from point_m2ae.feature_extraction_core_m2ae import PointM2AEFeatureExtractor, PointM2AEConfig


class PointM2AEFeatureDatasetSaver:
    """
    Feature dataset saver for Point-M2AE extraction pipeline.
    
    Creates structure: feature_extraction_point_m2ae/point_m2ae_encoder/feat_{approach}/
    """
    
    def __init__(self, data_path: str, output_base_path: str, model_name: str):
        """
        Initialize Point-M2AE feature saver.
        
        Args:
            data_path (str): Path to HCP OFC dataset
            output_base_path (str): Base output directory
            model_name (str): Model name for directory structure
        """
        self.data_path = data_path
        self.output_base_path = Path(output_base_path)
        self.model_name = model_name
        
        self.model_output_path = self.output_base_path / model_name
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = HCPOFCDataLoader(data_path)
    
    def save_feature_approach(self, approach_name: str, checkpoint_path: Path) -> None:
        """
        Extract and save features for specific approach.
        
        Args:
            approach_name (str): Feature approach ('feat_mean' or 'feat_mean_max')
            checkpoint_path (Path): Path to Point-M2AE checkpoint
        """
        print(f"Processing approach: {approach_name}")
        
        approach_dir = self.model_output_path / approach_name
        approach_dir.mkdir(exist_ok=True)
        
        # Initialize extractor
        cfg = PointM2AEConfig()
        extractor = PointM2AEFeatureExtractor(approach_name, checkpoint_path, cfg)
        
        # Process all splits
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
                volumes, labels, subject_ids = self.data_loader.load_split_as_tensor(split_name)
                features = extractor.extract_features_batch(volumes)
                
                self._save_split_data(features, labels, subject_ids, split_name, approach_dir)
                print(f"    Saved {len(subject_ids)} subjects, features shape: {features.shape}")
                
            except Exception as e:
                print(f"    Error processing {split_name}: {e}")
                continue
    
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
    required_sections = ['model', 'processing', 'data', 'feature_approaches']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    
    if not config['feature_approaches']:
        raise ValueError("No feature approaches specified")


def print_batch_summary(config: Dict) -> None:
    """
    Print batch processing summary.
    
    Args:
        config (Dict): Configuration dictionary
    """
    model_info = config['model']
    approaches = config['feature_approaches']
    data_info = config['data']
    
    print("=" * 60)
    print("POINT-M2AE FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Model: {model_info['name']} ({model_info['type']})")
    print(f"Checkpoint: {model_info['checkpoint_path']}")
    print(f"Data path: {data_info['dataset_path']}")
    print(f"Output: {data_info['output_base_path']}")
    print()
    print(f"Approaches ({len(approaches)}):")
    for i, (name, conf) in enumerate(approaches.items(), 1):
        print(f"  {i}. {name}: {conf['expected_output_dim']}D")
    print("=" * 60)


def run_batch_extraction(config_path: str) -> None:
    """
    Run complete batch feature extraction.
    
    Args:
        config_path (str): Path to configuration file
    """
    config = load_config(config_path)
    validate_config(config)
    
    print_batch_summary(config)
    
    # Initialize saver
    saver = PointM2AEFeatureDatasetSaver(
        data_path=config['data']['dataset_path'],
        output_base_path=config['data']['output_base_path'],
        model_name=config['model']['name']
    )
    
    # Process approaches
    checkpoint_path = Path(config['model']['checkpoint_path'])
    approaches = config['feature_approaches']
    
    start_time = time.time()
    
    for i, approach_name in enumerate(approaches.keys(), 1):
        print(f"\n--- APPROACH {i}/{len(approaches)} ---")
        
        try:
            saver.save_feature_approach(approach_name, checkpoint_path)
        except Exception as e:
            print(f"ERROR processing {approach_name}: {e}")
            continue
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("POINT-M2AE EXTRACTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Approaches processed: {len(approaches)}")
    print(f"Output: {config['data']['output_base_path']}")


def run_single_approach(config_path: str, approach_name: str) -> None:
    """
    Run extraction for single approach.
    
    Args:
        config_path (str): Path to configuration file
        approach_name (str): Approach to process
    """
    config = load_config(config_path)
    validate_config(config)
    
    if approach_name not in config['feature_approaches']:
        available = list(config['feature_approaches'].keys())
        print(f"ERROR: Approach '{approach_name}' not found")
        print(f"Available: {available}")
        sys.exit(1)
    
    print(f"Processing single approach: {approach_name}")
    
    saver = PointM2AEFeatureDatasetSaver(
        data_path=config['data']['dataset_path'],
        output_base_path=config['data']['output_base_path'],
        model_name=config['model']['name']
    )
    
    checkpoint_path = Path(config['model']['checkpoint_path'])
    
    try:
        saver.save_feature_approach(approach_name, checkpoint_path)
        print(f"Single approach extraction completed")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Point-M2AE feature extraction pipeline"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='feature_extraction_m2ae.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--approach',
        type=str,
        choices=['feat_mean', 'feat_mean_max'],
        help='Process specific approach only'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        if args.approach:
            run_single_approach(str(config_path), args.approach)
        else:
            run_batch_extraction(str(config_path))
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()