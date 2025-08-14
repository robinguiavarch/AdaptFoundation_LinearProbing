#!/usr/bin/env python3
"""
Feature extraction pipeline for S.C.-sylv dataset using SAM-Med3D standard model.

This script extracts features from S.C.-sylv 3D skeletal volumes using SAM-Med3D turbo
with standard flatten aggregation for regression task.

Usage:
python sam3d/SC_sylv/run_pipeline_feature_extraction_sc.py
python sam3d/SC_sylv/run_pipeline_feature_extraction_sc.py --config feature_extraction_sc.yaml
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
from datetime import datetime
from typing import Dict

try:
    from data.loader_sc import SCDataLoader
    from sam3d.SC_sylv.feature_extraction_core_sc import SAMMed3DStandardExtractor
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class SCFeatureDatasetSaver:
    """
    Feature dataset saver for S.C.-sylv dataset with simplified structure.
    
    Creates structure: feature_extraction_sam3d_sc/flatten/
    
    Attributes:
        data_path (str): Path to S.C.-sylv dataset
        output_base_path (Path): Base output directory
        flatten_dir (Path): Flatten approach directory
        data_loader (SCDataLoader): Data loader for S.C.-sylv dataset
    """
    
    def __init__(self, data_path: str, output_base_path: str):
        """
        Initialize S.C.-sylv feature dataset saver.
        
        Args:
            data_path (str): Path to S.C.-sylv dataset
            output_base_path (str): Base output directory
        """
        self.data_path = data_path
        self.output_base_path = Path(output_base_path)
        
        self.flatten_dir = self.output_base_path / "flatten"
        self.flatten_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = SCDataLoader(data_path)
        
        print(f"S.C.-sylv Feature Dataset Saver initialized")
        print(f"Source data: {data_path}")
        print(f"Output directory: {self.flatten_dir}")
    
    def save_features(self, extractor: SAMMed3DStandardExtractor) -> None:
        """
        Extract and save features for S.C.-sylv dataset.
        
        Args:
            extractor (SAMMed3DStandardExtractor): Feature extractor instance
        """
        print("Processing S.C.-sylv dataset with flatten aggregation")
        
        split_names = [
            "train_val_split_0.csv",
            "train_val_split_1.csv", 
            "train_val_split_2.csv",
            "train_val_split_3.csv",
            "train_val_split_4.csv",
            "test_split.csv"
        ]
        
        total_subjects = 0
        
        for split_name in split_names:
            print(f"Processing {split_name}")
            try:
                volumes, labels, subject_ids = self.data_loader.load_split_as_tensor(split_name)
                
                features = extractor.extract_features_batch(volumes)
                
                self._save_split_data(features, labels, subject_ids, split_name)
                
                total_subjects += len(subject_ids)
                print(f"  Saved {len(subject_ids)} subjects, features shape: {features.shape}")
                
            except Exception as e:
                print(f"  Error processing {split_name}: {e}")
                continue
        
        if total_subjects > 0:
            feature_dim = features.shape[1] if 'features' in locals() else 196608
            print(f"S.C.-sylv feature extraction completed")
            print(f"  Total subjects: {total_subjects}")
            print(f"  Feature dimension: {feature_dim}")
            print(f"  Saved to: {self.flatten_dir}")
        else:
            print("Warning: No splits processed successfully")
    
    def _save_split_data(self, features: torch.Tensor, labels: list, subject_ids: list, 
                        split_name: str) -> None:
        """
        Save features and metadata for a single split.
        
        Args:
            features (torch.Tensor): Extracted features
            labels (list): Subject labels (6D for SC_sylv regression)
            subject_ids (list): Subject identifiers
            split_name (str): Split filename
        """
        base_name = split_name.replace('.csv', '')
        
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        labels_np = np.array(labels)
        
        features_file = self.flatten_dir / f"{base_name}_features.npy"
        np.save(features_file, features_np)
        
        # Convert 6D labels to string representation for CSV storage
        metadata_df = pd.DataFrame({
            'Subject': subject_ids,
            'Label': [str(label.tolist()) for label in labels_np]
        })
        metadata_file = self.flatten_dir / f"{base_name}_metadata.csv"
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
    required_sections = ['model', 'processing', 'data', 'aggregation_configs']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    if 'flatten' not in config['aggregation_configs']:
        raise ValueError("Missing flatten aggregation configuration")
    
    print("Configuration validation passed")


def print_pipeline_summary(config: Dict) -> None:
    """
    Print summary of feature extraction pipeline.
    
    Args:
        config (Dict): Configuration dictionary
    """
    model_info = config['model']
    data_info = config['data']
    
    print("=" * 80)
    print("SAM-MED3D FEATURE EXTRACTION FOR S.C.-sylv DATASET")
    print("=" * 80)
    print(f"Model: {model_info['name']} ({model_info['type']})")
    print(f"Checkpoint: {model_info['checkpoint_path']}")
    print(f"Data path: {data_info['dataset_path']}")
    print(f"Output path: {data_info['output_base_path']}")
    print(f"Aggregation: flatten (196608D)")
    print(f"Input size: {config['processing']['input_size']}")
    print(f"Batch size: {config['processing']['batch_size']}")
    print("=" * 80)


def run_feature_extraction(config_path: str) -> None:
    """
    Run complete feature extraction pipeline for S.C.-sylv dataset.
    
    Args:
        config_path (str): Path to configuration YAML file
    """
    print("Loading S.C.-sylv feature extraction configuration")
    config = load_config(config_path)
    validate_config(config)
    
    print_pipeline_summary(config)
    
    data_path = config['data']['dataset_path']
    output_base_path = config['data']['output_base_path']
    
    print("Initializing S.C.-sylv feature dataset saver")
    saver = SCFeatureDatasetSaver(
        data_path=data_path,
        output_base_path=output_base_path
    )
    
    print("Initializing SAM-Med3D standard extractor")
    try:
        extractor = SAMMed3DStandardExtractor(config_path=config_path)
        print("Extractor initialized successfully")
    except Exception as e:
        print(f"Error initializing extractor: {e}")
        raise
    
    start_time = time.time()
    
    print(f"Starting feature extraction at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        saver.save_features(extractor)
        
        extraction_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print("S.C.-sylv FEATURE EXTRACTION COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {extraction_time:.2f}s ({extraction_time/60:.1f} minutes)")
        print(f"Output directory: {output_base_path}")
        print()
        print("Next steps:")
        print("1. Apply PCA dimensionality reduction")
        print("2. Run linear probing regression")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise


def main():
    """
    Main entry point for feature extraction script.
    """
    parser = argparse.ArgumentParser(
        description="Feature extraction for S.C.-sylv dataset using SAM-Med3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='feature_extraction_sc.yaml',
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        run_feature_extraction(str(config_path))
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()