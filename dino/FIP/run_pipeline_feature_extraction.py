"""
Pipeline 1: Feature extraction from DINOv2 Giant for F.I.P. dataset.

This script extracts raw feature map features using 2.5D overlapping methodology
without PCA reduction. Features are saved for subsequent PCA processing.
"""

import numpy as np
import pandas as pd
import torch
import yaml
import time
import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.loader_fip import FIPDataLoader
from feature_extraction_core import (
    Method25D, FeatureMapExtractor, SpatialAggregator
)


class FIPFeatureExtractionPipeline:
    """
    Pipeline 1: Raw feature extraction for F.I.P. dataset using DINOv2 Giant.
    
    This pipeline extracts feature maps for pooling variants without PCA reduction,
    saving raw features for subsequent processing by Pipeline 2.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        device (torch.device): Computation device
        data_loader (FIPDataLoader): Data loader for F.I.P. dataset
        dinov2_model (torch.nn.Module): Loaded DINOv2 model
        method_25d (Method25D): 2.5D preprocessing
        feature_extractor (FeatureMapExtractor): Feature map extraction
    """
    
    def __init__(self, config_path: str):
        """
        Initialize F.I.P. feature extraction pipeline.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data_loader = FIPDataLoader(self.config['dataset']['data_path'])
        
        model_name = self.config['model']['name']
        print(f"Loading {model_name}...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dinov2_model.eval()
        self.dinov2_model.to(self.device)
        
        self.method_25d = Method25D()
        self.feature_extractor = FeatureMapExtractor(device=self.device, 
                                                   batch_size=self.config['model']['batch_size'])
        
        print(f"F.I.P. feature extraction pipeline initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Variants: {len(self.config['configurations'])}")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load YAML configuration file.
        
        Args:
            config_path (str): Path to configuration file
        
        Returns:
            dict: Loaded configuration
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_single_subject(self, skeleton_volume: np.ndarray, variant_config: dict) -> np.ndarray:
        """
        Extract feature map features for a single subject.
        
        Args:
            skeleton_volume (np.ndarray): 3D skeleton volume with shape (39, 45, 44)
            variant_config (dict): Variant configuration
        
        Returns:
            np.ndarray: Extracted features for one subject
        """
        aggregation_method = variant_config['aggregation_method']
        use_25d_method = variant_config['use_25d_method']
        
        slices_dict = {}
        for axis in ['axial', 'coronal', 'sagittal']:
            if use_25d_method:
                slices_tensor = self.method_25d.create_25d_slices_adaptive(skeleton_volume, axis)
            else:
                slices_tensor = self.method_25d.create_standard_slices(skeleton_volume, axis)
            slices_dict[axis] = slices_tensor
        
        feature_maps_dict = {}
        for axis, slices_tensor in slices_dict.items():
            feature_maps = self.feature_extractor.extract_feature_maps(self.dinov2_model, slices_tensor)
            feature_maps_dict[axis] = feature_maps
        
        aggregator = SpatialAggregator(aggregation_method=aggregation_method)
        unified_features = aggregator.aggregate_triaxial(feature_maps_dict)
        
        return unified_features
    
    def extract_variant_features(self, variant_config: dict) -> None:
        """
        Extract raw features for one variant across all data splits.
        
        Args:
            variant_config (dict): Configuration for the variant to process
        """
        variant_name = variant_config['name']
        print(f"  Extracting {variant_name}...")
        
        output_dir = Path(self.config['output']['base_path']) / variant_name / "raw_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_splits = [f"train_val_split_{i}.csv" for i in range(5)] + ["test_split.csv"]
        
        for split_name in all_splits:
            skeleton_data, labels, subjects = self.data_loader.load_split(split_name)
            
            split_features = []
            for skeleton_volume in skeleton_data:
                features = self.extract_single_subject(skeleton_volume, variant_config)
                split_features.append(features)
            
            split_features_array = np.stack(split_features)
            
            if split_name == "test_split.csv":
                features_file = "test_split_raw_features.npy"
                metadata_file = "test_split_metadata.csv"
            else:
                split_idx = split_name.split('_')[3].split('.')[0]
                features_file = f"train_val_split_{split_idx}_raw_features.npy"
                metadata_file = f"train_val_split_{split_idx}_metadata.csv"
            
            np.save(output_dir / features_file, split_features_array.astype(np.float32))
            
            metadata_df = pd.DataFrame({'Subject': subjects, 'Label': labels})
            metadata_df.to_csv(output_dir / metadata_file, index=False)
        
        self._save_variant_metadata(variant_config, output_dir)
        print(f"    Raw features saved to {output_dir}")
    
    def _save_variant_metadata(self, variant_config: dict, output_dir: Path) -> None:
        """
        Save metadata for the extracted variant.
        
        Args:
            variant_config (dict): Variant configuration
            output_dir (Path): Output directory for metadata
        """
        metadata = {
            'variant_info': {
                'name': variant_config['name'],
                'aggregation_method': variant_config['aggregation_method'],
                'use_25d_method': variant_config['use_25d_method'],
                'model_name': self.config['model']['name']
            },
            'extraction_info': {
                'pipeline': 'fip_feature_extraction',
                'stage': 'raw_features_only',
                'pca_applied': False,
                'feature_dimension': self.config['model']['feature_dimension']
            },
            'dataset_info': {
                'name': self.config['dataset']['name'],
                'data_path': self.config['dataset']['data_path'],
                'volume_shape': [39, 45, 44],
                'overlapping_25d': True if variant_config['use_25d_method'] else False
            }
        }
        
        with open(output_dir / "extraction_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_extraction(self) -> None:
        """
        Run Pipeline 1: Extract raw feature map features for all F.I.P. variants.
        """
        configurations = self.config['configurations']
        
        print("=" * 80)
        print("PIPELINE 1: F.I.P. RAW FEATURE EXTRACTION")
        print("=" * 80)
        print(f"Variants to extract: {len(configurations)}")
        print("Raw features only - No PCA reduction")
        
        total_start_time = time.time()
        
        for variant_config in configurations:
            variant_start_time = time.time()
            
            try:
                self.extract_variant_features(variant_config)
                
                variant_time = time.time() - variant_start_time
                print(f"    Completed in {variant_time:.1f}s")
                
            except Exception as e:
                print(f"    ERROR in {variant_config['name']}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*80}")
        print("PIPELINE 1 COMPLETED")
        print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Output directory: {self.config['output']['base_path']}")
        print("Raw features ready for Pipeline 2 (PCA reduction)")
        print(f"{'='*80}")


def main():
    """
    Main entry point for F.I.P. feature extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="F.I.P. feature extraction - Pipeline 1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, 
                       default='feature_extraction.yaml',
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        pipeline = FIPFeatureExtractionPipeline(str(config_path))
        pipeline.run_extraction()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()