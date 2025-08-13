"""
Pipeline 1: Density-guided CLS token extraction from DINOv2.

This script extracts raw CLS token features using density-guided spatial optimization
approaches without PCA reduction. Features are saved for subsequent PCA processing.
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

from data.loaders import HCPOFCDataLoader
from dinov2_variantes.density_guided.density_guided_core import (
    DensityGuidedProcessor, CLSTokenExtractor, DensityGuidedAggregator
)


class DensityGuidedExtractionPipeline:
    """
    Pipeline 1: Raw feature extraction using density-guided CLS token strategies.
    
    This pipeline extracts CLS tokens for all variants without PCA reduction,
    saving raw features for subsequent processing by Pipeline 2.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        device (torch.device): Computation device
        data_loader (HCPOFCDataLoader): Data loader for HCP OFC dataset
        dinov2_model (torch.nn.Module): Loaded DINOv2 model
        density_processor (DensityGuidedProcessor): Density-guided preprocessing
        cls_extractor (CLSTokenExtractor): CLS token extraction
    """
    
    def __init__(self, config_path: str):
        """
        Initialize density-guided extraction pipeline.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data_loader = HCPOFCDataLoader(self.config['dataset']['data_path'])
        
        model_name = self.config['model']['name']
        print(f"Loading {model_name}...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dinov2_model.eval()
        self.dinov2_model.to(self.device)
        
        self.density_processor = DensityGuidedProcessor()
        self.cls_extractor = CLSTokenExtractor(device=self.device, 
                                             batch_size=self.config['model']['batch_size'])
        
        print(f"Density-guided extraction pipeline initialized:")
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
        Extract CLS token features for a single subject using density guidance.
        
        Args:
            skeleton_volume (np.ndarray): 3D skeleton volume
            variant_config (dict): Variant configuration
        
        Returns:
            np.ndarray: Extracted CLS token features for one subject
        """
        approach = variant_config['approach']
        aggregation_method = variant_config['aggregation_method']
        
        cls_tokens_dict = {}
        for axis in ['sagittal', 'coronal', 'axial']:
            slices_tensor = self.density_processor.create_slices(skeleton_volume, axis, variant_config)
            cls_tokens = self.cls_extractor.extract_cls_tokens(self.dinov2_model, slices_tensor)
            cls_tokens_dict[axis] = cls_tokens
        
        density_profiles = None
        if approach == 'linear_weighting':
            density_profiles = self.density_processor.density_profiles
        
        aggregator = DensityGuidedAggregator(
            aggregation_method=aggregation_method,
            approach=approach,
            density_profiles=density_profiles
        )
        
        unified_features = aggregator.aggregate_triaxial(cls_tokens_dict)
        
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
                'approach': variant_config['approach'],
                'aggregation_method': variant_config['aggregation_method'],
                'model_name': self.config['model']['name']
            },
            'extraction_info': {
                'pipeline': 'density_guided_extraction',
                'stage': 'raw_features_only',
                'pca_applied': False,
                'feature_dimension': self.config['model']['feature_dimension']
            },
            'density_guidance': {
                'approach': variant_config['approach'],
                'density_profiles_used': variant_config['approach'] == 'linear_weighting'
            }
        }
        
        with open(output_dir / "extraction_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_extraction(self) -> None:
        """
        Run Pipeline 1: Extract raw CLS token features for all density-guided variants.
        """
        configurations = self.config['configurations']
        
        print("=" * 80)
        print("PIPELINE 1: DENSITY-GUIDED RAW FEATURE EXTRACTION")
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
    Main entry point for density-guided CLS token extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Density-guided CLS token extraction - Pipeline 1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, 
                       default='dinov2_variantes/density_guided/feature_extraction_density.yaml',
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        pipeline = DensityGuidedExtractionPipeline(str(config_path))
        pipeline.run_extraction()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()