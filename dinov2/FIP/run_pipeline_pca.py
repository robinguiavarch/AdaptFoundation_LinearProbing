"""
Pipeline 2: PCA reduction for F.I.P. feature maps features.

This script applies PCA reduction to pre-extracted raw feature maps features
from Pipeline 1, generating final reduced features for classification.
"""

import numpy as np
import pandas as pd
import yaml
import time
import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from dinov2.FIP.feature_extraction_core import StandalonePCAProcessor


class FIPPCAPipeline:
    """
    Pipeline 2: PCA reduction for pre-extracted F.I.P. feature maps features.
    
    This pipeline loads raw features from Pipeline 1 and applies PCA reduction
    to generate final features ready for classification.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        base_path (Path): Base path for feature storage
    """
    
    def __init__(self, config_path: str):
        """
        Initialize F.I.P. PCA pipeline.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.base_path = Path(self.config['output']['base_path'])
        
        print(f"F.I.P. PCA pipeline initialized:")
        print(f"  Base path: {self.base_path}")
        print(f"  Variants: {len(self.config['configurations'])}")
        print(f"  PCA modes: {len(self.config['pca_modes'])}")
    
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
    
    def apply_pca_to_variant(self, variant_name: str, pca_config: dict) -> None:
        """
        Apply PCA reduction to one variant with one PCA configuration.
        
        Args:
            variant_name (str): Name of the variant to process
            pca_config (dict): PCA configuration parameters
        """
        print(f"    Applying {pca_config['description']} to {variant_name}...")
        
        raw_features_dir = self.base_path / variant_name / "raw_features"
        
        if not raw_features_dir.exists():
            print(f"      Raw features not found: {raw_features_dir}")
            return
        
        if pca_config['mode'] == 'fixed':
            pca_dir_name = f"PCA_{pca_config['n_components']}"
        else:
            pca_dir_name = f"PCA_{int(pca_config['variance_threshold']*100)}"
        
        output_dir = self.base_path / variant_name / pca_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pca_processor = StandalonePCAProcessor(pca_config)
        
        start_time = time.time()
        result = pca_processor.fit_and_transform_variant(raw_features_dir)
        pca_time = time.time() - start_time
        
        self._save_pca_results(result, output_dir, variant_name, pca_time)
        
        print(f"      Completed in {pca_time:.1f}s - {result['pca_info']['n_components']} components")
    
    def _save_pca_results(self, result: dict, output_dir: Path, variant_name: str, pca_time: float) -> None:
        """
        Save PCA-reduced features and metadata.
        
        Args:
            result (dict): PCA results containing features and metadata
            output_dir (Path): Output directory for PCA results
            variant_name (str): Name of the processed variant
            pca_time (float): Time taken for PCA processing
        """
        transformed_splits = result['transformed_splits']
        pca_info = result['pca_info']
        
        for split_name, split_data in transformed_splits.items():
            features_file = f"{split_name}_features.npy"
            metadata_file = f"{split_name}_metadata.csv"
            
            np.save(output_dir / features_file, split_data['features'])
            
            metadata_df = pd.read_csv(split_data['metadata_file'])
            metadata_df.to_csv(output_dir / metadata_file, index=False)
        
        final_metadata = {
            'variant_info': {
                'name': variant_name,
                'model_name': self.config['model']['name']
            },
            'pca_info': pca_info,
            'processing_info': {
                'pipeline': 'fip_pca',
                'stage': 'pca_reduced_features',
                'processing_time': pca_time
            },
            'feature_extraction': {
                'original_feature_dim': self.config['model']['feature_dimension'],
                'reduced_feature_dim': pca_info['n_components'],
                'compression_ratio': pca_info['original_dim'] / pca_info['n_components']
            }
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(final_metadata, f, indent=2)
    
    def run_pca_reduction(self) -> None:
        """
        Run Pipeline 2: Apply PCA reduction to all variants and PCA modes.
        """
        configurations = self.config['configurations']
        pca_modes = self.config['pca_modes']
        
        total_tasks = len(configurations) * len(pca_modes)
        
        print("=" * 80)
        print("PIPELINE 2: F.I.P. PCA REDUCTION")
        print("=" * 80)
        print(f"Total PCA tasks: {len(configurations)} variants × {len(pca_modes)} modes = {total_tasks}")
        print("Processing pre-extracted raw features")
        
        total_start_time = time.time()
        task_count = 0
        
        for variant_config in configurations:
            variant_name = variant_config['name']
            print(f"  Processing variant: {variant_name}")
            
            for pca_config in pca_modes:
                task_count += 1
                task_start_time = time.time()
                
                try:
                    self.apply_pca_to_variant(variant_name, pca_config)
                    
                    task_time = time.time() - task_start_time
                    print(f"    Task {task_count}/{total_tasks} completed in {task_time:.1f}s")
                    
                except Exception as e:
                    print(f"    ERROR in {variant_name} with {pca_config['description']}: {e}")
                    continue
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*80}")
        print("PIPELINE 2 COMPLETED")
        print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Output directory: {self.base_path}")
        print("PCA-reduced features ready for classification")
        print(f"{'='*80}")


def main():
    """
    Main entry point for F.I.P. PCA reduction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="F.I.P. PCA reduction - Pipeline 2",
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
        pipeline = FIPPCAPipeline(str(config_path))
        pipeline.run_pca_reduction()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()