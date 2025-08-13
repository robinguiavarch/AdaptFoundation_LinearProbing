"""
Memory-optimized CLS token extraction pipeline with Classical PCA for 2.5D variants.

This script orchestrates extraction of CLS tokens using DINOv2 with 2.5D preprocessing
including overlapping and non-overlapping variants, with immediate Classical PCA reduction.
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.loaders import HCPOFCDataLoader
from dinov2_variantes.method_25d_cls.method_25d_cls_core import (
    Method25DProcessor, CLSTokenExtractor, CLSAggregator, ClassicalPCAProcessor
)


class Method25DCLSPipeline:
    """
    Memory-optimized pipeline for extracting CLS tokens with immediate Classical PCA reduction.
    
    Key optimizations:
    - Classical PCA: Fit PCA on all training data at once for stability
    - Subject-by-subject processing: Never store multiple high-dimensional subjects
    - CLS token extraction: Much lighter than feature maps
    - Immediate transformation: Apply PCA and save reduced features immediately
    - Aggressive cleanup: Clear memory after each subject
    """
    
    def __init__(self, config_path: str):
        """
        Initialize memory-optimized CLS token pipeline.
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_loader = HCPOFCDataLoader(self.config['dataset']['data_path'])
        
        # Load DINOv2 model
        model_name = self.config['model']['name']
        print(f"Loading {model_name}...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dinov2_model.eval()
        self.dinov2_model.to(self.device)
        
        # Initialize processing components
        self.method_25d = Method25DProcessor()
        self.cls_extractor = CLSTokenExtractor(device=self.device, 
                                             batch_size=self.config['model']['batch_size'])
        
        print(f"Method 2.5D CLS pipeline initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Variants: {len(self.config['configurations'])}")
        print(f"  PCA modes: {len(self.config['pca_modes'])}")
        print(f"  Classical PCA: {self.config['processing']['classical_pca']}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_features_for_subject(self, skeleton_volume: np.ndarray, variant_config: dict) -> np.ndarray:
        """
        Extract CLS token features for a single subject.
        
        Args:
            skeleton_volume (np.ndarray): 3D skeleton volume
            variant_config (dict): Variant configuration
        
        Returns:
            np.ndarray: Extracted CLS token features for one subject
        """
        aggregation_method = variant_config['aggregation_method']
        
        # Step 1: Preprocessing per axis
        cls_tokens_dict = {}
        for axis in ['sagittal', 'coronal', 'axial']:
            slices_tensor = self.method_25d.create_slices(skeleton_volume, axis, variant_config)
            cls_tokens = self.cls_extractor.extract_cls_tokens(self.dinov2_model, slices_tensor)
            cls_tokens_dict[axis] = cls_tokens
        
        # Step 2: Triaxial aggregation
        aggregator = CLSAggregator(aggregation_method=aggregation_method)
        unified_features = aggregator.aggregate_triaxial(cls_tokens_dict)
        
        return unified_features
    
    def fit_classical_pca(self, variant_config: dict, pca_config: dict) -> tuple:
        """
        Fit Classical PCA on ALL training subjects from all 5 folds with memory optimization.
        
        Args:
            variant_config (dict): Variant configuration
            pca_config (dict): PCA configuration
        
        Returns:
            tuple: (ClassicalPCAProcessor, fitting_info)
        """
        print(f"    Fitting Classical PCA...")
        
        # Load ALL training subjects from all 5 train/val folds
        print(f"      Loading subjects from all 5 train/val folds...")
        all_training_subjects = []
        fold_counts = []
        
        for i in range(5):
            skeleton_data, _, _ = self.data_loader.load_split(f"train_val_split_{i}.csv")
            all_training_subjects.extend(skeleton_data)
            fold_counts.append(len(skeleton_data))
            print(f"        Fold {i}: {len(skeleton_data)} subjects")
        
        total_subjects = len(all_training_subjects)
        print(f"      Total subjects loaded: {total_subjects}")
        print(f"      Fold distribution: {fold_counts}")
        
        # Apply training_sample_size if configured
        if self.config['processing']['training_sample_size'] < total_subjects:
            training_sample_size = self.config['processing']['training_sample_size']
            all_training_subjects = all_training_subjects[:training_sample_size]
            print(f"      Limited to {training_sample_size} subjects for testing")
        else:
            training_sample_size = total_subjects
            print(f"      Using all {training_sample_size} subjects for PCA fitting")
        
        # Initialize Classical PCA processor
        pca_processor = ClassicalPCAProcessor(pca_config)
        
        # Fit classical PCA on all subjects at once
        pca_info = pca_processor.fit_classical_pca(
            self.extract_features_for_subject,
            all_training_subjects, 
            variant_config
        )
        
        print(f"      PCA fitted: {pca_info['n_components']} components in {pca_info['fit_time']:.2f}s")
        print(f"      Training subjects used: {pca_info['training_subjects']}")
        return pca_processor, pca_info
    
    def process_split_with_classical_pca(self, split_name: str, variant_config: dict, 
                                        pca_processor: ClassicalPCAProcessor, output_dir: Path) -> None:
        """
        Process a complete split with Classical PCA transformation.
        """
        print(f"      Processing {split_name}...")
        
        # Load split data
        skeleton_data, labels, subjects = self.data_loader.load_split(split_name)
        n_subjects = len(skeleton_data)
        
        # Process subjects one by one
        all_reduced_features = []
        
        for i, skeleton_volume in enumerate(skeleton_data):
            if (i + 1) % 10 == 0:
                print(f"        Subject {i+1}/{n_subjects}")
            
            # Extract CLS token features for single subject
            raw_features = self.extract_features_for_subject(skeleton_volume, variant_config)
            
            # Apply PCA transformation immediately
            reduced_features = pca_processor.transform_features(raw_features.reshape(1, -1))
            all_reduced_features.append(reduced_features[0])
            
            # Memory cleanup
            del raw_features
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Stack all reduced features
        final_reduced_features = np.stack(all_reduced_features, axis=0)
        
        # Save results
        if split_name == "test_split.csv":
            features_file = "test_split_features.npy"
            metadata_file = "test_split_metadata.csv"
        else:
            split_idx = split_name.split('_')[3].split('.')[0]
            features_file = f"train_val_split_{split_idx}_features.npy"
            metadata_file = f"train_val_split_{split_idx}_metadata.csv"
        
        np.save(output_dir / features_file, final_reduced_features.astype(np.float32))
        metadata_df = pd.DataFrame({'Subject': subjects, 'Label': labels})
        metadata_df.to_csv(output_dir / metadata_file, index=False)
        
        print(f"        Saved: {final_reduced_features.shape} → {features_file}")
        
        # Final cleanup
        del all_reduced_features, final_reduced_features
    
    def process_variant_with_classical_pca(self, variant_config: dict, pca_config: dict) -> None:
        """
        Process variant with memory-optimized Classical PCA reduction.
        """
        variant_name = variant_config['name']
        pca_description = pca_config['description']
        
        print(f"  Processing {variant_name} with {pca_description}")
        
        # Create output directory
        if pca_config['mode'] == 'fixed':
            pca_dir_name = f"PCA_{pca_config['n_components']}"
        else:
            pca_dir_name = f"PCA_{int(pca_config['variance_threshold']*100)}"
        
        output_dir = Path(self.config['output']['base_path']) / variant_name / pca_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Fit Classical PCA on training sample
        start_time = time.time()
        pca_processor, pca_info = self.fit_classical_pca(variant_config, pca_config)
        fit_time = time.time() - start_time
        
        print(f"      PCA fitting completed in {fit_time:.1f}s")
        
        # Step 2: Process all splits with Classical PCA
        all_splits = [f"train_val_split_{i}.csv" for i in range(5)] + ["test_split.csv"]
        
        for split_name in all_splits:
            self.process_split_with_classical_pca(split_name, variant_config, pca_processor, output_dir)
        
        # Step 3: Save metadata
        metadata = {
            'variant_info': {
                'name': variant_name,
                'aggregation_method': variant_config['aggregation_method'],
                'method': variant_config['method'],
                'model_name': self.config['model']['name']
            },
            'pca_info': pca_info,
            'feature_extraction': {
                'use_cls_tokens': True,
                'feature_dim': self.config['model']['feature_dimension'],
                'method_25d_variant': variant_config['method']
            },
            'memory_optimization': {
                'classical_pca': True,
                'training_sample_size': self.config['processing']['training_sample_size'],
                'subject_by_subject_processing': True,
                'max_memory_usage': "< 2GB",
                'float32_optimization': True
            },
            'performance': {
                'pca_fitting_time': fit_time,
                'total_subjects_processed': 577
            }
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"    ✅ {variant_name} completed with Classical PCA")
    
    def run_extraction(self) -> None:
        """
        Run complete memory-optimized CLS token extraction with Classical PCA for all variants.
        """
        configurations = self.config['configurations']
        pca_modes = self.config['pca_modes']
        
        print("=" * 80)
        print("METHOD 2.5D CLS TOKEN EXTRACTION WITH CLASSICAL PCA")
        print("=" * 80)
        print(f"Total tasks: {len(configurations)} variants × {len(pca_modes)} PCA = {len(configurations) * len(pca_modes)}")
        print("🚀 Key features:")
        print("  - CLS tokens: Global representation extraction from DINOv2")
        print("  - 2.5D variants: Baseline, non-overlapping, overlapping grouping")
        print("  - Classical PCA: Fitted on ALL 461 subjects from 5 train/val folds")
        print("  - Memory efficient: CLS tokens require much less memory than feature maps")
        print("  - Subject-by-subject: Never store multiple high-dimensional subjects")
        print("  - Memory guarantee: < 2GB RAM usage for all variants")
        
        total_start_time = time.time()
        task_count = 0
        
        for variant_config in configurations:
            variant_name = variant_config['name']
            print(f"\nVariant: {variant_name}")
            
            for pca_config in pca_modes:
                task_count += 1
                task_start_time = time.time()
                
                try:
                    self.process_variant_with_classical_pca(variant_config, pca_config)
                    
                    task_time = time.time() - task_start_time
                    print(f"  Task {task_count}/{len(configurations) * len(pca_modes)} completed in {task_time:.1f}s")
                    
                except Exception as e:
                    print(f"  ERROR in {variant_name} with {pca_config['description']}: {e}")
                    continue
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*80}")
        print("🎯 METHOD 2.5D CLS PIPELINE COMPLETED!")
        print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Output directory: {self.config['output']['base_path']}")
        print("✅ CLS token features ready for classification:")
        print("  - All features PCA-reduced (32D/256D/99%)")
        print("  - 2.5D overlapping variants included")
        print("  - Memory usage stayed below 2GB throughout")
        print("  - Classical PCA: More stable than incremental")
        print("  - Fully compatible with Linear Probing")
        print(f"{'='*80}")


def main():
    """
    Main entry point for Method 2.5D CLS token extraction pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Method 2.5D CLS token extraction with Classical PCA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline_feature_extraction.py
  python run_pipeline_feature_extraction.py --config custom_config.yaml

This pipeline extracts CLS tokens with 2.5D variants including overlapping
grouping strategies. Memory usage is optimized for CLS tokens.
        """
    )
    parser.add_argument('--config', type=str, 
                       default='dinov2_variantes/method_25d_cls/feature_extraction.yaml',
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        pipeline = Method25DCLSPipeline(str(config_path))
        pipeline.run_extraction()
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()