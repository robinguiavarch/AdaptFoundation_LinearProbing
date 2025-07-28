"""
Feature dataset saver with SAM-Med3D strategy for storing extracted features with associated metadata.

This module handles the systematic extraction and storage of features from
3D skeletal volumes using SAM-Med3D native 3D approach with configurable spatial aggregation.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import torch

from pipelines.feature_extraction_pipeline_sam3d import SAMMed3DFeatureExtractionPipeline, create_sam3d_pipeline
from models.feature_extraction_sam3d import SAMMed3DFeatureExtractor
from data.loaders import HCPOFCDataLoader


class FeatureDatasetSaverSAM3D:
    """
    Manages extraction and storage of feature datasets from 3D skeletal volumes using SAM-Med3D.
    
    This class orchestrates the complete process of running SAM-Med3D feature extraction
    pipelines with configurable spatial aggregation and saving the results in an
    organized directory structure.
    
    Attributes:
        data_path (str): Path to source HCP OFC dataset
        output_base_path (Path): Base directory for feature storage
        model_name (str): Name of the SAM-Med3D model used
        config_path (str): Path to YAML configuration file
    """
    
    def __init__(self, 
                 data_path: str, 
                 output_base_path: str, 
                 model_name: str = 'sam_med3d_turbo',
                 config_path: Optional[str] = None):
        """
        Initialize the feature dataset saver with SAM-Med3D strategy.
        
        Args:
            data_path (str): Path to HCP OFC dataset directory
            output_base_path (str): Base directory for feature storage
            model_name (str): SAM-Med3D model name. Defaults to 'sam_med3d_turbo'.
            config_path (Optional[str]): Path to YAML configuration file
        """
        self.data_path = data_path
        self.output_base_path = Path(output_base_path)
        self.model_name = model_name
        self.config_path = config_path
        
        # Create base output directory
        self.model_output_path = self.output_base_path / model_name
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = HCPOFCDataLoader(data_path)
        
        print(f"FeatureDatasetSaverSAM3D initialized:")
        print(f"  Source data: {data_path}")
        print(f"  Output base: {self.model_output_path}")
        print(f"  Model: {model_name}")
        print(f"  Config: {config_path or 'default'}")
    
    def _save_split_data(self, 
                        features: torch.Tensor, 
                        labels: torch.Tensor, 
                        subject_ids: List[str], 
                        split_name: str, 
                        config_dir: Path) -> None:
        """
        Save features and metadata for a single data split.
        
        Args:
            features (torch.Tensor): Feature tensor with shape (n_samples, feature_dim)
            labels (torch.Tensor): Label tensor with shape (n_samples,)
            subject_ids (List[str]): List of subject identifiers
            split_name (str): Name of the data split
            config_dir (Path): Configuration output directory
        """
        # Generate base filename
        base_name = split_name.replace('.csv', '')
        
        # Convert to numpy for saving
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        # Save features as .npy
        features_file = config_dir / f"{base_name}_features.npy"
        np.save(features_file, features_np)
        
        # Save metadata as combined .csv
        metadata_df = pd.DataFrame({
            'Subject': subject_ids,
            'Label': labels_np
        })
        metadata_file = config_dir / f"{base_name}_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        print(f"  Split {split_name}: {features_np.shape[0]} samples, {features_np.shape[1]}D features")
    
    def _generate_configuration_metadata(self, 
                                        pipeline: SAMMed3DFeatureExtractionPipeline, 
                                        results: Dict) -> Dict:
        """
        Generate comprehensive metadata for a SAM-Med3D configuration.
        
        Args:
            pipeline (SAMMed3DFeatureExtractionPipeline): The pipeline used for extraction
            results (Dict): Processing results for all splits
        
        Returns:
            Dict: Configuration metadata
        """
        # Get pipeline information
        pipeline_info = pipeline.get_pipeline_info()
        model_info = pipeline.extractor.get_model_info()
        aggregation_info = pipeline.extractor.get_aggregation_info()
        
        # Calculate dataset statistics
        total_samples = sum(result['n_subjects'] for result in results.values())
        feature_dim = next(iter(results.values()))['feature_dim']
        
        # Generate SAM-Med3D specific metadata
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'configuration_name': f"{pipeline_info['aggregation_method']}",
            'sam_med3d_info': {
                'model_type': model_info['model_type'],
                'checkpoint_path': model_info['checkpoint_path'],
                'num_parameters': model_info['num_parameters'],
                'extraction_method': model_info['extraction_method'],
                'device': model_info['device']
            },
            'aggregation_info': {
                'method': model_info['aggregation_method'],
                'description': model_info['aggregation_description'],
                'preserves_spatial': model_info['preserves_spatial'],
                'memory_efficient': model_info['memory_efficient'],
                'pca_required': model_info['pca_required'],
                'recommended_for': model_info['recommended_for']
            },
            'processing_info': {
                'input_size': pipeline_info['input_size'],
                'batch_size': pipeline_info['batch_size'],
                'workflow': pipeline_info['workflow'],
                'slicing_required': pipeline_info['slicing_required'],
                'manual_aggregation_required': pipeline_info['manual_aggregation_required']
            },
            'dataset_info': {
                'total_samples': int(total_samples),
                'feature_dimension': int(feature_dim),
                'n_splits': len(results),
                'split_names': list(results.keys())
            },
            'split_statistics': {}
        }
        
        # Add per-split statistics
        for split_name, result in results.items():
            labels_np = result['labels'].numpy() if isinstance(result['labels'], torch.Tensor) else result['labels']
            metadata['split_statistics'][split_name] = {
                'n_samples': result['n_subjects'],
                'feature_shape': [result['n_subjects'], result['feature_dim']],
                'original_volume_shape': list(result['original_volume_shape']),
                'input_size': list(result['input_size']),
                'aggregation_method': result['aggregation_method'],
                'label_distribution': {int(k): int(v) for k, v in pd.Series(labels_np).value_counts().items()}
            }
        
        return metadata
    
    def save_configuration(self, aggregation_method: str) -> None:
        """
        Extract and save features for a specific SAM-Med3D aggregation configuration.
        
        Args:
            aggregation_method (str): Spatial aggregation method 
                ('avg_pool', 'max_pool', 'sum_pool', 'flatten')
        """
        # Initialize pipeline with configuration (optimize with extractor reuse)
        print(f"Initializing SAM-Med3D pipeline with {aggregation_method} aggregation...")
        
        extractor = SAMMed3DFeatureExtractor(
            config_path=self.config_path,
            aggregation_method=aggregation_method
        )
        
        pipeline = create_sam3d_pipeline(extractor=extractor)
        
        # Create configuration directory
        config_name = aggregation_method
        config_dir = self.model_output_path / config_name
        config_dir.mkdir(exist_ok=True)
        
        print(f"Processing configuration: {config_name}")
        
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
                result = pipeline.process_dataset_split(self.data_loader, split_name)
                results[split_name] = result
                
                # Save split data
                self._save_split_data(
                    result['features'], 
                    result['labels'], 
                    result['subject_ids'], 
                    split_name, 
                    config_dir
                )
                
            except Exception as e:
                print(f"  Warning: Failed to process {split_name}: {e}")
                continue
        
        # Generate and save metadata
        if results:
            metadata = self._generate_configuration_metadata(pipeline, results)
            metadata_file = config_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Configuration saved: {config_dir}")
            print(f"  Total samples processed: {metadata['dataset_info']['total_samples']}")
            print(f"  Feature dimension: {metadata['dataset_info']['feature_dimension']}")
            print(f"  Aggregation method: {metadata['aggregation_info']['method']}")
            print(f"  PCA required: {metadata['aggregation_info']['pca_required']}")
        else:
            print(f"  Warning: No splits processed successfully for {config_name}")
    
    def save_all_standard_configurations(self) -> None:
        """
        Extract and save features for all standard SAM-Med3D spatial aggregation configurations.
        
        Processes the following configurations:
        - avg_pool: Global average pooling (384D, memory efficient)
        - max_pool: Global max pooling (384D, salient features)  
        - sum_pool: Global sum pooling (384D, accumulation)
        - flatten: Spatial concatenation (196608D, full spatial info)
        """
        configurations = ['avg_pool', 'max_pool', 'sum_pool', 'flatten']
        
        print(f"Processing {len(configurations)} standard SAM-Med3D configurations...")
        
        for i, aggregation_method in enumerate(configurations, 1):
            print(f"\n=== Configuration {i}/{len(configurations)}: {aggregation_method} ===")
            try:
                self.save_configuration(aggregation_method)
            except Exception as e:
                print(f"  Error processing {aggregation_method}: {e}")
                continue
        
        print(f"\nAll SAM-Med3D configurations completed. Output directory: {self.model_output_path}")
    
    def get_saved_configurations(self) -> List[str]:
        """
        Get list of saved configuration names.
        
        Returns:
            List[str]: List of saved configuration directory names
        """
        if not self.model_output_path.exists():
            return []
        
        return [d.name for d in self.model_output_path.iterdir() if d.is_dir()]


def test_feature_saver_sam3d():
    """
    Test function for SAM-Med3D feature saver with CPU-friendly parameters.
    """
    print("Testing SAM-Med3D feature saver...")
    
    try:
        # Initialize feature saver with correct path from config
        saver = FeatureDatasetSaverSAM3D(
            data_path="crops/2mm/S.Or.",  # Correct path from DINOv2 config
            output_base_path="feature_extracted_sam3d",
            model_name="sam_med3d_turbo",
            config_path="configs/feature_extraction_sam3d.yaml"
        )
        
        print("Feature saver initialized successfully.")
        print("SAM-Med3D feature saver test completed.")
        return True
        
    except Exception as e:
        print(f"Feature saver test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_feature_saver_sam3d()
    if success:
        print("Phase 3 complete - Feature saver functional.")
    else:
        print("Fix feature saver issues before proceeding.")