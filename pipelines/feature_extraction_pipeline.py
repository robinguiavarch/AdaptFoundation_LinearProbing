"""
Feature extraction pipeline for converting 3D skeletal volumes to aggregated features.

This module orchestrates the complete pipeline from 3D cortical skeleton volumes
to aggregated feature representations using foundation models.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import existing modules
from data.loaders import HCPOFCDataLoader
from models.slicing import SkeletonSlicer
from models.feature_extraction import DINOv2FeatureExtractor
from models.aggregation import FeatureAggregator


class FeatureExtractionPipeline:
    """
    Complete pipeline for extracting aggregated features from 3D skeletal volumes.
    
    This class orchestrates the full feature extraction workflow:
    1. Load 3D volumes from HCP OFC dataset
    2. Slice volumes along anatomical axes
    3. Extract features using DINOv2
    4. Aggregate features using specified pooling strategy
    
    Attributes:
        data_loader (HCPOFCDataLoader): Data loader for HCP OFC dataset
        slicer (SkeletonSlicer): 3D to 2D slicing component
        feature_extractor (DINOv2FeatureExtractor): DINOv2 feature extraction component
        aggregator (FeatureAggregator): Feature aggregation component
        config (Dict): Pipeline configuration parameters
    """
    
    def __init__(self, data_path: str, 
                 model_name: str = 'dinov2_vits14',
                 pooling_strategy: str = 'average',
                 required_axes: Optional[List[str]] = None,
                 device: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the feature extraction pipeline.
        
        Args:
            data_path (str): Path to HCP OFC dataset directory
            model_name (str): DINOv2 model variant name. Defaults to 'dinov2_vits14'.
            pooling_strategy (str): Pooling strategy for aggregation. Defaults to 'average'.
            required_axes (List[str], optional): Axes for aggregation. Defaults to all axes.
            device (str, optional): Computation device. Auto-detected if None.
            batch_size (int): Batch size for feature extraction. Defaults to 32.
        """
        self.config = {
            'model_name': model_name,
            'pooling_strategy': pooling_strategy,
            'required_axes': required_axes if required_axes is not None else ['axial', 'coronal', 'sagittal'],
            'device': device,
            'batch_size': batch_size
        }
        
        # Initialize components
        self.data_loader = HCPOFCDataLoader(data_path)
        self.slicer = SkeletonSlicer()
        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        self.aggregator = FeatureAggregator(
            pooling_strategy=pooling_strategy,
            required_axes=self.config['required_axes']
        )
        
        print(f"FeatureExtractionPipeline initialized:")
        print(f"  Model: {model_name}")
        print(f"  Pooling: {pooling_strategy}")
        print(f"  Axes: {self.config['required_axes']}")
        print(f"  Expected output dimension: {self.aggregator.get_expected_output_dimension(self.feature_extractor.feature_dim)}")
    
    def process_single_volume(self, volume_3d: np.ndarray) -> np.ndarray:
        """
        Process a single 3D volume through the complete pipeline.
        
        Args:
            volume_3d (np.ndarray): Input 3D volume with shape (H, W, D)
        
        Returns:
            np.ndarray: Aggregated feature representation
        """
        # Step 1: Slice volume along specified axes
        slices_dict = self.slicer.slice_volume(volume_3d, axes=self.config['required_axes'])
        
        # Step 2: Extract features using DINOv2
        features_dict = self.feature_extractor.extract_features_by_axis(slices_dict)
        
        # Step 3: Aggregate features
        unified_representation = self.aggregator.aggregate_multi_axes(features_dict)
        
        return unified_representation
    
    def process_split(self, split_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process a complete data split through the pipeline.
        
        Args:
            split_name (str): Name of the split file (e.g., 'train_val_split_0.csv')
        
        Returns:
            Tuple[np.ndarray, np.ndarray, List[str]]: Features, labels, and subject IDs
        """
        print(f"Processing split: {split_name}")
        
        # Load split data
        volumes, labels, subject_ids = self.data_loader.load_split(split_name)
        n_samples = len(volumes)
        
        print(f"  Loaded {n_samples} samples")
        
        # Determine output dimension
        output_dim = self.aggregator.get_expected_output_dimension(self.feature_extractor.feature_dim)
        
        # Initialize output arrays
        features_array = np.zeros((n_samples, output_dim), dtype=np.float32)
        
        # Process each volume
        for i, volume in enumerate(volumes):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing volume {i+1}/{n_samples}")
            
            features = self.process_single_volume(volume)
            features_array[i] = features
        
        print(f"  Completed processing {n_samples} volumes")
        print(f"  Output features shape: {features_array.shape}")
        
        return features_array, labels, subject_ids
    
    def process_all_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Process all data splits through the pipeline.
        
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]: Dictionary mapping 
                split names to (features, labels, subject_ids) tuples
        """
        results = {}
        
        # Process train/validation splits
        for i in range(5):
            split_name = f"train_val_split_{i}.csv"
            results[split_name] = self.process_split(split_name)
        
        # Process test split
        results["test_split.csv"] = self.process_split("test_split.csv")
        
        return results
    
    def get_configuration_name(self) -> str:
        """
        Generate configuration name for file organization.
        
        Returns:
            str: Configuration name string
        """
        axes_str = "_".join(self.config['required_axes'])
        if len(self.config['required_axes']) == 1:
            return f"single_axis_{axes_str}"
        else:
            return f"multi_axes_{self.config['pooling_strategy']}"
    
    def get_pipeline_info(self) -> Dict:
        """
        Get comprehensive information about the pipeline configuration.
        
        Returns:
            Dict: Pipeline configuration and component information
        """
        return {
            'pipeline_config': self.config.copy(),
            'slicer_info': {
                'target_size': self.slicer.target_size,
                'available_axes': self.slicer.axes
            },
            'extractor_info': self.feature_extractor.get_model_info(),
            'aggregator_info': {
                'pooling_strategy': self.aggregator.pooling_strategy,
                'required_axes': self.aggregator.required_axes,
                'expected_output_dim': self.aggregator.get_expected_output_dimension(
                    self.feature_extractor.feature_dim
                )
            }
        }