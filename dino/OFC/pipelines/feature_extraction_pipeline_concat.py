"""
Feature extraction pipeline with concatenation strategy for AdaptFoundation project.

This pipeline orchestrates the complete feature extraction process:
1. Load 3D skeleton data
2. Slice volumes into 2D representations
3. Extract features using foundation models
4. Aggregate features using concatenation strategy
5. Prepare data for downstream classification
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from data.loaders import HCPOFCDataLoader
from models.slicing import SkeletonSlicer
from models.feature_extraction import DINOv2FeatureExtractor
from models.aggregation_concat import FeatureAggregator


class FeatureExtractionPipeline:
    """
    Complete pipeline for extracting features from 3D skeletal data using concatenation.
    
    This pipeline coordinates all steps from 3D volume loading to feature aggregation,
    using concatenation strategy to preserve complete spatial information.
    
    Attributes:
        data_path (str): Path to HCP OFC dataset directory
        model_name (str): Foundation model name for feature extraction
        pooling_strategy (str): Pooling strategy (kept for compatibility, unused in concat)
        required_axes (List[str]): Axes for slicing and aggregation
        slicer (SkeletonSlicer): Volume slicing component
        extractor (DINOv2FeatureExtractor): Feature extraction component
        aggregator (FeatureAggregator): Feature aggregation component
    """
    
    def __init__(self, data_path: str, model_name: str = 'dinov2_vits14',
                 pooling_strategy: str = 'average', required_axes: Optional[List[str]] = None):
        """
        Initialize the feature extraction pipeline.
        
        Args:
            data_path (str): Path to HCP OFC dataset directory
            model_name (str): Foundation model name. Defaults to 'dinov2_vits14'.
            pooling_strategy (str): Kept for compatibility. Defaults to 'average'.
            required_axes (List[str], optional): Axes for processing. Defaults to all axes.
        """
        self.data_path = data_path
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.required_axes = required_axes
        
        # Initialize components
        self.slicer = SkeletonSlicer()
        self.extractor = DINOv2FeatureExtractor(model_name=model_name)
        self.aggregator = FeatureAggregator(required_axes=required_axes)
        
        # Load data loader
        self.data_loader = HCPOFCDataLoader(data_path)
        
        print(f"FeatureExtractionPipeline initialized:")
        print(f"  Model: {model_name}")
        print(f"  Required axes: {self.required_axes or 'all'}")
        print(f"  Dataset: {data_path}")
    
    def process_single_subject(self, skeleton_volume: np.ndarray) -> np.ndarray:
        """
        Process a single subject's skeleton volume through the complete pipeline.
        
        Args:
            skeleton_volume (np.ndarray): 3D skeleton volume with shape (H, W, D)
        
        Returns:
            np.ndarray: Aggregated feature vector using concatenation strategy
        """
        # Step 1: Slice 3D volume into 2D slices
        slices_dict = self.slicer.slice_volume(skeleton_volume, axes=self.required_axes)
        
        # Step 2: Extract features from slices using foundation model
        features_dict = self.extractor.extract_features_by_axis(slices_dict)
        
        # Step 3: Aggregate features using concatenation strategy
        unified_features = self.aggregator.aggregate_multi_axes(features_dict)
        
        return unified_features
    
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
        skeleton_data, labels, subject_ids = self.data_loader.load_split(split_name)
        n_subjects = len(skeleton_data)
        
        print(f"  Subjects: {n_subjects}")
        print(f"  Skeleton shape per subject: {skeleton_data[0].shape}")
        
        # Process each subject
        features_list = []
        start_time = time.time()
        
        for i, skeleton_volume in enumerate(skeleton_data):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (n_subjects - (i + 1)) / rate
                print(f"    Progress: {i+1}/{n_subjects} ({rate:.1f} subjects/min, ETA: {eta:.1f}min)")
            
            # Process single subject
            unified_features = self.process_single_subject(skeleton_volume)
            features_list.append(unified_features)
        
        # Combine all features
        features_array = np.stack(features_list, axis=0)
        
        total_time = time.time() - start_time
        print(f"  Split completed in {total_time:.2f}s")
        print(f"  Final features shape: {features_array.shape}")
        
        return features_array, labels, subject_ids
    
    def process_all_splits(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Process all data splits (training/validation and test).
        
        Returns:
            Dict[str, Tuple]: Dictionary mapping split names to (features, labels, subjects)
        """
        results = {}
        
        # Process training/validation splits
        for i in range(5):
            split_name = f"train_val_split_{i}.csv"
            results[split_name] = self.process_split(split_name)
        
        # Process test split
        test_split_name = "test_split.csv"
        results[test_split_name] = self.process_split(test_split_name)
        
        return results
    
    def get_configuration_name(self) -> str:
        """
        Generate configuration name based on pipeline settings.
        
        Returns:
            str: Configuration name for saving results
        """
        if self.required_axes is None or len(self.required_axes) == 3:
            return "multi_axes_concatenation"
        elif len(self.required_axes) == 1:
            axis_name = self.required_axes[0]
            return f"single_axis_{axis_name}_concatenation"
        else:
            axes_str = "_".join(sorted(self.required_axes))
            return f"custom_axes_{axes_str}_concatenation"
    
    def get_pipeline_info(self) -> Dict:
        """
        Get comprehensive information about the pipeline configuration.
        
        Returns:
            Dict: Pipeline configuration and component information
        """
        # Get expected output dimensions
        feature_dim = self.extractor.feature_dim
        expected_dim = self.aggregator.get_expected_output_dimension(feature_dim)
        
        # Get slice information from sample volume
        sample_volume = self.data_loader.skeletons[0]
        slice_info = self.slicer.get_slice_info(sample_volume)
        
        return {
            'pipeline_config': {
                'data_path': self.data_path,
                'model_name': self.model_name,
                'pooling_strategy': self.pooling_strategy,  # For compatibility
                'required_axes': self.required_axes or ['axial', 'coronal', 'sagittal'],
                'configuration_name': self.get_configuration_name()
            },
            'extractor_info': self.extractor.get_model_info(),
            'aggregator_info': {
                'strategy': 'concatenation',
                'required_axes': self.aggregator.required_axes,
                'expected_output_dim': expected_dim
            },
            'slice_info': slice_info,
            'data_info': {
                'total_subjects': len(self.data_loader.skeletons),
                'volume_shape': sample_volume.shape
            }
        }
    