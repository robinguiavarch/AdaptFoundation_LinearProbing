"""
Feature aggregation module with concatenation strategy for AdaptFoundation project.

This module implements feature aggregation by direct concatenation instead of pooling,
preserving the complete spatial information from multi-axis slice features.
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional


class FeatureAggregator:
    """
    Aggregates features extracted from multi-axis slices using direct concatenation.
    
    Uses concatenation strategy instead of pooling to preserve complete spatial
    information from different anatomical axes without compression.
    
    Attributes:
        required_axes (List[str]): Required anatomical axes for aggregation
    """
    
    def __init__(self, required_axes: Optional[List[str]] = None):
        """
        Initialize the FeatureAggregator with direct concatenation strategy.
        
        Args:
            required_axes (List[str], optional): Required axes for aggregation. 
                                               Defaults to all axes if None.
        """
        # Set required axes - use all if None provided
        if required_axes is None:
            self.required_axes = ['axial', 'coronal', 'sagittal']
        else:
            self.required_axes = required_axes
    
    def aggregate_multi_axes(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate features from multiple anatomical axes using direct concatenation.
        
        Concatenates all slice features from each required axis to preserve
        complete spatial information without compression loss.
        
        Args:
            features_dict (Dict[str, np.ndarray]): Dictionary mapping axis names to feature arrays.
                                                 Each array has shape (n_slices, feature_dim)
        
        Returns:
            np.ndarray: Concatenated representation vector of shape (total_slices * feature_dim,)
        """
        # Validate input
        if not isinstance(features_dict, dict):
            raise TypeError("features_dict must be a dictionary")
        
        missing_axes = set(self.required_axes) - set(features_dict.keys())
        if missing_axes:
            raise ValueError(f"Missing required axes: {missing_axes}")
        
        # Concatenate all slice features for each required axis (in order)
        concatenated_features = []
        for axis in self.required_axes:
            axis_features = features_dict[axis]
            
            if not isinstance(axis_features, np.ndarray):
                raise TypeError(f"Features for axis '{axis}' must be numpy array")
            
            if axis_features.ndim != 2:
                raise ValueError(f"Features for axis '{axis}' must be 2D array (n_slices, feature_dim)")
            
            # Flatten all slices for this axis: (n_slices, feature_dim) -> (n_slices * feature_dim,)
            flattened_axis = axis_features.flatten()
            concatenated_features.append(flattened_axis)
        
        # Concatenate features from all required axes
        unified_representation = np.concatenate(concatenated_features, axis=0)
        
        return unified_representation
    
    def get_expected_output_dimension(self, feature_dim: int, slice_counts: Dict[str, int] = None) -> int:
        """
        Calculate expected output dimension for concatenated representation.
        
        Args:
            feature_dim (int): Dimension of features from single slice
            slice_counts (Dict[str, int], optional): Number of slices per axis.
                                                   Defaults to standard HCP OFC counts.
        
        Returns:
            int: Expected output dimension (total_slices * feature_dim)
        """
        # Default slice counts for HCP OFC dataset
        if slice_counts is None:
            slice_counts = {
                'axial': 22,      # Z dimension
                'coronal': 38,    # Y dimension  
                'sagittal': 30    # X dimension
            }
        
        total_slices = sum(slice_counts[axis] for axis in self.required_axes)
        return total_slices * feature_dim