"""
Feature aggregation module for AdaptFoundation project.

This module implements feature aggregation strategies for combining 
multi-axis slice features into unified representations.
"""

"""
Feature aggregation module for AdaptFoundation project - FIXED VERSION

This module implements feature aggregation strategies for combining 
multi-axis slice features into unified representations.
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional


class FeatureAggregator:
    """
    Aggregates features extracted from multi-axis slices into unified representations.
    
    Supports multiple pooling strategies (average, max, add) for combining features
    from different anatomical axes before concatenation.
    
    Attributes:
        pooling_strategy (str): Pooling method to use ('average', 'max', 'add')
        required_axes (List[str]): Required anatomical axes for aggregation
    """
    
    def __init__(self, pooling_strategy: str = 'average', 
                 required_axes: Optional[List[str]] = None):
        """
        Initialize the FeatureAggregator with specified pooling strategy and axes.
        
        Args:
            pooling_strategy (str): Pooling method to use. Options: 'average', 'max', 'add'.
                                  Defaults to 'average'.
            required_axes (List[str], optional): Required axes for aggregation. 
                                               Defaults to all axes if None.
        """
        valid_strategies = ['average', 'max', 'add']
        if pooling_strategy not in valid_strategies:
            raise ValueError(f"pooling_strategy must be one of {valid_strategies}")
        
        self.pooling_strategy = pooling_strategy
        # Set required axes - use all if None provided
        if required_axes is None:
            self.required_axes = ['axial', 'coronal', 'sagittal']
        else:
            self.required_axes = required_axes
    
    def _pool_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply pooling strategy to features array.
        
        Args:
            features (np.ndarray): Features array of shape (n_slices, feature_dim)
        
        Returns:
            np.ndarray: Pooled features of shape (feature_dim,)
        """
        if self.pooling_strategy == 'average':
            return np.mean(features, axis=0)
        elif self.pooling_strategy == 'max':
            return np.max(features, axis=0)
        elif self.pooling_strategy == 'add':
            return np.sum(features, axis=0)
    
    def aggregate_multi_axes(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate features from multiple anatomical axes into unified representation.
        
        Args:
            features_dict (Dict[str, np.ndarray]): Dictionary mapping axis names to feature arrays.
                                                 Each array has shape (n_slices, feature_dim)
        
        Returns:
            np.ndarray: Unified representation vector of shape (n_required_axes * feature_dim,)
        """
        # Validate input
        if not isinstance(features_dict, dict):
            raise TypeError("features_dict must be a dictionary")
        
        missing_axes = set(self.required_axes) - set(features_dict.keys())
        if missing_axes:
            raise ValueError(f"Missing required axes: {missing_axes}")
        
        # Pool features for each required axis (in order)
        pooled_features = []
        for axis in self.required_axes:
            axis_features = features_dict[axis]
            
            if not isinstance(axis_features, np.ndarray):
                raise TypeError(f"Features for axis '{axis}' must be numpy array")
            
            if axis_features.ndim != 2:
                raise ValueError(f"Features for axis '{axis}' must be 2D array (n_slices, feature_dim)")
            
            pooled_axis = self._pool_features(axis_features)
            pooled_features.append(pooled_axis)
        
        # Concatenate pooled features from all required axes
        unified_representation = np.concatenate(pooled_features, axis=0)
        
        return unified_representation
    
    def get_expected_output_dimension(self, feature_dim: int) -> int:
        """
        Calculate expected output dimension for unified representation.
        
        Args:
            feature_dim (int): Dimension of features from single axis
        
        Returns:
            int: Expected output dimension (n_required_axes * feature_dim)
        """
        return len(self.required_axes) * feature_dim