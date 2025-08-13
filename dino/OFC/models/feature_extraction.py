"""
Feature extraction module using DINOv2 foundation model.

This module implements feature extraction from 2D slices using pre-trained DINOv2
models, with efficient batch processing and memory optimization.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional
import time

# =============================================================================
# DINOv2
# =============================================================================

class DINOv2FeatureExtractor:
    """
    Feature extractor using pre-trained DINOv2 model.
    
    This class handles loading of DINOv2 models and extraction of CLS token features
    from 2D image slices with optimized batch processing and memory management.
    
    Attributes:
        model_name (str): Name of the DINOv2 model variant
        feature_dim (int): Dimension of extracted features
        device (torch.device): Device for computation
        model (torch.nn.Module): Loaded DINOv2 model
        batch_size (int): Batch size for processing
    """
    
    def __init__(self, model_name: str = 'dinov2_vits14', 
                 device: Optional[Union[str, torch.device]] = None,
                 batch_size: int = 32):
        """
        Initialize DINOv2 feature extractor.
        
        Args:
            model_name (str): DINOv2 model variant. Defaults to 'dinov2_vits14' (384D).
            device (str or torch.device, optional): Computation device. Auto-detected if None.
            batch_size (int): Batch size for processing. Defaults to 32.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.feature_dim = self._get_feature_dimension()
        
        print(f"DINOv2FeatureExtractor initialized:")
        print(f"  Model: {model_name}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load pre-trained DINOv2 model from torch hub.
        
        Returns:
            torch.nn.Module: Loaded DINOv2 model
        """
        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()
        model.to(self.device)
        
        # Disable gradients for inference
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _get_feature_dimension(self) -> int:
        """
        Determine feature dimension of the loaded model.
        
        Returns:
            int: Feature dimension
        """
        # Create dummy input to get feature dimension
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            features = self.model(dummy_input)
            feature_dim = features.shape[-1]
        
        return feature_dim
    
    def extract_features(self, slices_2d: torch.Tensor) -> np.ndarray:
        """
        Extract DINOv2 CLS token features from 2D slices.
        
        Args:
            slices_2d (torch.Tensor): Input slices with shape (N, 3, 224, 224) in NCHW format
        
        Returns:
            np.ndarray: Extracted features with shape (N, feature_dim)
        """
        n_slices = slices_2d.shape[0]
        features_list = []
        
        print(f"Extracting features from {n_slices} slices...")
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, n_slices, self.batch_size):
                end_idx = min(i + self.batch_size, n_slices)
                batch = slices_2d[i:end_idx].to(self.device)
                
                # Extract features (CLS token)
                batch_features = self.model(batch)
                
                # Move back to CPU and convert to numpy
                batch_features_np = batch_features.cpu().numpy()
                features_list.append(batch_features_np)
                
                # Clear GPU memory
                del batch, batch_features
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate all features
        features = np.concatenate(features_list, axis=0)
        
        print(f"Feature extraction completed: {features.shape}")
        return features
    
    def extract_features_by_axis(self, slices_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Extract features for slices organized by anatomical axis.
        
        Args:
            slices_dict (Dict[str, torch.Tensor]): Dictionary mapping axis names to slice tensors
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping axis names to feature arrays
        """
        features_dict = {}
        
        for axis, slices in slices_dict.items():
            print(f"\nProcessing {axis} axis ({slices.shape[0]} slices)...")
            start_time = time.time()
            
            features = self.extract_features(slices)
            features_dict[axis] = features
            
            elapsed_time = time.time() - start_time
            print(f"{axis} axis completed in {elapsed_time:.2f}s")
        
        return features_dict
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Union[str, int]]: Model information
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'model_name': self.model_name,
            'feature_dimension': self.feature_dim,
            'total_parameters': total_params,
            'device': str(self.device),
            'batch_size': self.batch_size
        }


# =============================================================================
# CLIP
# =============================================================================