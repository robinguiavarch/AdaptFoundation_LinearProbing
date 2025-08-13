"""
3D to 2D adaptation module for cortical skeleton data.

This module implements slicing strategies to convert 3D cortical skeleton volumes
into 2D slices compatible with foundation models like DINOv2.
"""

import numpy as np
import torch
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional


class SkeletonSlicer:
    """
    Converts 3D cortical skeleton volumes to 2D slices for foundation model processing.
    
    This class implements multi-axis slicing strategies to extract 2D representations
    from 3D cortical skeleton data, with preprocessing for compatibility with 
    pre-trained vision models.
    
    Attributes:
        target_size (int): Target size for output slices (default: 224 for DINOv2)
        axes (List[str]): Available slicing axes
    """
    
    def __init__(self, target_size: int = 224):
        """
        Initialize the SkeletonSlicer with target output dimensions.
        
        Args:
            target_size (int): Target size for square output slices. Defaults to 224.
        """
        self.target_size = target_size
        self.axes = ['axial', 'coronal', 'sagittal']
    
    def slice_volume(self, volume_3d: np.ndarray, 
                    axes: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Extract 2D slices from 3D volume along specified anatomical axes.
        
        Args:
            volume_3d (np.ndarray): Input 3D volume with shape (H, W, D)
            axes (List[str], optional): Axes to slice along. Defaults to all axes.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping axis names to slice tensors.
                Each slice tensor has shape (n_slices, 3, target_size, target_size) in NCHW format
        """
        if axes is None:
            axes = self.axes
        
        slices_dict = {}
        
        for axis in axes:
            if axis == 'axial':
                slices = self._slice_axial(volume_3d)
            elif axis == 'coronal':
                slices = self._slice_coronal(volume_3d)
            elif axis == 'sagittal':
                slices = self._slice_sagittal(volume_3d)
            else:
                raise ValueError(f"Unknown axis: {axis}")
            
            slices_dict[axis] = slices
        
        return slices_dict
    
    def _slice_axial(self, volume_3d: np.ndarray) -> torch.Tensor:
        """
        Extract axial slices (XY planes, varying Z).
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (H, W, D)
        
        Returns:
            torch.Tensor: Processed slices with shape (D, 3, target_size, target_size)
        """
        slices = []
        for z in range(volume_3d.shape[2]):
            slice_2d = volume_3d[:, :, z]
            processed_slice = self._preprocess_slice(slice_2d)
            slices.append(processed_slice)
        
        return torch.stack(slices, dim=0)
    
    def _slice_coronal(self, volume_3d: np.ndarray) -> torch.Tensor:
        """
        Extract coronal slices (XZ planes, varying Y).
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (H, W, D)
        
        Returns:
            torch.Tensor: Processed slices with shape (W, 3, target_size, target_size)
        """
        slices = []
        for y in range(volume_3d.shape[1]):
            slice_2d = volume_3d[:, y, :]
            processed_slice = self._preprocess_slice(slice_2d)
            slices.append(processed_slice)
        
        return torch.stack(slices, dim=0)
    
    def _slice_sagittal(self, volume_3d: np.ndarray) -> torch.Tensor:
        """
        Extract sagittal slices (YZ planes, varying X).
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (H, W, D)
        
        Returns:
            torch.Tensor: Processed slices with shape (H, 3, target_size, target_size)
        """
        slices = []
        for x in range(volume_3d.shape[0]):
            slice_2d = volume_3d[x, :, :]
            processed_slice = self._preprocess_slice(slice_2d)
            slices.append(processed_slice)
        
        return torch.stack(slices, dim=0)
    
    def _preprocess_slice(self, slice_2d: np.ndarray) -> torch.Tensor:
        """
        Preprocess a 2D slice for foundation model input.
        
        Converts binary skeleton slice to 3-channel RGB tensor in NCHW format, 
        resizes to target dimensions and applies ImageNet normalization.
        
        Args:
            slice_2d (np.ndarray): Binary 2D slice with values in [0, 1]
        
        Returns:
            torch.Tensor: Preprocessed slice with shape (3, target_size, target_size)
                         normalized for DINOv2 input in NCHW format
        """
        # Resize binary slice directly (efficient for binary data)
        resized = cv2.resize(slice_2d.astype(np.float32), 
                           (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_NEAREST)
        
        # Convert to 3-channel RGB format (HW -> HWC)
        slice_rgb = np.stack([resized, resized, resized], axis=-1)
        
        # Apply ImageNet normalization (required for DINOv2 compatibility)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        slice_normalized = (slice_rgb - imagenet_mean) / imagenet_std
        
        # Convert to tensor and permute to NCHW format
        slice_tensor = torch.from_numpy(slice_normalized).float()
        slice_tensor = slice_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return slice_tensor
    
    def get_slice_info(self, volume_3d: np.ndarray) -> Dict[str, int]:
        """
        Get information about the number of slices per axis for a given volume.
        
        Args:
            volume_3d (np.ndarray): Input 3D volume with shape (H, W, D)
        
        Returns:
            Dict[str, int]: Dictionary mapping axis names to number of slices
        """
        return {
            'axial': volume_3d.shape[2],      # Z dimension
            'coronal': volume_3d.shape[1],    # Y dimension  
            'sagittal': volume_3d.shape[0]    # X dimension
        }