"""
Core module for 2.5D CLS token extraction from DINOv2.

This module implements CLS token extraction with 2.5D preprocessing variants
including overlapping and non-overlapping grouping strategies.
"""

import numpy as np
import torch
import cv2
import time
from typing import Dict, Optional
from sklearn.decomposition import PCA


class Method25DProcessor:
    """
    Handles 2.5D preprocessing with overlapping and non-overlapping variants.
    
    Attributes:
        target_size (int): Target image size for preprocessing
    """
    
    def __init__(self, target_size: int = 224):
        """
        Initialize Method25D processor.
        
        Args:
            target_size (int): Target size for output images. Defaults to 224.
        """
        self.target_size = target_size
    
    def create_slices(self, volume_3d: np.ndarray, axis: str, variant_config: dict) -> torch.Tensor:
        """
        Create slices based on variant configuration.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
            variant_config (dict): Configuration specifying slice method
        
        Returns:
            torch.Tensor: Processed slices tensor
        """
        method = variant_config.get('method', 'standard_slicing')
        
        if method == 'standard_slicing':
            return self.create_standard_slices(volume_3d, axis)
        elif method == 'adaptive_25d':
            return self.create_25d_slices_non_overlapping(volume_3d, axis)
        elif method == 'overlapping_25d':
            return self.create_25d_slices_overlapping(volume_3d, axis)
        else:
            raise ValueError(f"Unknown slicing method: {method}")
    
    def create_standard_slices(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create standard slices with single slice replication.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices with shape (n_slices, 3, 224, 224)
        """
        if axis == 'sagittal':
            slices_2d = [volume_3d[i, :, :] for i in range(30)]
        elif axis == 'coronal':
            slices_2d = [volume_3d[:, i, :] for i in range(38)]
        elif axis == 'axial':
            slices_2d = [volume_3d[:, :, i] for i in range(22)]
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        processed_slices = []
        for slice_2d in slices_2d:
            processed_slice = self._preprocess_standard_slice(slice_2d)
            processed_slices.append(processed_slice)
        
        return torch.stack(processed_slices, dim=0)
    
    def create_25d_slices_non_overlapping(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create 2.5D slices with non-overlapping grouping.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices with shape (n_groups, 3, 224, 224)
        """
        if axis == 'sagittal':
            n_groups = 30 // 3
            slices_2d = []
            for i in range(n_groups):
                start_idx = i * 3
                group_slices = np.stack([
                    volume_3d[start_idx, :, :],
                    volume_3d[start_idx + 1, :, :], 
                    volume_3d[start_idx + 2, :, :]
                ], axis=0)
                slices_2d.append(group_slices)
                
        elif axis == 'coronal':
            n_groups = (38 - 2) // 3
            slices_2d = []
            for i in range(n_groups):
                start_idx = 1 + i * 3
                group_slices = np.stack([
                    volume_3d[:, start_idx, :],
                    volume_3d[:, start_idx + 1, :], 
                    volume_3d[:, start_idx + 2, :]
                ], axis=0)
                slices_2d.append(group_slices)
                
        elif axis == 'axial':
            n_groups = (22 - 1) // 3
            slices_2d = []
            for i in range(n_groups):
                start_idx = 1 + i * 3
                group_slices = np.stack([
                    volume_3d[:, :, start_idx],
                    volume_3d[:, :, start_idx + 1], 
                    volume_3d[:, :, start_idx + 2]
                ], axis=0)
                slices_2d.append(group_slices)
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        processed_slices = []
        for slice_group in slices_2d:
            processed_slice = self._preprocess_25d_slice(slice_group)
            processed_slices.append(processed_slice)
        
        return torch.stack(processed_slices, dim=0)
    
    def create_25d_slices_overlapping(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create 2.5D slices with overlapping grouping from slice 0 to last slice.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices with shape (n_groups, 3, 224, 224)
        """
        if axis == 'sagittal':
            n_groups = 30 - 2
            slices_2d = []
            for i in range(n_groups):
                group_slices = np.stack([
                    volume_3d[i, :, :],
                    volume_3d[i + 1, :, :], 
                    volume_3d[i + 2, :, :]
                ], axis=0)
                slices_2d.append(group_slices)
                
        elif axis == 'coronal':
            n_groups = 38 - 2
            slices_2d = []
            for i in range(n_groups):
                group_slices = np.stack([
                    volume_3d[:, i, :],
                    volume_3d[:, i + 1, :], 
                    volume_3d[:, i + 2, :]
                ], axis=0)
                slices_2d.append(group_slices)
                
        elif axis == 'axial':
            n_groups = 22 - 2
            slices_2d = []
            for i in range(n_groups):
                group_slices = np.stack([
                    volume_3d[:, :, i],
                    volume_3d[:, :, i + 1], 
                    volume_3d[:, :, i + 2]
                ], axis=0)
                slices_2d.append(group_slices)
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        processed_slices = []
        for slice_group in slices_2d:
            processed_slice = self._preprocess_25d_slice(slice_group)
            processed_slices.append(processed_slice)
        
        return torch.stack(processed_slices, dim=0)
    
    def _preprocess_25d_slice(self, slice_group: np.ndarray) -> torch.Tensor:
        """
        Preprocess 2.5D slice group to RGB tensor.
        
        Args:
            slice_group (np.ndarray): Group of 3 consecutive slices with shape (3, H, W)
        
        Returns:
            torch.Tensor: Preprocessed tensor with shape (3, 224, 224)
        """
        rgb_channels = []
        for i in range(3):
            resized = cv2.resize(slice_group[i].astype(np.float32), 
                               (self.target_size, self.target_size), 
                               interpolation=cv2.INTER_NEAREST)
            rgb_channels.append(resized)
        
        slice_rgb = np.stack(rgb_channels, axis=-1)
        slice_normalized = self._apply_imagenet_normalization(slice_rgb)
        slice_tensor = torch.from_numpy(slice_normalized).float().permute(2, 0, 1)
        
        return slice_tensor
    
    def _preprocess_standard_slice(self, slice_2d: np.ndarray) -> torch.Tensor:
        """
        Preprocess standard slice to RGB tensor.
        
        Args:
            slice_2d (np.ndarray): Single 2D slice
        
        Returns:
            torch.Tensor: Preprocessed tensor with shape (3, 224, 224)
        """
        resized = cv2.resize(slice_2d.astype(np.float32), 
                           (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_NEAREST)
        
        slice_rgb = np.stack([resized, resized, resized], axis=-1)
        slice_normalized = self._apply_imagenet_normalization(slice_rgb)
        slice_tensor = torch.from_numpy(slice_normalized).float().permute(2, 0, 1)
        
        return slice_tensor
    
    def _apply_imagenet_normalization(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Apply ImageNet normalization to RGB image.
        
        Args:
            image_rgb (np.ndarray): RGB image with shape (H, W, 3)
        
        Returns:
            np.ndarray: Normalized image
        """
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        return (image_rgb - imagenet_mean) / imagenet_std


class CLSTokenExtractor:
    """
    Extracts CLS tokens from DINOv2 model.
    
    Attributes:
        device (torch.device): Computation device
        batch_size (int): Batch size for processing
    """
    
    def __init__(self, device: Optional[torch.device] = None, batch_size: int = 32):
        """
        Initialize CLS token extractor.
        
        Args:
            device (torch.device, optional): Computation device. Auto-detected if None.
            batch_size (int): Batch size for processing. Defaults to 32.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
    
    def extract_cls_tokens(self, model: torch.nn.Module, slices_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract CLS tokens from DINOv2 model.
        
        Args:
            model (torch.nn.Module): Loaded DINOv2 model
            slices_tensor (torch.Tensor): Input slices with shape (N, 3, 224, 224)
        
        Returns:
            np.ndarray: CLS tokens with shape (N, feature_dim)
        """
        n_slices = slices_tensor.shape[0]
        cls_tokens_list = []
        
        with torch.no_grad():
            for i in range(0, n_slices, self.batch_size):
                end_idx = min(i + self.batch_size, n_slices)
                batch = slices_tensor[i:end_idx].to(self.device)
                
                result = model.forward_features(batch)
                
                if isinstance(result, dict) and 'x_prenorm' in result:
                    all_tokens = result['x_prenorm']
                else:
                    all_tokens = result
                
                cls_tokens = all_tokens[:, 0, :]
                cls_tokens_list.append(cls_tokens.cpu().numpy())
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return np.concatenate(cls_tokens_list, axis=0)


class CLSAggregator:
    """
    Aggregates CLS tokens using concatenation or pooling strategies.
    
    Attributes:
        aggregation_method (str): Aggregation strategy ('concat' or 'pooling')
    """
    
    def __init__(self, aggregation_method: str = 'concat'):
        """
        Initialize CLS aggregator.
        
        Args:
            aggregation_method (str): Aggregation method. Either 'concat' or 'pooling'.
        """
        if aggregation_method not in ['concat', 'pooling']:
            raise ValueError("aggregation_method must be 'concat' or 'pooling'")
        self.aggregation_method = aggregation_method
    
    def aggregate_triaxial(self, cls_tokens_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate CLS tokens from three axes into unified representation.
        
        Args:
            cls_tokens_dict (Dict[str, np.ndarray]): CLS tokens per axis.
                Each value has shape (n_slices, feature_dim)
        
        Returns:
            np.ndarray: Aggregated CLS tokens
        """
        required_axes = ['axial', 'coronal', 'sagittal']
        missing_axes = set(required_axes) - set(cls_tokens_dict.keys())
        if missing_axes:
            raise ValueError(f"Missing required axes: {missing_axes}")
        
        if self.aggregation_method == 'concat':
            return self._concat_cls_triaxial(cls_tokens_dict)
        else:
            return self._pooling_cls_triaxial(cls_tokens_dict)
    
    def _concat_cls_triaxial(self, cls_tokens_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate all CLS tokens from three axes directly.
        
        Args:
            cls_tokens_dict (Dict[str, np.ndarray]): CLS tokens per axis
        
        Returns:
            np.ndarray: Concatenated CLS tokens
        """
        concatenated_tokens = []
        
        for axis in ['sagittal', 'coronal', 'axial']:
            axis_tokens = cls_tokens_dict[axis]
            concatenated_tokens.append(axis_tokens.flatten())
        
        return np.concatenate(concatenated_tokens, axis=0)
    
    def _pooling_cls_triaxial(self, cls_tokens_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply average pooling per axis then concatenate.
        
        Args:
            cls_tokens_dict (Dict[str, np.ndarray]): CLS tokens per axis
        
        Returns:
            np.ndarray: Pooled and concatenated CLS tokens with shape (3 * feature_dim,)
        """
        pooled_axes = []
        
        for axis in ['sagittal', 'coronal', 'axial']:
            axis_tokens = cls_tokens_dict[axis]
            avg_token = np.mean(axis_tokens, axis=0)
            pooled_axes.append(avg_token)
        
        return np.concatenate(pooled_axes, axis=0)


class ClassicalPCAProcessor:
    """
    Memory-optimized PCA processor using classical PCA for CLS token features.
    """
    
    def __init__(self, pca_config: dict):
        """
        Initialize Classical PCA processor.
        
        Args:
            pca_config (dict): PCA configuration from YAML
        """
        self.reduction_mode = pca_config['mode']
        self.pca_model = None
        
        if self.reduction_mode == 'fixed':
            self.n_components = pca_config['n_components']
            self.variance_threshold = None
        else:
            self.variance_threshold = pca_config['variance_threshold']
            self.n_components = None
    
    def fit_classical_pca(self, feature_extraction_func, training_subjects, variant_config) -> dict:
        """
        Fit classical PCA on all training subjects at once.
        
        Args:
            feature_extraction_func: Function to extract features for a single subject
            training_subjects: List of training subject skeleton volumes
            variant_config (dict): Variant configuration
        
        Returns:
            dict: PCA fitting information
        """
        print(f"      Fitting classical PCA on {len(training_subjects)} subjects...")
        
        all_features = []
        
        for i, subject in enumerate(training_subjects):
            if (i + 1) % 50 == 0:
                print(f"          Processed {i+1}/{len(training_subjects)} subjects")
            
            features = feature_extraction_func(subject, variant_config)
            all_features.append(features.astype(np.float32))
            
            del features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        training_array = np.stack(all_features).astype(np.float32)
        print(f"        Training array shape: {training_array.shape}")
        
        del all_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        original_dim = training_array.shape[1]
        
        if self.reduction_mode == 'variance':
            temp_pca = PCA()
            temp_pca.fit(training_array)
            
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
            print(f"        Estimated {self.n_components} components needed")
            del temp_pca
        
        start_time = time.time()
        
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(training_array)
        
        fit_time = time.time() - start_time
        
        final_variance = np.sum(self.pca_model.explained_variance_ratio_)
        
        print(f"        Classical PCA fitted in {fit_time:.2f}s")
        print(f"        Components: {self.n_components} (from {original_dim}D)")
        print(f"        Variance explained: {final_variance:.4f}")
        
        del training_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'reduction_mode': self.reduction_mode,
            'n_components': int(self.n_components),
            'variance_threshold': float(self.variance_threshold) if self.variance_threshold else None,
            'original_dim': int(original_dim),
            'actual_variance': float(final_variance),
            'classical_pca': True,
            'training_subjects': len(training_subjects),
            'fit_time': fit_time
        }
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted classical PCA model.
        
        Args:
            features (np.ndarray): Input features to transform
        
        Returns:
            np.ndarray: PCA-transformed features
        """
        if self.pca_model is None:
            raise ValueError("Classical PCA model not fitted. Call fit_classical_pca() first.")
        
        transformed = self.pca_model.transform(features)
        return transformed.astype(np.float32)