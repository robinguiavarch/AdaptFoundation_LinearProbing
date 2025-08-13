"""
Core module for density-guided CLS token extraction from DINOv2.

This module implements CLS token extraction with density-guided spatial optimization
using three approaches: central uniform selection, adaptive density selection,
and linear density weighting.
"""

import numpy as np
import torch
import cv2
import time
from typing import Dict, Optional
from pathlib import Path
from sklearn.decomposition import PCA


class DensityGuidedProcessor:
    """
    Handles density-guided preprocessing with three spatial optimization approaches.
    
    Attributes:
        target_size (int): Target image size for preprocessing
        density_profiles (dict): Loaded density profiles for each axis
    """
    
    def __init__(self, target_size: int = 224):
        """
        Initialize density-guided processor.
        
        Args:
            target_size (int): Target size for output images. Defaults to 224.
        """
        self.target_size = target_size
        self.density_profiles = self._load_density_profiles()
    
    def _load_density_profiles(self) -> dict:
        """
        Load pre-computed density profiles for spatial optimization.
        
        Returns:
            dict: Density profiles for each axis with keys 'x', 'y', 'z'
        """
        density_path = Path("density")
        
        profiles = {
            'x': np.load(density_path / "density_profile_x.npy"),  # Sagittal (30,)
            'y': np.load(density_path / "density_profile_y.npy"),  # Coronal (38,)
            'z': np.load(density_path / "density_profile_z.npy")   # Axial (22,)
        }
        
        return profiles
    
    def create_slices(self, volume_3d: np.ndarray, axis: str, variant_config: dict) -> torch.Tensor:
        """
        Create slices based on density-guided approach configuration.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
            variant_config (dict): Configuration specifying density approach
        
        Returns:
            torch.Tensor: Processed slices tensor
        """
        approach = variant_config.get('approach')
        
        if approach == 'central_uniform':
            return self._create_central_uniform_slices(volume_3d, axis)
        elif approach == 'adaptive_density':
            return self._create_adaptive_density_slices(volume_3d, axis)
        elif approach == 'linear_weighting':
            return self._create_linear_weighting_slices(volume_3d, axis)
        else:
            raise ValueError(f"Unknown density approach: {approach}")
    
    def _create_central_uniform_slices(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create slices using central uniform selection approach.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices with central selection
        """
        if axis == 'sagittal':
            indices = range(10, 21)  # 11 central slices
            slices_2d = [volume_3d[i, :, :] for i in indices]
        elif axis == 'coronal':
            indices = range(14, 25)  # 11 central slices
            slices_2d = [volume_3d[:, i, :] for i in indices]
        elif axis == 'axial':
            indices = range(6, 16)   # 10 central slices
            slices_2d = [volume_3d[:, :, i] for i in indices]
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        processed_slices = []
        for slice_2d in slices_2d:
            processed_slice = self._preprocess_slice(slice_2d)
            processed_slices.append(processed_slice)
        
        return torch.stack(processed_slices, dim=0)
    
    def _create_adaptive_density_slices(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create slices using adaptive density selection approach.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices with density-based selection
        """
        if axis == 'sagittal':
            indices = range(7, 26)   # 19 high-density slices
            slices_2d = [volume_3d[i, :, :] for i in indices]
        elif axis == 'coronal':
            indices = range(7, 34)   # 27 high-density slices
            slices_2d = [volume_3d[:, i, :] for i in indices]
        elif axis == 'axial':
            indices = range(4, 17)   # 13 high-density slices
            slices_2d = [volume_3d[:, :, i] for i in indices]
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        processed_slices = []
        for slice_2d in slices_2d:
            processed_slice = self._preprocess_slice(slice_2d)
            processed_slices.append(processed_slice)
        
        return torch.stack(processed_slices, dim=0)
    
    def _create_linear_weighting_slices(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create slices using all slices for linear weighting approach.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (30, 38, 22)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices for weighting
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
            processed_slice = self._preprocess_slice(slice_2d)
            processed_slices.append(processed_slice)
        
        return torch.stack(processed_slices, dim=0)
    
    def _preprocess_slice(self, slice_2d: np.ndarray) -> torch.Tensor:
        """
        Preprocess single slice to RGB tensor.
        
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
    
    def __init__(self, device: Optional[torch.device] = None, batch_size: int = 8):
        """
        Initialize CLS token extractor.
        
        Args:
            device (torch.device, optional): Computation device. Auto-detected if None.
            batch_size (int): Batch size for processing. Defaults to 8.
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


class DensityGuidedAggregator:
    """
    Aggregates CLS tokens using density-guided strategies.
    
    Attributes:
        aggregation_method (str): Aggregation strategy ('concat' or 'pooling')
        approach (str): Density approach for weighting
        density_profiles (dict): Density profiles for linear weighting
    """
    
    def __init__(self, aggregation_method: str, approach: str, density_profiles: Optional[dict] = None):
        """
        Initialize density-guided aggregator.
        
        Args:
            aggregation_method (str): Aggregation method ('concat' or 'pooling')
            approach (str): Density approach name
            density_profiles (dict, optional): Density profiles for weighting
        """
        if aggregation_method not in ['concat', 'pooling']:
            raise ValueError("aggregation_method must be 'concat' or 'pooling'")
        
        self.aggregation_method = aggregation_method
        self.approach = approach
        self.density_profiles = density_profiles
    
    def aggregate_triaxial(self, cls_tokens_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate CLS tokens from three axes with density guidance.
        
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
        
        if self.approach == 'linear_weighting':
            cls_tokens_dict = self._apply_linear_weighting(cls_tokens_dict)
        
        if self.aggregation_method == 'concat':
            return self._concat_cls_triaxial(cls_tokens_dict)
        else:
            return self._pooling_cls_triaxial(cls_tokens_dict)
    
    def _apply_linear_weighting(self, cls_tokens_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply linear density weighting to CLS tokens.
        
        Args:
            cls_tokens_dict (Dict[str, np.ndarray]): CLS tokens per axis
        
        Returns:
            Dict[str, np.ndarray]: Weighted CLS tokens
        """
        if self.density_profiles is None:
            raise ValueError("Density profiles required for linear weighting")
        
        weighted_tokens = {}
        
        for axis, tokens in cls_tokens_dict.items():
            if axis == 'sagittal':
                profile = self.density_profiles['x']
            elif axis == 'coronal':
                profile = self.density_profiles['y']
            elif axis == 'axial':
                profile = self.density_profiles['z']
            else:
                raise ValueError(f"Unknown axis: {axis}")
            
            weights = profile / np.max(profile)
            weighted_tokens[axis] = tokens * weights.reshape(-1, 1)
        
        return weighted_tokens
    
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
            np.ndarray: Pooled and concatenated CLS tokens
        """
        pooled_axes = []
        
        for axis in ['sagittal', 'coronal', 'axial']:
            axis_tokens = cls_tokens_dict[axis]
            avg_token = np.mean(axis_tokens, axis=0)
            pooled_axes.append(avg_token)
        
        return np.concatenate(pooled_axes, axis=0)


class StandalonePCAProcessor:
    """
    Standalone PCA processor for Pipeline 2 operations on pre-extracted features.
    
    Attributes:
        pca_config (dict): PCA configuration parameters
        pca_model (PCA): Fitted PCA model
    """
    
    def __init__(self, pca_config: dict):
        """
        Initialize standalone PCA processor.
        
        Args:
            pca_config (dict): PCA configuration from YAML
        """
        self.pca_config = pca_config
        self.pca_model = None
        
        if pca_config['mode'] == 'fixed':
            self.n_components = pca_config['n_components']
        else:
            self.variance_threshold = pca_config['variance_threshold']
            self.n_components = None
    
    def fit_and_transform_variant(self, raw_features_dir: Path) -> dict:
        """
        Fit PCA on training data and transform all splits for one variant.
        
        Args:
            raw_features_dir (Path): Directory containing raw features
        
        Returns:
            dict: PCA information and transformed features
        """
        training_features = self._load_training_features(raw_features_dir)
        
        if self.pca_config['mode'] == 'variance':
            self.n_components = self._estimate_components(training_features)
        
        start_time = time.time()
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(training_features)
        fit_time = time.time() - start_time
        
        transformed_splits = self._transform_all_splits(raw_features_dir)
        
        pca_info = {
            'mode': self.pca_config['mode'],
            'n_components': int(self.n_components),
            'original_dim': training_features.shape[1],
            'variance_explained': float(np.sum(self.pca_model.explained_variance_ratio_)),
            'fit_time': fit_time,
            'training_subjects': training_features.shape[0]
        }
        
        return {
            'pca_info': pca_info,
            'transformed_splits': transformed_splits
        }
    
    def _load_training_features(self, raw_features_dir: Path) -> np.ndarray:
        """
        Load training features from all train/val splits.
        
        Args:
            raw_features_dir (Path): Directory containing raw features
        
        Returns:
            np.ndarray: Combined training features
        """
        training_features = []
        
        for i in range(5):
            features_file = raw_features_dir / f"train_val_split_{i}_raw_features.npy"
            features = np.load(features_file)
            training_features.append(features)
        
        return np.concatenate(training_features, axis=0).astype(np.float32)
    
    def _estimate_components(self, training_features: np.ndarray) -> int:
        """
        Estimate number of components for variance threshold.
        
        Args:
            training_features (np.ndarray): Training features
        
        Returns:
            int: Estimated number of components
        """
        temp_pca = PCA()
        temp_pca.fit(training_features)
        
        cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        return n_components
    
    def _transform_all_splits(self, raw_features_dir: Path) -> dict:
        """
        Transform all data splits using fitted PCA.
        
        Args:
            raw_features_dir (Path): Directory containing raw features
        
        Returns:
            dict: Transformed features for all splits
        """
        transformed_splits = {}
        
        all_splits = [f"train_val_split_{i}" for i in range(5)] + ["test_split"]
        
        for split_name in all_splits:
            features_file = raw_features_dir / f"{split_name}_raw_features.npy"
            metadata_file = raw_features_dir / f"{split_name}_metadata.csv"
            
            raw_features = np.load(features_file)
            transformed_features = self.pca_model.transform(raw_features)
            
            transformed_splits[split_name] = {
                'features': transformed_features.astype(np.float32),
                'metadata_file': metadata_file
            }
        
        return transformed_splits