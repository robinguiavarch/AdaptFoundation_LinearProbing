"""
Feature extraction core module for feature maps and 2.5D variants.

This module implements feature map extraction from DINOv2 and 2.5D slice preprocessing
for cortical skeleton analysis with memory-optimized Classical PCA.
"""

import numpy as np
import torch
import cv2
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.decomposition import PCA


class Method25D:
    """
    Handles 2.5D preprocessing with adaptive slice grouping.
    
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
    
    def create_25d_slices_adaptive(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create 2.5D slices with adaptive overlapping grouping for F.I.P. dimensions.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (39, 45, 44)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices with overlapping step=2
        """
        if axis == 'sagittal':  # X axis - 39 slices (0-38)
            # Create overlapping groups: [0,1,2], [2,3,4], [4,5,6], ..., [36,37,38]
            # Max start_idx = 36 (pour avoir [36,37,38])
            slices_2d = []
            for start_idx in range(0, 37, 2):  # 0, 2, 4, ..., 36 → 19 groups
                group_slices = np.stack([
                    volume_3d[start_idx, :, :],
                    volume_3d[start_idx + 1, :, :], 
                    volume_3d[start_idx + 2, :, :]
                ], axis=0)
                slices_2d.append(group_slices)
                
        elif axis == 'coronal':  # Y axis - 45 slices (0-44)
            # Create overlapping groups: [0,1,2], [2,3,4], [4,5,6], ..., [42,43,44]
            # Max start_idx = 42 (pour avoir [42,43,44])
            slices_2d = []
            for start_idx in range(0, 43, 2):  # 0, 2, 4, ..., 42 → 22 groups
                group_slices = np.stack([
                    volume_3d[:, start_idx, :],
                    volume_3d[:, start_idx + 1, :], 
                    volume_3d[:, start_idx + 2, :]
                ], axis=0)
                slices_2d.append(group_slices)
                
        elif axis == 'axial':  # Z axis - 44 slices (0-43)
            # Create overlapping groups: [0,1,2], [2,3,4], [4,5,6], ..., [40,41,42]
            # Max start_idx = 40 (pour avoir [40,41,42])
            slices_2d = []
            for start_idx in range(0, 41, 2):  # 0, 2, 4, ..., 40 → 21 groups
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
    
    def create_standard_slices(self, volume_3d: np.ndarray, axis: str) -> torch.Tensor:
        """
        Create standard slices with single slice replication.
        
        Args:
            volume_3d (np.ndarray): Input volume with shape (39, 45, 44)
            axis (str): Slicing axis ('axial', 'coronal', 'sagittal')
        
        Returns:
            torch.Tensor: Processed slices with shape (n_slices, 3, 224, 224)
        """
        if axis == 'sagittal':
            slices_2d = [volume_3d[i, :, :] for i in range(39)]
        elif axis == 'coronal':
            slices_2d = [volume_3d[:, i, :] for i in range(45)]
        elif axis == 'axial':
            slices_2d = [volume_3d[:, :, i] for i in range(44)]
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        processed_slices = []
        for slice_2d in slices_2d:
            processed_slice = self._preprocess_standard_slice(slice_2d)
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


class FeatureMapExtractor:
    """
    Extracts feature maps from DINOv2 instead of CLS token.
    
    Attributes:
        device (torch.device): Computation device
        batch_size (int): Batch size for processing
    """
    
    def __init__(self, device: Optional[torch.device] = None, batch_size: int = 32):
        """
        Initialize feature map extractor.
        
        Args:
            device (torch.device, optional): Computation device. Auto-detected if None.
            batch_size (int): Batch size for processing. Defaults to 32.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
    
    def extract_feature_maps(self, model: torch.nn.Module, slices_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract feature maps from DINOv2 model.
        
        Args:
            model (torch.nn.Module): Loaded DINOv2 model
            slices_tensor (torch.Tensor): Input slices with shape (N, 3, 224, 224)
        
        Returns:
            np.ndarray: Feature maps with shape (N, 16, 16, feature_dim)
        """
        n_slices = slices_tensor.shape[0]
        feature_maps_list = []
        
        with torch.no_grad():
            for i in range(0, n_slices, self.batch_size):
                end_idx = min(i + self.batch_size, n_slices)
                batch = slices_tensor[i:end_idx].to(self.device)
                
                result = model.forward_features(batch)
                
                if isinstance(result, dict) and 'x_prenorm' in result:
                    all_tokens = result['x_prenorm']
                else:
                    all_tokens = result
                
                patch_tokens = all_tokens[:, 1:, :]  # Skip CLS token
                patch_maps = patch_tokens.view(batch.shape[0], 16, 16, -1)  # 16x16 patches!
                
                feature_maps_list.append(patch_maps.cpu().numpy())
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return np.concatenate(feature_maps_list, axis=0)
    
    def extract_cls_token(self, model: torch.nn.Module, slices_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract CLS token from DINOv2 model for comparison.
        
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
                
                cls_tokens = all_tokens[:, 0, :]  # CLS token only
                cls_tokens_list.append(cls_tokens.cpu().numpy())
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        return np.concatenate(cls_tokens_list, axis=0)


class SpatialAggregator:
    """
    Aggregates feature maps using concatenation or spatial pooling strategies.
    
    Attributes:
        aggregation_method (str): Aggregation strategy ('concat' or 'pooling')
    """
    
    def __init__(self, aggregation_method: str = 'concat'):
        """
        Initialize spatial aggregator.
        
        Args:
            aggregation_method (str): Aggregation method. Either 'concat' or 'pooling'.
        """
        if aggregation_method not in ['concat', 'pooling']:
            raise ValueError("aggregation_method must be 'concat' or 'pooling'")
        self.aggregation_method = aggregation_method
    
    def aggregate_triaxial(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate feature maps from three axes into unified representation.
        
        Args:
            features_dict (Dict[str, np.ndarray]): Features per axis.
                Each value has shape (n_slices, 16, 16, feature_dim)
        
        Returns:
            np.ndarray: Aggregated features for triaxial representation
        """
        required_axes = ['axial', 'coronal', 'sagittal']
        missing_axes = set(required_axes) - set(features_dict.keys())
        if missing_axes:
            raise ValueError(f"Missing required axes: {missing_axes}")
        
        if self.aggregation_method == 'concat':
            return self._concat_patches_triaxial(features_dict)
        else:
            return self._pooling_spatial_triaxial(features_dict)
    
    def _concat_patches_triaxial(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate all patches from three axes directly.
        
        Args:
            features_dict (Dict[str, np.ndarray]): Features per axis
        
        Returns:
            np.ndarray: Concatenated feature vector
        """
        concatenated_features = []
        
        for axis in ['axial', 'coronal', 'sagittal']:
            axis_features = features_dict[axis]  # (n_slices, 16, 16, feature_dim)
            flattened_axis = axis_features.flatten()
            concatenated_features.append(flattened_axis)
        
        return np.concatenate(concatenated_features, axis=0)
    
    def _pooling_spatial_triaxial(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply spatial pooling across patches and axes.
        
        Args:
            features_dict (Dict[str, np.ndarray]): Features per axis
        
        Returns:
            np.ndarray: Pooled feature vector with shape (16*16*feature_dim,)
        """
        all_patches = []
        
        for axis in ['axial', 'coronal', 'sagittal']:
            axis_features = features_dict[axis]  # (n_slices, 16, 16, feature_dim)
            all_patches.append(axis_features)
        
        # Concatenate all patches along slice dimension
        combined_patches = np.concatenate(all_patches, axis=0)  # (total_slices, 16, 16, feature_dim)
        
        # Average pool across all slices for each spatial position
        pooled_spatial = np.mean(combined_patches, axis=0)  # (16, 16, feature_dim)
        
        # Flatten to 1D vector
        return pooled_spatial.flatten()


class ClassicalPCAProcessor:
    """
    Memory-optimized PCA processor using classical PCA for DINOv2 Giant features.
    
    With DINOv2 Giant (1536D), memory requirements are manageable enough to use
    classical PCA on all training subjects at once, which is more stable and faster
    than IncrementalPCA.
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
        
        # Step 1: Collect ALL training features
        print(f"        Extracting features from all {len(training_subjects)} subjects...")
        all_features = []
        
        for i, subject in enumerate(training_subjects):
            if (i + 1) % 50 == 0:
                print(f"          Processed {i+1}/{len(training_subjects)} subjects")
            
            features = feature_extraction_func(subject, variant_config)
            all_features.append(features.astype(np.float32))  # Force float32 immediately
            
            # Cleanup individual subject
            del features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Step 2: Stack all features into training array
        print(f"        Stacking {len(training_subjects)} subjects into training array...")
        training_array = np.stack(all_features).astype(np.float32)
        print(f"        Training array shape: {training_array.shape}")
        print(f"        Memory usage: ~{training_array.nbytes / (1024**3):.1f} GB")
        
        # Cleanup features list
        del all_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 3: Determine number of components
        original_dim = training_array.shape[1]
        
        if self.reduction_mode == 'variance':
            # Estimate components for variance-based PCA
            print(f"        Estimating components for {self.variance_threshold:.0%} variance...")
            
            # Use full PCA to determine components
            temp_pca = PCA()
            temp_pca.fit(training_array)
            
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
            print(f"        Estimated {self.n_components} components needed")
            del temp_pca
        
        # Step 4: Fit final PCA with determined components
        start_time = time.time()
        
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(training_array)
        
        fit_time = time.time() - start_time
        
        # Calculate final variance explained
        final_variance = np.sum(self.pca_model.explained_variance_ratio_)
        
        print(f"        Classical PCA fitted in {fit_time:.2f}s")
        print(f"        Components: {self.n_components} (from {original_dim}D)")
        print(f"        Variance explained: {final_variance:.4f}")
        
        # Cleanup training array
        del training_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return fitting info
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
            np.ndarray: PCA-transformed features (float32 for memory efficiency)
        """
        if self.pca_model is None:
            raise ValueError("Classical PCA model not fitted. Call fit_classical_pca() first.")
        
        transformed = self.pca_model.transform(features)
        return transformed.astype(np.float32)  # Force float32 for memory efficiency


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