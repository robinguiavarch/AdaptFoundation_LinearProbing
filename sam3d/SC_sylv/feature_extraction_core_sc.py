"""
SAM-Med3D standard feature extraction for S.C.-sylv dataset.

This module provides feature extraction using SAM-Med3D turbo model
with standard flatten aggregation for regression task.
"""

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path (adaptfoundation_linearprobing)
project_root = Path(__file__).parent.parent.parent  # sam3d/SC_sylv/ -> adaptfoundation_linearprobing/
sys.path.append(str(project_root))

# Add SAM-Med3D to path for segment_anything imports
SAM_MED3D_PATH = '/home/ids/guiavarch-24/SAM-Med3D'
if SAM_MED3D_PATH not in sys.path:
    sys.path.insert(0, SAM_MED3D_PATH)

from segment_anything.build_sam3D import sam_model_registry3D


def load_sam_med3d_model(model_type='vit_b', checkpoint_path=None):
    """
    Load SAM-Med3D model with optional checkpoint.
    
    Args:
        model_type (str): Model type
        checkpoint_path (str): Path to checkpoint file
    
    Returns:
        SAM-Med3D model
    """
    model = sam_model_registry3D[model_type]()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


class SAMMed3DFeatureExtractor:
    """
    SAM-Med3D feature extractor with YAML configuration support.
    
    Attributes:
        config (dict): Configuration dictionary from YAML
        model_type (str): SAM-Med3D model variant
        checkpoint_path (Optional[Path]): Path to checkpoint file
        device (torch.device): Computation device
        aggregation_method (str): Method for spatial aggregation
        model: Full SAM-Med3D model
        image_encoder: Extracted 3D image encoder
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 aggregation_method: Optional[str] = None,
                 **override_params):
        """
        Initialize SAM-Med3D feature extractor with YAML configuration.
        
        Args:
            config_path (Optional[str]): Path to YAML configuration file
            aggregation_method (Optional[str]): Override aggregation method from config
            **override_params: Additional parameters to override config values
        """
        self.config = self._load_config(config_path)
        
        if aggregation_method:
            self.aggregation_method = aggregation_method
        else:
            self.aggregation_method = self.config.get('default_config', 'flatten')
        
        self._apply_overrides(override_params)
        
        self.model_type = self.config['model']['type']
        self.checkpoint_path = Path(self.config['model']['checkpoint_path']) if self.config['model']['checkpoint_path'] else None
        
        device_config = self.config['processing']['device']
        if device_config == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        valid_methods = list(self.config['aggregation_configs'].keys())
        if self.aggregation_method not in valid_methods:
            raise ValueError(f"aggregation_method must be one of {valid_methods}")
        
        self.aggregation_config = self.config['aggregation_configs'][self.aggregation_method]
        
        self.model = None
        self.image_encoder = None
        self.optimal_input_size = tuple(self.config['processing']['input_size'])
        self.feature_dim = self.aggregation_config['output_dim']
        
        self._load_sam_med3d_model()
        
        print(f"SAM-Med3D Feature Extractor initialized")
        print(f"Model type: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Aggregation method: {self.aggregation_method}")
        print(f"Feature dimension: {self.feature_dim}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (Optional[str]): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "feature_extraction_sam3d.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _apply_overrides(self, override_params: Dict[str, Any]):
        """
        Apply parameter overrides to configuration.
        
        Args:
            override_params (Dict[str, Any]): Parameters to override
        """
        for key, value in override_params.items():
            if key in ['batch_size']:
                self.config['processing'][key] = value
            elif key in ['device']:
                self.config['processing'][key] = value
            elif key in ['model_type']:
                self.config['model']['type'] = value
            elif key in ['checkpoint_path']:
                self.config['model']['checkpoint_path'] = value
    
    def _load_sam_med3d_model(self):
        """Load the SAM-Med3D model and extract image encoder."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                self.model = load_sam_med3d_model(
                    model_type='vit_b_ori',
                    checkpoint_path=str(self.checkpoint_path)
                )
            except Exception as e:
                self.model = sam_model_registry3D[self.model_type]()
        else:
            self.model = sam_model_registry3D[self.model_type]()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.image_encoder = self.model.image_encoder
    
    def preprocess_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Preprocess volume for SAM-Med3D input.
        
        Args:
            volume (torch.Tensor): Input volume
        
        Returns:
            torch.Tensor: Preprocessed volume ready for SAM-Med3D
        """
        if volume.dim() == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.dim() == 4:
            volume = volume.unsqueeze(1)
        elif volume.dim() == 5:
            if volume.shape[1] != 1:
                volume = volume[:, :1]
        
        current_size = volume.shape[2:]
        if current_size != self.optimal_input_size:
            volume = F.interpolate(
                volume,
                size=self.optimal_input_size,
                mode='trilinear',
                align_corners=False
            )
        
        volume = volume.float().to(self.device)
        
        return self._apply_normalization(volume)
    
    def _apply_normalization(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization based on SAM-Med3D training.
        
        Args:
            volume (torch.Tensor): Input volume
        
        Returns:
            torch.Tensor: Normalized volume
        """
        if volume.max() <= 1.0 and volume.min() >= 0.0:
            return volume
        
        volume_min = volume.min()
        volume_max = volume.max()
        
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        return volume
    
    def extract_features(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Extract features using SAM-Med3D image encoder.
        
        Args:
            volume (torch.Tensor): Preprocessed volume [B, C, H, W, D]
        
        Returns:
            torch.Tensor: Extracted features
        """
        with torch.no_grad():
            features = self.image_encoder(volume)
            features = self._process_encoder_output(features)
        
        return features
    
    def _process_encoder_output(self, encoder_output) -> torch.Tensor:
        """
        Process the raw output from SAM-Med3D image encoder using configured aggregation.
        
        Args:
            encoder_output: Raw output from image encoder
        
        Returns:
            torch.Tensor: Processed features for regression
        """
        if isinstance(encoder_output, torch.Tensor):
            if encoder_output.dim() == 5:
                return self._apply_spatial_aggregation(encoder_output)
            elif encoder_output.dim() == 3:
                batch_size, num_patches, embed_dim = encoder_output.shape
                spatial_size = int(round(num_patches ** (1/3)))
                if spatial_size ** 3 == num_patches:
                    spatial_features = encoder_output.view(batch_size, spatial_size, spatial_size, spatial_size, embed_dim)
                    spatial_features = spatial_features.permute(0, 4, 1, 2, 3)
                    return self._apply_spatial_aggregation(spatial_features)
                else:
                    return encoder_output.mean(dim=1)
            elif encoder_output.dim() == 2:
                return encoder_output
            else:
                raise ValueError(f"Unexpected encoder output shape: {encoder_output.shape}")
        elif isinstance(encoder_output, (list, tuple)):
            return self._process_encoder_output(encoder_output[-1])
        else:
            raise ValueError(f"Unexpected encoder output type: {type(encoder_output)}")
    
    def _apply_spatial_aggregation(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Apply configured spatial aggregation method to feature maps.
        
        Args:
            spatial_features (torch.Tensor): Features [B, C, H, W, D]
        
        Returns:
            torch.Tensor: Aggregated features
        """
        method = self.aggregation_method
        
        if method == 'avg_pool':
            features = F.adaptive_avg_pool3d(spatial_features, (1, 1, 1))
            return features.flatten(1)
        elif method == 'max_pool':
            features = F.adaptive_max_pool3d(spatial_features, (1, 1, 1))
            return features.flatten(1)
        elif method == 'sum_pool':
            features = spatial_features.sum(dim=(2, 3, 4))
            return features
        elif method == 'flatten':
            return spatial_features.flatten(1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def extract_features_batch(self, volumes: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Extract features from multiple volumes with batch processing.
        
        Args:
            volumes (torch.Tensor): Multiple volumes [N, H, W, D] or [N, C, H, W, D]
            batch_size (Optional[int]): Batch size for processing
        
        Returns:
            torch.Tensor: Features for all volumes [N, feature_dim]
        """
        if volumes.dim() == 4:
            volumes = volumes.unsqueeze(1)
        
        n_volumes = volumes.shape[0]
        
        if batch_size is None:
            batch_size = self.config['processing']['batch_size']
        
        all_features = []
        
        for i in range(0, n_volumes, batch_size):
            end_idx = min(i + batch_size, n_volumes)
            batch_volumes = volumes[i:end_idx]
            
            preprocessed_batch = []
            for j in range(batch_volumes.shape[0]):
                preprocessed_vol = self.preprocess_volume(batch_volumes[j:j+1])
                preprocessed_batch.append(preprocessed_vol)
            
            batch_tensor = torch.cat(preprocessed_batch, dim=0)
            batch_features = self.extract_features(batch_tensor)
            all_features.append(batch_features.cpu())
        
        features = torch.cat(all_features, dim=0)
        return features
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded SAM-Med3D model and configuration.
        
        Returns:
            dict: Model information including configuration details
        """
        return {
            'model_type': self.model_type,
            'checkpoint_path': str(self.checkpoint_path) if self.checkpoint_path else None,
            'device': str(self.device),
            'aggregation_method': self.aggregation_method,
            'aggregation_description': self.aggregation_config['description'],
            'image_encoder_type': type(self.image_encoder).__name__,
            'num_parameters': sum(p.numel() for p in self.image_encoder.parameters()),
            'optimal_input_size': self.optimal_input_size,
            'feature_dim': self.feature_dim,
            'pca_required': self.aggregation_config['pca_required'],
            'preserves_spatial': self.aggregation_config['preserves_spatial'],
            'memory_efficient': self.aggregation_config['memory_efficient'],
            'recommended_for': self.aggregation_config['recommended_for'],
            'extraction_method': 'authentic_sam_med3d_yaml_config',
            'batch_size': self.config['processing']['batch_size']
        }
    
    def get_aggregation_info(self) -> dict:
        """
        Get detailed information about the current aggregation configuration.
        
        Returns:
            dict: Aggregation configuration details
        """
        return self.aggregation_config


class SAMMed3DStandardExtractor:
    """
    SAM-Med3D standard feature extractor for S.C.-sylv dataset.
    
    Provides feature extraction using SAM-Med3D turbo model with 
    standard flatten spatial aggregation for regression task.
    
    Attributes:
        base_extractor (SAMMed3DFeatureExtractor): Base SAM-Med3D extractor
        device (torch.device): Computation device
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 base_extractor: Optional[SAMMed3DFeatureExtractor] = None):
        """
        Initialize SAM-Med3D standard extractor.
        
        Args:
            config_path (str, optional): Path to YAML configuration file
            base_extractor (SAMMed3DFeatureExtractor, optional): Pre-initialized extractor
        """
        if base_extractor is not None:
            self.base_extractor = base_extractor
        else:
            self.base_extractor = SAMMed3DFeatureExtractor(
                config_path=config_path,
                aggregation_method='flatten'
            )
        
        self.device = self.base_extractor.device
        
        print(f"SAM-Med3D Standard Extractor initialized")
        print(f"Model: {self.base_extractor.model_type}")
        print(f"Device: {self.device}")
    
    def extract_features(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Extract features using standard SAM-Med3D flatten aggregation.
        
        Args:
            volume (torch.Tensor): Input 3D volume [H,W,D] or [1,H,W,D]
        
        Returns:
            torch.Tensor: Extracted features [1, 196608]
        """
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        volume = volume.to(self.device)
        
        preprocessed_volume = self.base_extractor.preprocess_volume(volume)
        features = self.base_extractor.extract_features(preprocessed_volume)
        
        return features
    
    def extract_features_batch(self, volumes: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Extract features from multiple volumes with batch processing.
        
        Args:
            volumes (torch.Tensor): Multiple volumes [N, H, W, D] or [N, C, H, W, D]
            batch_size (int, optional): Batch size for processing
        
        Returns:
            torch.Tensor: Features for all volumes [N, 196608]
        """
        return self.base_extractor.extract_features_batch(volumes, batch_size)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the SAM-Med3D model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration and statistics
        """
        base_info = self.base_extractor.get_model_info()
        
        return {
            'model_type': base_info['model_type'],
            'feature_dim': 196608,  # 384 * 8 * 8 * 8
            'patch_grid_size': (8, 8, 8),
            'aggregation_method': 'flatten',
            'device': str(self.device),
            'checkpoint_path': base_info.get('checkpoint_path', 'unknown'),
            'dataset': 'S.C.-sylv',
            'task_type': 'regression'
        }