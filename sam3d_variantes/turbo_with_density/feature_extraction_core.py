"""
SAM-Med3D turbo feature extraction with density-based spatial optimization.
FIXED VERSION: Corrected baseline implementation to avoid dimension errors.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.feature_extraction_sam3d import SAMMed3DFeatureExtractor


class SAMMed3DTurboDensityExtractor:
    """
    SAM-Med3D turbo feature extractor with density-based spatial optimization.
    
    Fixed implementation with consistent processing pipeline for all approaches.
    """
    
    def __init__(self, 
                 approach: str = 'baseline',
                 config_path: Optional[str] = None,
                 base_extractor: Optional[SAMMed3DFeatureExtractor] = None):
        """
        Initialize density-aware SAM-Med3D extractor.
        
        Args:
            approach (str): Optimization approach ('baseline', 'masking', 'linear_weighting')
            config_path (Optional[str]): Path to YAML configuration file
            base_extractor (Optional[SAMMed3DFeatureExtractor]): Pre-initialized extractor
        """
        # Validate approach
        valid_approaches = {'baseline', 'masking', 'linear_weighting'}
        if approach not in valid_approaches:
            raise ValueError(f"approach must be one of {valid_approaches}")
        
        self.approach = approach
        
        # Initialize or reuse base extractor
        if base_extractor is not None:
            self.base_extractor = base_extractor
        else:
            self.base_extractor = SAMMed3DFeatureExtractor(
                config_path=config_path,
                aggregation_method='flatten'
            )
        
        # Load pre-calculated patch density map
        density_path = Path("density/patch_density_map_8x8x8.npy")
        if not density_path.exists():
            raise FileNotFoundError(f"Patch density map not found: {density_path}")
        
        self.patch_density_map = np.load(density_path)
        
        # Validate density map
        if self.patch_density_map.shape != (8, 8, 8):
            raise ValueError(f"Expected patch density map shape (8,8,8), got {self.patch_density_map.shape}")
        
        if not (np.all(self.patch_density_map >= 0) and np.all(self.patch_density_map <= 1)):
            raise ValueError("Patch density map values must be in range [0,1]")
        
        # Create valid patches mask for masking approach
        self.valid_patches_mask = self.patch_density_map > 0
        self.n_valid_patches = np.sum(self.valid_patches_mask)
        
        print(f"SAM-Med3D Density Extractor initialized")
        print(f"Approach: {self.approach}")
        print(f"Valid patches: {self.n_valid_patches}/512")
        print(f"Density range: [{np.min(self.patch_density_map):.4f}, {np.max(self.patch_density_map):.4f}]")
    
    def extract_features(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Extract features using unified pipeline for all approaches.
        
        This implementation uses the same successful pattern for all approaches,
        avoiding the problematic base_extractor.extract_features() method.
        
        Args:
            volume (torch.Tensor): Input 3D volume [H,W,D] or [1,H,W,D]
        
        Returns:
            torch.Tensor: Extracted features with density optimization applied
        """
        # Ensure volume has batch dimension
        if volume.dim() == 3:  # [H, W, D]
            volume = volume.unsqueeze(0)  # [1, H, W, D]
        
        # Move volume to same device as base extractor
        volume = volume.to(self.base_extractor.device)
        
        # Use unified processing pipeline for ALL approaches
        # This avoids the problematic base_extractor.extract_features()
        
        # Step 1: Preprocess volume (resize to 128³ and add channel dimension)
        preprocessed_volume = self._preprocess_volume_unified(volume)
        
        # Step 2: Extract spatial features using encoder
        spatial_features = self._extract_spatial_features(preprocessed_volume)
        
        # Step 3: Apply approach-specific aggregation
        if self.approach == 'baseline':
            return self._apply_baseline_aggregation(spatial_features)
        elif self.approach == 'masking':
            return self._apply_masking_aggregation(spatial_features)
        elif self.approach == 'linear_weighting':
            return self._apply_linear_weighting_aggregation(spatial_features)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
    def _preprocess_volume_unified(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Unified preprocessing that works consistently for all approaches.
        
        Args:
            volume (torch.Tensor): Input volume [B, H, W, D]
            
        Returns:
            torch.Tensor: Preprocessed volume [B, 1, 128, 128, 128]
        """
        # Add channel dimension if needed
        if volume.dim() == 4:  # [B, H, W, D]
            volume = volume.unsqueeze(1)  # [B, 1, H, W, D]
        
        # Resize to 128³ using trilinear interpolation
        target_size = (128, 128, 128)
        
        if volume.shape[2:] != target_size:
            volume = torch.nn.functional.interpolate(
                volume,
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
        
        return volume
    
    def _extract_spatial_features(self, preprocessed_volume: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features using the image encoder.
        
        Args:
            preprocessed_volume (torch.Tensor): Volume [B, 1, 128, 128, 128]
            
        Returns:
            torch.Tensor: Spatial features [B, C, 8, 8, 8]
        """
        with torch.no_grad():
            # Get encoder output
            encoder_output = self.base_extractor.image_encoder(preprocessed_volume)
            
            # Convert to spatial format [B, C, H, W, D]
            spatial_features = self._convert_to_spatial_format(encoder_output)
            
        return spatial_features
    
    def _convert_to_spatial_format(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Convert encoder output to consistent spatial format.
        
        Args:
            encoder_output: Output from SAM-Med3D encoder
            
        Returns:
            torch.Tensor: Spatial features [B, C, 8, 8, 8]
        """
        if isinstance(encoder_output, torch.Tensor):
            if encoder_output.dim() == 5:  # Already [B, C, H, W, D]
                return encoder_output
            elif encoder_output.dim() == 3:  # [B, N, C] patch tokens
                batch_size, num_patches, embed_dim = encoder_output.shape
                
                # SAM-Med3D uses 8x8x8 patch grid
                if num_patches == 512:  # 8³
                    # Reshape to spatial format
                    spatial_features = encoder_output.view(batch_size, 8, 8, 8, embed_dim)
                    # Permute to [B, C, H, W, D]
                    spatial_features = spatial_features.permute(0, 4, 1, 2, 3)
                    return spatial_features
                else:
                    raise ValueError(f"Unexpected number of patches: {num_patches}")
            elif encoder_output.dim() == 2:  # [B, C] already aggregated
                # This shouldn't happen with our setup, but handle gracefully
                raise ValueError("Encoder output already aggregated, cannot extract spatial features")
            else:
                raise ValueError(f"Unexpected encoder output shape: {encoder_output.shape}")
        else:
            raise ValueError(f"Unexpected encoder output type: {type(encoder_output)}")
    
    def _apply_baseline_aggregation(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Apply baseline flatten aggregation (identical to SAM-Med3D standard).
        
        Args:
            spatial_features (torch.Tensor): Spatial features [B, C, 8, 8, 8]
        
        Returns:
            torch.Tensor: Flattened features [B, C*8*8*8]
        """
        # Simple flatten - this is what SAM-Med3D does by default
        return spatial_features.flatten(1)
    
    def _apply_masking_aggregation(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Apply masking aggregation excluding zero-density patches.
        
        Args:
            spatial_features (torch.Tensor): Spatial features [B, C, 8, 8, 8]
        
        Returns:
            torch.Tensor: Features from valid patches only [B, n_valid_patches*C]
        """
        batch_size, channels, h, w, d = spatial_features.shape
        
        # Extract features only from valid patches
        valid_features = []
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    if self.valid_patches_mask[i, j, k]:
                        patch_features = spatial_features[:, :, i, j, k]  # [B, C]
                        valid_features.append(patch_features)
        
        # Concatenate valid patch features
        if valid_features:
            concatenated_features = torch.cat(valid_features, dim=1)  # [B, n_valid_patches*C]
        else:
            # Fallback (should not happen with real data)
            concatenated_features = torch.zeros(batch_size, channels, device=spatial_features.device)
        
        return concatenated_features
    
    def _apply_linear_weighting_aggregation(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Apply linear weighting aggregation with density-based patch weighting.
        
        Args:
            spatial_features (torch.Tensor): Spatial features [B, C, 8, 8, 8]
        
        Returns:
            torch.Tensor: Density-weighted flattened features [B, C*8*8*8]
        """
        # Convert density map to torch tensor
        density_weights = torch.from_numpy(self.patch_density_map).float().to(spatial_features.device)
        
        # Expand density weights to match feature dimensions
        batch_size, channels, h, w, d = spatial_features.shape
        
        # Density weights: [8, 8, 8] -> [1, 1, 8, 8, 8] -> [B, C, 8, 8, 8]
        density_weights = density_weights.unsqueeze(0).unsqueeze(0)
        density_weights = density_weights.expand(batch_size, channels, h, w, d)
        
        # Apply weights to spatial features
        weighted_features = spatial_features * density_weights
        
        # Flatten weighted features
        return weighted_features.flatten(1)
    
    def extract_features_batch(self, volumes: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Extract features from multiple volumes with batch processing.
        
        Uses the unified pipeline for consistent results across all approaches.
        
        Args:
            volumes (torch.Tensor): Multiple volumes [N, H, W, D] or [N, C, H, W, D]
            batch_size (Optional[int]): Batch size for processing
        
        Returns:
            torch.Tensor: Features for all volumes [N, feature_dim]
        """
        if volumes.dim() == 4:
            volumes = volumes.unsqueeze(1)  # Add channel dimension
        
        n_volumes = volumes.shape[0]
        
        if batch_size is None:
            batch_size = self.base_extractor.config['processing']['batch_size']
        
        all_features = []
        
        for i in range(0, n_volumes, batch_size):
            end_idx = min(i + batch_size, n_volumes)
            batch_volumes = volumes[i:end_idx]
            
            # Process batch using unified pipeline
            batch_features = self.extract_features(batch_volumes)
            all_features.append(batch_features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def get_approach_info(self) -> Dict[str, Any]:
        """
        Get information about the current density optimization approach.
        
        Returns:
            Dict[str, Any]: Approach configuration and statistics
        """
        base_info = self.base_extractor.get_model_info()
        
        # Calculate expected feature dimensions
        if self.approach == 'masking':
            feature_dim = self.n_valid_patches * 384  # Only valid patches
        else:
            feature_dim = 8 * 8 * 8 * 384  # 196608 for baseline and linear_weighting
        
        return {
            'approach': self.approach,
            'base_model': base_info['model_type'],
            'feature_dim': feature_dim,
            'patch_grid_size': (8, 8, 8),
            'total_patches': 512,
            'valid_patches': self.n_valid_patches,
            'invalid_patches': 512 - self.n_valid_patches,
            'density_min': float(np.min(self.patch_density_map)),
            'density_max': float(np.max(self.patch_density_map)),
            'density_mean': float(np.mean(self.patch_density_map)),
            'uses_density_optimization': self.approach != 'baseline',
            'unified_pipeline': True  # New flag indicating fixed implementation
        }