"""
SAM-Med3D feature extraction pipeline for AdaptFoundation.

This module implements the simplified 3D native pipeline for extracting features
using SAM-Med3D without slicing or aggregation steps.
"""

import torch
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_extraction_sam3d import SAMMed3DFeatureExtractor


class SAMMed3DFeatureExtractionPipeline:
    """
    Feature extraction pipeline for SAM-Med3D integration.
    
    This pipeline processes 3D volumes directly through SAM-Med3D without
    the slicing and aggregation steps required by 2D approaches.
    
    Attributes:
        extractor (SAMMed3DFeatureExtractor): SAM-Med3D feature extractor
        config (dict): Pipeline configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SAM-Med3D feature extraction pipeline.
        
        Args:
            config (Dict[str, Any]): Pipeline configuration containing model and processing parameters
        """
        self.config = config
        
        # Initialize SAM-Med3D feature extractor
        checkpoint_path = config.get('checkpoint_path', 'ckpt/sam_med3d_turbo.pth')
        device = config.get('device', 'auto')
        
        self.extractor = SAMMed3DFeatureExtractor(
            checkpoint_path=checkpoint_path,
            device=device
        )
        
        # Pipeline parameters
        self.batch_size = config.get('batch_size', 8)
        
    def process_volume(self, volume: torch.Tensor, subject_id: str) -> Dict[str, Any]:
        """
        Process a single 3D volume through SAM-Med3D feature extraction.
        
        Args:
            volume (torch.Tensor): Input 3D volume
            subject_id (str): Subject identifier
        
        Returns:
            Dict[str, Any]: Processing results containing features and metadata
        """
        # Preprocess volume
        preprocessed_volume = self.extractor.preprocess_volume(volume)
        
        # Extract features
        with torch.no_grad():
            features = self.extractor.extract_features(preprocessed_volume)
        
        # Prepare results
        results = {
            'features': features.cpu(),
            'subject_id': subject_id,
            'original_shape': volume.shape,
            'preprocessed_shape': preprocessed_volume.shape,
            'feature_dim': features.shape[-1]
        }
        
        return results
    
    def process_batch(self, volumes: torch.Tensor, subject_ids: list) -> Dict[str, Any]:
        """
        Process a batch of 3D volumes through SAM-Med3D feature extraction.
        
        Args:
            volumes (torch.Tensor): Batch of 3D volumes [N, H, W, D]
            subject_ids (list): List of subject identifiers
        
        Returns:
            Dict[str, Any]: Batch processing results
        """
        # Extract features for the entire batch
        batch_features = self.extractor.extract_features_batch(
            volumes, 
            batch_size=self.batch_size
        )
        
        # Prepare batch results
        results = {
            'features': batch_features,
            'subject_ids': subject_ids,
            'batch_size': len(subject_ids),
            'original_shape': volumes.shape,
            'feature_dim': batch_features.shape[-1]
        }
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dict[str, Any]: Pipeline information
        """
        return {
            'pipeline_type': 'sam_med3d_native',
            'model_name': 'sam_med3d_turbo',
            'target_size': self.extractor.target_size,
            'patch_size': self.extractor.patch_size,
            'feature_dim': self.extractor.feature_dim,
            'batch_size': self.batch_size,
            'device': str(self.extractor.device),
            'slicing_required': False,
            'aggregation_required': False
        }


def create_sam3d_pipeline(config: Dict[str, Any]) -> SAMMed3DFeatureExtractionPipeline:
    """
    Factory function to create SAM-Med3D feature extraction pipeline.
    
    Args:
        config (Dict[str, Any]): Pipeline configuration
    
    Returns:
        SAMMed3DFeatureExtractionPipeline: Configured pipeline instance
    """
    return SAMMed3DFeatureExtractionPipeline(config)