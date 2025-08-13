"""
SAM-Med3D feature extraction pipeline for AdaptFoundation.

This module implements the simplified 3D native pipeline for extracting features
using SAM-Med3D without slicing or aggregation steps.

Supports YAML configuration with 4 spatial aggregation methods.
"""

import torch
import yaml
from typing import Dict, Any, Tuple, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_extraction_sam3d import SAMMed3DFeatureExtractor
from data.loaders import HCPOFCDataLoader


class SAMMed3DFeatureExtractionPipeline:
    """
    Feature extraction pipeline for SAM-Med3D integration.
    
    This pipeline processes 3D volumes directly through SAM-Med3D without
    the slicing and aggregation steps required by 2D approaches.
    
    Attributes:
        extractor (SAMMed3DFeatureExtractor): SAM-Med3D feature extractor
        config (dict): Pipeline configuration from YAML
        aggregation_method (str): Current aggregation method
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 aggregation_method: Optional[str] = None,
                 extractor: Optional[SAMMed3DFeatureExtractor] = None):
        """
        Initialize the SAM-Med3D feature extraction pipeline.
        
        Args:
            config_path (Optional[str]): Path to YAML configuration file
            aggregation_method (Optional[str]): Override aggregation method from config
            extractor (Optional[SAMMed3DFeatureExtractor]): Pre-initialized extractor (avoids reloading)
        """
        # Load configuration
        self.config_path = config_path
        self.aggregation_method = aggregation_method
        
        # Use pre-initialized extractor or create new one
        if extractor is not None:
            print("Using pre-initialized SAM-Med3D extractor (faster)")
            self.extractor = extractor
            # Verify aggregation method compatibility
            if aggregation_method and aggregation_method != self.extractor.aggregation_method:
                print(f"Warning: Requested {aggregation_method}, but extractor uses {self.extractor.aggregation_method}")
        else:
            print("Initializing new SAM-Med3D extractor...")
            # Initialize SAM-Med3D feature extractor with YAML configuration
            self.extractor = SAMMed3DFeatureExtractor(
                config_path=config_path,
                aggregation_method=aggregation_method
            )
        
        # Get configuration from extractor
        self.config = self.extractor.config
        self.aggregation_config = self.extractor.aggregation_config
        
        # Pipeline parameters from configuration
        self.batch_size = self.config['processing']['batch_size']
        self.input_size = tuple(self.config['processing']['input_size'])
        
        print(f"SAM-Med3D Pipeline initialized")
        print(f"Aggregation method: {self.extractor.aggregation_method}")
        print(f"Batch size: {self.batch_size}")
        print(f"Input size: {self.input_size}")
        
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
            'feature_dim': features.shape[-1],
            'aggregation_method': self.extractor.aggregation_method
        }
        
        return results
    
    def process_batch(self, volumes: torch.Tensor, subject_ids: List[str]) -> Dict[str, Any]:
        """
        Process a batch of 3D volumes through SAM-Med3D feature extraction.
        
        Args:
            volumes (torch.Tensor): Batch of 3D volumes [N, H, W, D]
            subject_ids (List[str]): List of subject identifiers
        
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
            'feature_dim': batch_features.shape[-1],
            'aggregation_method': self.extractor.aggregation_method
        }
        
        return results
    
    def process_dataset_split(self, 
                            data_loader: HCPOFCDataLoader, 
                            split_name: str) -> Dict[str, Any]:
        """
        Process an entire dataset split through SAM-Med3D feature extraction.
        
        Args:
            data_loader (HCPOFCDataLoader): Data loader instance
            split_name (str): Name of the split to process
        
        Returns:
            Dict[str, Any]: Complete split processing results
        """
        print(f"Processing split: {split_name}")
        
        # Load split data
        volumes, labels, subject_ids = data_loader.load_split_as_tensor(split_name)
        print(f"Loaded {len(subject_ids)} subjects from {split_name}")
        print(f"Volume shape: {volumes.shape}")
        
        # Process in batches
        all_features = []
        all_subject_ids = []
        all_labels = []
        
        n_volumes = volumes.shape[0]
        
        for i in range(0, n_volumes, self.batch_size):
            end_idx = min(i + self.batch_size, n_volumes)
            batch_volumes = volumes[i:end_idx]
            batch_subject_ids = subject_ids[i:end_idx]
            batch_labels = labels[i:end_idx]
            
            # Process batch
            batch_results = self.process_batch(batch_volumes, batch_subject_ids)
            
            # Collect results
            all_features.append(batch_results['features'])
            all_subject_ids.extend(batch_results['subject_ids'])
            all_labels.extend(batch_labels)
            
            print(f"Processed batch {i//self.batch_size + 1}/{(n_volumes-1)//self.batch_size + 1}")
        
        # Concatenate all features
        final_features = torch.cat(all_features, dim=0)
        
        # Prepare final results
        results = {
            'features': final_features,
            'labels': torch.tensor(all_labels),
            'subject_ids': all_subject_ids,
            'split_name': split_name,
            'n_subjects': len(all_subject_ids),
            'feature_dim': final_features.shape[-1],
            'aggregation_method': self.extractor.aggregation_method,
            'aggregation_config': self.aggregation_config,
            'original_volume_shape': volumes.shape[1:],
            'input_size': self.input_size
        }
        
        print(f"Split {split_name} processed: {final_features.shape}")
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dict[str, Any]: Pipeline information
        """
        model_info = self.extractor.get_model_info()
        
        return {
            'pipeline_type': 'sam_med3d_native',
            'model_name': self.config['model']['name'],
            'model_type': self.config['model']['type'],
            'checkpoint_path': self.config['model']['checkpoint_path'],
            'input_size': self.input_size,
            'feature_dim': model_info['feature_dim'],
            'batch_size': self.batch_size,
            'device': str(self.extractor.device),
            'aggregation_method': self.extractor.aggregation_method,
            'aggregation_description': self.aggregation_config['description'],
            'pca_required': self.aggregation_config['pca_required'],
            'preserves_spatial': self.aggregation_config['preserves_spatial'],
            'memory_efficient': self.aggregation_config['memory_efficient'],
            'slicing_required': False,
            'manual_aggregation_required': False,
            'workflow': 'Volume 3D → SAM-Med3D → Features (Direct)',
            'config_path': self.config_path
        }


def create_sam3d_pipeline(config_path: Optional[str] = None,
                         aggregation_method: Optional[str] = None,
                         extractor: Optional[SAMMed3DFeatureExtractor] = None) -> SAMMed3DFeatureExtractionPipeline:
    """
    Factory function to create SAM-Med3D feature extraction pipeline.
    
    Args:
        config_path (Optional[str]): Path to YAML configuration file
        aggregation_method (Optional[str]): Override aggregation method from config
        extractor (Optional[SAMMed3DFeatureExtractor]): Pre-initialized extractor for efficiency
    
    Returns:
        SAMMed3DFeatureExtractionPipeline: Configured pipeline instance
    """
    return SAMMed3DFeatureExtractionPipeline(
        config_path=config_path,
        aggregation_method=aggregation_method,
        extractor=extractor
    )


def test_sam3d_pipeline():
    """
    Test function for SAM-Med3D pipeline with CPU-friendly parameters.
    
    Tests pipeline functionality without requiring extensive computation time.
    Tests both fresh initialization and pre-initialized extractor reuse.
    """
    print("=" * 60)
    print("TESTING SAM-MED3D PIPELINE (CPU-FRIENDLY)")
    print("=" * 60)
    
    try:
        # Test 1: Fresh initialization (slower)
        print("\n1. Testing fresh pipeline initialization...")
        
        pipeline1 = create_sam3d_pipeline(
            config_path="configs/feature_extraction_sam3d.yaml",
            aggregation_method="avg_pool"
        )
        
        # Get pipeline info
        print("\n2. Pipeline Information:")
        info = pipeline1.get_pipeline_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test single volume processing
        print("\n3. Testing single volume processing...")
        test_volume = torch.randint(0, 2, (64, 64, 64), dtype=torch.float32)  # Smaller for CPU
        test_subject_id = "test_subject_001"
        
        result = pipeline1.process_volume(test_volume, test_subject_id)
        
        print(f"  Original shape: {result['original_shape']}")
        print(f"  Preprocessed shape: {result['preprocessed_shape']}")
        print(f"  Features shape: {result['features'].shape}")
        print(f"  Feature dimension: {result['feature_dim']}")
        print(f"  Aggregation method: {result['aggregation_method']}")
        
        # Test 2: Reuse extractor (faster - no model reloading)
        print("\n4. Testing extractor reuse (OPTIMIZATION)...")
        
        pipeline2 = create_sam3d_pipeline(
            extractor=pipeline1.extractor  # Reuse the already loaded model!
        )
        
        result2 = pipeline2.process_volume(test_volume, "test_subject_002")
        print(f"  Reused extractor features shape: {result2['features'].shape}")
        print("  ✅ No model reloading - much faster!")
        
        # Test small batch processing
        print("\n5. Testing small batch processing...")
        test_batch = torch.randint(0, 2, (3, 64, 64, 64), dtype=torch.float32)  # Small batch
        test_subject_ids = ["test_001", "test_002", "test_003"]
        
        batch_result = pipeline2.process_batch(test_batch, test_subject_ids)
        
        print(f"  Batch features shape: {batch_result['features'].shape}")
        print(f"  Batch size: {batch_result['batch_size']}")
        print(f"  Subject IDs: {batch_result['subject_ids']}")
        
        # Test different aggregation method with same extractor
        print("\n6. Testing different aggregation (requires new extractor)...")
        extractor_max = SAMMed3DFeatureExtractor(
            config_path="configs/feature_extraction_sam3d.yaml",
            aggregation_method="max_pool"
        )
        
        pipeline_max = create_sam3d_pipeline(extractor=extractor_max)
        result_max = pipeline_max.process_volume(test_volume, test_subject_id)
        print(f"  Max pool features shape: {result_max['features'].shape}")
        print(f"  Max pool feature dim: {result_max['feature_dim']}")
        
        print("\n" + "=" * 60)
        print("✅ SAM-MED3D PIPELINE CPU TESTING COMPLETED!")
        print("✅ Pipeline ready for Phase 3 (Feature Saver integration)")
        print("✅ Extractor reuse optimization validated!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sam3d_pipeline()
    if success:
        print("\n🚀 Phase 2 COMPLETE - Pipeline fully functional!")
        print("🚀 Ready for Phase 3 Feature Saver Development!")
    else:
        print("\n⚠️ Fix pipeline issues before proceeding to Phase 3")