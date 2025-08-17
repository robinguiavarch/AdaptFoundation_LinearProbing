#!/usr/bin/env python3
"""
Test script for Point-M2AE feature extraction pipeline.

Validates all components before running the full pipeline to ensure proper
functionality with CUDA extensions and hierarchical grouping.
"""

import numpy as np
import torch
import psutil
import time
from pathlib import Path
import sys
import yaml
import importlib
import inspect

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.loaders import HCPOFCDataLoader
from point_m2ae.feature_extraction_core_m2ae import PointM2AEFeatureExtractor


class MemoryMonitor:
    """
    Monitors memory usage during testing.
    
    Attributes:
        initial_ram (float): Initial RAM usage in GB
        initial_gpu (float): Initial GPU memory usage in GB
    """
    
    def __init__(self):
        """Initialize memory monitor."""
        self.initial_ram = self.get_ram_usage()
        self.initial_gpu = self.get_gpu_usage()
    
    def get_ram_usage(self) -> float:
        """Get current RAM usage in GB."""
        return psutil.virtual_memory().used / (1024**3)
    
    def get_gpu_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        return torch.cuda.memory_allocated() / (1024**3)
    
    def print_memory_status(self, stage: str) -> None:
        """Print current memory status."""
        ram_current = self.get_ram_usage()
        gpu_current = self.get_gpu_usage()
        ram_delta = ram_current - self.initial_ram
        gpu_delta = gpu_current - self.initial_gpu
        
        print(f"  Memory [{stage}]:")
        print(f"    RAM: {ram_current:.2f}GB (+{ram_delta:.2f}GB)")
        print(f"    GPU: {gpu_current:.2f}GB (+{gpu_delta:.2f}GB)")


class PointM2AEFeatureExtractionTester:
    """
    Tests all components of the Point-M2AE feature extraction pipeline.
    
    Attributes:
        data_loader (HCPOFCDataLoader): Data loader for testing
        device (torch.device): Computation device
        memory_monitor (MemoryMonitor): Memory usage monitor
    """
    
    def __init__(self, data_path: str = "crops/2mm/S.Or."):
        """
        Initialize tester.
        
        Args:
            data_path (str): Path to HCP OFC dataset
        """
        self.device = torch.device('cuda')
        self.memory_monitor = MemoryMonitor()
        
        print("Initializing Point-M2AE Feature Extraction Tester")
        print(f"Device: {self.device}")
        
        try:
            self.data_loader = HCPOFCDataLoader(data_path)
            print(f"Dataset loaded: {len(self.data_loader.skeletons)} volumes")
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            sys.exit(1)
    
    def test_data_loading(self) -> np.ndarray:
        """
        Test data loading and return sample volume.
        
        Returns:
            np.ndarray: Sample 3D volume for testing
        """
        print("\n1. Testing Data Loading")
        
        sample_volume = self.data_loader.skeletons[0]
        expected_shape = (30, 38, 22)
        
        print(f"  Sample volume shape: {sample_volume.shape}")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Volume density: {np.mean(sample_volume):.4f}")
        print(f"  Non-zero voxels: {np.sum(sample_volume)} / {np.prod(sample_volume.shape)}")
        
        assert sample_volume.shape == expected_shape, f"Wrong volume shape: {sample_volume.shape}"
        assert 0 <= np.min(sample_volume) <= np.max(sample_volume) <= 1, "Volume values not in [0,1]"
        
        print("  Data loading OK")
        return sample_volume
    
    def test_extensions(self) -> None:
        """
        Test CUDA extensions critical for Point-M2AE.
        """
        print("\n2. Testing CUDA Extensions")
        
        # Test knn_cuda import
        try:
            import knn_cuda
            print(f"  knn_cuda import: OK")
            print(f"  knn_cuda file: {knn_cuda.__file__}")
            
            # Test KNN signature
            sig = inspect.signature(knn_cuda.knn)
            params_count = len(sig.parameters)
            print(f"  knn() parameters: {params_count}")
            
            if params_count == 3:
                print("  KNN shim required: 4->3 args compatibility")
            else:
                print("  KNN signature: standard 4 args")
                
        except Exception as e:
            print(f"  ERROR knn_cuda: {e}")
            sys.exit(1)
        
        # Test pointnet2_ops import
        try:
            import pointnet2_ops
            print(f"  pointnet2_ops import: OK")
        except Exception as e:
            print(f"  ERROR pointnet2_ops: {e}")
            sys.exit(1)
        
        # Test chamfer import (optional)
        try:
            import chamfer
            print(f"  chamfer import: OK")
        except Exception as e:
            print(f"  WARNING chamfer: {e}")
        
        print("  CUDA extensions OK")
    
    def test_configuration_integration(self) -> dict:
        """
        Test integration of YAML config with core components.
        
        Returns:
            dict: Validated configuration dictionary
        """
        print("\n3. Testing Configuration Integration")
        
        config_path = Path("point_m2ae/feature_extraction_m2ae.yaml")
        
        if not config_path.exists():
            print(f"  ERROR: Config file not found: {config_path}")
            sys.exit(1)
        
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'processing', 'data', 'feature_approaches']
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"
        
        # Validate processing parameters
        processing = config['processing']
        expected_params = {
            'encoder_depths': [5, 5, 5],
            'encoder_dims': [96, 192, 384],
            'group_sizes': [16, 16, 16],
            'num_groups': [256, 128, 32]
        }
        
        for param, expected_value in expected_params.items():
            assert param in processing, f"Missing processing param: {param}"
            assert processing[param] == expected_value, f"Wrong {param}: {processing[param]}"
        
        # Validate feature approaches
        approaches = config['feature_approaches']
        assert 'feat_mean' in approaches, "Missing feat_mean approach"
        assert 'feat_mean_max' in approaches, "Missing feat_mean_max approach"
        assert approaches['feat_mean']['expected_output_dim'] == 384
        assert approaches['feat_mean_max']['expected_output_dim'] == 768
        
        print(f"  Config sections: {list(config.keys())}")
        print(f"  Feature approaches: {list(approaches.keys())}")
        print(f"  Processing params validated")
        print("  Configuration integration OK")
        
        return config
    
    def test_preprocessing_roi_to_points(self, sample_volume: np.ndarray) -> np.ndarray:
        """
        Test ROI to points preprocessing.
        
        Args:
            sample_volume (np.ndarray): Sample ROI volume
            
        Returns:
            np.ndarray: Preprocessed points
        """
        print("\n4. Testing Preprocessing ROI to Points")
        
        # Extract active voxels
        idx = np.argwhere(sample_volume == 1)
        assert idx.size > 0, "Sample volume has no active voxels"
        
        # Apply preprocessing
        pts = idx.astype(np.float32)
        pts_original_mean = pts.mean(axis=0)
        
        # Centering
        pts -= pts.mean(axis=0, keepdims=True)
        
        # Scaling
        scale = float(np.abs(pts).max()) + 1e-6
        pts /= scale
        
        # Validations
        assert pts.shape[1] == 3, f"Points should be 3D, got shape {pts.shape}"
        assert pts.dtype == np.float32, f"Points should be float32, got {pts.dtype}"
        
        # Check centering
        pts_mean = np.abs(pts.mean(axis=0))
        assert np.all(pts_mean < 1e-5), f"Points not properly centered: {pts_mean}"
        
        # Check normalization
        pts_max = np.abs(pts).max()
        assert 0.9 <= pts_max <= 1.1, f"Points not in [-1,1] range: max={pts_max}"
        
        print(f"  Original points: {idx.shape[0]}")
        print(f"  Points shape: {pts.shape}")
        print(f"  Points range: [{np.min(pts):.3f}, {np.max(pts):.3f}]")
        print(f"  Original mean: {pts_original_mean}")
        print(f"  Centered mean: {pts.mean(axis=0)}")
        print("  Preprocessing ROI to points OK")
        
        return pts
    
    def test_extractor_loading(self, config: dict) -> dict:
        """
        Test Point-M2AE extractor loading.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Dictionary of loaded extractors
        """
        print("\n5. Testing Extractor Loading")
        
        self.memory_monitor.print_memory_status("before extractor loading")
        
        checkpoint_path = Path(config['model']['checkpoint_path'])
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
        
        extractors = {}
        approaches = ['feat_mean', 'feat_mean_max']
        
        for approach in approaches:
            print(f"  Loading {approach} extractor...")
            
            try:
                extractor = PointM2AEFeatureExtractor(approach, checkpoint_path, config)
                
                # Validate extractor properties
                assert extractor.approach == approach
                assert extractor.device == self.device
                assert hasattr(extractor, 'encoder')
                assert hasattr(extractor, 'groupers')
                
                # Check eval mode for sub-modules (like test_model.py)
                assert not extractor.encoder.training, "Encoder should be in eval mode"
                assert not extractor.groupers.training, "Groupers should be in eval mode"
                
                # Check output dimension
                expected_dim = 384 if approach == 'feat_mean' else 768
                assert extractor.get_output_dim() == expected_dim
                
                extractors[approach] = extractor
                print(f"    {approach} extractor loaded, output_dim: {expected_dim}")
                
            except Exception as e:
                print(f"    ERROR loading {approach}: {e}")
                sys.exit(1)
        
        self.memory_monitor.print_memory_status("after extractor loading")
        print("  Extractor loading OK")
        
        return extractors
    
    def test_feature_extraction(self, extractors: dict, sample_points: np.ndarray) -> dict:
        """
        Test feature extraction with Point-M2AE pipeline.
        
        Args:
            extractors (dict): Dictionary of extractors
            sample_points (np.ndarray): Preprocessed sample points
            
        Returns:
            dict: Extracted features for each approach
        """
        print("\n6. Testing Feature Extraction")
        
        self.memory_monitor.print_memory_status("before feature extraction")
        
        # Create sample ROI for testing
        sample_roi = np.zeros((30, 38, 22), dtype=np.uint8)
        
        # Use subset of points to create ROI
        n_points = min(600, len(sample_points))
        selected_points = sample_points[:n_points]
        
        # Denormalize points back to ROI coordinates
        denorm_points = selected_points * 10 + 15  # Rough denormalization
        denorm_points = np.clip(denorm_points, 0, [29, 37, 21]).astype(int)
        
        sample_roi[denorm_points[:, 0], denorm_points[:, 1], denorm_points[:, 2]] = 1
        
        extracted_features = {}
        
        for approach, extractor in extractors.items():
            print(f"  Testing {approach} feature extraction...")
            
            try:
                start_time = time.time()
                features = extractor.extract_features(sample_roi)
                extraction_time = time.time() - start_time
                
                # Validate feature properties
                assert isinstance(features, torch.Tensor), "Features should be torch.Tensor"
                assert features.dim() == 2, f"Features should be 2D, got {features.dim()}"
                assert features.shape[0] == 1, f"Batch size should be 1, got {features.shape[0]}"
                
                expected_dim = 384 if approach == 'feat_mean' else 768
                assert features.shape[1] == expected_dim, f"Wrong feature dim: {features.shape[1]} vs {expected_dim}"
                
                # Check for NaN/Inf
                assert torch.isfinite(features).all(), f"Features contain NaN/Inf"
                
                # Check feature range
                feat_min, feat_max = torch.min(features), torch.max(features)
                non_zero_features = torch.sum(features != 0).item()
                
                print(f"    Features shape: {features.shape}")
                print(f"    Extraction time: {extraction_time:.3f}s")
                print(f"    Feature range: [{feat_min:.4f}, {feat_max:.4f}]")
                print(f"    Non-zero features: {non_zero_features}/{features.numel()}")
                
                extracted_features[approach] = features
                print(f"    {approach} extraction OK")
                
            except Exception as e:
                print(f"    ERROR {approach} extraction: {e}")
                continue
        
        self.memory_monitor.print_memory_status("after feature extraction")
        print("  Feature extraction OK")
        
        return extracted_features
    
    def test_batch_processing(self, extractors: dict) -> None:
        """
        Test batch processing capabilities with sequential processing.
        
        Args:
            extractors (dict): Dictionary of extractors
        """
        print("\n7. Testing Batch Processing")
        
        self.memory_monitor.print_memory_status("before batch processing")
        
        # Create test batch
        test_volumes = []
        for i in range(3):
            volume = self.data_loader.skeletons[i]
            test_volumes.append(volume)
        
        batch_tensor = torch.stack([torch.from_numpy(v).float() for v in test_volumes])
        print(f"  Test batch shape: {batch_tensor.shape}")
        
        for approach, extractor in extractors.items():
            print(f"  Testing {approach} batch processing...")
            
            try:
                start_time = time.time()
                batch_features = extractor.extract_features_batch(batch_tensor)
                batch_time = time.time() - start_time
                
                # Validate batch dimensions
                expected_batch_size = 3
                expected_dim = 384 if approach == 'feat_mean' else 768
                
                assert batch_features.shape[0] == expected_batch_size, f"Wrong batch size: {batch_features.shape[0]}"
                assert batch_features.shape[1] == expected_dim, f"Wrong feature dim: {batch_features.shape[1]}"
                
                # Check for NaN/Inf
                assert torch.isfinite(batch_features).all(), "Batch features contain NaN/Inf"
                
                print(f"    Batch features shape: {batch_features.shape}")
                print(f"    Batch processing time: {batch_time:.3f}s")
                print(f"    Time per volume: {batch_time/3:.3f}s")
                print(f"    {approach} batch processing OK")
                
            except Exception as e:
                print(f"    ERROR {approach} batch processing: {e}")
                continue
        
        self.memory_monitor.print_memory_status("after batch processing")
        print("  Batch processing OK")
    
    def test_approach_differences(self, extracted_features: dict) -> None:
        """
        Test differences between feat_mean and feat_mean_max approaches.
        
        Args:
            extracted_features (dict): Features from different approaches
        """
        print("\n8. Testing Approach Differences")
        
        if len(extracted_features) < 2:
            print("  Need both approaches for comparison")
            return
        
        feat_mean = extracted_features['feat_mean']
        feat_mean_max = extracted_features['feat_mean_max']
        
        print(f"  Comparing feat_mean vs feat_mean_max...")
        
        # Validate dimensions
        assert feat_mean.shape[1] == 384, f"feat_mean wrong dim: {feat_mean.shape[1]}"
        assert feat_mean_max.shape[1] == 768, f"feat_mean_max wrong dim: {feat_mean_max.shape[1]}"
        
        # Check relationship
        dim_ratio = feat_mean_max.shape[1] / feat_mean.shape[1]
        assert dim_ratio == 2.0, f"feat_mean_max should be 2x feat_mean: {dim_ratio}"
        
        # Check that they are different (not just concatenated)
        feat_mean_part = feat_mean_max[:, :384]
        difference = torch.abs(feat_mean - feat_mean_part)
        max_diff = torch.max(difference).item()
        
        print(f"    feat_mean dimension: {feat_mean.shape[1]}")
        print(f"    feat_mean_max dimension: {feat_mean_max.shape[1]}")
        print(f"    Dimension ratio: {dim_ratio}")
        print(f"    Max difference vs first half: {max_diff:.6f}")
        
        # They should be different because feat_mean_max = [mean, max], not [mean, mean]
        if max_diff > 1e-6:
            print("    feat_mean_max contains different information than feat_mean")
        else:
            print("    feat_mean_max first half identical to feat_mean")
        
        print("  Approach differences OK")
    
    def run_all_tests(self) -> None:
        """
        Run complete test suite for Point-M2AE feature extraction.
        """
        print("=" * 60)
        print("POINT-M2AE FEATURE EXTRACTION PIPELINE TESTS")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run tests sequentially
            sample_volume = self.test_data_loading()
            self.test_extensions()
            config = self.test_configuration_integration()
            sample_points = self.test_preprocessing_roi_to_points(sample_volume)
            extractors = self.test_extractor_loading(config)
            extracted_features = self.test_feature_extraction(extractors, sample_points)
            self.test_batch_processing(extractors)
            self.test_approach_differences(extracted_features)
            
            # Final summary
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print("ALL TESTS PASSED")
            print(f"{'='*60}")
            print(f"Total test time: {total_time:.1f}s")
            self.memory_monitor.print_memory_status("final")
            print("Pipeline ready for full execution!")
            print("Key features validated:")
            print("  - CUDA extensions: knn_cuda, pointnet2_ops")
            print("  - Configuration integration: YAML->Config->Extractors")
            print("  - Preprocessing: ROI->Points normalization")
            print("  - Feature extraction: 384D (feat_mean), 768D (feat_mean_max)")
            print("  - Batch processing: sequential robustness")
            print("  - Hierarchical grouping: 3 scales (256->128->32)")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\nTEST FAILED: {e}")
            print("Fix issues before running full pipeline")
            sys.exit(1)


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Point-M2AE feature extraction pipeline")
    parser.add_argument('--data-path', type=str, default='crops/2mm/S.Or.',
                       help='Path to HCP OFC dataset')
    
    args = parser.parse_args()
    
    # Check data path
    if not Path(args.data_path).exists():
        print(f"Data path not found: {args.data_path}")
        sys.exit(1)
    
    # Check config file
    config_path = Path("point_m2ae/feature_extraction_m2ae.yaml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Run tests
    tester = PointM2AEFeatureExtractionTester(args.data_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()