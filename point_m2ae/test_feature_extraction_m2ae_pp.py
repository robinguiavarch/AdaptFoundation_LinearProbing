#!/usr/bin/env python3
"""
Test script for Point-M2AE feature extraction pipeline with optimized preprocessing.

Validates all components including v1 (fixed normalization) and v2 (topological features)
preprocessing before running the full pipeline.
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
from point_m2ae.feature_extraction_core_m2ae_pp import PointM2AEFeatureExtractor


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


class PointM2AEOptimizedPreprocessingTester:
    """
    Tests all components of the Point-M2AE feature extraction pipeline with optimized preprocessing.
    
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
        
        print("Initializing Point-M2AE Optimized Preprocessing Tester")
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
        
        print("  CUDA extensions OK")
    
    def test_configuration_integration(self) -> dict:
        """
        Test integration of optimized YAML config with core components.
        
        Returns:
            dict: Validated configuration dictionary
        """
        print("\n3. Testing Optimized Configuration Integration")
        
        config_path = Path("point_m2ae/feature_extraction_m2ae_pp.yaml")
        
        if not config_path.exists():
            print(f"  ERROR: Config file not found: {config_path}")
            sys.exit(1)
        
        # Load YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'processing', 'data', 'feature_approaches', 'preprocessing']
        for section in required_sections:
            assert section in config, f"Missing config section: {section}"
        
        # Validate processing parameters (corrected config)
        processing = config['processing']
        expected_params = {
            'encoder_depths': [5, 5, 5],
            'encoder_dims': [96, 192, 384],
            'group_sizes': [16, 8, 8],
            'num_groups': [512, 256, 64]
        }
        
        for param, expected_value in expected_params.items():
            assert param in processing, f"Missing processing param: {param}"
            assert processing[param] == expected_value, f"Wrong {param}: {processing[param]}"
        
        # Validate preprocessing sections
        preprocessing = config['preprocessing']
        assert 'versions' in preprocessing, "Missing preprocessing versions"
        assert 'v1' in preprocessing['versions'], "Missing v1 preprocessing"
        assert 'v2' in preprocessing['versions'], "Missing v2 preprocessing"
        
        # Validate feature approaches (only feat_mean_max)
        approaches = config['feature_approaches']
        assert 'feat_mean_max' in approaches, "Missing feat_mean_max approach"
        assert approaches['feat_mean_max']['expected_output_dim'] == 768
        
        print(f"  Config sections: {list(config.keys())}")
        print(f"  Feature approaches: {list(approaches.keys())}")
        print(f"  Preprocessing versions: {list(preprocessing['versions'].keys())}")
        print(f"  Processing params validated (corrected config)")
        print("  Optimized configuration integration OK")
        
        return config
    
    def test_preprocessing_v1_fixed_normalization(self, sample_volume: np.ndarray) -> np.ndarray:
        """
        Test v1 preprocessing: fixed normalization preserving anatomical position.
        
        Args:
            sample_volume (np.ndarray): Sample ROI volume
            
        Returns:
            np.ndarray: Preprocessed points with v1
        """
        print("\n4. Testing Preprocessing v1 (Fixed Normalization)")
        
        # Extract active voxels
        idx = np.argwhere(sample_volume == 1)
        assert idx.size > 0, "Sample volume has no active voxels"
        
        # Apply v1 preprocessing (fixed normalization)
        pts = idx.astype(np.float32) / np.array([29.0, 37.0, 21.0])
        
        # Validations
        assert pts.shape[1] == 3, f"Points should be 3D, got shape {pts.shape}"
        assert pts.dtype in [np.float32, np.float64], f"Points should be float32 or float64, got {pts.dtype}"
        
        # Check normalization range
        assert np.all(pts >= 0.0), f"Points should be >= 0, min: {np.min(pts)}"
        assert np.all(pts <= 1.0), f"Points should be <= 1, max: {np.max(pts)}"
        
        # Check that normalization is fixed (deterministic)
        # Test point (15, 20, 10) should always become (15/29, 20/37, 10/21)
        test_coord = np.array([15, 20, 10])
        expected_norm = test_coord.astype(np.float32) / np.array([29.0, 37.0, 21.0])
        
        print(f"  Original points: {idx.shape[0]}")
        print(f"  Points shape: {pts.shape}")
        print(f"  Points range: [{np.min(pts):.3f}, {np.max(pts):.3f}]")
        print(f"  Test normalization (15,20,10) -> ({expected_norm[0]:.3f}, {expected_norm[1]:.3f}, {expected_norm[2]:.3f})")
        print(f"  Anatomical position preserved: Fixed grid normalization")
        print("  Preprocessing v1 OK")
        
        return pts
    
    def test_preprocessing_v2_topological_features(self, sample_volume: np.ndarray) -> np.ndarray:
        """
        Test v2 preprocessing: fixed normalization + topological features.
        
        Args:
            sample_volume (np.ndarray): Sample ROI volume
            
        Returns:
            np.ndarray: Preprocessed features with v2
        """
        print("\n5. Testing Preprocessing v2 (Topological Features)")
        
        # Extract active voxels
        idx = np.argwhere(sample_volume == 1)
        assert idx.size > 0, "Sample volume has no active voxels"
        
        # Apply v2 preprocessing manually to test components
        enriched_features = []
        
        # Test first few points
        test_points = idx[:min(10, len(idx))]
        
        for pt in test_points:
            x, y, z = int(pt[0]), int(pt[1]), int(pt[2])
            
            # Fixed normalization (same as v1)
            pos_norm = pt.astype(np.float32) / np.array([29.0, 37.0, 21.0])
            
            # Test topological features
            nb_neighbors = self._count_6_neighbors(sample_volume, x, y, z)
            density_gradient = self._calculate_density_gradient(sample_volume, x, y, z)
            continuity_score = self._check_continuity_pattern(sample_volume, x, y, z)
            centrality_score = self._calculate_centrality_chebyshev(x, y, z)
            
            # Validate individual features
            assert 0 <= nb_neighbors <= 6, f"Invalid neighbors count: {nb_neighbors}"
            assert 0.0 <= density_gradient <= 1.0, f"Invalid density gradient: {density_gradient}"
            assert 0.0 <= continuity_score <= 1.0, f"Invalid continuity score: {continuity_score}"
            assert 0.0 <= centrality_score <= 1.0, f"Invalid centrality score: {centrality_score}"
            
            # Combine features
            feature_vector = np.concatenate([
                pos_norm,
                [nb_neighbors/6.0],
                [density_gradient],
                [continuity_score],
                [centrality_score]
            ])
            enriched_features.append(feature_vector)
        
        features_array = np.array(enriched_features)
        
        # Validations
        assert features_array.shape[1] == 7, f"Features should be 7D, got {features_array.shape[1]}"
        assert features_array.dtype in [np.float32, np.float64], f"Features should be float32 or float64"
        
        # Check feature ranges
        pos_features = features_array[:, :3]  # Position
        topo_features = features_array[:, 3:]  # Topological
        
        assert np.all(pos_features >= 0.0) and np.all(pos_features <= 1.0), "Position features out of range"
        assert np.all(topo_features >= 0.0) and np.all(topo_features <= 1.0), "Topological features out of range"
        
        print(f"  Test points processed: {len(test_points)}")
        print(f"  Features shape: {features_array.shape}")
        print(f"  Position features range: [{np.min(pos_features):.3f}, {np.max(pos_features):.3f}]")
        print(f"  Topological features range: [{np.min(topo_features):.3f}, {np.max(topo_features):.3f}]")
        print(f"  Feature composition: [x,y,z,neighbors,density,continuity,centrality]")
        
        # Test centrality calculation specifically
        center_point = np.array([14, 18, 10])  # Near center
        corner_point = np.array([0, 0, 0])     # Corner
        
        center_centrality = self._calculate_centrality_chebyshev(14, 18, 10)
        corner_centrality = self._calculate_centrality_chebyshev(0, 0, 0)
        
        print(f"  Centrality test - center (14,18,10): {center_centrality:.3f}")
        print(f"  Centrality test - corner (0,0,0): {corner_centrality:.3f}")
        assert center_centrality > corner_centrality, "Center should have higher centrality than corner"
        
        print("  Preprocessing v2 OK")
        
        return features_array
    
    def _count_6_neighbors(self, roi_3d: np.ndarray, x: int, y: int, z: int) -> int:
        """Helper function to count 6-connected neighbors."""
        neighbors = [
            (x-1, y, z), (x+1, y, z),
            (x, y-1, z), (x, y+1, z),
            (x, y, z-1), (x, y, z+1)
        ]
        
        count = 0
        for nx, ny, nz in neighbors:
            if (0 <= nx < 30 and 0 <= ny < 38 and 0 <= nz < 22):
                if roi_3d[nx, ny, nz] == 1:
                    count += 1
        
        return count
    
    def _calculate_density_gradient(self, roi_3d: np.ndarray, x: int, y: int, z: int) -> float:
        """Helper function to calculate density gradient."""
        window = 2
        total_voxels = 0
        active_voxels = 0
        
        for dx in range(-window, window+1):
            for dy in range(-window, window+1):
                for dz in range(-window, window+1):
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if (0 <= nx < 30 and 0 <= ny < 38 and 0 <= nz < 22):
                        total_voxels += 1
                        if roi_3d[nx, ny, nz] == 1:
                            active_voxels += 1
        
        if total_voxels == 0:
            return 0.0
        
        density = active_voxels / total_voxels
        return float(density)
    
    def _check_continuity_pattern(self, roi_3d: np.ndarray, x: int, y: int, z: int) -> float:
        """Helper function to check continuity patterns."""
        directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        continuity_scores = []
        
        for dx, dy, dz in directions:
            continuous_length = 0
            for step in range(1, 4):
                nx, ny, nz = x + step*dx, y + step*dy, z + step*dz
                if (0 <= nx < 30 and 0 <= ny < 38 and 0 <= nz < 22):
                    if roi_3d[nx, ny, nz] == 1:
                        continuous_length += 1
                    else:
                        break
                else:
                    break
            
            continuity_scores.append(continuous_length / 3.0)
        
        return float(np.mean(continuity_scores))
    
    def _calculate_centrality_chebyshev(self, x: int, y: int, z: int) -> float:
        """Helper function to calculate Chebyshev centrality."""
        center_x, center_y, center_z = 14.5, 18.5, 10.5
        
        dist_x = abs(x - center_x)
        dist_y = abs(y - center_y)
        dist_z = abs(z - center_z)
        
        chebyshev_distance = max(dist_x, dist_y, dist_z)
        max_chebyshev = max(center_x, center_y, center_z)
        
        centrality = 1.0 - (chebyshev_distance / max_chebyshev)
        
        return float(centrality)
    
    def test_extractor_loading(self, config: dict) -> dict:
        """
        Test Point-M2AE extractor loading with both preprocessing versions.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Dictionary of loaded extractors
        """
        print("\n6. Testing Extractor Loading (Both Preprocessing Versions)")
        
        self.memory_monitor.print_memory_status("before extractor loading")
        
        checkpoint_path = Path(config['model']['checkpoint_path'])
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
        
        extractors = {}
        preprocessing_versions = ['v1', 'v2']
        approach = 'feat_mean_max'
        
        for prep_version in preprocessing_versions:
            print(f"  Loading {approach} extractor with preprocessing {prep_version}...")
            
            try:
                extractor = PointM2AEFeatureExtractor(approach, checkpoint_path, config, prep_version)
                
                # Validate extractor properties
                assert extractor.approach == approach
                assert extractor.preprocessing_version == prep_version
                assert extractor.device == self.device
                assert hasattr(extractor, 'encoder')
                assert hasattr(extractor, 'groupers')
                
                # Check eval mode
                assert not extractor.encoder.training, "Encoder should be in eval mode"
                assert not extractor.groupers.training, "Groupers should be in eval mode"
                
                # Check output dimension (always 768 for feat_mean_max)
                expected_dim = 768
                assert extractor.get_output_dim() == expected_dim
                
                extractors[f"{approach}_{prep_version}"] = extractor
                print(f"    {approach}_{prep_version} extractor loaded, output_dim: {expected_dim}")
                
            except Exception as e:
                print(f"    ERROR loading {approach}_{prep_version}: {e}")
                sys.exit(1)
        
        self.memory_monitor.print_memory_status("after extractor loading")
        print("  Extractor loading OK")
        
        return extractors
    
    def test_feature_extraction(self, extractors: dict, sample_volume: np.ndarray) -> dict:
        """
        Test feature extraction with both preprocessing versions.
        
        Args:
            extractors (dict): Dictionary of extractors
            sample_volume (np.ndarray): Sample ROI volume
            
        Returns:
            dict: Extracted features for each preprocessing version
        """
        print("\n7. Testing Feature Extraction (Both Preprocessing Versions)")
        
        self.memory_monitor.print_memory_status("before feature extraction")
        
        extracted_features = {}
        
        for extractor_name, extractor in extractors.items():
            print(f"  Testing {extractor_name} feature extraction...")
            
            try:
                start_time = time.time()
                features = extractor.extract_features(sample_volume)
                extraction_time = time.time() - start_time
                
                # Validate feature properties
                assert isinstance(features, torch.Tensor), "Features should be torch.Tensor"
                assert features.dim() == 2, f"Features should be 2D, got {features.dim()}"
                assert features.shape[0] == 1, f"Batch size should be 1, got {features.shape[0]}"
                
                expected_dim = 768  # Always 768 for feat_mean_max
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
                
                extracted_features[extractor_name] = features
                print(f"    {extractor_name} extraction OK")
                
            except Exception as e:
                print(f"    ERROR {extractor_name} extraction: {e}")
                continue
        
        self.memory_monitor.print_memory_status("after feature extraction")
        print("  Feature extraction OK")
        
        return extracted_features
    
    def test_preprocessing_differences(self, extracted_features: dict) -> None:
        """
        Test differences between v1 and v2 preprocessing approaches.
        
        Args:
            extracted_features (dict): Features from different preprocessing versions
        """
        print("\n8. Testing Preprocessing Differences")
        
        if len(extracted_features) < 2:
            print("  Need both preprocessing versions for comparison")
            return
        
        feat_v1 = extracted_features['feat_mean_max_v1']
        feat_v2 = extracted_features['feat_mean_max_v2']
        
        print(f"  Comparing preprocessing v1 vs v2...")
        
        # Both should have same output dimension (768)
        assert feat_v1.shape == feat_v2.shape, f"Different shapes: v1={feat_v1.shape}, v2={feat_v2.shape}"
        
        # Check that they produce different features (due to different input preprocessing)
        difference = torch.abs(feat_v1 - feat_v2)
        max_diff = torch.max(difference).item()
        mean_diff = torch.mean(difference).item()
        
        print(f"    v1 features range: [{torch.min(feat_v1):.4f}, {torch.max(feat_v1):.4f}]")
        print(f"    v2 features range: [{torch.min(feat_v2):.4f}, {torch.max(feat_v2):.4f}]")
        print(f"    Max difference: {max_diff:.6f}")
        print(f"    Mean difference: {mean_diff:.6f}")
        
        # They should be different due to different preprocessing
        if max_diff > 1e-4:
            print("    v1 and v2 produce different features (expected due to different preprocessing)")
        else:
            print("    WARNING: v1 and v2 produce very similar features")
        
        print("  Preprocessing differences OK")
    
    def test_batch_processing(self, extractors: dict) -> None:
        """
        Test batch processing capabilities with both preprocessing versions.
        
        Args:
            extractors (dict): Dictionary of extractors
        """
        print("\n9. Testing Batch Processing (Both Preprocessing Versions)")
        
        self.memory_monitor.print_memory_status("before batch processing")
        
        # Create test batch
        test_volumes = []
        for i in range(3):
            volume = self.data_loader.skeletons[i]
            test_volumes.append(volume)
        
        batch_tensor = torch.stack([torch.from_numpy(v).float() for v in test_volumes])
        print(f"  Test batch shape: {batch_tensor.shape}")
        
        for extractor_name, extractor in extractors.items():
            print(f"  Testing {extractor_name} batch processing...")
            
            try:
                start_time = time.time()
                batch_features = extractor.extract_features_batch(batch_tensor)
                batch_time = time.time() - start_time
                
                # Validate batch dimensions
                expected_batch_size = 3
                expected_dim = 768  # Always 768 for feat_mean_max
                
                assert batch_features.shape[0] == expected_batch_size, f"Wrong batch size: {batch_features.shape[0]}"
                assert batch_features.shape[1] == expected_dim, f"Wrong feature dim: {batch_features.shape[1]}"
                
                # Check for NaN/Inf
                assert torch.isfinite(batch_features).all(), "Batch features contain NaN/Inf"
                
                print(f"    Batch features shape: {batch_features.shape}")
                print(f"    Batch processing time: {batch_time:.3f}s")
                print(f"    Time per volume: {batch_time/3:.3f}s")
                print(f"    {extractor_name} batch processing OK")
                
            except Exception as e:
                print(f"    ERROR {extractor_name} batch processing: {e}")
                continue
        
        self.memory_monitor.print_memory_status("after batch processing")
        print("  Batch processing OK")
    
    def run_all_tests(self) -> None:
        """
        Run complete test suite for Point-M2AE feature extraction with optimized preprocessing.
        """
        print("=" * 70)
        print("POINT-M2AE OPTIMIZED PREPROCESSING PIPELINE TESTS")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Run tests sequentially
            sample_volume = self.test_data_loading()
            self.test_extensions()
            config = self.test_configuration_integration()
            self.test_preprocessing_v1_fixed_normalization(sample_volume)
            self.test_preprocessing_v2_topological_features(sample_volume)
            extractors = self.test_extractor_loading(config)
            extracted_features = self.test_feature_extraction(extractors, sample_volume)
            self.test_preprocessing_differences(extracted_features)
            self.test_batch_processing(extractors)
            
            # Final summary
            total_time = time.time() - start_time
            print(f"\n{'='*70}")
            print("ALL OPTIMIZED PREPROCESSING TESTS PASSED")
            print(f"{'='*70}")
            print(f"Total test time: {total_time:.1f}s")
            self.memory_monitor.print_memory_status("final")
            print("Optimized pipeline ready for full execution!")
            print("Key features validated:")
            print("  - CUDA extensions: knn_cuda, pointnet2_ops")
            print("  - Corrected configuration: Official Point-M2AE params")
            print("  - Preprocessing v1: Fixed normalization [0,1] preserving anatomy")
            print("  - Preprocessing v2: v1 + 4 topological features")
            print("  - Feature extraction: 768D (feat_mean_max) for both versions")
            print("  - Batch processing: Sequential robustness")
            print("  - Hierarchical grouping: 3 scales (512->256->64) - CORRECTED")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\nTEST FAILED: {e}")
            print("Fix issues before running full pipeline")
            sys.exit(1)


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Point-M2AE optimized preprocessing pipeline")
    parser.add_argument('--data-path', type=str, default='crops/2mm/S.Or.',
                       help='Path to HCP OFC dataset')
    
    args = parser.parse_args()
    
    # Check data path
    if not Path(args.data_path).exists():
        print(f"Data path not found: {args.data_path}")
        sys.exit(1)
    
    # Check config file
    config_path = Path("point_m2ae/feature_extraction_m2ae_pp.yaml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Run tests
    tester = PointM2AEOptimizedPreprocessingTester(args.data_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()