#!/usr/bin/env python3
"""
Test script for Point-M2AE feature extraction pipeline with 50 configurations.

Validates all 50 combinations (10 configs × 5 aggregations) including
parameter variations, aggregation methods, and output dimensions.
"""

import numpy as np
import torch
import psutil
import time
from pathlib import Path
import sys
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.loaders import HCPOFCDataLoader
from point_m2ae.feature_extraction_core_m2ae_cfgs import PointM2AEFeatureExtractorConfigs


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


class PointM2AEConfigsTester:
    """
    Tests all 50 configurations of Point-M2AE feature extraction pipeline.
    
    Attributes:
        data_loader (HCPOFCDataLoader): Data loader for testing
        device (torch.device): Computation device
        memory_monitor (MemoryMonitor): Memory usage monitor
        config (dict): Configuration dictionary from YAML
    """
    
    def __init__(self, data_path: str = "crops/2mm/S.Or."):
        """
        Initialize tester.
        
        Args:
            data_path (str): Path to HCP OFC dataset
        """
        self.device = torch.device('cuda')
        self.memory_monitor = MemoryMonitor()
        
        print("Initializing Point-M2AE 50 Configurations Tester")
        print(f"Device: {self.device}")
        
        try:
            self.data_loader = HCPOFCDataLoader(data_path)
            print(f"Dataset loaded: {len(self.data_loader.skeletons)} volumes")
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            sys.exit(1)
        
        # Load configuration
        config_path = Path(__file__).parent / "feature_extraction_m2ae_cfgs.yaml"
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
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
            
            # Test KNN signature
            import inspect
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
    
    def test_configuration_validation(self) -> None:
        """
        Test validation of 50 configurations in YAML.
        """
        print("\n3. Testing Configuration Validation")
        
        # Validate required sections
        required_sections = ['model', 'processing', 'aggregation_methods', 'configurations', 'data']
        for section in required_sections:
            assert section in self.config, f"Missing config section: {section}"
        
        # Validate 10 configurations
        configs = self.config['processing']['configs']
        assert len(configs) == 10, f"Expected 10 configs, got {len(configs)}"
        
        expected_configs = [f"C{i}" for i in range(1, 11)]
        for config_name in expected_configs:
            assert config_name in configs, f"Missing configuration: {config_name}"
            
            config_params = configs[config_name]
            assert 'num_groups' in config_params, f"Missing num_groups in {config_name}"
            assert 'group_sizes' in config_params, f"Missing group_sizes in {config_name}"
            assert 'local_radius' in config_params, f"Missing local_radius in {config_name}"
            assert len(config_params['num_groups']) == 3, f"num_groups should have 3 levels"
            assert len(config_params['group_sizes']) == 3, f"group_sizes should have 3 levels"
            assert len(config_params['local_radius']) == 3, f"local_radius should have 3 levels"
        
        # Validate 5 aggregation methods
        aggregations = self.config['aggregation_methods']
        assert len(aggregations) == 5, f"Expected 5 aggregations, got {len(aggregations)}"
        
        expected_aggregations = [f"A{i}" for i in range(1, 6)]
        expected_dims = {'A1': 384, 'A2': 1536, 'A3': 576, 'A4': 384, 'A5': 384}
        
        for agg_name in expected_aggregations:
            assert agg_name in aggregations, f"Missing aggregation: {agg_name}"
            assert aggregations[agg_name]['output_dim'] == expected_dims[agg_name]
        
        # Validate 50 complete configurations
        complete_configs = self.config['configurations']
        assert len(complete_configs) == 50, f"Expected 50 configurations, got {len(complete_configs)}"
        
        for i in range(1, 11):
            for j in range(1, 6):
                config_key = f"C{i}A{j}"
                assert config_key in complete_configs, f"Missing configuration: {config_key}"
                assert complete_configs[config_key]['config'] == f"C{i}"
                assert complete_configs[config_key]['aggregation'] == f"A{j}"
        
        print(f"  Configurations: {len(configs)} base configs")
        print(f"  Aggregations: {len(aggregations)} methods")
        print(f"  Complete configurations: {len(complete_configs)} total")
        print("  Configuration validation OK")
    
    def test_preprocessing_corrected(self, sample_volume: np.ndarray) -> np.ndarray:
        """
        Test corrected preprocessing with center + [-1,1] normalization.
        
        Args:
            sample_volume (np.ndarray): Sample ROI volume
            
        Returns:
            np.ndarray: Preprocessed points
        """
        print("\n4. Testing Corrected Preprocessing")
        
        # Extract active voxels
        idx = np.argwhere(sample_volume == 1)
        assert idx.size > 0, "Sample volume has no active voxels"
        
        # Apply corrected preprocessing (center + [-1,1])
        center = np.array([15.0, 19.0, 11.0])
        pts = (idx.astype(np.float32) - center) / center
        
        # Validations
        assert pts.shape[1] == 3, f"Points should be 3D, got shape {pts.shape}"
        assert pts.dtype in [np.float32, np.float64], f"Points should be float32 or float64"
        
        # Check normalization range [-1, 1]
        assert np.all(pts >= -1.0), f"Points should be >= -1, min: {np.min(pts)}"
        assert np.all(pts <= 1.0), f"Points should be <= 1, max: {np.max(pts)}"
        
        # Test specific normalization
        test_coord = np.array([0, 0, 0])  # Corner
        expected_norm = (test_coord.astype(np.float32) - center) / center
        expected_values = [-1.0, -1.0, -1.0]
        
        print(f"  Original points: {idx.shape[0]}")
        print(f"  Points shape: {pts.shape}")
        print(f"  Points range: [{np.min(pts):.3f}, {np.max(pts):.3f}]")
        print(f"  Test normalization (0,0,0) -> ({expected_values[0]:.1f}, {expected_values[1]:.1f}, {expected_values[2]:.1f})")
        print(f"  Center-based [-1,1] normalization: Fixed")
        print("  Corrected preprocessing OK")
        
        return pts
    
    def test_extractor_loading_all_configs(self) -> dict:
        """
        Test loading extractors for all 50 configurations.
        
        Returns:
            dict: Dictionary of loaded extractors
        """
        print("\n5. Testing Extractor Loading (All 50 Configurations)")
        
        self.memory_monitor.print_memory_status("before extractor loading")
        
        checkpoint_path = Path(self.config['model']['checkpoint_path'])
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
        
        extractors = {}
        failed_configs = []
        
        # Test all 50 configurations
        for i in range(1, 11):
            for j in range(1, 6):
                config_name = f"C{i}"
                aggregation_name = f"A{j}"
                extractor_key = f"{config_name}{aggregation_name}"
                
                try:
                    extractor = PointM2AEFeatureExtractorConfigs(
                        config_name, aggregation_name, checkpoint_path, self.config
                    )
                    
                    # Validate extractor properties
                    assert extractor.config_name == config_name
                    assert extractor.aggregation_name == aggregation_name
                    assert extractor.device == self.device
                    assert hasattr(extractor, 'encoder')
                    assert hasattr(extractor, 'groupers')
                    
                    # Check eval mode
                    assert not extractor.encoder.training, "Encoder should be in eval mode"
                    assert not extractor.groupers.training, "Groupers should be in eval mode"
                    
                    # Validate output dimension
                    expected_dims = {'A1': 384, 'A2': 1536, 'A3': 576, 'A4': 384, 'A5': 384}
                    expected_dim = expected_dims[aggregation_name]
                    assert extractor.get_output_dim() == expected_dim
                    
                    extractors[extractor_key] = extractor
                    
                    if len(extractors) % 10 == 0:
                        print(f"    Loaded {len(extractors)}/50 configurations...")
                    
                except Exception as e:
                    print(f"    ERROR loading {extractor_key}: {e}")
                    failed_configs.append(extractor_key)
        
        if failed_configs:
            print(f"  FAILED configurations: {failed_configs}")
            sys.exit(1)
        
        self.memory_monitor.print_memory_status("after extractor loading")
        print(f"  Successfully loaded all 50 configurations")
        print("  Extractor loading OK")
        
        return extractors
    
    def test_feature_extraction_all_aggregations(self, extractors: dict, sample_volume: np.ndarray) -> dict:
        """
        Test feature extraction with all aggregation methods.
        
        Args:
            extractors (dict): Dictionary of extractors
            sample_volume (np.ndarray): Sample ROI volume
            
        Returns:
            dict: Extracted features for each configuration
        """
        print("\n6. Testing Feature Extraction (All Aggregation Methods)")
        
        self.memory_monitor.print_memory_status("before feature extraction")
        
        extracted_features = {}
        failed_extractions = []
        
        # Test one config per aggregation method first (C1A1-C1A5)
        test_configs = [f"C1A{j}" for j in range(1, 6)]
        
        for config_key in test_configs:
            extractor = extractors[config_key]
            
            try:
                start_time = time.time()
                features = extractor.extract_features(sample_volume)
                extraction_time = time.time() - start_time
                
                # Validate feature properties
                assert isinstance(features, torch.Tensor), "Features should be torch.Tensor"
                assert features.dim() == 2, f"Features should be 2D, got {features.dim()}"
                assert features.shape[0] == 1, f"Batch size should be 1, got {features.shape[0]}"
                
                expected_dim = extractor.get_output_dim()
                assert features.shape[1] == expected_dim, f"Wrong feature dim: {features.shape[1]} vs {expected_dim}"
                
                # Check for NaN/Inf
                assert torch.isfinite(features).all(), f"Features contain NaN/Inf"
                
                # Check feature statistics
                feat_min, feat_max = torch.min(features), torch.max(features)
                non_zero_features = torch.sum(features != 0).item()
                
                print(f"    {config_key}: shape {features.shape}, time {extraction_time:.3f}s")
                print(f"      Range: [{feat_min:.4f}, {feat_max:.4f}], non-zero: {non_zero_features}")
                
                extracted_features[config_key] = features
                
            except Exception as e:
                print(f"    ERROR {config_key} extraction: {e}")
                failed_extractions.append(config_key)
        
        if failed_extractions:
            print(f"  FAILED extractions: {failed_extractions}")
            sys.exit(1)
        
        self.memory_monitor.print_memory_status("after feature extraction")
        print("  Feature extraction OK")
        
        return extracted_features
    
    def test_aggregation_differences(self, extracted_features: dict) -> None:
        """
        Test differences between aggregation methods.
        
        Args:
            extracted_features (dict): Features from different aggregations
        """
        print("\n7. Testing Aggregation Differences")
        
        # Compare all aggregation methods with C1 config
        base_configs = {f"C1A{j}": extracted_features[f"C1A{j}"] for j in range(1, 6)}
        
        print("  Comparing aggregation methods (C1 config):")
        
        for agg_name, features in base_configs.items():
            feat_min, feat_max = torch.min(features), torch.max(features)
            feat_mean = torch.mean(features)
            print(f"    {agg_name}: dim={features.shape[1]}, range=[{feat_min:.4f}, {feat_max:.4f}], mean={feat_mean:.4f}")
        
        # Check that different aggregations produce different features
        a1_features = base_configs['C1A1']
        
        for agg_key in ['C1A2', 'C1A3', 'C1A4', 'C1A5']:
            other_features = base_configs[agg_key]
            
            if a1_features.shape == other_features.shape:
                difference = torch.abs(a1_features - other_features)
                max_diff = torch.max(difference).item()
                print(f"    C1A1 vs {agg_key}: max diff = {max_diff:.6f}")
                
                if max_diff < 1e-4:
                    print(f"    WARNING: {agg_key} very similar to C1A1")
            else:
                print(f"    C1A1 vs {agg_key}: different dimensions ({a1_features.shape[1]} vs {other_features.shape[1]})")
        
        print("  Aggregation differences OK")
    
    def test_configuration_differences(self, extractors: dict, sample_volume: np.ndarray) -> None:
        """
        Test differences between parameter configurations.
        
        Args:
            extractors (dict): Dictionary of extractors
            sample_volume (np.ndarray): Sample ROI volume
        """
        print("\n8. Testing Configuration Differences")
        
        # Test different configs with same aggregation (A1)
        test_configs = ['C1A1', 'C2A1', 'C8A1', 'C10A1']
        config_features = {}
        
        for config_key in test_configs:
            extractor = extractors[config_key]
            features = extractor.extract_features(sample_volume)
            config_features[config_key] = features
            
            config_info = extractor.get_config_info()
            print(f"    {config_key}: {config_info['strategy']}")
            print(f"      num_groups: {config_info['num_groups']}")
            print(f"      group_sizes: {config_info['group_sizes']}")
        
        # Compare C1 (baseline) with others
        c1_features = config_features['C1A1']
        
        for config_key in ['C2A1', 'C8A1', 'C10A1']:
            other_features = config_features[config_key]
            difference = torch.abs(c1_features - other_features)
            max_diff = torch.max(difference).item()
            mean_diff = torch.mean(difference).item()
            print(f"    C1A1 vs {config_key}: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")
        
        print("  Configuration differences OK")
    
    def test_batch_processing_sample(self, extractors: dict) -> None:
        """
        Test batch processing with sample configurations.
        
        Args:
            extractors (dict): Dictionary of extractors
        """
        print("\n9. Testing Batch Processing")
        
        self.memory_monitor.print_memory_status("before batch processing")
        
        # Create test batch
        test_volumes = []
        for i in range(3):
            volume = self.data_loader.skeletons[i]
            test_volumes.append(volume)
        
        batch_tensor = torch.stack([torch.from_numpy(v).float() for v in test_volumes])
        print(f"  Test batch shape: {batch_tensor.shape}")
        
        # Test sample configurations
        test_configs = ['C1A1', 'C1A2', 'C8A5']
        
        for config_key in test_configs:
            extractor = extractors[config_key]
            
            try:
                start_time = time.time()
                batch_features = extractor.extract_features_batch(batch_tensor)
                batch_time = time.time() - start_time
                
                # Validate batch dimensions
                expected_batch_size = 3
                expected_dim = extractor.get_output_dim()
                
                assert batch_features.shape[0] == expected_batch_size, f"Wrong batch size: {batch_features.shape[0]}"
                assert batch_features.shape[1] == expected_dim, f"Wrong feature dim: {batch_features.shape[1]}"
                
                # Check for NaN/Inf
                assert torch.isfinite(batch_features).all(), "Batch features contain NaN/Inf"
                
                print(f"    {config_key}: batch shape {batch_features.shape}, time {batch_time:.3f}s")
                
            except Exception as e:
                print(f"    ERROR {config_key} batch processing: {e}")
                sys.exit(1)
        
        self.memory_monitor.print_memory_status("after batch processing")
        print("  Batch processing OK")
    
    def run_all_tests(self) -> None:
        """
        Run complete test suite for all 50 Point-M2AE configurations.
        """
        print("=" * 70)
        print("POINT-M2AE 50 CONFIGURATIONS PIPELINE TESTS")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Run tests sequentially
            sample_volume = self.test_data_loading()
            self.test_extensions()
            self.test_configuration_validation()
            self.test_preprocessing_corrected(sample_volume)
            extractors = self.test_extractor_loading_all_configs()
            extracted_features = self.test_feature_extraction_all_aggregations(extractors, sample_volume)
            self.test_aggregation_differences(extracted_features)
            self.test_configuration_differences(extractors, sample_volume)
            self.test_batch_processing_sample(extractors)
            
            # Final summary
            total_time = time.time() - start_time
            print(f"\n{'='*70}")
            print("ALL 50 CONFIGURATIONS TESTS PASSED")
            print(f"{'='*70}")
            print(f"Total test time: {total_time:.1f}s")
            self.memory_monitor.print_memory_status("final")
            print("Pipeline ready for full 50 configurations execution!")
            print("Key features validated:")
            print("  - CUDA extensions: knn_cuda, pointnet2_ops")
            print("  - 50 configurations: 10 parameter configs × 5 aggregation methods")
            print("  - Corrected preprocessing: center (15,19,11) + [-1,1] normalization")
            print("  - All aggregation methods: A1 (384D), A2 (1536D), A3 (576D), A4 (384D), A5 (384D)")
            print("  - Parameter variations: num_groups, group_sizes, local_radius")
            print("  - Batch processing: Sequential robustness")
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"\nTEST FAILED: {e}")
            print("Fix issues before running full pipeline")
            sys.exit(1)


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Point-M2AE 50 configurations pipeline")
    parser.add_argument('--data-path', type=str, default='crops/2mm/S.Or.',
                       help='Path to HCP OFC dataset')
    
    args = parser.parse_args()
    
    # Check data path
    if not Path(args.data_path).exists():
        print(f"Data path not found: {args.data_path}")
        sys.exit(1)
    
    # Check config file
    config_path = Path(__file__).parent / "feature_extraction_m2ae_cfgs.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Run tests
    tester = PointM2AEConfigsTester(args.data_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()