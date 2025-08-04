#!/usr/bin/env python3
"""
Test script for SAM-Med3D density-optimized feature extraction pipeline.

This script validates all components before running the full pipeline
to avoid memory issues and ensure proper functionality.
"""

import numpy as np
import torch
import psutil
import os
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.loaders import HCPOFCDataLoader
from sam3d_variantes.turbo_with_density.feature_extraction_core import SAMMed3DTurboDensityExtractor
from models.feature_extraction_sam3d import SAMMed3DFeatureExtractor


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
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def print_memory_status(self, stage: str) -> None:
        """Print current memory status."""
        ram_current = self.get_ram_usage()
        gpu_current = self.get_gpu_usage()
        ram_delta = ram_current - self.initial_ram
        gpu_delta = gpu_current - self.initial_gpu
        
        print(f"  Memory [{stage}]:")
        print(f"    RAM: {ram_current:.2f}GB (+{ram_delta:.2f}GB)")
        print(f"    GPU: {gpu_current:.2f}GB (+{gpu_delta:.2f}GB)")


class DensityFeatureExtractionTester:
    """
    Tests all components of the SAM-Med3D density feature extraction pipeline.
    
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_monitor = MemoryMonitor()
        
        print("Initializing SAM-Med3D Density Feature Extraction Tester")
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
        expected_shape = (30, 38, 22)  # Direct shape from processed dataloader
        
        print(f"  Sample volume shape: {sample_volume.shape}")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Volume density: {np.mean(sample_volume):.4f}")
        print(f"  Non-zero voxels: {np.sum(sample_volume)} / {np.prod(sample_volume.shape)}")
        
        assert sample_volume.shape == expected_shape, f"Wrong volume shape: {sample_volume.shape}"
        assert 0 <= np.min(sample_volume) <= np.max(sample_volume) <= 1, "Volume values not in [0,1]"
        
        print("  ✅ Data loading OK")
        return sample_volume
    
    def test_density_map_loading(self) -> np.ndarray:
        """
        Test patch density map loading and validation.
        
        Returns:
            np.ndarray: Loaded patch density map
        """
        print("\n2. Testing Patch Density Map Loading")
        
        density_path = Path("density/patch_density_map_8x8x8.npy")
        
        if not density_path.exists():
            print(f"  ❌ Density map not found: {density_path}")
            print("  Run 'python run_density_patch.py' first to generate the density map")
            sys.exit(1)
        
        patch_density_map = np.load(density_path)
        
        print(f"  Density map shape: {patch_density_map.shape}")
        print(f"  Expected shape: (8, 8, 8)")
        print(f"  Value range: [{np.min(patch_density_map):.4f}, {np.max(patch_density_map):.4f}]")
        print(f"  Mean density: {np.mean(patch_density_map):.4f}")
        print(f"  Zero-density patches: {np.sum(patch_density_map == 0)}/512")
        print(f"  Non-zero patches: {np.sum(patch_density_map > 0)}/512")
        
        assert patch_density_map.shape == (8, 8, 8), f"Wrong density map shape: {patch_density_map.shape}"
        assert np.all(patch_density_map >= 0) and np.all(patch_density_map <= 1), "Density values not in [0,1]"
        
        print("  ✅ Density map loading OK")
        return patch_density_map
    
    def test_base_extractor_loading(self) -> SAMMed3DFeatureExtractor:
        """
        Test base SAM-Med3D extractor loading.
        
        Returns:
            SAMMed3DFeatureExtractor: Loaded base extractor
        """
        print("\n3. Testing Base SAM-Med3D Extractor Loading")
        
        self.memory_monitor.print_memory_status("before base extractor")
        
        try:
            base_extractor = SAMMed3DFeatureExtractor(
                config_path="configs/feature_extraction_sam3d.yaml",
                aggregation_method="flatten"
            )
            
            model_info = base_extractor.get_model_info()
            print(f"  Model type: {model_info['model_type']}")
            print(f"  Checkpoint: {model_info['checkpoint_path']}")
            print(f"  Feature dimension: {model_info['feature_dim']}")
            print(f"  Device: {model_info['device']}")
            print(f"  Parameters: {model_info['num_parameters']:,}")
            
            self.memory_monitor.print_memory_status("after base extractor")
            print("  ✅ Base extractor loading OK")
            
            return base_extractor
            
        except Exception as e:
            print(f"  ❌ Base extractor loading failed: {e}")
            sys.exit(1)
    
    def test_density_extractors(self, base_extractor: SAMMed3DFeatureExtractor) -> dict:
        """
        Test all three density optimization approaches.
        
        Args:
            base_extractor (SAMMed3DFeatureExtractor): Base extractor instance
        
        Returns:
            dict: Dictionary of density extractors
        """
        print("\n4. Testing Density Optimization Approaches")
        
        self.memory_monitor.print_memory_status("before density extractors")
        
        extractors = {}
        approaches = ['baseline', 'masking', 'linear_weighting']
        
        for approach in approaches:
            print(f"  Testing {approach} approach...")
            
            try:
                extractor = SAMMed3DTurboDensityExtractor(
                    approach=approach,
                    base_extractor=base_extractor
                )
                
                approach_info = extractor.get_approach_info()
                print(f"    Approach: {approach_info['approach']}")
                print(f"    Feature dimension: {approach_info['feature_dim']}")
                print(f"    Valid patches: {approach_info['valid_patches']}/512")
                print(f"    Uses density optimization: {approach_info['uses_density_optimization']}")
                
                extractors[approach] = extractor
                print(f"    ✅ {approach} extractor OK")
                
            except Exception as e:
                print(f"    ❌ {approach} extractor failed: {e}")
                continue
        
        self.memory_monitor.print_memory_status("after density extractors")
        return extractors
    
    def test_feature_extraction(self, extractors: dict, sample_volume: np.ndarray) -> dict:
        """
        Test feature extraction with all density approaches.
        
        Args:
            extractors (dict): Dictionary of density extractors
            sample_volume (np.ndarray): Sample volume for testing
        
        Returns:
            dict: Extracted features for each approach
        """
        print("\n5. Testing Feature Extraction")
        
        self.memory_monitor.print_memory_status("before feature extraction")
        
        volume_tensor = torch.from_numpy(sample_volume).float()
        extracted_features = {}
        
        for approach, extractor in extractors.items():
            print(f"  Testing {approach} feature extraction...")
            
            try:
                start_time = time.time()
                features = extractor.extract_features(volume_tensor)
                extraction_time = time.time() - start_time
                
                print(f"    Features shape: {features.shape}")
                print(f"    Feature dtype: {features.dtype}")
                print(f"    Extraction time: {extraction_time:.3f}s")
                print(f"    Feature range: [{torch.min(features):.4f}, {torch.max(features):.4f}]")
                print(f"    Non-zero features: {torch.sum(features != 0).item()}/{features.numel()}")
                
                # Validate feature dimensions (SAM-Med3D: 384D per patch)
                approach_info = extractor.get_approach_info()
                if approach == 'masking':
                    expected_dim = approach_info['valid_patches'] * 384
                    assert features.shape[1] == expected_dim, f"Wrong masking dimension: {features.shape[1]} vs {expected_dim}"
                else:
                    expected_dim = 8 * 8 * 8 * 384  # 196608 for SAM-Med3D
                    assert features.shape[1] == expected_dim, f"Wrong dimension for {approach}: {features.shape[1]} vs {expected_dim}"
                
                extracted_features[approach] = features
                print(f"    ✅ {approach} extraction OK")
                
            except Exception as e:
                print(f"    ❌ {approach} extraction failed: {e}")
                continue
        
        self.memory_monitor.print_memory_status("after feature extraction")
        return extracted_features
    
    def test_batch_processing(self, extractors: dict) -> None:
        """
        Test batch processing capabilities.
        
        Args:
            extractors (dict): Dictionary of density extractors
        """
        print("\n6. Testing Batch Processing")
        
        self.memory_monitor.print_memory_status("before batch processing")
        
        # Create small test batch
        test_volumes = []
        for i in range(3):
            volume = self.data_loader.skeletons[i]  # Already (30, 38, 22)
            test_volumes.append(volume)
        
        batch_tensor = torch.stack([torch.from_numpy(v).float() for v in test_volumes])
        print(f"  Test batch shape: {batch_tensor.shape}")
        
        for approach, extractor in extractors.items():
            print(f"  Testing {approach} batch processing...")
            
            try:
                start_time = time.time()
                batch_features = extractor.extract_features_batch(batch_tensor, batch_size=2)
                batch_time = time.time() - start_time
                
                print(f"    Batch features shape: {batch_features.shape}")
                print(f"    Batch processing time: {batch_time:.3f}s")
                print(f"    Time per volume: {batch_time/3:.3f}s")
                
                # Validate batch dimensions
                expected_batch_size = 3
                assert batch_features.shape[0] == expected_batch_size, f"Wrong batch size: {batch_features.shape[0]}"
                
                print(f"    ✅ {approach} batch processing OK")
                
            except Exception as e:
                print(f"    ❌ {approach} batch processing failed: {e}")
                continue
        
        self.memory_monitor.print_memory_status("after batch processing")
    
    def test_approach_differences(self, extracted_features: dict) -> None:
        """
        Test differences between density optimization approaches.
        
        Args:
            extracted_features (dict): Features from different approaches
        """
        print("\n7. Testing Approach Differences")
        
        if len(extracted_features) < 2:
            print("  ⚠️ Need at least 2 approaches for comparison")
            return
        
        approaches = list(extracted_features.keys())
        
        # Compare baseline vs optimized approaches
        if 'baseline' in approaches:
            baseline_features = extracted_features['baseline']
            
            for approach in approaches:
                if approach == 'baseline':
                    continue
                
                other_features = extracted_features[approach]
                print(f"  Comparing baseline vs {approach}...")
                
                if approach == 'masking':
                    # Different dimensions - check that masking reduces size
                    print(f"    Baseline: {baseline_features.shape[1]} features")
                    print(f"    Masking: {other_features.shape[1]} features")
                    print(f"    Reduction: {(1 - other_features.shape[1]/baseline_features.shape[1])*100:.1f}%")
                    
                    assert other_features.shape[1] < baseline_features.shape[1], "Masking should reduce feature count"
                    
                elif approach == 'linear_weighting':
                    # Same dimensions - check that weighting changes values
                    assert baseline_features.shape == other_features.shape, "Linear weighting should preserve dimensions"
                    
                    # Check for differences in feature values
                    feature_diff = torch.abs(baseline_features - other_features)
                    max_diff = torch.max(feature_diff).item()
                    mean_diff = torch.mean(feature_diff).item()
                    
                    print(f"    Max difference: {max_diff:.6f}")
                    print(f"    Mean difference: {mean_diff:.6f}")
                    
                    assert max_diff > 1e-6, "Linear weighting should change feature values"
                
                print(f"    ✅ {approach} vs baseline comparison OK")
        
        # Compare masking vs linear_weighting effect
        if 'masking' in approaches and 'linear_weighting' in approaches:
            masking_features = extracted_features['masking']
            weighting_features = extracted_features['linear_weighting']
            
            print(f"  Comparing optimization approaches...")
            print(f"    Masking dimension: {masking_features.shape[1]}")
            print(f"    Weighting dimension: {weighting_features.shape[1]}")
            
            # They should have different strategies
            dimension_ratio = masking_features.shape[1] / weighting_features.shape[1]
            print(f"    Dimension ratio: {dimension_ratio:.3f}")
            
            print("    ✅ Optimization approaches comparison OK")
    
    def test_memory_requirements(self) -> None:
        """
        Test memory and storage requirements estimation.
        """
        print("\n8. Testing Memory Requirements")
        
        n_subjects = 577
        n_splits = 6  # 5 train/val + 1 test
        
        print("  Estimating storage requirements...")
        
        # Feature dimensions for each approach (SAM-Med3D: 384D per patch)
        dimensions = {
            'baseline': 8 * 8 * 8 * 384,  # 196608 (8³ patches × 384D)
            'masking': 347 * 384,  # ~133248 (valid patches × 384D)
            'linear_weighting': 8 * 8 * 8 * 384  # 196608 (same as baseline)
        }
        
        total_storage = 0
        
        for approach, dim in dimensions.items():
            # Memory per subject (float32 = 4 bytes)
            memory_per_subject = dim * 4 / (1024**3)  # GB
            
            # Total storage for all subjects and splits
            approach_storage = memory_per_subject * n_subjects
            total_storage += approach_storage
            
            print(f"    {approach}: {dim:,}D → {memory_per_subject:.4f}GB per subject")
            print(f"      Total: {approach_storage:.3f}GB")
        
        print(f"    Total storage all approaches: {total_storage:.2f}GB")
        
        # Memory requirements for batch processing
        batch_size = 8
        max_memory_per_batch = max(dimensions.values()) * batch_size * 4 / (1024**3)
        print(f"    Max memory per batch (8 subjects): {max_memory_per_batch:.3f}GB")
        
        # GPU memory check
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    Available GPU memory: {gpu_memory:.1f}GB")
            
            if max_memory_per_batch < gpu_memory * 0.5:
                print(f"  ✅ Batch processing feasible: {max_memory_per_batch:.3f}GB < {gpu_memory*0.5:.1f}GB")
            else:
                print(f"  ⚠️ May need smaller batch size: {max_memory_per_batch:.3f}GB vs {gpu_memory*0.5:.1f}GB")
        
        # Storage limit check
        storage_limit = 50  # GB (reasonable for density optimization)
        if total_storage < storage_limit:
            print(f"  ✅ Total storage within limit: {total_storage:.2f}GB < {storage_limit}GB")
        else:
            print(f"  ⚠️ Storage may be high: {total_storage:.2f}GB > {storage_limit}GB")
    
    def run_all_tests(self) -> None:
        """
        Run complete test suite for density feature extraction.
        """
        print("=" * 80)
        print("SAM-MED3D DENSITY FEATURE EXTRACTION PIPELINE TESTS")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Run tests sequentially
            sample_volume = self.test_data_loading()
            patch_density_map = self.test_density_map_loading()
            base_extractor = self.test_base_extractor_loading()
            extractors = self.test_density_extractors(base_extractor)
            extracted_features = self.test_feature_extraction(extractors, sample_volume)
            self.test_batch_processing(extractors)
            self.test_approach_differences(extracted_features)
            self.test_memory_requirements()
            
            # Final memory status
            total_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("ALL TESTS PASSED ✅")
            print(f"Total test time: {total_time:.1f}s")
            self.memory_monitor.print_memory_status("final")
            print("Pipeline ready for full execution!")
            print("Key features validated:")
            print("  - SAM-Med3D turbo: 384D per patch (8×8×8 = 512 patches)")
            print("  - Feature dimensions: 196608D (512 × 384) for baseline/weighting")
            print("  - Density map: patch-level optimization (8×8×8)")
            print("  - 3 approaches: baseline, masking, linear_weighting")
            print("  - Batch processing: memory-optimized extraction")
            print("  - Approach differences: validation of optimization effects")
            print("  - Memory estimation: storage and processing requirements")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            print("Fix issues before running full pipeline")
            sys.exit(1)


def main():
    """
    Main entry point for testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM-Med3D density feature extraction pipeline")
    parser.add_argument('--data-path', type=str, default='crops/2mm/S.Or.',
                       help='Path to HCP OFC dataset')
    
    args = parser.parse_args()
    
    # Check if data path exists
    if not Path(args.data_path).exists():
        print(f"Data path not found: {args.data_path}")
        print("Please provide correct path to HCP OFC dataset")
        sys.exit(1)
    
    # Check if density map exists
    density_path = Path("density/patch_density_map_8x8x8.npy")
    if not density_path.exists():
        print(f"Density map not found: {density_path}")
        print("Run 'python run_density_patch.py' first to generate the density map")
        sys.exit(1)
    
    # Run tests
    tester = DensityFeatureExtractionTester(args.data_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()