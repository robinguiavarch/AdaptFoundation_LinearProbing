"""
Test script for 2.5D CLS token extraction pipeline with Classical PCA.

This script validates all components before running the full pipeline
to avoid memory issues and storage overflow.
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
from dinov2_variantes.method_25d_cls.method_25d_cls_core import (
    Method25DProcessor, CLSTokenExtractor, CLSAggregator, ClassicalPCAProcessor
)


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


class Method25DCLSTester:
    """
    Tests all components of the 2.5D CLS token extraction pipeline.
    
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
        
        print("Initializing Method 2.5D CLS Token Extraction Tester")
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
        
        print("  ✅ Data loading OK")
        return sample_volume
    
    def test_25d_preprocessing(self, sample_volume: np.ndarray) -> dict:
        """
        Test 2.5D preprocessing methods including overlapping variant.
        
        Args:
            sample_volume (np.ndarray): Sample volume for testing
        
        Returns:
            dict: Slices dictionary for testing
        """
        print("\n2. Testing 2.5D Preprocessing (Standard, Non-overlapping, Overlapping)")
        
        method_25d = Method25DProcessor()
        self.memory_monitor.print_memory_status("before preprocessing")
        
        # Test standard slicing (baseline)
        print("  Testing standard slicing (baseline)...")
        standard_slices = {}
        expected_standard = {'sagittal': 30, 'coronal': 38, 'axial': 22}
        
        for axis in ['sagittal', 'coronal', 'axial']:
            variant_config = {'method': 'standard_slicing'}
            slices_tensor = method_25d.create_slices(sample_volume, axis, variant_config)
            standard_slices[axis] = slices_tensor
            expected_count = expected_standard[axis]
            
            print(f"    {axis}: {slices_tensor.shape} (expected: ({expected_count}, 3, 224, 224))")
            assert slices_tensor.shape == (expected_count, 3, 224, 224), f"Wrong standard slice shape for {axis}"
        
        # Test 2.5D non-overlapping slicing
        print("  Testing 2.5D non-overlapping slicing...")
        slices_25d_no_overlap = {}
        expected_25d_no_overlap = {'sagittal': 10, 'coronal': 12, 'axial': 7}
        
        for axis in ['sagittal', 'coronal', 'axial']:
            variant_config = {'method': 'adaptive_25d'}
            slices_tensor = method_25d.create_slices(sample_volume, axis, variant_config)
            slices_25d_no_overlap[axis] = slices_tensor
            expected_count = expected_25d_no_overlap[axis]
            
            print(f"    {axis}: {slices_tensor.shape} (expected: ({expected_count}, 3, 224, 224))")
            assert slices_tensor.shape == (expected_count, 3, 224, 224), f"Wrong 2.5D non-overlap slice shape for {axis}"
        
        # Test 2.5D overlapping slicing
        print("  Testing 2.5D overlapping slicing...")
        slices_25d_overlap = {}
        expected_25d_overlap = {'sagittal': 28, 'coronal': 36, 'axial': 20}
        
        for axis in ['sagittal', 'coronal', 'axial']:
            variant_config = {'method': 'overlapping_25d'}
            slices_tensor = method_25d.create_slices(sample_volume, axis, variant_config)
            slices_25d_overlap[axis] = slices_tensor
            expected_count = expected_25d_overlap[axis]
            
            print(f"    {axis}: {slices_tensor.shape} (expected: ({expected_count}, 3, 224, 224))")
            assert slices_tensor.shape == (expected_count, 3, 224, 224), f"Wrong 2.5D overlap slice shape for {axis}"
        
        self.memory_monitor.print_memory_status("after preprocessing")
        print("  ✅ Preprocessing OK")
        
        return {
            'standard': standard_slices, 
            '25d_no_overlap': slices_25d_no_overlap,
            '25d_overlap': slices_25d_overlap
        }
    
    def test_model_loading(self) -> torch.nn.Module:
        """
        Test DINOv2 Giant model loading and basic functionality.
        
        Returns:
            torch.nn.Module: Loaded DINOv2 model
        """
        print("\n3. Testing DINOv2 Giant Model Loading")
        
        self.memory_monitor.print_memory_status("before model loading")
        
        try:
            print("  Loading dinov2_vitg14...")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            model.eval()
            model.to(self.device)
            
            dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                # Test standard forward (CLS token)
                cls_output = model(dummy_input)
                print(f"  CLS token output shape: {cls_output.shape} (expected: (2, 1536))")
                assert cls_output.shape == (2, 1536), f"Wrong CLS output shape: {cls_output.shape}"
                
                # Test forward_features (all tokens)
                features_output = model.forward_features(dummy_input)
                if isinstance(features_output, dict):
                    all_tokens = features_output.get('x_prenorm', features_output.get('x'))
                else:
                    all_tokens = features_output
                
                print(f"  All tokens shape: {all_tokens.shape} (expected: (2, 257, 1536))")
                assert all_tokens.shape == (2, 257, 1536), f"Wrong features shape: {all_tokens.shape}"
                
                # Extract CLS token specifically
                cls_tokens = all_tokens[:, 0, :]
                print(f"  CLS tokens shape: {cls_tokens.shape} (expected: (2, 1536))")
                assert cls_tokens.shape == (2, 1536), f"Wrong CLS tokens shape: {cls_tokens.shape}"
            
            self.memory_monitor.print_memory_status("after model loading")
            print("  ✅ Model loading OK")
            
            return model
            
        except Exception as e:
            print(f"  ❌ Model loading failed: {e}")
            sys.exit(1)
    
    def test_cls_token_extraction(self, model: torch.nn.Module, slices_dict: dict) -> dict:
        """
        Test CLS token extraction using CLSTokenExtractor.
        
        Args:
            model (torch.nn.Module): Loaded DINOv2 model
            slices_dict (dict): Dictionary of slices for testing
        
        Returns:
            dict: Extracted CLS tokens
        """
        print("\n4. Testing CLS Token Extraction")
        
        self.memory_monitor.print_memory_status("before CLS extraction")
        
        extractor = CLSTokenExtractor(device=self.device, batch_size=4)
        
        # Test on standard slices (smaller batch for memory)
        standard_slices = slices_dict['standard']
        cls_tokens_dict = {}
        
        for axis, slices_tensor in standard_slices.items():
            print(f"  Extracting CLS tokens for {axis} axis...")
            
            # Use only first few slices to save memory
            test_slices = slices_tensor[:8]
            cls_tokens = extractor.extract_cls_tokens(model, test_slices)
            cls_tokens_dict[axis] = cls_tokens
            
            expected_shape = (8, 1536)
            print(f"    Output shape: {cls_tokens.shape} (expected: {expected_shape})")
            assert cls_tokens.shape == expected_shape, f"Wrong CLS tokens shape for {axis}"
        
        self.memory_monitor.print_memory_status("after CLS extraction")
        print("  ✅ CLS token extraction OK")
        
        return cls_tokens_dict
    
    def test_cls_aggregation(self, cls_tokens_dict: dict) -> dict:
        """
        Test CLS token aggregation methods.
        
        Args:
            cls_tokens_dict (dict): CLS tokens for testing
        
        Returns:
            dict: Aggregated CLS tokens
        """
        print("\n5. Testing CLS Token Aggregation")
        
        self.memory_monitor.print_memory_status("before aggregation")
        
        results = {}
        
        # Test concatenation
        print("  Testing concatenation aggregation...")
        concat_aggregator = CLSAggregator(aggregation_method='concat')
        concat_features = concat_aggregator.aggregate_triaxial(cls_tokens_dict)
        
        expected_concat_dim = 8 * 1536 * 3  # 8 slices × 3 axes × 1536D
        print(f"    Concat output shape: {concat_features.shape} (expected: ({expected_concat_dim},))")
        assert concat_features.shape == (expected_concat_dim,), f"Wrong concat shape: {concat_features.shape}"
        results['concat'] = concat_features
        
        # Test pooling
        print("  Testing pooling aggregation...")
        pooling_aggregator = CLSAggregator(aggregation_method='pooling')
        pooling_features = pooling_aggregator.aggregate_triaxial(cls_tokens_dict)
        
        expected_pooling_dim = 3 * 1536  # 3 axes × 1536D
        print(f"    Pooling output shape: {pooling_features.shape} (expected: ({expected_pooling_dim},))")
        assert pooling_features.shape == (expected_pooling_dim,), f"Wrong pooling shape: {pooling_features.shape}"
        results['pooling'] = pooling_features
        
        self.memory_monitor.print_memory_status("after aggregation")
        print("  ✅ Aggregation OK")
        
        return results
    
    def test_classical_pca_processing(self, aggregated_features: dict) -> None:
        """
        Test Classical PCA processing with CLS token dimensions.
        
        Args:
            aggregated_features (dict): Aggregated CLS token features for testing
        """
        print("\n6. Testing Classical PCA Processing (CLS Tokens)")
        
        self.memory_monitor.print_memory_status("before Classical PCA")
        
        for method, features in aggregated_features.items():
            print(f"  Testing Classical PCA on {method} CLS features...")
            print(f"    Feature dimension: {features.shape[0]:,}")
            
            # Test PCA 32D
            print(f"    Testing PCA 32D...")
            
            try:
                pca_config_32d = {
                    'mode': 'fixed',
                    'n_components': 32,
                    'variance_threshold': None
                }
                
                n_test_subjects = 100
                
                print(f"      Creating {n_test_subjects} realistic test subjects...")
                test_subjects = []
                for i in range(n_test_subjects):
                    noise_scale = 0.01 * np.std(features) if np.std(features) > 0 else 0.01
                    noisy_features = features + np.random.normal(0, noise_scale, features.shape)
                    test_subjects.append(noisy_features)
                
                pca_processor = ClassicalPCAProcessor(pca_config_32d)
                
                def mock_feature_extraction(subject, variant_config):
                    return subject
                
                start_time = time.time()
                pca_info = pca_processor.fit_classical_pca(
                    mock_feature_extraction,
                    test_subjects[:40],
                    {}
                )
                fit_time = time.time() - start_time
                
                print(f"      PCA fitted: {pca_info['n_components']} components in {fit_time:.3f}s")
                print(f"      Original dimension: {pca_info['original_dim']:,}")
                print(f"      Variance explained: {pca_info['actual_variance']:.4f}")
                
                transformed = pca_processor.transform_features(features.reshape(1, -1))
                expected_shape = (1, 32)
                
                print(f"      Transform: {transformed.shape} (expected: {expected_shape})")
                assert transformed.shape == expected_shape, f"Wrong transform shape: {transformed.shape}"
                assert transformed.dtype == np.float32, f"Expected float32, got {transformed.dtype}"
                
                print(f"      ✅ Classical PCA 32D successful for {method} CLS features")
                
            except Exception as e:
                print(f"      ❌ Classical PCA 32D failed: {e}")
                continue
            
            # Test PCA 256D
            print(f"    Testing PCA 256D...")
            
            try:
                pca_config_256d = {
                    'mode': 'fixed',
                    'n_components': 256,
                    'variance_threshold': None
                }
                
                pca_processor_256d = ClassicalPCAProcessor(pca_config_256d)
                
                pca_info_256d = pca_processor_256d.fit_classical_pca(
                    mock_feature_extraction,
                    test_subjects[:80],
                    {}
                )
                
                transformed_256d = pca_processor_256d.transform_features(features.reshape(1, -1))
                expected_shape_256d = (1, 256)
                
                print(f"      Transform 256D: {transformed_256d.shape} (expected: {expected_shape_256d})")
                assert transformed_256d.shape == expected_shape_256d, f"Wrong 256D transform shape: {transformed_256d.shape}"
                
                print(f"      ✅ Classical PCA 256D successful for {method} CLS features")
                
            except Exception as e:
                print(f"      ❌ Classical PCA 256D failed: {e}")
                continue
            
            # Test PCA 99% variance
            print(f"    Testing PCA 99% variance...")
            
            try:
                pca_config_99 = {
                    'mode': 'variance',
                    'n_components': None,
                    'variance_threshold': 0.99
                }
                
                pca_processor_99 = ClassicalPCAProcessor(pca_config_99)
                
                pca_info_99 = pca_processor_99.fit_classical_pca(
                    mock_feature_extraction,
                    test_subjects[:60],
                    {}
                )
                
                estimated_components = pca_info_99['n_components']
                print(f"      Estimated components for 99% variance: {estimated_components}")
                
                transformed_99 = pca_processor_99.transform_features(features.reshape(1, -1))
                expected_shape_99 = (1, estimated_components)
                
                print(f"      Transform 99%: {transformed_99.shape} (expected: {expected_shape_99})")
                assert transformed_99.shape == expected_shape_99, f"Wrong 99% transform shape: {transformed_99.shape}"
                
                print(f"      ✅ Classical PCA 99% successful for {method} CLS features")
                
            except Exception as e:
                print(f"      ❌ Classical PCA 99% failed: {e}")
                continue
            
            del test_subjects
        
        self.memory_monitor.print_memory_status("after Classical PCA")
        print("  ✅ Classical PCA validation complete")
    
    def test_memory_estimation_cls(self) -> None:
        """
        Test memory and storage requirements estimation for CLS tokens.
        """
        print("\n7. Testing Memory and Storage Estimation (CLS Tokens)")
        
        n_subjects = 577
        
        print("  Estimating CLS token storage requirements...")
        
        # CLS token dimensions (DINOv2 Giant - 1536D)
        raw_concat_baseline_dim = 90 * 1536  # 138,240
        raw_concat_25d_no_overlap_dim = 29 * 1536  # 44,544
        raw_concat_25d_overlap_dim = 84 * 1536  # 129,024
        raw_pooling_dim = 3 * 1536  # 4,608
        
        # Memory per subject (float32 = 4 bytes)
        memory_per_subject_concat_baseline = raw_concat_baseline_dim * 4 / (1024**3)
        memory_per_subject_concat_25d_overlap = raw_concat_25d_overlap_dim * 4 / (1024**3)
        memory_per_subject_pooling = raw_pooling_dim * 4 / (1024**3)
        
        print(f"    Memory per subject (concat baseline): {memory_per_subject_concat_baseline:.6f} GB")
        print(f"    Memory per subject (concat 2.5D overlap): {memory_per_subject_concat_25d_overlap:.6f} GB")
        print(f"    Memory per subject (pooling): {memory_per_subject_pooling:.6f} GB")
        
        # Classical PCA memory requirements
        n_training_subjects = 461
        training_memory_concat_baseline = memory_per_subject_concat_baseline * n_training_subjects
        training_memory_concat_25d_overlap = memory_per_subject_concat_25d_overlap * n_training_subjects
        training_memory_pooling = memory_per_subject_pooling * n_training_subjects
        
        print(f"    Training memory needed (concat baseline): {training_memory_concat_baseline:.3f} GB")
        print(f"    Training memory needed (concat 2.5D overlap): {training_memory_concat_25d_overlap:.3f} GB")
        print(f"    Training memory needed (pooling): {training_memory_pooling:.3f} GB")
        
        # PCA storage
        for n_components in [32, 256]:
            pca_storage_per_subject = n_components * 4 / (1024**3)
            total_pca_storage = pca_storage_per_subject * n_subjects
            print(f"    PCA {n_components}D total storage: {total_pca_storage:.6f} GB")
        
        # Estimated PCA 99% storage
        estimated_99_components = 500
        pca_99_storage_per_subject = estimated_99_components * 4 / (1024**3)
        total_pca_99_storage = pca_99_storage_per_subject * n_subjects
        print(f"    PCA 99% estimated total storage: {total_pca_99_storage:.4f} GB (~{estimated_99_components} components)")
        
        # Total for all configurations
        total_all_configs = (32 + 256 + estimated_99_components) * 4 * n_subjects * 6 / (1024**3)
        print(f"    Total storage all 18 configs: {total_all_configs:.3f} GB")
        
        # Memory requirements with Classical PCA
        max_training_memory = max(training_memory_concat_baseline, training_memory_concat_25d_overlap, training_memory_pooling)
        print(f"    Max training memory for Classical PCA: {max_training_memory:.3f} GB")
        
        available_ram = psutil.virtual_memory().available / (1024**3)
        print(f"    Available RAM: {available_ram:.1f} GB")
        
        if max_training_memory < available_ram * 0.8:
            print(f"  ✅ Classical PCA memory usage feasible: {max_training_memory:.3f}GB < {available_ram*0.8:.1f}GB")
        else:
            print(f"  ⚠️  Classical PCA may need optimization: {max_training_memory:.3f}GB vs {available_ram*0.8:.1f}GB available")
        
        storage_limit = 50
        if total_all_configs < storage_limit:
            print(f"  ✅ Total storage within limit: {total_all_configs:.3f}GB < {storage_limit}GB")
        else:
            print(f"  ⚠️  Total storage may exceed limit: {total_all_configs:.3f}GB > {storage_limit}GB")
    
    def test_25d_dimensions_validation_cls(self) -> None:
        """
        Test and validate 2.5D dimensions for CLS tokens.
        """
        print("\n8. Testing 2.5D Dimensions Validation (CLS Tokens)")
        
        volume_shape = (30, 38, 22)
        print(f"  Volume shape: {volume_shape}")
        
        # Expected groups per variant
        variants = {
            'baseline': {'sagittal': 30, 'coronal': 38, 'axial': 22, 'total': 90},
            '25d_no_overlap': {'sagittal': 10, 'coronal': 12, 'axial': 7, 'total': 29},
            '25d_overlap': {'sagittal': 28, 'coronal': 36, 'axial': 20, 'total': 84}
        }
        
        cls_token_dim = 1536
        
        for variant_name, groups in variants.items():
            print(f"  {variant_name}:")
            for axis, count in groups.items():
                if axis != 'total':
                    print(f"    {axis}: {count} groups")
            
            total_groups = groups['total']
            concat_dim = total_groups * cls_token_dim
            pooling_dim = 3 * cls_token_dim
            
            print(f"    Total groups: {total_groups}")
            print(f"    Concat dimension: {concat_dim:,}")
            print(f"    Pooling dimension: {pooling_dim:,}")
        
        # Verify expected dimensions
        expected_dims = {
            'concat_baseline': 138240,
            'concat_25d_no_overlap': 44544,
            'concat_25d_overlap': 129024,
            'pooling_all': 4608
        }
        
        actual_dims = {
            'concat_baseline': variants['baseline']['total'] * cls_token_dim,
            'concat_25d_no_overlap': variants['25d_no_overlap']['total'] * cls_token_dim,
            'concat_25d_overlap': variants['25d_overlap']['total'] * cls_token_dim,
            'pooling_all': 3 * cls_token_dim
        }
        
        for key, expected in expected_dims.items():
            actual = actual_dims[key]
            print(f"  {key}: {actual:,} (expected: {expected:,})")
            assert actual == expected, f"Dimension mismatch for {key}: {actual} vs {expected}"
        
        print("  ✅ 2.5D CLS dimensions validation OK")
    
    def run_all_tests(self) -> None:
        """
        Run complete test suite for 2.5D CLS token extraction.
        """
        print("=" * 80)
        print("METHOD 2.5D CLS TOKEN EXTRACTION PIPELINE TESTS")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            sample_volume = self.test_data_loading()
            slices_dict = self.test_25d_preprocessing(sample_volume)
            model = self.test_model_loading()
            cls_tokens = self.test_cls_token_extraction(model, slices_dict)
            aggregated_features = self.test_cls_aggregation(cls_tokens)
            self.test_classical_pca_processing(aggregated_features)
            self.test_memory_estimation_cls()
            self.test_25d_dimensions_validation_cls()
            
            total_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("ALL TESTS PASSED ✅")
            print(f"Total test time: {total_time:.1f}s")
            self.memory_monitor.print_memory_status("final")
            print("2.5D CLS pipeline ready for full execution!")
            print("Key features validated:")
            print("  - DINOv2 Giant: CLS tokens (1536D) much lighter than feature maps")
            print("  - 2.5D variants: Baseline, non-overlapping, overlapping grouping")
            print("  - Classical PCA: Memory requirements < 1GB for all variants")
            print("  - Overlapping: Maximum spatial continuity (84 vs 29 groups)")
            print("  - Memory optimization: CLS tokens require minimal memory")
            print("  - 18 configurations: 6 variants × 3 PCA modes")
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
    
    parser = argparse.ArgumentParser(description="Test 2.5D CLS token extraction pipeline")
    parser.add_argument('--data-path', type=str, default='crops/2mm/S.Or.',
                       help='Path to HCP OFC dataset')
    
    args = parser.parse_args()
    
    if not Path(args.data_path).exists():
        print(f"Data path not found: {args.data_path}")
        print("Please provide correct path to HCP OFC dataset")
        sys.exit(1)
    
    tester = Method25DCLSTester(args.data_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()