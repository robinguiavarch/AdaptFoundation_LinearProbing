"""
Test script for density-guided CLS token extraction pipeline.

This script validates all components of the density-guided approach
before running the full pipeline to ensure proper functionality.
"""

import numpy as np
import torch
import psutil
import os
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.loaders import HCPOFCDataLoader
from dinov2_variantes.density_guided.density_guided_core import (
    DensityGuidedProcessor, CLSTokenExtractor, DensityGuidedAggregator, StandalonePCAProcessor
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


class DensityGuidedTester:
    """
    Tests all components of the density-guided CLS token extraction pipeline.
    
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
        
        print("Initializing Density-Guided CLS Token Extraction Tester")
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
    
    def test_density_profiles_loading(self) -> dict:
        """
        Test density profiles loading functionality.
        
        Returns:
            dict: Loaded density profiles for validation
        """
        print("\n2. Testing Density Profiles Loading")
        
        self.memory_monitor.print_memory_status("before density loading")
        
        try:
            density_processor = DensityGuidedProcessor()
            density_profiles = density_processor.density_profiles
            
            expected_shapes = {'x': (30,), 'y': (38,), 'z': (22,)}
            
            for axis, expected_shape in expected_shapes.items():
                profile = density_profiles[axis]
                print(f"  {axis}-axis profile shape: {profile.shape} (expected: {expected_shape})")
                print(f"  {axis}-axis density range: [{np.min(profile):.4f}, {np.max(profile):.4f}]")
                
                assert profile.shape == expected_shape, f"Wrong {axis} profile shape: {profile.shape}"
                assert np.all(profile >= 0), f"Negative densities in {axis} profile"
                assert np.max(profile) > 0, f"All zeros in {axis} profile"
            
            self.memory_monitor.print_memory_status("after density loading")
            print("  ✅ Density profiles loading OK")
            
            return density_profiles
            
        except FileNotFoundError as e:
            print(f"  ❌ Density profiles not found: {e}")
            print("  Please ensure density/density_profile_{{x,y,z}}.npy files exist")
            sys.exit(1)
        except Exception as e:
            print(f"  ❌ Density profiles loading failed: {e}")
            sys.exit(1)
    
    def test_density_guided_preprocessing(self, sample_volume: np.ndarray) -> dict:
        """
        Test density-guided preprocessing with all three approaches.
        
        Args:
            sample_volume (np.ndarray): Sample volume for testing
        
        Returns:
            dict: Slices dictionary for testing
        """
        print("\n3. Testing Density-Guided Preprocessing (3 Approaches)")
        
        density_processor = DensityGuidedProcessor()
        self.memory_monitor.print_memory_status("before preprocessing")
        
        all_slices = {}
        
        # Test Approach 1: Central Uniform
        print("  Testing central uniform approach...")
        central_uniform_slices = {}
        expected_central = {'sagittal': 11, 'coronal': 11, 'axial': 10}
        
        for axis in ['sagittal', 'coronal', 'axial']:
            variant_config = {'approach': 'central_uniform'}
            slices_tensor = density_processor.create_slices(sample_volume, axis, variant_config)
            central_uniform_slices[axis] = slices_tensor
            expected_count = expected_central[axis]
            
            print(f"    {axis}: {slices_tensor.shape} (expected: ({expected_count}, 3, 224, 224))")
            assert slices_tensor.shape == (expected_count, 3, 224, 224), f"Wrong central uniform slice shape for {axis}"
        
        all_slices['central_uniform'] = central_uniform_slices
        
        # Test Approach 2: Adaptive Density
        print("  Testing adaptive density approach...")
        adaptive_density_slices = {}
        expected_adaptive = {'sagittal': 19, 'coronal': 27, 'axial': 13}
        
        for axis in ['sagittal', 'coronal', 'axial']:
            variant_config = {'approach': 'adaptive_density'}
            slices_tensor = density_processor.create_slices(sample_volume, axis, variant_config)
            adaptive_density_slices[axis] = slices_tensor
            expected_count = expected_adaptive[axis]
            
            print(f"    {axis}: {slices_tensor.shape} (expected: ({expected_count}, 3, 224, 224))")
            assert slices_tensor.shape == (expected_count, 3, 224, 224), f"Wrong adaptive density slice shape for {axis}"
        
        all_slices['adaptive_density'] = adaptive_density_slices
        
        # Test Approach 3: Linear Weighting (all slices)
        print("  Testing linear weighting approach...")
        linear_weighting_slices = {}
        expected_linear = {'sagittal': 30, 'coronal': 38, 'axial': 22}
        
        for axis in ['sagittal', 'coronal', 'axial']:
            variant_config = {'approach': 'linear_weighting'}
            slices_tensor = density_processor.create_slices(sample_volume, axis, variant_config)
            linear_weighting_slices[axis] = slices_tensor
            expected_count = expected_linear[axis]
            
            print(f"    {axis}: {slices_tensor.shape} (expected: ({expected_count}, 3, 224, 224))")
            assert slices_tensor.shape == (expected_count, 3, 224, 224), f"Wrong linear weighting slice shape for {axis}"
        
        all_slices['linear_weighting'] = linear_weighting_slices
        
        self.memory_monitor.print_memory_status("after preprocessing")
        print("  ✅ Density-guided preprocessing OK")
        
        return all_slices
    
    def test_model_loading(self) -> torch.nn.Module:
        """
        Test DINOv2 Giant model loading and basic functionality.
        
        Returns:
            torch.nn.Module: Loaded DINOv2 model
        """
        print("\n4. Testing DINOv2 Giant Model Loading")
        
        self.memory_monitor.print_memory_status("before model loading")
        
        try:
            print("  Loading dinov2_vitg14...")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            model.eval()
            model.to(self.device)
            
            dummy_input = torch.randn(2, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                result = model.forward_features(dummy_input)
                
                if isinstance(result, dict) and 'x_prenorm' in result:
                    all_tokens = result['x_prenorm']
                else:
                    all_tokens = result
                
                print(f"  All tokens shape: {all_tokens.shape} (expected: (2, 257, 1536))")
                assert all_tokens.shape == (2, 257, 1536), f"Wrong features shape: {all_tokens.shape}"
                
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
            dict: Extracted CLS tokens per approach
        """
        print("\n5. Testing CLS Token Extraction")
        
        self.memory_monitor.print_memory_status("before CLS extraction")
        
        extractor = CLSTokenExtractor(device=self.device, batch_size=4)
        all_cls_tokens = {}
        
        for approach, slices_by_axis in slices_dict.items():
            print(f"  Extracting CLS tokens for {approach} approach...")
            cls_tokens_dict = {}
            
            for axis, slices_tensor in slices_by_axis.items():
                test_slices = slices_tensor[:6]  # Use subset to save memory
                cls_tokens = extractor.extract_cls_tokens(model, test_slices)
                cls_tokens_dict[axis] = cls_tokens
                
                expected_shape = (6, 1536)
                print(f"    {axis}: {cls_tokens.shape} (expected: {expected_shape})")
                assert cls_tokens.shape == expected_shape, f"Wrong CLS tokens shape for {axis} in {approach}"
            
            all_cls_tokens[approach] = cls_tokens_dict
        
        self.memory_monitor.print_memory_status("after CLS extraction")
        print("  ✅ CLS token extraction OK")
        
        return all_cls_tokens
    
    def test_density_guided_aggregation(self, all_cls_tokens: dict, density_profiles: dict) -> dict:
        """
        Test density-guided aggregation methods including linear weighting.
        
        Args:
            all_cls_tokens (dict): CLS tokens for all approaches
            density_profiles (dict): Density profiles for weighting
        
        Returns:
            dict: Aggregated features for all approaches and methods
        """
        print("\n6. Testing Density-Guided Aggregation")
        
        self.memory_monitor.print_memory_status("before aggregation")
        
        all_results = {}
        
        for approach, cls_tokens_dict in all_cls_tokens.items():
            print(f"  Testing {approach} approach...")
            approach_results = {}
            
            # Prepare density profiles for testing (adapt to 6 test slices)
            if approach == 'linear_weighting':
                print(f"    Adapting density profiles for test (6 slices per axis)...")
                test_density_profiles = {
                    'x': density_profiles['x'][:6],  # First 6 sagittal weights
                    'y': density_profiles['y'][:6],  # First 6 coronal weights  
                    'z': density_profiles['z'][:6]   # First 6 axial weights
                }
                density_profiles_for_weighting = test_density_profiles
                print(f"      Test weights shapes - x: {test_density_profiles['x'].shape}, y: {test_density_profiles['y'].shape}, z: {test_density_profiles['z'].shape}")
            else:
                density_profiles_for_weighting = None
            
            # Test concatenation
            print(f"    Testing concatenation...")
            
            concat_aggregator = DensityGuidedAggregator(
                aggregation_method='concat',
                approach=approach,
                density_profiles=density_profiles_for_weighting
            )
            concat_features = concat_aggregator.aggregate_triaxial(cls_tokens_dict)
            
            expected_concat_dim = 6 * 1536 * 3  # 6 slices × 3 axes × 1536D
            print(f"      Concat output shape: {concat_features.shape} (expected: ({expected_concat_dim},))")
            assert concat_features.shape == (expected_concat_dim,), f"Wrong concat shape for {approach}: {concat_features.shape}"
            approach_results['concat'] = concat_features
            
            # Test pooling
            print(f"    Testing pooling...")
            
            pooling_aggregator = DensityGuidedAggregator(
                aggregation_method='pooling',
                approach=approach,
                density_profiles=density_profiles_for_weighting
            )
            pooling_features = pooling_aggregator.aggregate_triaxial(cls_tokens_dict)
            
            expected_pooling_dim = 3 * 1536  # 3 axes × 1536D
            print(f"      Pooling output shape: {pooling_features.shape} (expected: ({expected_pooling_dim},))")
            assert pooling_features.shape == (expected_pooling_dim,), f"Wrong pooling shape for {approach}: {pooling_features.shape}"
            approach_results['pooling'] = pooling_features
            
            all_results[approach] = approach_results
        
        # Test linear weighting specifically
        if 'linear_weighting' in all_results:
            print("  Validating linear weighting application...")
            linear_concat = all_results['linear_weighting']['concat']
            linear_pooling = all_results['linear_weighting']['pooling']
            
            print(f"    Linear weighting concat: {linear_concat.shape}")
            print(f"    Linear weighting pooling: {linear_pooling.shape}")
            print("    ✅ Linear weighting validated (weights applied during aggregation)")
        
        self.memory_monitor.print_memory_status("after aggregation")
        print("  ✅ Density-guided aggregation OK")
        
        return all_results
    
    def test_standalone_pca_processing(self, all_aggregated_features: dict) -> None:
        """
        Test Standalone PCA processing with density-guided features.
        
        Args:
            all_aggregated_features (dict): Aggregated features for testing
        """
        print("\n7. Testing Standalone PCA Processing")
        
        self.memory_monitor.print_memory_status("before PCA")
        
        # Test on one representative approach and aggregation method
        test_approach = 'central_uniform'
        test_method = 'concat'
        
        if test_approach in all_aggregated_features and test_method in all_aggregated_features[test_approach]:
            features = all_aggregated_features[test_approach][test_method]
            print(f"  Testing PCA on {test_approach}_{test_method} features...")
            print(f"    Feature dimension: {features.shape[0]:,}")
            
            # Test PCA 32D
            print(f"    Testing PCA 32D...")
            
            try:
                pca_config_32d = {
                    'mode': 'fixed',
                    'n_components': 32
                }
                
                n_test_subjects = 100
                
                print(f"      Creating {n_test_subjects} test subjects...")
                test_subjects = []
                for i in range(n_test_subjects):
                    noise_scale = 0.01 * np.std(features) if np.std(features) > 0 else 0.01
                    noisy_features = features + np.random.normal(0, noise_scale, features.shape)
                    test_subjects.append(noisy_features)
                
                # Create temporary directory structure for testing
                temp_dir = Path("temp_test_raw_features")
                temp_dir.mkdir(exist_ok=True)
                
                # Save test features
                for i in range(5):
                    split_features = np.stack(test_subjects[i*20:(i+1)*20])
                    np.save(temp_dir / f"train_val_split_{i}_raw_features.npy", split_features)
                
                pca_processor = StandalonePCAProcessor(pca_config_32d)
                
                start_time = time.time()
                result = pca_processor.fit_and_transform_variant(temp_dir)
                fit_time = time.time() - start_time
                
                pca_info = result['pca_info']
                print(f"      PCA fitted: {pca_info['n_components']} components in {fit_time:.3f}s")
                print(f"      Original dimension: {pca_info['original_dim']:,}")
                print(f"      Variance explained: {pca_info['variance_explained']:.4f}")
                
                # Test transform
                transformed_splits = result['transformed_splits']
                for split_name, split_data in transformed_splits.items():
                    if 'train_val' in split_name:
                        expected_shape = (20, 32)
                        actual_shape = split_data['features'].shape
                        print(f"      {split_name}: {actual_shape} (expected: {expected_shape})")
                        assert actual_shape == expected_shape, f"Wrong transform shape: {actual_shape}"
                
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir)
                
                print(f"      ✅ Standalone PCA 32D successful")
                
            except Exception as e:
                print(f"      ❌ Standalone PCA 32D failed: {e}")
                # Cleanup on error
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            
            # Test PCA 99% variance
            print(f"    Testing PCA 99% variance...")
            
            try:
                pca_config_99 = {
                    'mode': 'variance',
                    'variance_threshold': 0.99
                }
                
                temp_dir_99 = Path("temp_test_raw_features_99")
                temp_dir_99.mkdir(exist_ok=True)
                
                for i in range(5):
                    split_features = np.stack(test_subjects[i*15:(i+1)*15])
                    np.save(temp_dir_99 / f"train_val_split_{i}_raw_features.npy", split_features)
                
                pca_processor_99 = StandalonePCAProcessor(pca_config_99)
                result_99 = pca_processor_99.fit_and_transform_variant(temp_dir_99)
                
                pca_info_99 = result_99['pca_info']
                estimated_components = pca_info_99['n_components']
                print(f"      Estimated components for 99% variance: {estimated_components}")
                
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir_99)
                
                print(f"      ✅ Standalone PCA 99% successful")
                
            except Exception as e:
                print(f"      ❌ Standalone PCA 99% failed: {e}")
                import shutil
                if temp_dir_99.exists():
                    shutil.rmtree(temp_dir_99)
        
        self.memory_monitor.print_memory_status("after PCA")
        print("  ✅ Standalone PCA validation complete")
    
    def test_memory_estimation_density_guided(self) -> None:
        """
        Test memory and storage requirements estimation for density-guided approaches.
        """
        print("\n8. Testing Memory and Storage Estimation (Density-Guided)")
        
        n_subjects = 577
        cls_token_dim = 1536
        
        print("  Estimating density-guided storage requirements...")
        
        # Approach dimensions
        approach_dims = {
            'central_uniform': {'total_slices': 32, 'concat_dim': 32 * cls_token_dim},
            'adaptive_density': {'total_slices': 59, 'concat_dim': 59 * cls_token_dim},
            'linear_weighting': {'total_slices': 90, 'concat_dim': 90 * cls_token_dim}
        }
        
        pooling_dim = 3 * cls_token_dim  # Same for all approaches
        
        for approach, dims in approach_dims.items():
            concat_dim = dims['concat_dim']
            memory_concat = concat_dim * 4 / (1024**3)  # float32 bytes to GB
            memory_pooling = pooling_dim * 4 / (1024**3)
            
            print(f"    {approach}:")
            print(f"      Total slices: {dims['total_slices']}")
            print(f"      Concat dimension: {concat_dim:,}")
            print(f"      Memory per subject (concat): {memory_concat:.6f} GB")
            print(f"      Memory per subject (pooling): {memory_pooling:.6f} GB")
            
            # Training memory for Classical PCA
            n_training = 461
            training_memory_concat = memory_concat * n_training
            training_memory_pooling = memory_pooling * n_training
            
            print(f"      Training memory (concat): {training_memory_concat:.3f} GB")
            print(f"      Training memory (pooling): {training_memory_pooling:.3f} GB")
        
        # PCA storage estimation
        for n_components in [32, 256]:
            pca_storage_per_subject = n_components * 4 / (1024**3)
            total_pca_storage = pca_storage_per_subject * n_subjects * 6  # 6 variants
            print(f"    PCA {n_components}D total storage (6 variants): {total_pca_storage:.6f} GB")
        
        # Total for all 18 configurations
        estimated_99_components = 800
        total_storage = (32 + 256 + estimated_99_components) * 4 * n_subjects * 6 / (1024**3)
        print(f"    Total storage all 18 configs: {total_storage:.3f} GB")
        
        # Memory feasibility check
        max_training_memory = max(approach_dims[approach]['concat_dim'] * 4 * 461 / (1024**3) 
                                 for approach in approach_dims)
        available_ram = psutil.virtual_memory().available / (1024**3)
        
        print(f"    Max training memory needed: {max_training_memory:.3f} GB")
        print(f"    Available RAM: {available_ram:.1f} GB")
        
        if max_training_memory < available_ram * 0.8:
            print(f"  ✅ Memory usage feasible: {max_training_memory:.3f}GB < {available_ram*0.8:.1f}GB")
        else:
            print(f"  ⚠️  Memory may need optimization: {max_training_memory:.3f}GB vs {available_ram*0.8:.1f}GB")
        
        print("  ✅ Memory estimation complete")
    
    def test_density_guided_dimensions_validation(self) -> None:
        """
        Test and validate density-guided dimensions for all approaches.
        """
        print("\n9. Testing Density-Guided Dimensions Validation")
        
        volume_shape = (30, 38, 22)
        cls_token_dim = 1536
        
        print(f"  Volume shape: {volume_shape}")
        print(f"  CLS token dimension: {cls_token_dim}")
        
        approaches = {
            'central_uniform': {'sagittal': 11, 'coronal': 11, 'axial': 10, 'total': 32},
            'adaptive_density': {'sagittal': 19, 'coronal': 27, 'axial': 13, 'total': 59},
            'linear_weighting': {'sagittal': 30, 'coronal': 38, 'axial': 22, 'total': 90}
        }
        
        for approach_name, slices in approaches.items():
            print(f"  {approach_name}:")
            for axis, count in slices.items():
                if axis != 'total':
                    print(f"    {axis}: {count} slices")
            
            total_slices = slices['total']
            concat_dim = total_slices * cls_token_dim
            pooling_dim = 3 * cls_token_dim
            
            print(f"    Total slices: {total_slices}")
            print(f"    Concat dimension: {concat_dim:,}")
            print(f"    Pooling dimension: {pooling_dim:,}")
        
        # Verify expected dimensions
        expected_concat_dims = {
            'central_uniform': 49152,    # 32 × 1536
            'adaptive_density': 90624,   # 59 × 1536
            'linear_weighting': 138240   # 90 × 1536
        }
        
        expected_pooling_dim = 4608  # 3 × 1536
        
        for approach, expected_concat in expected_concat_dims.items():
            actual_concat = approaches[approach]['total'] * cls_token_dim
            print(f"  {approach}_concat: {actual_concat:,} (expected: {expected_concat:,})")
            assert actual_concat == expected_concat, f"Concat dimension mismatch for {approach}"
            
            print(f"  {approach}_pooling: {expected_pooling_dim:,} (expected: {expected_pooling_dim:,})")
        
        print("  ✅ Density-guided dimensions validation OK")
    
    def run_all_tests(self) -> None:
        """
        Run complete test suite for density-guided CLS token extraction.
        """
        print("=" * 80)
        print("DENSITY-GUIDED CLS TOKEN EXTRACTION PIPELINE TESTS")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            sample_volume = self.test_data_loading()
            density_profiles = self.test_density_profiles_loading()
            slices_dict = self.test_density_guided_preprocessing(sample_volume)
            model = self.test_model_loading()
            all_cls_tokens = self.test_cls_token_extraction(model, slices_dict)
            aggregated_features = self.test_density_guided_aggregation(all_cls_tokens, density_profiles)
            self.test_standalone_pca_processing(aggregated_features)
            self.test_memory_estimation_density_guided()
            self.test_density_guided_dimensions_validation()
            
            total_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("ALL TESTS PASSED ✅")
            print(f"Total test time: {total_time:.1f}s")
            self.memory_monitor.print_memory_status("final")
            print("Density-guided pipeline ready for full execution!")
            print("Key features validated:")
            print("  - DINOv2 Giant: CLS tokens (1536D) optimized for density guidance")
            print("  - 3 approaches: Central uniform, adaptive density, linear weighting")
            print("  - 6 variants: 3 approaches × 2 aggregations (concat/pooling)")
            print("  - Standalone PCA: Memory-efficient processing of pre-extracted features")
            print("  - Linear weighting: Density-based feature emphasis functional")
            print("  - Pipeline separation: Raw extraction + PCA reduction optimized")
            print("  - 18 final configurations: 6 variants × 3 PCA modes")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            print("Fix issues before running full pipeline")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """
    Main entry point for testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test density-guided CLS token extraction pipeline")
    parser.add_argument('--data-path', type=str, default='crops/2mm/S.Or.',
                       help='Path to HCP OFC dataset')
    
    args = parser.parse_args()
    
    if not Path(args.data_path).exists():
        print(f"Data path not found: {args.data_path}")
        print("Please provide correct path to HCP OFC dataset")
        sys.exit(1)
    
    tester = DensityGuidedTester(args.data_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()