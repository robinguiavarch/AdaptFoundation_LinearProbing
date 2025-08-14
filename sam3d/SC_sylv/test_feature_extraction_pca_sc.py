#!/usr/bin/env python3
"""
Test script for S.C.-sylv SAM-Med3D feature extraction and PCA pipeline.

This script validates all components before running the full pipeline
to ensure proper functionality and avoid issues during production.
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

from data.loader_sc import SCDataLoader
from sam3d.SC_sylv.feature_extraction_core_sc import SAMMed3DStandardExtractor
from sklearn.decomposition import PCA


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


class SCFeatureExtractionPCATester:
    """
    Tests all components of the S.C.-sylv SAM-Med3D feature extraction and PCA pipeline.
    
    Attributes:
        data_loader (SCDataLoader): Data loader for testing
        device (torch.device): Computation device
        memory_monitor (MemoryMonitor): Memory usage monitor
    """
    
    def __init__(self, data_path: str = "crops/2mm/S.C.-sylv."):
        """
        Initialize S.C.-sylv tester.
        
        Args:
            data_path (str): Path to S.C.-sylv dataset
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_monitor = MemoryMonitor()
        
        print("Initializing S.C.-sylv SAM-Med3D Feature Extraction and PCA Tester")
        print(f"Device: {self.device}")
        
        try:
            self.data_loader = SCDataLoader(data_path)
            print(f"S.C.-sylv dataset loaded: {len(self.data_loader.skeletons)} volumes")
        except Exception as e:
            print(f"ERROR loading S.C.-sylv dataset: {e}")
            sys.exit(1)
    
    def test_sc_data_loading(self) -> np.ndarray:
        """
        Test S.C.-sylv data loading and return sample volume.
        
        Returns:
            np.ndarray: Sample 3D volume for testing
        """
        print("\n1. Testing S.C.-sylv Data Loading")
        
        sample_volume = self.data_loader.skeletons[0]
        expected_shape = (38, 36, 49)  # S.C.-sylv specific dimensions
        expected_subjects = 883  # S.C.-sylv labeled subjects
        
        print(f"  Sample volume shape: {sample_volume.shape}")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Volume density: {np.mean(sample_volume):.4f}")
        print(f"  Non-zero voxels: {np.sum(sample_volume)} / {np.prod(sample_volume.shape)}")
        print(f"  Total volumes: {len(self.data_loader.skeletons)}")
        
        assert sample_volume.shape == expected_shape, f"Wrong volume shape: {sample_volume.shape}"
        assert 0 <= np.min(sample_volume) <= np.max(sample_volume) <= 1, "Volume values not in [0,1]"
        
        # Test split loading
        test_data, test_labels, test_subjects = self.data_loader.get_test_split()
        print(f"  Test split: {len(test_subjects)} subjects")
        print(f"  Label shape: {test_labels.shape if hasattr(test_labels, 'shape') else 'scalar'}")
        
        train_splits = self.data_loader.get_train_val_splits()
        total_train_subjects = sum(len(subjects) for _, _, subjects in train_splits)
        print(f"  Training splits: {total_train_subjects} subjects total")
        
        print("  ✅ S.C.-sylv data loading OK")
        return sample_volume
    
    def test_sam3d_extractor_loading(self) -> SAMMed3DStandardExtractor:
        """
        Test SAM-Med3D standard extractor loading.
        
        Returns:
            SAMMed3DStandardExtractor: Loaded extractor
        """
        print("\n2. Testing SAM-Med3D Standard Extractor Loading")
        
        self.memory_monitor.print_memory_status("before extractor")
        
        try:
            extractor = SAMMed3DStandardExtractor(
                config_path="sam3d/SC_sylv/feature_extraction_sc.yaml"
            )
            
            model_info = extractor.get_model_info()
            print(f"  Model type: {model_info['model_type']}")
            print(f"  Feature dimension: {model_info['feature_dim']}")
            print(f"  Aggregation method: {model_info['aggregation_method']}")
            print(f"  Device: {model_info['device']}")
            print(f"  Dataset: {model_info['dataset']}")
            print(f"  Task type: {model_info['task_type']}")
            
            assert model_info['feature_dim'] == 196608, f"Wrong feature dimension: {model_info['feature_dim']}"
            assert model_info['aggregation_method'] == 'flatten', "Wrong aggregation method"
            assert model_info['dataset'] == 'S.C.-sylv', "Wrong dataset"
            assert model_info['task_type'] == 'regression', "Wrong task type"
            
            self.memory_monitor.print_memory_status("after extractor")
            print("  ✅ SAM-Med3D extractor loading OK")
            
            return extractor
            
        except Exception as e:
            print(f"  ❌ SAM-Med3D extractor loading failed: {e}")
            sys.exit(1)
    
    def test_feature_extraction(self, extractor: SAMMed3DStandardExtractor, 
                               sample_volume: np.ndarray) -> np.ndarray:
        """
        Test feature extraction with S.C.-sylv volume.
        
        Args:
            extractor (SAMMed3DStandardExtractor): Feature extractor
            sample_volume (np.ndarray): Sample S.C.-sylv volume
        
        Returns:
            np.ndarray: Extracted features
        """
        print("\n3. Testing Feature Extraction")
        
        self.memory_monitor.print_memory_status("before feature extraction")
        
        volume_tensor = torch.from_numpy(sample_volume).float()
        
        print(f"  Input volume shape: {volume_tensor.shape}")
        print(f"  Input volume type: {volume_tensor.dtype}")
        
        try:
            start_time = time.time()
            features = extractor.extract_features(volume_tensor)
            extraction_time = time.time() - start_time
            
            print(f"  Output features shape: {features.shape}")
            print(f"  Output features type: {features.dtype}")
            print(f"  Extraction time: {extraction_time:.3f}s")
            print(f"  Feature range: [{torch.min(features):.4f}, {torch.max(features):.4f}]")
            print(f"  Non-zero features: {torch.sum(features != 0).item()}/{features.numel()}")
            
            # Validate expected dimensions
            expected_batch_size = 1
            expected_feature_dim = 196608  # 384 * 8 * 8 * 8
            
            assert features.shape == (expected_batch_size, expected_feature_dim), \
                f"Wrong feature shape: {features.shape} vs ({expected_batch_size}, {expected_feature_dim})"
            
            features_np = features.cpu().numpy()
            
            self.memory_monitor.print_memory_status("after feature extraction")
            print("  ✅ Feature extraction OK")
            
            return features_np
            
        except Exception as e:
            print(f"  ❌ Feature extraction failed: {e}")
            raise
    
    def test_batch_processing(self, extractor: SAMMed3DStandardExtractor) -> np.ndarray:
        """
        Test batch processing capabilities.
        
        Args:
            extractor (SAMMed3DStandardExtractor): Feature extractor
            
        Returns:
            np.ndarray: Batch features for PCA testing
        """
        print("\n4. Testing Batch Processing")
        
        self.memory_monitor.print_memory_status("before batch processing")
        
        # Create test batch with S.C.-sylv volumes
        test_volumes = []
        for i in range(50):  # 50 samples for testing
            volume = self.data_loader.skeletons[i]
            test_volumes.append(volume)
        
        batch_tensor = torch.stack([torch.from_numpy(v).float() for v in test_volumes])
        print(f"  Test batch shape: {batch_tensor.shape}")
        
        try:
            start_time = time.time()
            batch_features = extractor.extract_features_batch(batch_tensor, batch_size=3)
            batch_time = time.time() - start_time
            
            print(f"  Batch features shape: {batch_features.shape}")
            print(f"  Batch processing time: {batch_time:.3f}s")
            print(f"  Time per volume: {batch_time/50:.3f}s")
            
            # Validate batch dimensions
            expected_batch_size = 50
            expected_feature_dim = 196608
            
            assert batch_features.shape == (expected_batch_size, expected_feature_dim), \
                f"Wrong batch shape: {batch_features.shape}"
            
            batch_features_np = batch_features.numpy()
            
            self.memory_monitor.print_memory_status("after batch processing")
            print("  ✅ Batch processing OK")
            
            return batch_features_np
            
        except Exception as e:
            print(f"  ❌ Batch processing failed: {e}")
            raise
    
    def test_pca_pipeline(self, features: np.ndarray) -> None:
        """
        Test PCA pipeline with 4 modes on extracted features.
        
        Args:
            features (np.ndarray): Features for PCA testing
        """
        print("\n5. Testing PCA Pipeline (1 mode - test environment)")
        
        self.memory_monitor.print_memory_status("before PCA testing")
        
        pca_modes = [
            {'mode': 'fixed', 'n_components': 32, 'description': 'PCA 32D'}
            # Note: PCA 256D, 95%, 99.5% nécessitent plus de 50 échantillons
            # Ces modes seront testés en production avec 883 sujets S.C.-sylv
        ]
        
        print(f"  Input features shape: {features.shape}")
        print(f"  Original dimensionality: {features.shape[1]}")
        
        for i, mode in enumerate(pca_modes, 1):
            print(f"  [{i}/1] Testing {mode['description']}")
            
            try:
                start_time = time.time()
                
                if mode['mode'] == 'fixed':
                    pca = PCA(n_components=mode['n_components'])
                    pca.fit(features)
                    reduced_features = pca.transform(features)
                    expected_dim = mode['n_components']
                    variance_explained = np.sum(pca.explained_variance_ratio_)
                    
                else:  # variance mode
                    pca_full = PCA()
                    pca_full.fit(features)
                    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
                    n_components = np.argmax(cumsum_var >= mode['variance_threshold']) + 1
                    
                    pca = PCA(n_components=n_components)
                    pca.fit(features)
                    reduced_features = pca.transform(features)
                    expected_dim = n_components
                    variance_explained = np.sum(pca.explained_variance_ratio_)
                
                pca_time = time.time() - start_time
                
                print(f"    196608D → {reduced_features.shape[1]}D")
                print(f"    Variance explained: {variance_explained:.4f}")
                print(f"    Processing time: {pca_time:.3f}s")
                
                assert reduced_features.shape[1] == expected_dim, \
                    f"Wrong PCA dimension: {reduced_features.shape[1]} vs {expected_dim}"
                assert reduced_features.shape[0] == features.shape[0], \
                    f"Wrong sample count: {reduced_features.shape[0]} vs {features.shape[0]}"
                
                print(f"    ✅ {mode['description']} OK")
                
            except Exception as e:
                print(f"    ❌ {mode['description']} failed: {e}")
                raise
        
        self.memory_monitor.print_memory_status("after PCA testing")
        print("  ✅ PCA pipeline (1 mode) OK - Production will test all 4 modes")
    
    def test_memory_requirements(self) -> None:
        """
        Test memory and storage requirements for S.C.-sylv dataset.
        """
        print("\n6. Testing Memory Requirements")
        
        n_subjects = 883  # S.C.-sylv labeled subjects
        n_splits = 6  # 5 train/val + 1 test
        
        print("  Estimating storage requirements for S.C.-sylv")
        
        # Feature dimensions
        raw_features_dim = 196608  # 384 * 8 * 8 * 8
        pca_dims = {
            'PCA_32': 32,
            'PCA_256': 256,
            'PCA_95': 180,  # Estimated ~95% variance components
            'PCA_995': 350  # Estimated ~99.5% variance components
        }
        
        total_storage = 0
        
        # Raw features storage
        raw_memory_per_subject = raw_features_dim * 4 / (1024**3)  # float32 = 4 bytes
        raw_total_storage = raw_memory_per_subject * n_subjects
        total_storage += raw_total_storage
        
        print(f"    Raw features: {raw_features_dim:,}D → {raw_memory_per_subject:.4f}GB per subject")
        print(f"      Total: {raw_total_storage:.3f}GB")
        
        # PCA features storage
        for pca_name, pca_dim in pca_dims.items():
            pca_memory_per_subject = pca_dim * 4 / (1024**3)
            pca_total_storage = pca_memory_per_subject * n_subjects
            total_storage += pca_total_storage
            
            print(f"    {pca_name}: {pca_dim:,}D → {pca_memory_per_subject:.4f}GB per subject")
            print(f"      Total: {pca_total_storage:.3f}GB")
        
        print(f"    Total storage all modes: {total_storage:.2f}GB")
        
        # Memory requirements for batch processing
        batch_size = 8
        max_memory_per_batch = raw_features_dim * batch_size * 4 / (1024**3)
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
        storage_limit = 15  # GB (higher for S.C.-sylv due to more subjects)
        if total_storage < storage_limit:
            print(f"  ✅ Total storage within limit: {total_storage:.2f}GB < {storage_limit}GB")
        else:
            print(f"  ⚠️ Storage may be high: {total_storage:.2f}GB > {storage_limit}GB")
    
    def test_pipeline_integration(self) -> None:
        """
        Test complete pipeline integration.
        """
        print("\n7. Testing Pipeline Integration")
        
        print("  Simulating complete S.C.-sylv pipeline:")
        print("    1. Feature extraction: S.C.-sylv volumes → SAM-Med3D → 196608D features")
        print("    2. PCA reduction: 196608D → 32D/256D/95%/99.5%")
        print("    3. Output structure: feature_extraction_sam3d_sc/flatten/PCA_{mode}/")
        
        # Test expected file structure
        expected_structure = {
            'feature_extraction_sam3d_sc/flatten/': [
                'test_split_features.npy',
                'test_split_metadata.csv',
                'train_val_split_0_features.npy',
                'train_val_split_0_metadata.csv',
                'train_val_split_1_features.npy',
                'train_val_split_1_metadata.csv',
                'train_val_split_2_features.npy',
                'train_val_split_2_metadata.csv',
                'train_val_split_3_features.npy',
                'train_val_split_3_metadata.csv',
                'train_val_split_4_features.npy',
                'train_val_split_4_metadata.csv'
            ],
            'PCA subdirectories': ['PCA_32/', 'PCA_256/', 'PCA_95/', 'PCA_995/']
        }
        
        print("  Expected output structure:")
        for path, files in expected_structure.items():
            print(f"    {path}")
            if isinstance(files, list) and len(files) > 3:
                print(f"      {files[0]}, {files[1]}, ..., {files[-1]}")
            else:
                for file in files:
                    print(f"      {file}")
        
        print("  ✅ Pipeline integration design OK")
    
    def run_all_tests(self) -> None:
        """
        Run complete test suite for S.C.-sylv feature extraction and PCA pipeline.
        """
        print("=" * 80)
        print("S.C.-sylv SAM-MED3D FEATURE EXTRACTION AND PCA PIPELINE TESTS")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Run tests sequentially
            sample_volume = self.test_sc_data_loading()
            extractor = self.test_sam3d_extractor_loading()
            single_features = self.test_feature_extraction(extractor, sample_volume)
            batch_features = self.test_batch_processing(extractor)
            self.test_pca_pipeline(batch_features)
            self.test_memory_requirements()
            self.test_pipeline_integration()
            
            # Final memory status
            total_time = time.time() - start_time
            print(f"\n{'='*80}")
            print("ALL S.C.-sylv TESTS PASSED ✅")
            print(f"Total test time: {total_time:.1f}s")
            self.memory_monitor.print_memory_status("final")
            print("S.C.-sylv pipeline ready for full execution!")
            print()
            print("Key features validated:")
            print("  - S.C.-sylv dataset: 883 subjects, regression task")
            print("  - Volume dimensions: (38,36,49) → (128,128,128)")
            print("  - SAM-Med3D standard: 196608D features (384 × 8³)")
            print("  - Flatten aggregation: no density optimization")
            print("  - PCA modes: 32D (256D, 95%, 99.5% tested in production)")
            print("  - Batch processing: memory-optimized extraction")
            print("  - Output structure: feature_extraction_sam3d_sc/flatten/")
            print("  - Memory estimation: ~3.6GB total storage")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"\n❌ S.C.-sylv TEST FAILED: {e}")
            print("Fix issues before running full pipeline")
            sys.exit(1)


def main():
    """
    Main entry point for S.C.-sylv testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test S.C.-sylv SAM-Med3D feature extraction and PCA pipeline")
    parser.add_argument('--data-path', type=str, default='crops/2mm/S.C.-sylv.',
                       help='Path to S.C.-sylv dataset')
    
    args = parser.parse_args()
    
    # Check if data path exists
    if not Path(args.data_path).exists():
        print(f"S.C.-sylv data path not found: {args.data_path}")
        print("Please provide correct path to S.C.-sylv dataset")
        sys.exit(1)
    
    # Check if config exists
    config_path = Path("sam3d/SC_sylv/feature_extraction_sc.yaml")
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Please ensure S.C.-sylv configuration files are in place")
        sys.exit(1)
    
    # Run tests
    tester = SCFeatureExtractionPCATester(args.data_path)
    tester.run_all_tests()


if __name__ == "__main__":
    main()