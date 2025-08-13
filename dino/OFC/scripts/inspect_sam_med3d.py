#!/usr/bin/env python3
"""
SAM-Med3D architecture inspection script for AdaptFoundation - Phase 0.

This script analyzes the SAM-Med3D architecture and tests feature extraction
from the 3D Image Encoder only, without prompt encoder or mask decoder.

Based on Nan Mo et al. (ICBB 2025) paper that uses SAM-Med3D for multi-scale
feature extraction without segmentation.

Usage:
    python scripts/inspect_sam_med3d.py
"""

import torch
import numpy as np
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Add SAM-Med3D to Python path
sam_med3d_path = Path.home() / "SAM-Med3D"
if sam_med3d_path.exists():
    sys.path.insert(0, str(sam_med3d_path))
    print(f"Added SAM-Med3D to path: {sam_med3d_path}")

# SAM-Med3D imports based on repository structure
try:
    from segment_anything.build_sam3D import sam_model_registry3d
    print("Successfully imported SAM-Med3D components")
except ImportError as e:
    print(f"SAM-Med3D import error: {e}")
    print("Will use direct torch.load approach")


class SAMMed3DInspector:
    """
    Inspector for analyzing SAM-Med3D architecture and testing feature extraction.
    
    This class loads SAM-Med3D-turbo checkpoint, extracts the 3D Image Encoder,
    and tests multi-scale feature extraction capabilities for integration into
    the AdaptFoundation pipeline.
    
    Attributes:
        checkpoint_path (Path): Path to SAM-Med3D-turbo checkpoint
        device (torch.device): Computation device (CUDA or CPU)
        model: Loaded SAM-Med3D model
        image_encoder: Extracted 3D Image Encoder component
    """
    
    def __init__(self, checkpoint_path="ckpt/sam_med3d_turbo.pth"):
        """
        Initialize the SAM-Med3D inspector.
        
        Args:
            checkpoint_path (str): Path to SAM-Med3D-turbo checkpoint file
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.image_encoder = None
        
        print(f"SAM-Med3D Inspector initialized")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")
    
    def load_model(self):
        """
        Load the complete SAM-Med3D model from checkpoint.
        
        Raises:
            FileNotFoundError: If checkpoint file does not exist
            Exception: If model loading fails
        """
        print("\n=== SAM-MED3D MODEL LOADING ===")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        try:
            print("Loading checkpoint...")
            
            # Load checkpoint using torch.load
            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Try to build model using SAM-Med3D registry
            try:
                # Check if sam_model_registry3d is available
                if 'sam_model_registry3d' in globals():
                    print("Attempting to use sam_model_registry3d...")
                    # Common model types in SAM-Med3D
                    possible_model_types = ["sam_med3d_turbo", "sam_med3d", "vit_b", "vit_l", "vit_h"]
                    
                    for model_type in possible_model_types:
                        try:
                            if model_type in sam_model_registry3d:
                                print(f"Trying model type: {model_type}")
                                self.model = sam_model_registry3d[model_type](checkpoint=str(self.checkpoint_path))
                                print(f"Model loaded successfully with type: {model_type}")
                                break
                        except Exception as e:
                            print(f"Failed with model type {model_type}: {e}")
                            continue
                else:
                    raise ImportError("sam_model_registry3d not available")
                    
            except Exception as e:
                print(f"Registry loading failed: {e}")
                print("Using direct checkpoint analysis...")
                
                # Analyze checkpoint structure
                self._analyze_checkpoint_structure(checkpoint)
                
                # For now, store the checkpoint for analysis
                self.model = checkpoint
            
            print("SAM-Med3D loaded successfully")
            
        except Exception as e:
            print(f"Error during model loading: {e}")
            raise
    
    def _analyze_checkpoint_structure(self, checkpoint):
        """
        Analyze the structure of the checkpoint to understand the model.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
        """
        print("\n=== CHECKPOINT STRUCTURE ANALYSIS ===")
        
        if isinstance(checkpoint, dict):
            print("Checkpoint is a dictionary with keys:")
            for key in checkpoint.keys():
                print(f"  - {key}")
                
                # Look for state_dict
                if key == 'state_dict' or key == 'model_state_dict':
                    print(f"    Found state dict with {len(checkpoint[key])} parameters")
                    
                    # Sample some parameter names
                    param_names = list(checkpoint[key].keys())[:10]
                    print("    Sample parameter names:")
                    for param_name in param_names:
                        print(f"      - {param_name}")
                        
                elif isinstance(checkpoint[key], dict):
                    print(f"    Dictionary with {len(checkpoint[key])} items")
                elif isinstance(checkpoint[key], (list, tuple)):
                    print(f"    List/tuple with {len(checkpoint[key])} items")
                else:
                    print(f"    Type: {type(checkpoint[key])}")
        else:
            print(f"Checkpoint is of type: {type(checkpoint)}")
    
    def analyze_architecture(self):
        """
        Analyze the complete SAM-Med3D architecture and extract image encoder.
        
        Raises:
            ValueError: If model is not loaded
        """
        print("\n=== SAM-MED3D ARCHITECTURE ANALYSIS ===")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Handle different model types
        if hasattr(self.model, 'named_children'):
            # Standard PyTorch model
            print("SAM-Med3D components:")
            for name, module in self.model.named_children():
                print(f"  - {name}: {type(module).__name__}")
                
                n_params = sum(p.numel() for p in module.parameters())
                print(f"    Parameters: {n_params:,}")
            
            # Look for image encoder
            if hasattr(self.model, 'image_encoder'):
                self.image_encoder = self.model.image_encoder
                print(f"\n3D Image Encoder found: {type(self.image_encoder).__name__}")
                
                print("3D Image Encoder structure:")
                for name, module in self.image_encoder.named_children():
                    print(f"  - {name}: {type(module).__name__}")
                    
            elif hasattr(self.model, 'image_encoder3d'):
                self.image_encoder = self.model.image_encoder3d
                print(f"\n3D Image Encoder found (image_encoder3d): {type(self.image_encoder).__name__}")
            else:
                print("3D Image Encoder not found in standard locations")
                print("Available attributes:")
                for attr in dir(self.model):
                    if not attr.startswith('_') and not callable(getattr(self.model, attr)):
                        print(f"  - {attr}")
                        
        elif isinstance(self.model, dict):
            # Checkpoint dictionary
            print("Model is a checkpoint dictionary")
            print("Cannot analyze architecture without building the model")
            print("Need to implement model building from state_dict")
            
        else:
            print(f"Unknown model type: {type(self.model)}")
    
    def test_feature_extraction(self):
        """
        Test feature extraction with dummy volumes.
        
        Tests different input sizes and analyzes the multi-scale feature pyramid
        as described in Nan Mo et al. paper (32³, 16³, 8³ expected resolutions).
        
        Raises:
            ValueError: If image encoder is not available
        """
        print("\n=== FEATURE EXTRACTION TESTING ===")
        
        if self.image_encoder is None:
            print("Image encoder not available - cannot test feature extraction")
            print("This may require building the model from checkpoint first")
            return
        
        print("Generating dummy volume 128³...")
        
        test_sizes = [
            (64, 64, 64),
            (128, 128, 128),
        ]
        
        for h, w, d in test_sizes:
            print(f"\nTesting with volume {h}×{w}×{d}:")
            
            dummy_volume = torch.randint(0, 2, (1, 1, h, w, d), dtype=torch.float32)
            dummy_volume = dummy_volume.to(self.device)
            
            print(f"  Input shape: {dummy_volume.shape}")
            print(f"  Input type: {dummy_volume.dtype}")
            print(f"  Input range: [{dummy_volume.min():.1f}, {dummy_volume.max():.1f}]")
            
            try:
                start_time = time.time()
                
                with torch.no_grad():
                    features = self.image_encoder(dummy_volume)
                
                extraction_time = time.time() - start_time
                
                self._analyze_extracted_features(features, extraction_time)
                
            except Exception as e:
                print(f"  Extraction error: {e}")
                
    def _analyze_extracted_features(self, features, extraction_time):
        """
        Analyze features extracted by the image encoder.
        
        Args:
            features: Extracted features (single tensor or list/tuple of tensors)
            extraction_time (float): Extraction time in seconds
        """
        print(f"  Extraction time: {extraction_time:.3f}s")
        
        if isinstance(features, (list, tuple)):
            print(f"  Multi-scale features: {len(features)} levels")
            
            for i, feat in enumerate(features):
                if isinstance(feat, torch.Tensor):
                    print(f"    Level {i}: {feat.shape} | {feat.dtype}")
                    print(f"    Range: [{feat.min():.3f}, {feat.max():.3f}]")
                    
                    memory_mb = feat.numel() * feat.element_size() / (1024**2)
                    print(f"    Memory: {memory_mb:.1f} MB")
                else:
                    print(f"    Level {i}: {type(feat)} (non-tensor)")
            
        elif isinstance(features, torch.Tensor):
            print(f"  Single tensor features: {features.shape}")
            print(f"  Type: {features.dtype}")
            print(f"  Range: [{features.min():.3f}, {features.max():.3f}]")
            
            memory_mb = features.numel() * features.element_size() / (1024**2)
            print(f"  Memory: {memory_mb:.1f} MB")
            
        else:
            print(f"  Unexpected features type: {type(features)}")
    
    def test_batch_processing(self):
        """
        Test batch processing capabilities for GPU A100 optimization.
        
        Tests different batch sizes to determine optimal configuration
        for the AdaptFoundation pipeline.
        """
        print("\n=== BATCH PROCESSING TESTING ===")
        
        if self.image_encoder is None:
            print("Image encoder not available - skipping batch processing test")
            return
        
        batch_sizes = [1, 2, 4, 8]
        volume_size = (128, 128, 128)
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size={batch_size}:")
            
            try:
                batch_volumes = torch.randint(0, 2, (batch_size, 1, *volume_size), dtype=torch.float32)
                batch_volumes = batch_volumes.to(self.device)
                
                print(f"  Batch shape: {batch_volumes.shape}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated() / (1024**2)
                    print(f"  GPU memory before: {memory_before:.1f} MB")
                
                start_time = time.time()
                
                with torch.no_grad():
                    batch_features = self.image_encoder(batch_volumes)
                
                extraction_time = time.time() - start_time
                
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated() / (1024**2)
                    memory_used = memory_after - memory_before
                    print(f"  GPU memory after: {memory_after:.1f} MB (+{memory_used:.1f} MB)")
                
                print(f"  Time: {extraction_time:.3f}s ({extraction_time/batch_size:.3f}s/volume)")
                
                if isinstance(batch_features, (list, tuple)):
                    print(f"  Batch features: {len(batch_features)} levels")
                    for i, feat in enumerate(batch_features):
                        if isinstance(feat, torch.Tensor):
                            print(f"    Level {i}: {feat.shape}")
                elif isinstance(batch_features, torch.Tensor):
                    print(f"  Batch features: {batch_features.shape}")
                
                print("  Batch processing successful")
                
            except Exception as e:
                print(f"  Error batch_size={batch_size}: {e}")
                
                if "out of memory" in str(e).lower():
                    print("  Memory limit reached")
                    break
    
    def determine_extraction_strategy(self):
        """
        Determine the recommended extraction strategy based on test results.
        
        Provides implementation recommendations for the AdaptFoundation
        SAM-Med3D feature extractor based on inspection results.
        """
        print("\n=== RECOMMENDED EXTRACTION STRATEGY ===")
        
        print("Based on checkpoint analysis:")
        print()
        
        if self.image_encoder is not None:
            print("DIRECT EXTRACTION successful:")
            print("   volume_3d → model.image_encoder(volume_3d) → features")
            print()
            print("FIXED WEIGHTS approach:")
            print("   model.image_encoder.eval()")
            print("   with torch.no_grad(): ...")
        else:
            print("MODEL BUILDING REQUIRED:")
            print("   Need to build model from checkpoint state_dict")
            print("   Investigate model architecture definition")
        
        print()
        print("MULTI-SCALE FEATURES expected:")
        print("   - High resolution: fine details (gyrus, sulci)")
        print("   - Low resolution: global morphology")
        print()
        print("BATCH PROCESSING:")
        print("   Optimal batch size to be determined (target: 8 for A100)")
        print()
        print("NEXT STEPS:")
        print("   1. Implement proper model building from checkpoint")
        print("   2. Extract image encoder successfully")
        print("   3. Test feature extraction pipeline")
        print("   4. Integrate into AdaptFoundation")
    
    def run_complete_inspection(self):
        """
        Execute complete SAM-Med3D inspection workflow.
        
        Runs all inspection phases sequentially and handles errors gracefully.
        """
        print("SAM-MED3D COMPLETE INSPECTION - PHASE 0")
        print("=" * 50)
        
        try:
            self.load_model()
            self.analyze_architecture()
            self.test_feature_extraction()
            self.test_batch_processing()
            self.determine_extraction_strategy()
            
            print("\nINSPECTION COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            print(f"\nERROR DURING INSPECTION: {e}")
            print("\nThis is expected for Phase 0 - we're learning the architecture")
            print("The analysis above provides valuable information for implementation")


def main():
    """
    Main entry point for SAM-Med3D inspection.
    """
    checkpoint_path = "ckpt/sam_med3d_turbo.pth"
    
    inspector = SAMMed3DInspector(checkpoint_path)
    inspector.run_complete_inspection()


if __name__ == "__main__":
    main()