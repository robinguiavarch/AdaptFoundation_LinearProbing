#!/usr/bin/env python3
"""
Simple SAM-Med3D checkpoint inspection without imports.

This script analyzes the SAM-Med3D checkpoint structure to understand
the model architecture without requiring SAM-Med3D dependencies.

Usage:
    python scripts/inspect_checkpoint_simple.py
"""

import torch
from pathlib import Path


def analyze_checkpoint(checkpoint_path="ckpt/sam_med3d_turbo.pth"):
    """
    Analyze SAM-Med3D checkpoint structure.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print("SAM-MED3D CHECKPOINT ANALYSIS")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Size: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        # Analyze each key
        for key, value in checkpoint.items():
            print(f"\n--- {key} ---")
            print(f"Type: {type(value)}")
            
            if isinstance(value, dict):
                print(f"Dict with {len(value)} items")
                
                # If it looks like state_dict, analyze parameters
                if 'weight' in str(value.keys()) or 'bias' in str(value.keys()):
                    print("Looks like model state_dict:")
                    
                    # Group parameters by module
                    modules = {}
                    for param_name in value.keys():
                        module_name = param_name.split('.')[0]
                        if module_name not in modules:
                            modules[module_name] = []
                        modules[module_name].append(param_name)
                    
                    print(f"Found {len(modules)} main modules:")
                    for module_name, params in modules.items():
                        print(f"  {module_name}: {len(params)} parameters")
                        
                        # Show first few parameters for each module
                        for param in params[:3]:
                            param_shape = value[param].shape if hasattr(value[param], 'shape') else 'Unknown'
                            print(f"    {param}: {param_shape}")
                        
                        if len(params) > 3:
                            print(f"    ... and {len(params) - 3} more")
                
            elif isinstance(value, (list, tuple)):
                print(f"List/tuple with {len(value)} items")
                if len(value) > 0:
                    print(f"First item type: {type(value[0])}")
                    
            elif hasattr(value, 'shape'):
                print(f"Tensor shape: {value.shape}")
                print(f"Tensor dtype: {value.dtype}")
                
            else:
                print(f"Value: {value}")
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("\nKey findings:")
    print("- Checkpoint structure understood")
    print("- Ready for Phase 1: Model building")


if __name__ == "__main__":
    analyze_checkpoint()