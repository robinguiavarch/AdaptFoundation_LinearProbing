"""
Test script for validating all dataset loaders.
"""

import numpy as np
import torch
from pathlib import Path
from loader_sc import SCDataLoader
from loader_lc import LCDataLoader
from loader_fip import FIPDataLoader


def test_sc_loader(data_path):
    """
    Test SCDataLoader validation.
    
    Args:
        data_path (str or Path): Path to S.C.-sylv dataset directory
        
    Returns:
        bool: True if all tests pass
    """
    print("="*60)
    print("TESTING S.C.-sylv DATASET LOADER")
    print("="*60)
    
    try:
        loader = SCDataLoader(data_path)
        
        # Validate skeleton dimensions
        expected_shape = (1114, 38, 36, 49)
        print(f"Skeleton shape: {loader.skeletons.shape}")
        assert loader.skeletons.shape == expected_shape, f"Expected shape {expected_shape}, got {loader.skeletons.shape}"
        print("✓ Skeleton dimensions validated")
        
        # Validate binary data
        unique_values = np.unique(loader.skeletons)
        print(f"Unique voxel values: {unique_values}")
        assert np.array_equal(unique_values, [0, 1]), f"Expected [0, 1], got {unique_values}"
        print("✓ Binary data validated")
        
        # Calculate density
        density = np.mean(loader.skeletons) * 100
        print(f"Density (%): mean = {density:.2f}")
        
        # Validate number of subjects with labels
        subjects_with_skeletons = len(loader.subjects_df)
        subjects_with_labels = len(loader.labels_df)
        print(f"Subjects with skeletons: {subjects_with_skeletons}")
        print(f"Subjects with labels: {subjects_with_labels}")
        
        # Test intersection
        skeleton_subjects = set(loader.subjects_df['Subject'].astype(str))
        label_subjects = set(loader.labels_df['Subject'].astype(str))
        intersection = len(skeleton_subjects.intersection(label_subjects))
        print(f"Intersection: {intersection}")
        assert intersection == 883, f"Expected 883 subjects with both data and labels, got {intersection}"
        print("✓ Subject intersection validated")
        
        # Test splits
        split_files = ['test_split.csv'] + [f'train_val_split_{i}.csv' for i in range(5)]
        print("Split composition:")
        for split_file in split_files:
            try:
                data, labels, subjects = loader.load_split(split_file)
                print(f"  {split_file:<25} → subjects: {len(subjects)}, labels shape: {labels.shape}")
                assert labels.shape[1] == 6, f"Expected 6D labels, got shape {labels.shape}"
            except Exception as e:
                print(f"  {split_file:<25} → ERROR: {e}")
                return False
        
        print("✓ All splits loaded successfully")
        
        # Test tensor conversion
        tensor_data, tensor_labels, _ = loader.load_split_as_tensor('test_split.csv')
        print(f"Tensor data type: {tensor_data.dtype}")
        print(f"Tensor labels type: {tensor_labels.dtype}")
        assert tensor_labels.dtype == torch.float32, f"Expected float32 labels for regression, got {tensor_labels.dtype}"
        print("✓ Tensor conversion validated")
        
        print("✓ S.C.-sylv loader validation PASSED")
        return True
        
    except Exception as e:
        print(f"✗ S.C.-sylv loader validation FAILED: {e}")
        return False


def test_lc_loader(data_path):
    """
    Test LCDataLoader validation.
    
    Args:
        data_path (str or Path): Path to LARGE_CINGULATE dataset directory
        
    Returns:
        bool: True if all tests pass
    """
    print("="*60)
    print("TESTING LARGE_CINGULATE DATASET LOADER")
    print("="*60)
    
    try:
        loader = LCDataLoader(data_path)
        
        # Validate skeleton dimensions
        expected_shape = (1114, 18, 73, 57)
        print(f"Skeleton shape: {loader.skeletons.shape}")
        assert loader.skeletons.shape == expected_shape, f"Expected shape {expected_shape}, got {loader.skeletons.shape}"
        print("✓ Skeleton dimensions validated")
        
        # Validate binary data
        unique_values = np.unique(loader.skeletons)
        print(f"Unique voxel values: {unique_values}")
        assert np.array_equal(unique_values, [0, 1]), f"Expected [0, 1], got {unique_values}"
        print("✓ Binary data validated")
        
        # Calculate density
        density = np.mean(loader.skeletons) * 100
        print(f"Density (%): mean = {density:.2f}")
        
        # Validate number of subjects with labels
        subjects_with_skeletons = len(loader.subjects_df)
        subjects_with_labels = len(loader.labels_df)
        print(f"Subjects with skeletons: {subjects_with_skeletons}")
        print(f"Subjects with labels: {subjects_with_labels}")
        
        # Test intersection
        skeleton_subjects = set(loader.subjects_df['Subject'].astype(str))
        label_subjects = set(loader.labels_df['long_name'].astype(str))
        intersection = len(skeleton_subjects.intersection(label_subjects))
        print(f"Intersection: {intersection}")
        assert intersection == 341, f"Expected 341 subjects with both data and labels, got {intersection}"
        print("✓ Subject intersection validated")
        
        # Test splits
        split_files = ['test_split.csv'] + [f'train_val_split_{i}.csv' for i in range(5)]
        print("Split composition:")
        for split_file in split_files:
            try:
                data, labels, subjects = loader.load_split(split_file)
                print(f"  {split_file:<25} → subjects: {len(subjects)}, labels: {np.unique(labels, return_counts=True)}")
                assert set(np.unique(labels)).issubset({0, 1}), f"Expected binary labels [0,1], got {np.unique(labels)}"
            except Exception as e:
                print(f"  {split_file:<25} → ERROR: {e}")
                return False
        
        print("✓ All splits loaded successfully")
        
        # Test tensor conversion
        tensor_data, tensor_labels, _ = loader.load_split_as_tensor('test_split.csv')
        print(f"Tensor data type: {tensor_data.dtype}")
        print(f"Tensor labels type: {tensor_labels.dtype}")
        assert tensor_labels.dtype == torch.int64, f"Expected int64 labels for classification, got {tensor_labels.dtype}"
        print("✓ Tensor conversion validated")
        
        print("✓ LARGE_CINGULATE loader validation PASSED")
        return True
        
    except Exception as e:
        print(f"✗ LARGE_CINGULATE loader validation FAILED: {e}")
        return False


def test_fip_loader(data_path):
    """
    Test FIPDataLoader validation.
    
    Args:
        data_path (str or Path): Path to F.I.P dataset directory
        
    Returns:
        bool: True if all tests pass
    """
    print("="*60)
    print("TESTING F.I.P DATASET LOADER")
    print("="*60)
    
    try:
        loader = FIPDataLoader(data_path)
        
        # Validate skeleton dimensions
        expected_shape = (1114, 39, 45, 44)
        print(f"Skeleton shape: {loader.skeletons.shape}")
        assert loader.skeletons.shape == expected_shape, f"Expected shape {expected_shape}, got {loader.skeletons.shape}"
        print("✓ Skeleton dimensions validated")
        
        # Validate binary data
        unique_values = np.unique(loader.skeletons)
        print(f"Unique voxel values: {unique_values}")
        assert np.array_equal(unique_values, [0, 1]), f"Expected [0, 1], got {unique_values}"
        print("✓ Binary data validated")
        
        # Calculate density
        density = np.mean(loader.skeletons) * 100
        print(f"Density (%): mean = {density:.2f}")
        
        # Validate number of subjects with labels
        subjects_with_skeletons = len(loader.subjects_df)
        subjects_with_labels = len(loader.labels_df)
        print(f"Subjects with skeletons: {subjects_with_skeletons}")
        print(f"Subjects with labels: {subjects_with_labels}")
        
        # Test intersection
        skeleton_subjects = set(loader.subjects_df['Subject'].astype(str))
        label_subjects = set(loader.labels_df['Subject'].astype(str))
        intersection = len(skeleton_subjects.intersection(label_subjects))
        print(f"Intersection: {intersection}")
        assert intersection == 390, f"Expected 390 subjects with both data and labels, got {intersection}"
        print("✓ Subject intersection validated")
        
        # Test splits
        split_files = ['test_split.csv'] + [f'train_val_split_{i}.csv' for i in range(5)]
        print("Split composition:")
        for split_file in split_files:
            try:
                data, labels, subjects = loader.load_split(split_file)
                print(f"  {split_file:<25} → subjects: {len(subjects)}, labels: {np.unique(labels, return_counts=True)}")
                assert set(np.unique(labels)).issubset({0, 1}), f"Expected binary labels [0,1], got {np.unique(labels)}"
            except Exception as e:
                print(f"  {split_file:<25} → ERROR: {e}")
                return False
        
        print("✓ All splits loaded successfully")
        
        # Test tensor conversion
        tensor_data, tensor_labels, _ = loader.load_split_as_tensor('test_split.csv')
        print(f"Tensor data type: {tensor_data.dtype}")
        print(f"Tensor labels type: {tensor_labels.dtype}")
        assert tensor_labels.dtype == torch.int64, f"Expected int64 labels for classification, got {tensor_labels.dtype}"
        print("✓ Tensor conversion validated")
        
        print("✓ F.I.P loader validation PASSED")
        return True
        
    except Exception as e:
        print(f"✗ F.I.P loader validation FAILED: {e}")
        return False


def main():
    """
    Main function to test all loaders.
    """
    print("ADAPTFOUNDATION - DATASET LOADERS VALIDATION")
    print("=" * 80)
    
    # Define dataset paths (relative to parent directory since script is in data/)
    base_path = Path("../crops/2mm")
    datasets = {
        "S.C.-sylv": base_path / "S.C.-sylv.",
        "LARGE_CINGULATE": base_path / "LARGE_CINGULATE.",
        "F.I.P": base_path / "F.I.P."
    }
    
    results = {}
    
    # Test each dataset
    for dataset_name, dataset_path in datasets.items():
        if dataset_name == "S.C.-sylv":
            results[dataset_name] = test_sc_loader(dataset_path)
        elif dataset_name == "LARGE_CINGULATE":
            results[dataset_name] = test_lc_loader(dataset_path)
        elif dataset_name == "F.I.P":
            results[dataset_name] = test_fip_loader(dataset_path)
        
        print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for dataset_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {dataset_name:<20} : {status}")
        if not passed:
            all_passed = False
    
    print("-" * 80)
    if all_passed:
        print("✓ ALL DATASET LOADERS VALIDATED SUCCESSFULLY")
    else:
        print("✗ SOME DATASET LOADERS FAILED VALIDATION")
    
    print("=" * 80)


if __name__ == "__main__":
    main()