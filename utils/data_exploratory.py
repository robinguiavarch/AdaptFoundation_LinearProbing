"""
Visualization utilities for HCP OFC skeletal dataset analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_skeleton_data(skeleton_path):
    """
    Load skeleton data from numpy file.
    
    Args:
        skeleton_path (Path): Path to skeleton .npy file
        
    Returns:
        np.ndarray: Loaded skeleton data
    """
    return np.load(skeleton_path)


def load_subject_mapping(subject_path):
    """
    Load subject mapping from CSV file.
    
    Args:
        subject_path (Path): Path to subject mapping CSV
        
    Returns:
        pd.DataFrame: Subject mapping dataframe
    """
    return pd.read_csv(subject_path)


def load_labels(labels_path):
    """
    Load labels from CSV file.
    
    Args:
        labels_path (Path): Path to labels CSV
        
    Returns:
        pd.DataFrame: Labels dataframe
    """
    return pd.read_csv(labels_path)


def analyze_data_correspondence(subjects_df, labels_df):
    """
    Analyze correspondence between subjects and labels.
    
    Args:
        subjects_df (pd.DataFrame): Subject mapping dataframe
        labels_df (pd.DataFrame): Labels dataframe
        
    Returns:
        tuple: (subjects_with_skeletons, subjects_with_labels, intersection)
    """
    subjects_with_skeletons = set(subjects_df['Subject'].astype(str))
    subjects_with_labels = set(labels_df['Subject'].astype(str))
    intersection = subjects_with_skeletons & subjects_with_labels
    
    return subjects_with_skeletons, subjects_with_labels, intersection


def analyze_splits(splits_path, subjects_with_labels):
    """
    Analyze split files and their correspondence with labeled data.
    
    Args:
        splits_path (Path): Path to splits directory
        subjects_with_labels (set): Set of subjects with labels
        
    Returns:
        dict: Split analysis results
    """
    split_files = list(splits_path.glob("*.csv"))
    split_analysis = {}
    
    for split_file in sorted(split_files):
        split_df = pd.read_csv(split_file, header=None, names=['Subject'])
        split_subjects = set(split_df['Subject'].astype(str))
        with_labels = split_subjects & subjects_with_labels
        
        split_analysis[split_file.name] = {
            'total_subjects': len(split_subjects),
            'subjects_with_labels': len(with_labels)
        }
    
    return split_analysis


def get_labeled_indices(subjects_df, subjects_with_labels, max_samples=6):
    """
    Get indices of subjects that have labels for visualization.
    
    Args:
        subjects_df (pd.DataFrame): Subject mapping dataframe
        subjects_with_labels (set): Set of subjects with labels
        max_samples (int): Maximum number of samples to return
        
    Returns:
        list: List of indices with labels
    """
    labeled_indices = []
    for i, subject in enumerate(subjects_df['Subject']):
        if str(subject) in subjects_with_labels:
            labeled_indices.append(i)
        if len(labeled_indices) >= max_samples:
            break
    return labeled_indices


def visualize_skeletons(skeletons, subjects_df, labels_df, labeled_indices):
    """
    Visualize skeleton samples with their labels.
    
    Args:
        skeletons (np.ndarray): Skeleton data
        subjects_df (pd.DataFrame): Subject mapping dataframe
        labels_df (pd.DataFrame): Labels dataframe
        labeled_indices (list): Indices to visualize
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, idx in enumerate(labeled_indices):
        skeleton = skeletons[idx]
        subject_id = subjects_df.iloc[idx]['Subject']
        
        label_row = labels_df[labels_df['Subject'] == subject_id]
        label = label_row['Left_OFC'].iloc[0] if not label_row.empty else "No label"
        
        mid_slice = skeleton.shape[2] // 2
        slice_data = skeleton[:, :, mid_slice]
        
        axes[i].imshow(slice_data, cmap='gray')
        axes[i].set_title(f'Subject {subject_id}\nLabel: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_skeleton_statistics(skeletons, max_samples=100):
    """
    Calculate basic statistics on skeleton data.
    
    Args:
        skeletons (np.ndarray): Skeleton data
        max_samples (int): Maximum samples to analyze
        
    Returns:
        dict: Statistical summary
    """
    densities = []
    for i in range(min(max_samples, len(skeletons))):
        density = np.count_nonzero(skeletons[i]) / skeletons[i].size * 100
        densities.append(density)
    
    return {
        'mean_density': np.mean(densities),
        'median_density': np.median(densities),
        'min_density': np.min(densities),
        'max_density': np.max(densities)
    }


def explore_hcp_ofc_dataset(base_path):
    """
    Complete exploration of HCP OFC dataset.
    
    Args:
        base_path (str or Path): Base path to dataset
    """
    base_path = Path(base_path)
    skeleton_file = base_path / "Lskeleton.npy"
    subject_file = base_path / "Lskeleton_subject.csv"
    labels_file = base_path / "hcp_OFC_labels.csv"
    splits_path = base_path / "splits"
    
    print("Loading skeleton data...")
    skeletons = load_skeleton_data(skeleton_file)
    print(f"Skeleton shape: {skeletons.shape}")
    
    print("Loading subject mapping...")
    subjects_df = load_subject_mapping(subject_file)
    print(f"Number of mapped subjects: {len(subjects_df)}")
    
    print("Loading labels...")
    labels_df = load_labels(labels_file)
    print(f"Number of labeled subjects: {len(labels_df)}")
    print("Left_OFC distribution:")
    print(labels_df['Left_OFC'].value_counts().sort_index())
    
    print("Analyzing data correspondence...")
    subjects_with_skeletons, subjects_with_labels, intersection = analyze_data_correspondence(subjects_df, labels_df)
    print(f"Subjects with skeletons: {len(subjects_with_skeletons)}")
    print(f"Subjects with labels: {len(subjects_with_labels)}")
    print(f"Intersection: {len(intersection)}")
    
    print("Analyzing splits...")
    split_analysis = analyze_splits(splits_path, subjects_with_labels)
    for split_name, analysis in split_analysis.items():
        print(f"{split_name}: {analysis['total_subjects']} total, {analysis['subjects_with_labels']} with labels")
    
    print("Preparing visualization...")
    labeled_indices = get_labeled_indices(subjects_df, subjects_with_labels)
    visualize_skeletons(skeletons, subjects_df, labels_df, labeled_indices)
    
    print("Calculating statistics...")
    stats = calculate_skeleton_statistics(skeletons)
    print(f"Mean density: {stats['mean_density']:.2f}%")
    print(f"Density range: {stats['min_density']:.2f}% - {stats['max_density']:.2f}%")
    
    print("Data integrity check...")
    unique_values = np.unique(skeletons)
    print(f"Unique values: {unique_values}")
    print(f"Data consistency: {'OK' if len(skeletons) == len(subjects_df) else 'ERROR'}")


def main():
    """
    Main function for standalone script execution.
    """
    # Configuration - adapt this path
    base_path = "crops/2mm/S.Or"
    
    print("=== HCP OFC Dataset Exploration ===")
    explore_hcp_ofc_dataset(base_path)
    print("=== Exploration Complete ===")


if __name__ == "__main__":
    main()