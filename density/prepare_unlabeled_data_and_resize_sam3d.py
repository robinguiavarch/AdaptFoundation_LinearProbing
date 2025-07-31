"""
Data preparation script for unlabeled subjects density analysis.
Extracts unlabeled subjects, resizes to 128^3, and saves for density mapping.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import zoom
import time

# Configuration
BASE_PATH = Path("crops/2mm/S.Or.")
SKELETON_FILE = BASE_PATH / "Lskeleton.npy"
SUBJECT_FILE = BASE_PATH / "Lskeleton_subject.csv"
LABELS_FILE = BASE_PATH / "hcp_OFC_labels.csv"
OUTPUT_DIR = Path("density")
TARGET_SIZE = (128, 128, 128)


def get_unlabeled_subjects():
    """
    Identify subjects that have skeleton data but no labels.
    
    Returns:
        tuple: (unlabeled_indices, unlabeled_subject_ids)
    """
    print("Loading subject mapping...")
    subjects_df = pd.read_csv(SUBJECT_FILE)
    all_subjects = set(subjects_df['Subject'].astype(str))
    
    print("Loading labels...")
    labels_df = pd.read_csv(LABELS_FILE)
    labeled_subjects = set(labels_df['Subject'].astype(str))
    
    unlabeled_subjects = all_subjects - labeled_subjects
    print(f"Found {len(unlabeled_subjects)} unlabeled subjects")
    
    unlabeled_indices = []
    unlabeled_subject_ids = []
    for i, subject in enumerate(subjects_df['Subject']):
        if str(subject) in unlabeled_subjects:
            unlabeled_indices.append(i)
            unlabeled_subject_ids.append(str(subject))
    
    return unlabeled_indices, unlabeled_subject_ids


def resize_volume(volume, target_size):
    """
    Resize 3D volume to target size using scipy zoom.
    
    Args:
        volume (np.ndarray): Input volume of shape (H, W, D)
        target_size (tuple): Target size (H_new, W_new, D_new)
        
    Returns:
        np.ndarray: Resized volume
    """
    current_size = volume.shape
    zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
    return zoom(volume, zoom_factors, order=1)


def process_and_save_unlabeled_data():
    """
    Process unlabeled subjects: extract, resize to 128^3, and save.
    """
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get unlabeled subjects
    unlabeled_indices, unlabeled_subject_ids = get_unlabeled_subjects()
    
    print(f"Loading skeleton data...")
    skeletons = np.load(SKELETON_FILE)
    print(f"Original skeleton shape: {skeletons.shape}")
    
    # Extract unlabeled data
    unlabeled_skeletons = skeletons[unlabeled_indices]
    print(f"Unlabeled skeletons shape: {unlabeled_skeletons.shape}")
    
    # Remove singleton dimension and resize
    print(f"Resizing {len(unlabeled_skeletons)} volumes to {TARGET_SIZE}...")
    resized_volumes = []
    
    start_time = time.time()
    for i, skeleton in enumerate(unlabeled_skeletons):
        # Remove last dimension (30, 38, 22, 1) -> (30, 38, 22)
        volume_3d = skeleton.squeeze()
        
        # Resize to 128^3
        resized_volume = resize_volume(volume_3d, TARGET_SIZE)
        resized_volumes.append(resized_volume)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i + 1}/{len(unlabeled_skeletons)} volumes ({elapsed:.1f}s)")
    
    # Convert to numpy array
    resized_volumes = np.array(resized_volumes)
    print(f"Final resized shape: {resized_volumes.shape}")
    
    # Save resized data
    resized_file = OUTPUT_DIR / "unlabeled_skeletons_128.npy"
    print(f"Saving resized data to {resized_file}...")
    np.save(resized_file, resized_volumes)
    
    # Save subject IDs
    subjects_file = OUTPUT_DIR / "unlabeled_subject_ids.txt"
    print(f"Saving subject IDs to {subjects_file}...")
    with open(subjects_file, 'w') as f:
        for subject_id in unlabeled_subject_ids:
            f.write(f"{subject_id}\n")
    
    print(f"Data preparation completed successfully!")
    print(f"Resized volumes: {resized_file}")
    print(f"Subject IDs: {subjects_file}")
    
    # Basic statistics
    mean_density = np.mean(resized_volumes > 0) * 100
    print(f"Average density: {mean_density:.2f}%")


if __name__ == "__main__":
    process_and_save_unlabeled_data()