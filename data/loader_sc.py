"""
Data loading utilities for HCP S.C.-sylv dataset.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path


class SCDataLoader:
    """
    Data loader for HCP S.C.-sylv skeletal dataset (regression task).
    
    Attributes:
        data_path (Path): Path to dataset directory
        skeletons (np.ndarray): Loaded skeleton data
        subjects_df (pd.DataFrame): Subject mapping dataframe
        labels_df (pd.DataFrame): Labels dataframe
    """
    
    def __init__(self, data_path):
        """
        Initialize SC data loader.
        
        Args:
            data_path (str or Path): Path to dataset directory
        """
        self.data_path = Path(data_path)
        self.skeletons = None
        self.subjects_df = None
        self.labels_df = None
        
        self._load_data()
    
    def _load_data(self):
        """
        Load all dataset components.
        """
        skeleton_file = self.data_path / "Lskeleton.npy"
        subject_file = self.data_path / "Lskeleton_subject.csv"
        labels_file = self.data_path / "hcp_isomap_labels_SC-sylv_left.csv"
        
        self.skeletons = np.load(skeleton_file)
        self.subjects_df = pd.read_csv(subject_file)
        self.labels_df = pd.read_csv(labels_file)
        
        if self.skeletons.ndim == 5 and self.skeletons.shape[-1] == 1:
            self.skeletons = self.skeletons.squeeze(-1)
    
    def load_split(self, split_name):
        """
        Load specific data split.
        
        Args:
            split_name (str): Split filename (e.g., 'train_val_split_0.csv', 'test_split.csv')
            
        Returns:
            tuple: (skeleton_data, labels, subject_ids)
        """
        split_file = self.data_path / "splits" / split_name
        split_df = pd.read_csv(split_file, header=None, names=['Subject'])
        split_subjects = split_df['Subject'].astype(str).tolist()
        
        subject_indices = []
        split_labels = []
        split_subject_ids = []
        
        label_columns = [
            'Isomap_central_left_dim1', 'Isomap_central_left_dim2', 'Isomap_central_left_dim3',
            'Isomap_central_left_dim4', 'Isomap_central_left_dim5', 'Isomap_central_left_dim6'
        ]
        
        for i, subject in enumerate(self.subjects_df['Subject']):
            if str(subject) in split_subjects:
                label_row = self.labels_df[self.labels_df['Subject'] == subject]
                if not label_row.empty:
                    subject_indices.append(i)
                    label_values = label_row[label_columns].iloc[0].values
                    split_labels.append(label_values)
                    split_subject_ids.append(subject)
        
        skeleton_data = self.skeletons[subject_indices]
        labels = np.array(split_labels)
        
        return skeleton_data, labels, split_subject_ids
    
    def load_split_as_tensor(self, split_name):
        """
        Load split and return as torch tensor.
        
        Args:
            split_name (str): Split filename (e.g., 'train_val_split_0.csv', 'test_split.csv')
            
        Returns:
            tuple: (skeleton_tensor, labels_tensor, subject_ids)
        """
        skeleton_data, labels, subject_ids = self.load_split(split_name)
        
        tensor_data = torch.from_numpy(skeleton_data).float()
        labels_tensor = torch.from_numpy(labels).float()
        
        return tensor_data, labels_tensor, subject_ids
    
    def get_train_val_splits(self):
        """
        Load all training/validation splits for cross-validation.
        
        Returns:
            list: List of (skeleton_data, labels, subject_ids) tuples for each split
        """
        splits = []
        for i in range(5):
            split_name = f"train_val_split_{i}.csv"
            splits.append(self.load_split(split_name))
        return splits
    
    def get_train_val_splits_as_tensor(self):
        """
        Load all training/validation splits as tensors for cross-validation.
        
        Returns:
            list: List of (skeleton_tensor, labels_tensor, subject_ids) tuples for each split
        """
        splits = []
        for i in range(5):
            split_name = f"train_val_split_{i}.csv"
            splits.append(self.load_split_as_tensor(split_name))
        return splits
    
    def get_test_split(self):
        """
        Load test split.
        
        Returns:
            tuple: (skeleton_data, labels, subject_ids)
        """
        return self.load_split("test_split.csv")
    
    def get_test_split_as_tensor(self):
        """
        Load test split as tensor.
        
        Returns:
            tuple: (skeleton_tensor, labels_tensor, subject_ids)
        """
        return self.load_split_as_tensor("test_split.csv")


def load_sc_dataset(data_path):
    """
    Load SC dataset with all splits.
    
    Args:
        data_path (str or Path): Path to dataset directory
        
    Returns:
        SCDataLoader: Initialized data loader
    """
    return SCDataLoader(data_path)