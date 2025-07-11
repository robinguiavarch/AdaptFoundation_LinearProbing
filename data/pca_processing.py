"""
PCA dimensionality reduction module for feature preprocessing.

This module implements PCA-based dimensionality reduction for extracted features
with configurable variance thresholds and unsupervised processing.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA


class DimensionalityReducer:
    """
    Handles dimensionality reduction of extracted features using PCA.
    
    Implements unsupervised PCA with configurable variance threshold for 
    feature preprocessing in linear probing scenarios. Ensures no label
    information leakage during dimensionality reduction.
    
    Attributes:
        features_base_path (Path): Base path to extracted features directory
        model_name (str): Name of the foundation model
        variance_threshold (float): Cumulative variance threshold for component selection
        pca_model (PCA): Fitted PCA model
        n_components (int): Number of components selected based on variance threshold
    """
    
    def __init__(self, features_base_path: str, model_name: str = 'dinov2_vits14',
                 variance_threshold: float = None, n_components: int = None):
        """
        Initialize the dimensionality reducer.
        
        Args:
            features_base_path (str): Base path to feature_extracted directory
            model_name (str): Foundation model name. Defaults to 'dinov2_vits14'.
            variance_threshold (float, optional): Cumulative variance threshold mode
            n_components (int, optional): Fixed number of components mode
        """
        self.features_base_path = Path(features_base_path)
        self.model_name = model_name
        
        # Validate reduction mode
        if variance_threshold is not None and n_components is not None:
            raise ValueError("Cannot specify both variance_threshold and n_components")
        if variance_threshold is None and n_components is None:
            raise ValueError("Must specify either variance_threshold or n_components")
        
        self.variance_threshold = variance_threshold
        self.fixed_n_components = n_components
        self.pca_model = None
        self.n_components = None
        self.reduction_mode = "variance" if variance_threshold is not None else "fixed_components"
        
        # Validate paths
        self.model_path = self.features_base_path / model_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
    
    def _load_split_features(self, config_name: str, split_name: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load features and metadata for a specific configuration and split.
        
        Args:
            config_name (str): Configuration name (e.g., 'multi_axes_average')
            split_name (str): Split name (e.g., 'train_val_split_0')
        
        Returns:
            Tuple[np.ndarray, pd.DataFrame]: Features array and metadata dataframe
        """
        config_path = self.model_path / config_name
        
        features_file = config_path / f"{split_name}_features.npy"
        metadata_file = config_path / f"{split_name}_metadata.csv"
        
        if not features_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Missing files for {config_name}/{split_name}")
        
        features = np.load(features_file)
        metadata = pd.read_csv(metadata_file)
        
        return features, metadata
    
    def _get_training_data(self, config_name: str) -> np.ndarray:
        """
        Concatenate all training splits for PCA fitting.
        
        Args:
            config_name (str): Configuration name
        
        Returns:
            np.ndarray: Concatenated training features
        """
        training_features = []
        
        for i in range(5):
            split_name = f"train_val_split_{i}"
            features, _ = self._load_split_features(config_name, split_name)
            training_features.append(features)
        
        return np.concatenate(training_features, axis=0)
    
    def fit_pca(self, config_name: str) -> Dict:
        """
        Fit PCA model on training data with variance threshold or fixed components.
        
        Args:
            config_name (str): Configuration name to fit PCA on
        
        Returns:
            Dict: PCA fitting information including variance analysis
        """
        print(f"Fitting PCA on configuration: {config_name}")
        print(f"Mode: {self.reduction_mode}")
        
        # Load all training data
        training_features = self._get_training_data(config_name)
        original_dim = training_features.shape[1]
        
        print(f"Training data shape: {training_features.shape}")
        print(f"Original dimensionality: {original_dim}")
        
        if self.reduction_mode == "variance":
            # Variance threshold mode (existing logic)
            print(f"Target variance: {self.variance_threshold}")
            
            # Fit PCA with all components first to analyze variance
            pca_full = PCA()
            pca_full.fit(training_features)
            
            # Calculate cumulative variance
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Find number of components for desired variance threshold
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
        else:
            # Fixed components mode
            self.n_components = min(self.fixed_n_components, original_dim)
            print(f"Target components: {self.n_components}")
            
            if self.n_components >= original_dim:
                print(f"Warning: Requested {self.fixed_n_components} components but only {original_dim} available")
        
        # Fit final PCA with selected number of components
        self.pca_model = PCA(n_components=self.n_components)
        self.pca_model.fit(training_features)
        
        # Calculate final variance explained
        final_variance = np.sum(self.pca_model.explained_variance_ratio_)
        
        # Calculate cumulative variance for all components (for metadata)
        pca_full = PCA()
        pca_full.fit(training_features)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        print(f"Selected {self.n_components} components ({self.n_components}/{original_dim})")
        print(f"Variance explained: {final_variance:.4f}")
        print(f"Dimensionality reduction: {original_dim} -> {self.n_components}")
        
        return {
            'reduction_mode': self.reduction_mode,
            'original_dim': int(original_dim),
            'reduced_dim': int(self.n_components),
            'variance_threshold': float(self.variance_threshold) if self.variance_threshold else None,
            'fixed_n_components': int(self.fixed_n_components) if self.fixed_n_components else None,
            'actual_variance': float(final_variance),
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
            'cumulative_variance': cumulative_variance.tolist()
        }
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA model.
        
        Args:
            features (np.ndarray): Input features to transform
        
        Returns:
            np.ndarray: PCA-transformed features
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")
        
        return self.pca_model.transform(features)
    
    def process_configuration(self, config_name: str) -> Dict:
        """
        Complete PCA processing for a configuration: fit on training, transform all splits.
        
        Args:
            config_name (str): Configuration name to process
        
        Returns:
            Dict: Processing results and PCA information
        """
        # Fit PCA on training data
        pca_info = self.fit_pca(config_name)
        
        # Create output directory with appropriate name
        if self.reduction_mode == "variance":
            output_dir_name = f"PCA_{int(self.variance_threshold*100)}"
        else:
            output_dir_name = f"PCA_{self.n_components}"
        
        output_dir = self.model_path / config_name / output_dir_name
        output_dir.mkdir(exist_ok=True)
        
        print(f"Processing and saving PCA-transformed features to: {output_dir}")
        
        # Process all splits
        split_info = {}
        
        # Process training/validation splits
        for i in range(5):
            split_name = f"train_val_split_{i}"
            features, metadata = self._load_split_features(config_name, split_name)
            
            # Transform features
            transformed_features = self.transform_features(features)
            
            # Save transformed features and metadata
            np.save(output_dir / f"{split_name}_features.npy", transformed_features)
            metadata.to_csv(output_dir / f"{split_name}_metadata.csv", index=False)
            
            split_info[split_name] = {
                'original_shape': list(features.shape),
                'transformed_shape': list(transformed_features.shape)
            }
        
        # Process test split
        test_features, test_metadata = self._load_split_features(config_name, "test_split")
        transformed_test = self.transform_features(test_features)
        
        np.save(output_dir / "test_split_features.npy", transformed_test)
        test_metadata.to_csv(output_dir / "test_split_metadata.csv", index=False)
        
        split_info["test_split"] = {
            'original_shape': list(test_features.shape),
            'transformed_shape': list(transformed_test.shape)
        }
        
        # Save only metadata (no PCA model)
        metadata_info = {
            'pca_info': pca_info,
            'split_info': split_info,
            'processing_config': {
                'reduction_mode': self.reduction_mode,
                'variance_threshold': float(self.variance_threshold) if self.variance_threshold else None,
                'fixed_n_components': int(self.fixed_n_components) if self.fixed_n_components else None,
                'config_name': config_name,
                'model_name': self.model_name
            }
        }
        
        with open(output_dir / "pca_metadata.json", 'w') as f:
            json.dump(metadata_info, f, indent=2)
        
        print(f"PCA processing completed for {config_name}")
        print(f"Saved PCA-transformed features and metadata in: {output_dir}")
        
        return metadata_info
    
    def get_available_configurations(self) -> List[str]:
        """
        Get list of available configurations for PCA processing.
        
        Returns:
            List[str]: List of available configuration names
        """
        configs = []
        for path in self.model_path.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                configs.append(path.name)
        return configs