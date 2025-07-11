"""
Feature dataset saver for storing extracted features with associated metadata.

This module handles the systematic extraction and storage of features from
3D skeletal volumes using various pipeline configurations.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from pipelines.feature_extraction_pipeline import FeatureExtractionPipeline


class FeatureDatasetSaver:
    """
    Manages extraction and storage of feature datasets from 3D skeletal volumes.
    
    This class orchestrates the complete process of running feature extraction
    pipelines with different configurations and saving the results in an
    organized directory structure.
    
    Attributes:
        data_path (str): Path to source HCP OFC dataset
        output_base_path (Path): Base directory for feature storage
        model_name (str): Name of the foundation model used
    """
    
    def __init__(self, data_path: str, output_base_path: str, model_name: str = 'dinov2_vits14'):
        """
        Initialize the feature dataset saver.
        
        Args:
            data_path (str): Path to HCP OFC dataset directory
            output_base_path (str): Base directory for feature storage
            model_name (str): Foundation model name. Defaults to 'dinov2_vits14'.
        """
        self.data_path = data_path
        self.output_base_path = Path(output_base_path)
        self.model_name = model_name
        
        # Create base output directory
        self.model_output_path = self.output_base_path / model_name
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"FeatureDatasetSaver initialized:")
        print(f"  Source data: {data_path}")
        print(f"  Output base: {self.model_output_path}")
    
    def _save_split_data(self, features: np.ndarray, labels: np.ndarray, 
                        subject_ids: List[str], split_name: str, config_dir: Path) -> None:
        """
        Save features and metadata for a single data split.
        
        Args:
            features (np.ndarray): Feature array with shape (n_samples, feature_dim)
            labels (np.ndarray): Label array with shape (n_samples,)
            subject_ids (List[str]): List of subject identifiers
            split_name (str): Name of the data split
            config_dir (Path): Configuration output directory
        """
        # Generate base filename
        base_name = split_name.replace('.csv', '')
        
        # Save features as .npy
        features_file = config_dir / f"{base_name}_features.npy"
        np.save(features_file, features)
        
        # Save metadata as combined .csv
        metadata_df = pd.DataFrame({
            'Subject': subject_ids,
            'Label': labels
        })
        metadata_file = config_dir / f"{base_name}_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
    
    def _generate_configuration_metadata(self, pipeline: FeatureExtractionPipeline, 
                                        results: Dict) -> Dict:
        """
        Generate comprehensive metadata for a configuration.
        
        Args:
            pipeline (FeatureExtractionPipeline): The pipeline used for extraction
            results (Dict): Processing results for all splits
        
        Returns:
            Dict: Configuration metadata
        """
        # Get pipeline information
        pipeline_info = pipeline.get_pipeline_info()
        
        # Calculate dataset statistics
        total_samples = sum(len(labels) for _, labels, _ in results.values())
        feature_dim = next(iter(results.values()))[0].shape[1]
        
        # Generate metadata
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'configuration_name': pipeline.get_configuration_name(),
            'pipeline_config': pipeline_info['pipeline_config'],
            'extractor_info': pipeline_info['extractor_info'],
            'aggregator_info': pipeline_info['aggregator_info'],
            'dataset_info': {
                'total_samples': int(total_samples),
                'feature_dimension': int(feature_dim),
                'n_splits': len(results),
                'split_names': list(results.keys())
            },
            'split_statistics': {}
        }
        
        # Add per-split statistics
        for split_name, (features, labels, subject_ids) in results.items():
            metadata['split_statistics'][split_name] = {
                'n_samples': len(features),
                'feature_shape': list(features.shape),
                'label_distribution': {int(k): int(v) for k, v in pd.Series(labels).value_counts().items()}
            }
        
        return metadata
    
    def save_configuration(self, pooling_strategy: str, 
                          required_axes: Optional[List[str]] = None) -> None:
        """
        Extract and save features for a specific configuration.
        
        Args:
            pooling_strategy (str): Pooling strategy for feature aggregation
            required_axes (List[str], optional): Axes for processing. Defaults to all axes.
        """
        # Initialize pipeline with configuration
        pipeline = FeatureExtractionPipeline(
            data_path=self.data_path,
            model_name=self.model_name,
            pooling_strategy=pooling_strategy,
            required_axes=required_axes
        )
        
        # Create configuration directory
        config_name = pipeline.get_configuration_name()
        config_dir = self.model_output_path / config_name
        config_dir.mkdir(exist_ok=True)
        
        print(f"Processing configuration: {config_name}")
        
        # Process all splits
        results = pipeline.process_all_splits()
        
        # Save data for each split
        for split_name, (features, labels, subject_ids) in results.items():
            self._save_split_data(features, labels, subject_ids, split_name, config_dir)
        
        # Generate and save metadata
        metadata = self._generate_configuration_metadata(pipeline, results)
        metadata_file = config_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Configuration saved: {config_dir}")
        print(f"  Total samples processed: {metadata['dataset_info']['total_samples']}")
        print(f"  Feature dimension: {metadata['dataset_info']['feature_dimension']}")
    
    def save_all_standard_configurations(self) -> None:
        """
        Extract and save features for all standard configurations.
        
        Processes the following configurations:
        - Multi-axes with average pooling
        - Multi-axes with max pooling  
        - Multi-axes with add pooling
        - Single axis (axial)
        - Single axis (coronal)
        - Single axis (sagittal)
        """
        configurations = [
            # Multi-axes configurations
            ('average', None),
            ('max', None),
            ('add', None),
            # Single-axis configurations
            ('average', ['axial']),
            ('average', ['coronal']),
            ('average', ['sagittal'])
        ]
        
        print(f"Processing {len(configurations)} standard configurations...")
        
        for i, (pooling_strategy, required_axes) in enumerate(configurations, 1):
            print(f"\n=== Configuration {i}/{len(configurations)} ===")
            self.save_configuration(pooling_strategy, required_axes)
        
        print(f"\nAll configurations completed. Output directory: {self.model_output_path}")
    
    def get_saved_configurations(self) -> List[str]:
        """
        Get list of saved configuration names.
        
        Returns:
            List[str]: List of saved configuration directory names
        """
        if not self.model_output_path.exists():
            return []
        
        return [d.name for d in self.model_output_path.iterdir() if d.is_dir()]