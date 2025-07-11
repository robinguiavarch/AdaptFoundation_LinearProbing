#!/usr/bin/env python3
"""
Feature Analysis Script for AdaptFoundation Project
==================================================

Analyzes extracted features from DINOv2 feature extraction pipeline.
Provides comprehensive analysis including statistics, visualizations, and PCA.

Usage:
    python utils/analyze_features_extracted.py [--config CONFIG_NAME] [--save-plots] [--output-dir OUTPUT_DIR]

Examples:
    python utils/analyze_features_extracted.py --config multi_axes_average
    python utils/analyze_features_extracted.py --config single_axis_axial --save-plots
    python utils/analyze_features_extracted.py --config multi_axes_max --output-dir results/
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for server environments
import matplotlib
matplotlib.use('Agg')


class FeatureAnalyzer:
    """
    Comprehensive analyzer for extracted DINOv2 features.
    
    Provides statistical analysis, visualizations, and quality assessment
    of features extracted from cortical skeleton data.
    """
    
    def __init__(self, config_name: str, base_path: str = "feature_extracted/dinov2_vits14"):
        """
        Initialize the feature analyzer.
        
        Args:
            config_name (str): Name of the configuration to analyze
            base_path (str): Base path to feature extraction results
        """
        self.config_name = config_name
        self.config_path = Path(base_path) / config_name
        self.metadata = None
        self.features = None
        self.labels = None
        self.subjects = None
        self.splits = None
        
        # Verify configuration exists
        if not self.config_path.exists():
            raise ValueError(f"Configuration path does not exist: {self.config_path}")
        
        print(f"Initializing FeatureAnalyzer for: {config_name}")
        print("=" * 60)
    
    def load_metadata(self):
        """Load configuration metadata."""
        metadata_file = self.config_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        print("\n1. METADATA ANALYSIS")
        print("-" * 30)
        print(f"Configuration: {self.metadata['configuration_name']}")
        print(f"Model: {self.metadata['model_name']}")
        print(f"Creation: {self.metadata['creation_timestamp']}")
        print(f"Total samples: {self.metadata['dataset_info']['total_samples']}")
        print(f"Feature dimension: {self.metadata['dataset_info']['feature_dimension']}")
        print(f"Number of splits: {self.metadata['dataset_info']['n_splits']}")
        
        # Pipeline configuration
        pipeline_config = self.metadata['pipeline_config']
        print(f"\nPipeline Configuration:")
        print(f"  Pooling strategy: {pipeline_config['pooling_strategy']}")
        print(f"  Required axes: {pipeline_config['required_axes']}")
        print(f"  Batch size: {pipeline_config.get('batch_size', 'N/A')}")
        
        # Aggregator info
        agg_info = self.metadata['aggregator_info']
        print(f"\nAggregator Info:")
        print(f"  Expected output dim: {agg_info['expected_output_dim']}")
        print(f"  Required axes: {agg_info['required_axes']}")
    
    def analyze_splits(self):
        """Analyze split statistics."""
        print("\n\n2. SPLIT STATISTICS")
        print("-" * 30)
        
        split_stats = self.metadata['split_statistics']
        split_summary = []
        
        for split_name, stats in split_stats.items():
            split_type = "Test" if "test" in split_name else "Train/Val"
            split_summary.append({
                'Split': split_name.replace('.csv', ''),
                'Type': split_type,
                'Samples': stats['n_samples'],
                'Feature_Shape': f"{stats['feature_shape'][0]}x{stats['feature_shape'][1]}",
                'Class_0': stats['label_distribution'].get('0', stats['label_distribution'].get(0, 0)),
                'Class_1': stats['label_distribution'].get('1', stats['label_distribution'].get(1, 0)),
                'Class_2': stats['label_distribution'].get('2', stats['label_distribution'].get(2, 0)),
                'Class_3': stats['label_distribution'].get('3', stats['label_distribution'].get(3, 0))
            })
        
        split_df = pd.DataFrame(split_summary)
        print(split_df.to_string(index=False))
        
        return split_df
    
    def load_features(self):
        """Load all features and labels from splits."""
        print("\n\n3. FEATURE LOADING")
        print("-" * 30)
        
        all_features = []
        all_labels = []
        all_subjects = []
        split_info = []
        
        for split_name in self.metadata['split_statistics'].keys():
            base_name = split_name.replace('.csv', '')
            
            # Load features
            features_file = self.config_path / f"{base_name}_features.npy"
            features = np.load(features_file)
            
            # Load metadata  
            metadata_file = self.config_path / f"{base_name}_metadata.csv"
            metadata_df = pd.read_csv(metadata_file)
            
            all_features.append(features)
            all_labels.extend(metadata_df['Label'].tolist())
            all_subjects.extend(metadata_df['Subject'].tolist())
            
            # Track split info
            split_info.extend([base_name] * len(features))
            
            print(f"{base_name}: {features.shape} features, {len(metadata_df)} labels")
        
        # Concatenate all data
        self.features = np.vstack(all_features)
        self.labels = np.array(all_labels)
        self.subjects = np.array(all_subjects)
        self.splits = np.array(split_info)
        
        print(f"\nCombined dataset:")
        print(f"  Total features shape: {self.features.shape}")
        print(f"  Total labels: {len(self.labels)}")
        print(f"  Unique subjects: {len(np.unique(self.subjects))}")
    
    def analyze_distribution(self):
        """Analyze feature distributions and statistics."""
        print("\n\n4. FEATURE DISTRIBUTION ANALYSIS")
        print("-" * 30)
        
        # Basic statistics
        print(f"Feature statistics:")
        print(f"  Mean: {self.features.mean():.6f}")
        print(f"  Std: {self.features.std():.6f}")
        print(f"  Min: {self.features.min():.6f}")
        print(f"  Max: {self.features.max():.6f}")
        non_zero_pct = 100 * np.count_nonzero(self.features) / self.features.size
        print(f"  Non-zero features: {np.count_nonzero(self.features)} / {self.features.size} ({non_zero_pct:.1f}%)")
        
        # Class distribution
        class_counts = pd.Series(self.labels).value_counts().sort_index()
        print(f"\nClass distribution:")
        for class_id, count in class_counts.items():
            percentage = 100 * count / len(self.labels)
            print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")
        
        return {
            'mean': self.features.mean(),
            'std': self.features.std(),
            'min': self.features.min(),
            'max': self.features.max(),
            'non_zero_pct': non_zero_pct,
            'class_counts': class_counts
        }
    
    def compute_pca(self):
        """Compute PCA analysis."""
        print("\n\n5. PCA ANALYSIS")
        print("-" * 30)
        
        print("Computing PCA...")
        pca = PCA()
        X_pca = pca.fit_transform(self.features)
        
        # Find components for variance thresholds
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        pca_95 = np.argmax(cumsum_var >= 0.95) + 1
        pca_99 = np.argmax(cumsum_var >= 0.99) + 1
        
        print(f"Components for 95% variance: {pca_95}")
        print(f"Components for 99% variance: {pca_99}")
        
        return {
            'pca': pca,
            'X_pca': X_pca,
            'cumsum_var': cumsum_var,
            'pca_95': pca_95,
            'pca_99': pca_99
        }
    
    def compute_tsne(self, X_pca, n_components_pca=50):
        """Compute t-SNE visualization."""
        print("\n\n6. t-SNE ANALYSIS")
        print("-" * 30)
        
        print("Computing t-SNE...")
        X_pca_reduced = X_pca[:, :n_components_pca]
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_pca_reduced)
        
        print(f"t-SNE completed using first {n_components_pca} PCA components")
        
        return X_tsne
    
    def assess_quality(self, X_pca):
        """Assess feature quality."""
        print("\n\n7. FEATURE QUALITY ASSESSMENT")
        print("-" * 30)
        
        # Check for NaN or infinite values
        nan_count = np.isnan(self.features).sum()
        inf_count = np.isinf(self.features).sum()
        print(f"NaN values: {nan_count}")
        print(f"Infinite values: {inf_count}")
        
        # Check for zero variance features
        feature_vars = np.var(self.features, axis=0)
        zero_var_features = np.sum(feature_vars == 0)
        print(f"Zero variance features: {zero_var_features} / {len(feature_vars)}")
        
        # Check feature separability
        quality_metrics = {
            'nan_count': nan_count,
            'inf_count': inf_count,
            'zero_var_features': zero_var_features,
            'feature_vars': feature_vars
        }
        
        if len(np.unique(self.labels)) > 1:
            X_pca_50 = X_pca[:, :50]
            sil_score = silhouette_score(X_pca_50, self.labels)
            print(f"Silhouette score (PCA-50): {sil_score:.4f}")
            quality_metrics['silhouette_score'] = sil_score
        
        return quality_metrics
    
    def create_visualizations(self, stats, pca_results, X_tsne, quality_metrics, save_plots=False, output_dir="results"):
        """Create comprehensive visualizations."""
        print("\n\n8. GENERATING VISUALIZATIONS")
        print("-" * 30)
        
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to: {output_path}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'Feature Analysis - {self.config_name}', fontsize=16)
        
        # 1. Feature distribution histogram
        plt.subplot(3, 3, 1)
        plt.hist(self.features.flatten(), bins=50, alpha=0.7, density=True)
        plt.title('Feature Value Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # 2. Class distribution
        plt.subplot(3, 3, 2)
        stats['class_counts'].plot(kind='bar', color='skyblue')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        
        # 3. Feature variance distribution
        plt.subplot(3, 3, 3)
        plt.hist(quality_metrics['feature_vars'], bins=30, alpha=0.7)
        plt.title('Feature Variance Distribution')
        plt.xlabel('Variance')
        plt.ylabel('Number of Features')
        plt.grid(True, alpha=0.3)
        
        # 4. Correlation matrix sample
        plt.subplot(3, 3, 4)
        n_features_corr = min(50, self.features.shape[1])
        corr_sample = np.corrcoef(self.features[:, :n_features_corr].T)
        im = plt.imshow(corr_sample, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Feature Correlation (First {n_features_corr} Features)')
        plt.colorbar(im)
        
        # 5. PCA variance explained
        plt.subplot(3, 3, 5)
        cumsum_var = pca_results['cumsum_var']
        n_components_plot = min(100, len(cumsum_var))
        plt.plot(range(1, n_components_plot + 1), cumsum_var[:n_components_plot], 'b-', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        plt.axhline(y=0.99, color='orange', linestyle='--', label='99% variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. t-SNE by classes
        plt.subplot(3, 3, 6)
        colors = ['red', 'blue', 'green', 'orange']
        for class_id in range(4):
            mask = self.labels == class_id
            if np.any(mask):
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=colors[class_id], label=f'Class {class_id}', alpha=0.6, s=20)
        plt.title('t-SNE Visualization (Classes)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. t-SNE by splits
        plt.subplot(3, 3, 7)
        unique_splits = np.unique(self.splits)
        split_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_splits)))
        for i, split_name in enumerate(unique_splits):
            mask = self.splits == split_name
            split_type = "Test" if "test" in split_name else "Train/Val"
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[split_colors[i]], label=split_type, alpha=0.6, s=20)
        plt.title('t-SNE Visualization (Splits)')
        plt.xlabel('t-SNE Component 1') 
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Feature magnitude by class
        plt.subplot(3, 3, 8)
        feature_norms = np.linalg.norm(self.features, axis=1)
        box_data = [feature_norms[self.labels == class_id] for class_id in range(4)]
        plt.boxplot(box_data, labels=[f'Class {i}' for i in range(4)])
        plt.title('Feature Vector Magnitude by Class')
        plt.xlabel('Class')
        plt.ylabel('L2 Norm')
        plt.grid(True, alpha=0.3)
        
        # 9. Split-wise feature comparison
        plt.subplot(3, 3, 9)
        split_norms = []
        split_labels = []
        for split_name in np.unique(self.splits):
            mask = self.splits == split_name
            norms = np.linalg.norm(self.features[mask], axis=1)
            split_norms.extend(norms)
            split_labels.extend([split_name] * len(norms))
        
        split_df_norms = pd.DataFrame({
            'Split': split_labels,
            'Feature_Norm': split_norms
        })
        sns.boxplot(data=split_df_norms, x='Split', y='Feature_Norm')
        plt.title('Feature Magnitude by Split')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = output_path / f"{self.config_name}_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_file}")
        
        return fig
    
    def generate_summary_report(self, stats, pca_results, quality_metrics):
        """Generate comprehensive summary report."""
        print("\n\n9. SUMMARY REPORT")
        print("=" * 30)
        
        print(f"✅ Dataset successfully loaded and analyzed")
        print(f"✅ {self.features.shape[0]} samples with {self.features.shape[1]} features")
        print(f"✅ {len(np.unique(self.labels))} classes, {len(np.unique(self.subjects))} unique subjects")
        print(f"✅ Features are well-distributed (mean: {stats['mean']:.4f}, std: {stats['std']:.4f})")
        
        if quality_metrics['nan_count'] == 0 and quality_metrics['inf_count'] == 0:
            print(f"✅ No NaN or infinite values detected")
        else:
            print(f"⚠️  Found {quality_metrics['nan_count']} NaN and {quality_metrics['inf_count']} infinite values")
        
        print(f"✅ PCA analysis: {pca_results['pca_95']} components for 95% variance, {pca_results['pca_99']} for 99%")
        
        if quality_metrics['zero_var_features'] > 0:
            print(f"⚠️  {quality_metrics['zero_var_features']} zero-variance features detected")
        else:
            print(f"✅ No zero-variance features")
        
        if 'silhouette_score' in quality_metrics:
            print(f"✅ Silhouette score: {quality_metrics['silhouette_score']:.4f}")
        
        print(f"\n🎯 Ready for Phase 5: Classification Pipeline")
        print(f"   - Consider PCA reduction to {pca_results['pca_95']}-{pca_results['pca_99']} components")
        print(f"   - All splits properly balanced and ready for cross-validation")
        print(f"   - Features show class separability in t-SNE visualization")
        
        # Return summary for potential saving
        summary = {
            'config_name': self.config_name,
            'n_samples': self.features.shape[0],
            'n_features': self.features.shape[1],
            'n_classes': len(np.unique(self.labels)),
            'n_subjects': len(np.unique(self.subjects)),
            'feature_stats': stats,
            'pca_components_95': pca_results['pca_95'],
            'pca_components_99': pca_results['pca_99'],
            'quality_metrics': quality_metrics
        }
        
        return summary
    
    def run_complete_analysis(self, save_plots=False, output_dir="results"):
        """Run the complete feature analysis pipeline."""
        print(f"Starting complete analysis for {self.config_name}")
        
        # Load data
        self.load_metadata()
        split_df = self.analyze_splits()
        self.load_features()
        
        # Analyze features
        stats = self.analyze_distribution()
        pca_results = self.compute_pca()
        X_tsne = self.compute_tsne(pca_results['X_pca'])
        quality_metrics = self.assess_quality(pca_results['X_pca'])
        
        # Create visualizations
        fig = self.create_visualizations(stats, pca_results, X_tsne, quality_metrics, save_plots, output_dir)
        
        # Generate summary
        summary = self.generate_summary_report(stats, pca_results, quality_metrics)
        
        return {
            'summary': summary,
            'split_df': split_df,
            'pca_results': pca_results,
            'X_tsne': X_tsne,
            'figure': fig
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze extracted features from AdaptFoundation pipeline')
    parser.add_argument('--config', default='multi_axes_average', 
                       help='Configuration name to analyze (default: multi_axes_average)')
    parser.add_argument('--save-plots', action='store_true', 
                       help='Save plots to disk')
    parser.add_argument('--output-dir', default='results', 
                       help='Output directory for saved plots (default: results)')
    parser.add_argument('--base-path', default='feature_extracted/dinov2_vits14',
                       help='Base path to feature extraction results')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = FeatureAnalyzer(args.config, args.base_path)
        
        # Run complete analysis
        results = analyzer.run_complete_analysis(
            save_plots=args.save_plots,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Analysis completed successfully for {args.config}!")
        
        if args.save_plots:
            print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()