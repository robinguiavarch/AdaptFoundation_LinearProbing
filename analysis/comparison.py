"""
Comparison analysis module for AdaptFoundation project.

This module implements comprehensive analysis of classification results
across different foundation models, configurations, and PCA strategies.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ComparisonAnalyzer:
    """
    Analyzer for comparing classification results across different configurations.
    
    This class handles loading and parsing of classification results from
    multiple foundation models, configurations, and PCA strategies.
    
    Attributes:
        features_base_path (Path): Base path to feature_extracted directory
    """
    
    def __init__(self, features_base_path: str):
        """
        Initialize the comparison analyzer.
        
        Args:
            features_base_path (str): Path to feature_extracted directory
        """
        self.features_base_path = Path(features_base_path)
        
        if not self.features_base_path.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_base_path}")
    
    def _parse_consolidated_file(self, filepath: Path, pca_mode: str) -> List[Dict]:
        """
        Parse a consolidated classification_results.json file.
        
        Args:
            filepath (Path): Path to consolidated JSON file
            pca_mode (str): PCA mode (none, 32, 256, 95)
            
        Returns:
            List[Dict]: List of result dictionaries for each config/classifier
        """
        if not filepath.exists():
            return []
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            results = []
            
            # Extract model name from filename
            filename = filepath.name
            if filename.startswith('classification_results_pca_'):
                parts = filename.replace('.json', '').split('_')
                model = '_'.join(parts[4:])
            elif filename.startswith('classification_results_'):
                parts = filename.replace('.json', '').split('_')
                model = '_'.join(parts[2:])
            else:
                return []
            
            # Navigate JSON structure: {config: {classifier: results}}
            for config_name, config_data in data.items():
                if isinstance(config_data, dict):
                    for classifier, result in config_data.items():
                        if isinstance(result, dict) and 'test_metrics' in result:
                            
                            # Extract feature dimension
                            if 'data_info' in result:
                                feature_dim = result['data_info']['train_val_shape'][1]
                            else:
                                feature_dim = None
                            
                            # Extract best parameters
                            best_params = str(result.get('best_params', {}))
                            
                            # Extract diagnostics
                            diagnostics = result.get('diagnostics', {})
                            cv_metrics = result.get('cv_metrics', {})
                            
                            # Handle convergence_warning
                            convergence_warning = diagnostics.get('convergence_warning', False)
                            if isinstance(convergence_warning, str):
                                convergence_ok = convergence_warning.lower() != 'true'
                            else:
                                convergence_ok = not convergence_warning
                            
                            parsed_result = {
                                'model': model,
                                'config': config_name,
                                'pca_mode': pca_mode,
                                'classifier': classifier,
                                'best_params': best_params,
                                'test_roc_auc': result['test_metrics']['roc_auc_weighted'],
                                'cv_roc_auc': result['best_cv_score'],
                                'overfitting_gap': cv_metrics.get('overfitting_gap', None),
                                'feature_dim': feature_dim,
                                'convergence_ok': convergence_ok,
                                'cv_stability': cv_metrics.get('cv_stability', None)
                            }
                            
                            results.append(parsed_result)
            
            return results
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return []
    
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect all classification results from consolidated JSON files.
        
        Parses consolidated classification_results*.json files at the root
        of features_base_path directory.
        
        Returns:
            pd.DataFrame: Complete dataset with all experimental results
        """
        all_results = []
        
        # Define consolidated file patterns
        file_patterns = [
            ('classification_results_*.json', 'none'),
            ('classification_results_pca_32_*.json', '32'),
            ('classification_results_pca_95_*.json', '95'),
            ('classification_results_pca_256_*.json', '256')
        ]
        
        # Process consolidated files
        for pattern, pca_mode in file_patterns:
            files = list(self.features_base_path.glob(pattern))
            
            for filepath in files:
                results = self._parse_consolidated_file(filepath, pca_mode)
                all_results.extend(results)
                print(f"Parsed {len(results)} results from {filepath.name}")
        
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            print(f"Collected {len(df)} experimental results")
            print(f"Models: {df['model'].nunique()}")
            print(f"Configurations: {df['config'].nunique()}")
            print(f"PCA modes: {df['pca_mode'].nunique()}")
            print(f"Classifiers: {df['classifier'].nunique()}")
            print(f"Unique classifiers: {df['classifier'].unique().tolist()}")
        else:
            print("No results collected - check file paths and structure")
        
        return df
    
    def filter_logistic_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset to keep only logistic regression results.
        
        Args:
            df (pd.DataFrame): Complete results dataset
            
        Returns:
            pd.DataFrame: Filtered dataset with logistic regression only
        """
        logistic_df = df[df['classifier'] == 'logistic'].copy()
        
        print(f"Filtered to {len(logistic_df)} logistic regression results")
        
        return logistic_df
    
    def get_top_10_configurations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get top 10 configurations ranked by test ROC-AUC performance.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Top 10 configurations sorted by test_roc_auc descending
        """
        if df.empty:
            print("Warning: Empty dataframe provided to get_top_10_configurations")
            return df
        
        top10_df = df.nlargest(10, 'test_roc_auc').copy()
        
        print(f"Selected top {len(top10_df)} configurations")
        if not top10_df.empty:
            print(f"Best performance: {top10_df.iloc[0]['test_roc_auc']:.4f}")
            if len(top10_df) >= 10:
                print(f"Worst in top 10: {top10_df.iloc[-1]['test_roc_auc']:.4f}")
        
        return top10_df
    
    def create_top10_table(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create styled table for top 10 configurations ranking.
        
        Args:
            df (pd.DataFrame): Top 10 dataset to visualize
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df.empty:
            print("Warning: Empty dataframe for top 10 table")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
            ax.set_title('Top 10 Configurations - Logistic Regression')
            ax.axis('off')
            return fig
        
        # Prepare data for table
        df_table = df.copy().reset_index(drop=True)
        
        # Create table data
        table_data = []
        for i, row in df_table.iterrows():
            model_clean = row['model'].replace('dinov2_', '').upper()
            config_clean = row['config'].replace('_', ' ').replace('multi axes', 'Multi-Axes').replace('single axis', 'Single-Axis')
            pca_clean = 'None' if row['pca_mode'] == 'none' else f"{row['pca_mode']}D"
            gap = f"{row['overfitting_gap']:.3f}" if pd.notna(row['overfitting_gap']) else 'N/A'
            cv_roc_auc = f"{row['cv_roc_auc']:.4f}"
            test_roc_auc = f"{row['test_roc_auc']:.4f}"
            
            table_data.append([
                f"#{i+1}",
                model_clean, 
                config_clean,
                pca_clean,
                gap,
                cv_roc_auc,
                test_roc_auc
            ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('off')
        
        # Define column headers
        columns = ['Rank', 'Model', 'Configuration', 'PCA', 'Overfitting Gap during CV', 'CV ROC-AUC', 'Test ROC-AUC']
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Header styling
        for i in range(len(columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.12)
        
        # Row styling with alternating colors
        for i in range(1, len(table_data) + 1):
            row_color = '#F8F9FA' if i % 2 == 1 else '#E9ECEF'
            
            for j in range(len(columns)):
                cell = table[(i, j)]
                cell.set_facecolor(row_color)
                cell.set_height(0.08)
                
                # Special styling for rank column
                if j == 0:
                    if i == 1:
                        cell.set_facecolor('#FFD700')
                        cell.set_text_props(weight='bold', color='#8B4513')
                    elif i == 2:
                        cell.set_facecolor('#C0C0C0')
                        cell.set_text_props(weight='bold', color='#2F4F4F')
                    elif i == 3:
                        cell.set_facecolor('#CD7F32')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_text_props(weight='bold')
                
                # Special styling for metrics columns
                elif j == 4:
                    cell.set_text_props(weight='bold', color='#D32F2F')
                elif j == 5:
                    cell.set_text_props(weight='bold', color='#1565C0')
                elif j == 6:
                    cell.set_text_props(weight='bold', color='#1B5E20')
                
                # Border styling
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        # Title
        ax.set_title('Top 10 Configurations - Logistic Regression\nRanked by Test ROC-AUC Performance', 
                    fontsize=16, fontweight='bold', pad=30)
        
        # Add subtle note
        fig.text(0.5, 0.02, 'Higher ROC-AUC scores indicate better classification performance', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#666666')
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_heatmap_grid(self, df: pd.DataFrame, title: str) -> plt.Figure:
        """
        Create a 2x3 grid of heatmaps with shared colorbar for comprehensive comparison.
        
        Top row displays Test ROC-AUC for each classifier (Logistic, KNN, SVM).
        Bottom row displays CV ROC-AUC for each classifier (Logistic, KNN, SVM).
        All heatmaps share the same color scale for direct comparison.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            title (str): Title for the figure
            
        Returns:
            plt.Figure: Matplotlib figure containing the 2x3 grid with shared colorbar
        """
        if df.empty:
            print(f"Warning: Empty dataframe for comprehensive heatmap '{title}'")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
        
        # Get unique classifiers and sort them for consistent ordering
        classifiers = sorted(df['classifier'].unique())
        
        # Create the 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # Create combined configuration + PCA labels
        df = df.copy()
        df['config_pca'] = df['config'] + '_' + df['pca_mode']
        
        # Find global min/max for consistent colorbar scale
        all_test_values = df['test_roc_auc'].dropna()
        all_cv_values = df['cv_roc_auc'].dropna()
        global_min = min(all_test_values.min(), all_cv_values.min())
        global_max = max(all_test_values.max(), all_cv_values.max())
        
        # Store all heatmap data for consistent colorbar
        heatmaps = []
        
        # TOP ROW: Test ROC-AUC for each classifier
        for i, classifier in enumerate(classifiers):
            classifier_df = df[df['classifier'] == classifier].copy()
            
            # Create pivot table for test ROC-AUC
            test_data = classifier_df.pivot_table(
                index='config_pca',
                columns='model', 
                values='test_roc_auc',
                aggfunc='first'
            )
            
            # Create heatmap (top row)
            ax_top = axes[0, i]
            heatmap_test = sns.heatmap(
                test_data,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn',
                vmin=global_min,
                vmax=global_max,
                cbar=False,
                ax=ax_top
            )
            heatmaps.append(heatmap_test)
            
            ax_top.set_title(f'{classifier.upper()}\nTest ROC-AUC', fontsize=14, fontweight='bold')
            ax_top.set_xlabel('Foundation Models', fontsize=12)
            if i == 0:
                ax_top.set_ylabel('Configuration + PCA Mode', fontsize=12)
            else:
                ax_top.set_ylabel('')
            
            # Rotate labels
            ax_top.tick_params(axis='x', rotation=45)
            ax_top.tick_params(axis='y', rotation=0)
        
        # BOTTOM ROW: CV ROC-AUC for each classifier
        for i, classifier in enumerate(classifiers):
            classifier_df = df[df['classifier'] == classifier].copy()
            
            # Create pivot table for CV ROC-AUC
            cv_data = classifier_df.pivot_table(
                index='config_pca',
                columns='model', 
                values='cv_roc_auc',
                aggfunc='first'
            )
            
            # Create heatmap (bottom row)
            ax_bottom = axes[1, i]
            heatmap_cv = sns.heatmap(
                cv_data,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn',
                vmin=global_min,
                vmax=global_max,
                cbar=False,
                ax=ax_bottom
            )
            heatmaps.append(heatmap_cv)
            
            ax_bottom.set_title(f'{classifier.upper()}\nCV ROC-AUC', fontsize=14, fontweight='bold')
            ax_bottom.set_xlabel('Foundation Models', fontsize=12)
            if i == 0:
                ax_bottom.set_ylabel('Configuration + PCA Mode', fontsize=12)
            else:
                ax_bottom.set_ylabel('')
            
            # Rotate labels
            ax_bottom.tick_params(axis='x', rotation=45)
            ax_bottom.tick_params(axis='y', rotation=0)
        
        # Add a single colorbar on the right side
        plt.subplots_adjust(right=0.92)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        
        # Use the first heatmap to create the colorbar
        cbar = plt.colorbar(heatmaps[0].collections[0], cax=cbar_ax)
        cbar.set_label('ROC-AUC Score', fontsize=14, fontweight='bold')
        
        # Main title
        fig.suptitle(f'Comprehensive Performance Analysis: {title}', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # Add subtitle explaining the layout
        fig.text(0.5, 0.02, 
                 'Top row: Test set performance | Bottom row: Cross-validation performance', 
                 ha='center', va='bottom', fontsize=12, style='italic')
        
        plt.tight_layout()
        
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete analysis workflow with comprehensive visualization.
        
        Returns:
            Tuple containing:
                - Complete DataFrame (all classifiers)
                - Logistic-only DataFrame  
                - Top-10 DataFrame
                - List of matplotlib figures
        """
        # Collect all results
        df_all = self.collect_all_results()
        
        # Filter to logistic regression only
        df_logistic = self.filter_logistic_only(df_all)
        
        # Get top 10 configurations
        df_top10 = self.get_top_10_configurations(df_logistic)
        
        # Create visualizations
        figures = []
        
        if not df_all.empty:
            # Create comprehensive 2x3 heatmap grid
            fig_comprehensive = self.create_comprehensive_heatmap_grid(df_all, "All Foundation Models")
            figures.append(fig_comprehensive)
        
        if not df_top10.empty:
            fig_top10 = self.create_top10_table(df_top10)
            figures.append(fig_top10)
        
        return df_all, df_logistic, df_top10, figures