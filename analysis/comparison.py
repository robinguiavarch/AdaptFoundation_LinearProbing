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
            # e.g., "classification_results_pca_32_dinov2_vitb14.json"
            filename = filepath.name
            if filename.startswith('classification_results_pca_'):
                # With PCA: classification_results_pca_32_dinov2_vitb14.json
                parts = filename.replace('.json', '').split('_')
                model = '_'.join(parts[4:])  # dinov2_vitb14
            elif filename.startswith('classification_results_'):
                # Without PCA: classification_results_dinov2_vitb14.json
                parts = filename.replace('.json', '').split('_')
                model = '_'.join(parts[2:])  # dinov2_vitb14
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
                            
                            # Handle convergence_warning (can be string or boolean)
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
                if j == 0:  # Rank column
                    if i == 1:  # First place
                        cell.set_facecolor('#FFD700')  # Gold
                        cell.set_text_props(weight='bold', color='#8B4513')
                    elif i == 2:  # Second place
                        cell.set_facecolor('#C0C0C0')  # Silver
                        cell.set_text_props(weight='bold', color='#2F4F4F')
                    elif i == 3:  # Third place
                        cell.set_facecolor('#CD7F32')  # Bronze
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_text_props(weight='bold')
                
                # Special styling for ROC-AUC columns
                elif j == 4:  # Overfitting Gap column
                    cell.set_text_props(weight='bold', color='#D32F2F')
                elif j == 5:  # CV ROC-AUC column
                    cell.set_text_props(weight='bold', color='#1565C0')
                elif j == 6:  # Test ROC-AUC column (final column)
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
    
    def create_performance_heatmap(self, df: pd.DataFrame, title: str) -> plt.Figure:
        """
        Create performance heatmap for models vs configurations.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            title (str): Title for the heatmap
            
        Returns:
            plt.Figure: Matplotlib figure containing the heatmap
        """
        if df.empty:
            print(f"Warning: Empty dataframe for heatmap '{title}'")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
        
        # Create combined configuration + PCA labels
        df = df.copy()
        df['config_pca'] = df['config'] + '_' + df['pca_mode']
        
        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            index='config_pca',
            columns='model', 
            values='test_roc_auc',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Test ROC-AUC'},
            ax=ax
        )
        
        ax.set_title(f'Performance Heatmap: {title}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Foundation Models', fontsize=12)
        ax.set_ylabel('Configuration + PCA Mode', fontsize=12)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        return fig
        """
        Create performance heatmap for models vs configurations.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            title (str): Title for the heatmap
            
        Returns:
            plt.Figure: Matplotlib figure containing the heatmap
        """
        if df.empty:
            print(f"Warning: Empty dataframe for heatmap '{title}'")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
        
        # Create combined configuration + PCA labels
        df = df.copy()
        df['config_pca'] = df['config'] + '_' + df['pca_mode']
        
        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            index='config_pca',
            columns='model', 
            values='test_roc_auc',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Test ROC-AUC'},
            ax=ax
        )
        
        ax.set_title(f'Performance Heatmap: {title}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Foundation Models', fontsize=12)
        ax.set_ylabel('Configuration + PCA Mode', fontsize=12)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete analysis workflow.
        
        Returns:
            Tuple containing:
                - Complete DataFrame (all classifiers)
                - Logistic-only DataFrame  
                - Top-10 DataFrame
                - List of matplotlib figures (heatmaps)
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
            fig_a = self.create_performance_heatmap(df_all, "All Classifiers")
            figures.append(fig_a)
        
        if not df_logistic.empty:
            fig_b = self.create_performance_heatmap(df_logistic, "Logistic Regression Only")
            figures.append(fig_b)
            
        if not df_top10.empty:
            fig_c = self.create_top10_table(df_top10)
            figures.append(fig_c)
        
        return df_all, df_logistic, df_top10, figures