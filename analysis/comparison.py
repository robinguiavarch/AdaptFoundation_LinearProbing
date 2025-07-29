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
from matplotlib.colors import LinearSegmentedColormap


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
            
            # Configuration mapping for anatomical axes
            config_mapping = {
                'single_axis_sagittal': 'axis_X',
                'single_axis_coronal': 'axis_Y', 
                'single_axis_axial': 'axis_Z'
            }
            
            # Navigate JSON structure: {config: {classifier: results}}
            for config_name, config_data in data.items():
                if isinstance(config_data, dict):
                    for classifier, result in config_data.items():
                        # Skip SVM classifier
                        if classifier == 'svm_linear':
                            continue
                            
                        if isinstance(result, dict) and 'test_metrics' in result:
                            
                            # Apply configuration mapping
                            mapped_config = config_mapping.get(config_name, config_name)
                            
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
                                'config': mapped_config,
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
        
        print(f"Selected top {len(top10_df)} configurations (Test ROC-AUC)")
        if not top10_df.empty:
            print(f"Best Test performance: {top10_df.iloc[0]['test_roc_auc']:.4f}")
            if len(top10_df) >= 10:
                print(f"Worst in top 10: {top10_df.iloc[-1]['test_roc_auc']:.4f}")
        
        return top10_df
    
    def get_top_10_configurations_cv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get top 10 configurations ranked by CV ROC-AUC performance.
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Top 10 configurations sorted by cv_roc_auc descending
        """
        if df.empty:
            print("Warning: Empty dataframe provided to get_top_10_configurations_cv")
            return df
        
        top10_cv_df = df.nlargest(10, 'cv_roc_auc').copy()
        
        print(f"Selected top {len(top10_cv_df)} configurations (CV ROC-AUC)")
        if not top10_cv_df.empty:
            print(f"Best CV performance: {top10_cv_df.iloc[0]['cv_roc_auc']:.4f}")
            if len(top10_cv_df) >= 10:
                print(f"Worst in top 10: {top10_cv_df.iloc[-1]['cv_roc_auc']:.4f}")
        
        return top10_cv_df
    
    def create_top10_table(self, df: pd.DataFrame, metric_type: str = "test") -> plt.Figure:
        """
        Create styled table for top 10 configurations ranking.
        
        Args:
            df (pd.DataFrame): Top 10 dataset to visualize
            metric_type (str): Type of metric used for ranking ("test" or "cv")
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df.empty:
            print("Warning: Empty dataframe for top 10 table")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
            ax.set_title(f'Top 10 Configurations - Logistic Regression ({metric_type.upper()})')
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
                elif j == 5:  # CV ROC-AUC column
                    if metric_type == "cv":
                        cell.set_text_props(weight='bold', color='#1B5E20')  # Green for primary metric
                    else:
                        cell.set_text_props(weight='bold', color='#1565C0')  # Blue for secondary
                elif j == 6:  # Test ROC-AUC column
                    if metric_type == "test":
                        cell.set_text_props(weight='bold', color='#1B5E20')  # Green for primary metric
                    else:
                        cell.set_text_props(weight='bold', color='#1565C0')  # Blue for secondary
                
                # Border styling
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        # Title with metric type
        metric_display = "Test ROC-AUC" if metric_type == "test" else "CV ROC-AUC"
        ax.set_title(f'Top 10 Configurations - Logistic Regression\nRanked by {metric_display} Performance', 
                    fontsize=16, fontweight='bold', pad=30)
        
        # Add subtle note
        fig.text(0.5, 0.02, f'Ranked by {metric_display} - Higher scores indicate better classification performance', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#666666')
        
        plt.tight_layout()
        return fig
    
    def create_simple_grid_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create a 2x2 grid heatmap with simple column labels and color-coded model names.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            
        Returns:
            plt.Figure: Matplotlib figure containing the grid
        """
        if df.empty:
            print("Warning: Empty dataframe for grid heatmap")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
            return fig
        
        # Filter out SVM and get classifiers
        df_filtered = df[df['classifier'] != 'svm_linear'].copy()
        classifiers = sorted(df_filtered['classifier'].unique())
        
        # Create the 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(32, 16))
        
        # Define configuration order for better readability
        config_order = ['multi_axes_add', 'multi_axes_average', 'multi_axes_max', 
                       'axis_Z', 'axis_Y', 'axis_X']
        
        # Create nested column structure: model_pca
        df_filtered['model_pca'] = df_filtered['model'] + '_' + df_filtered['pca_mode']
        
        # Define model and PCA order for consistent display
        models = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
        pca_modes = ['none', '256', '32', '95']
        
        # Create ordered column list for nested structure
        ordered_columns = []
        for model in models:
            for pca in pca_modes:
                ordered_columns.append(f"{model}_{pca}")
        
        # Create human-readable column labels with color coding
        column_labels = []
        model_colors = {
            'dinov2_vits14': '#1f4e79',  # Bleu foncé
            'dinov2_vitb14': '#2d5016',  # Vert foncé
            'dinov2_vitl14': '#4a148c',  # Violet foncé
            'dinov2_vitg14': '#5d4037'   # Marron foncé
        }
        
        for model in models:
            model_clean = model.replace('dinov2_', '').upper()
            for pca in pca_modes:
                pca_clean = 'None' if pca == 'none' else f'PCA{pca}'
                column_labels.append(f"{model_clean}_{pca_clean}")
        
        # Find global min/max for consistent colorbar scale
        all_test_values = df_filtered['test_roc_auc'].dropna()
        all_cv_values = df_filtered['cv_roc_auc'].dropna()
        global_min = min(all_test_values.min(), all_cv_values.min())
        global_max = max(all_test_values.max(), all_cv_values.max())
        
        # Create custom colormap (Rouge → Orange → Jaune → Vert pâle → Vert foncé)
        colors = ['#d32f2f', '#ff6f00', '#ffd600', '#8bc34a', '#2e7d32']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Store heatmaps for shared colorbar
        heatmaps = []
        
        # Process each subplot (2x2 grid)
        for row in range(2):
            for col, classifier in enumerate(classifiers):
                ax = axes[row, col]
                classifier_df = df_filtered[df_filtered['classifier'] == classifier].copy()
                
                # Choose metric based on row
                metric = 'test_roc_auc' if row == 0 else 'cv_roc_auc'
                
                # Create pivot table
                pivot_data = classifier_df.pivot_table(
                    index='config',
                    columns='model_pca', 
                    values=metric,
                    aggfunc='first'
                )
                
                # Reorder rows and columns
                pivot_data = pivot_data.reindex(index=config_order)
                pivot_data = pivot_data.reindex(columns=ordered_columns)
                
                # Create heatmap
                heatmap = sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt='.4f',
                    cmap=cmap,
                    vmin=global_min,
                    vmax=global_max,
                    cbar=False,
                    ax=ax,
                    annot_kws={'size': 8},
                    linewidths=0.2,       
                    linecolor='black'     
                )
                heatmaps.append(heatmap)
                
                # Set title
                metric_label = 'Test ROC-AUC' if row == 0 else 'CV ROC-AUC'
                ax.set_title(f'{classifier.upper()}\n{metric_label}', fontsize=14, fontweight='bold')
                
                # Configure axes
                ax.set_xlabel('')
                if col == 0:
                    ax.set_ylabel('Configuration', fontsize=12)
                else:
                    ax.set_ylabel('')
                ax.tick_params(axis='y', rotation=0)
                
                # Set custom x-tick labels with color coding
                ax.set_xticks(np.arange(len(column_labels)) + 0.5)  
                ax.set_xticklabels(column_labels, rotation=45, ha='center', fontsize=9)  
                
                # Color-code the x-tick labels by model
                for i, label in enumerate(ax.get_xticklabels()):
                    model_idx = i // len(pca_modes)  # Which model group (0,1,2,3)
                    model_key = models[model_idx]
                    color = model_colors[model_key]
                    label.set_color(color)
                    label.set_fontweight('bold')
        
        # Adjust layout for better spacing
        plt.subplots_adjust(left=0.06, right=0.85, top=0.92, bottom=0.15, hspace=0.3, wspace=0.15)
        
        # Add single colorbar on the right side
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(heatmaps[0].collections[0], cax=cbar_ax)
        cbar.set_label('ROC-AUC Score', fontsize=14, fontweight='bold')
        
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete analysis workflow with simple grid visualization.
        
        Returns:
            Tuple containing:
                - Complete DataFrame (KNN + Logistic only)
                - Logistic-only DataFrame  
                - Top-10 DataFrame (Test ROC-AUC)
                - Top-10 DataFrame (CV ROC-AUC)
                - List of matplotlib figures
        """
        # Collect all results
        df_all = self.collect_all_results()
        
        # Filter to logistic regression only
        df_logistic = self.filter_logistic_only(df_all)
        
        # Get top 10 configurations for both metrics
        df_top10_test = self.get_top_10_configurations(df_logistic)
        df_top10_cv = self.get_top_10_configurations_cv(df_logistic)
        
        # Create visualizations
        figures = []
        
        if not df_all.empty:
            # Create simple grid heatmap (2x2, no SVM)
            fig_grid = self.create_simple_grid_heatmap(df_all)
            figures.append(fig_grid)
        
        if not df_top10_test.empty:
            fig_top10_test = self.create_top10_table(df_top10_test, metric_type="test")
            figures.append(fig_top10_test)
        
        if not df_top10_cv.empty:
            fig_top10_cv = self.create_top10_table(df_top10_cv, metric_type="cv")
            figures.append(fig_top10_cv)
        
        return df_all, df_logistic, df_top10_test, df_top10_cv, figures