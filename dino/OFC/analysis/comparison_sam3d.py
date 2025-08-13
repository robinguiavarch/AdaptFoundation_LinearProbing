"""
Comparison analysis module for AdaptFoundation project - SAM-MED3D STRATEGY.

This module implements comprehensive analysis of SAM-Med3D-based classification results
across different aggregation methods and PCA strategies, including PCA 99%.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class ComparisonAnalyzerSAM3D:
    """
    Analyzer for comparing SAM-Med3D-based classification results across different configurations.
    
    This class handles loading and parsing of classification results from
    SAM-Med3D with different aggregation methods and PCA strategies, including PCA 99%.
    
    Attributes:
        features_base_path (Path): Base path to feature_extracted_sam3d directory
    """
    
    def __init__(self, features_base_path: str):
        """
        Initialize the SAM-Med3D comparison analyzer.
        
        Args:
            features_base_path (str): Path to feature_extracted_sam3d directory
        """
        self.features_base_path = Path(features_base_path)
        
        if not self.features_base_path.exists():
            raise FileNotFoundError(f"SAM-Med3D features directory not found: {self.features_base_path}")
    
    def _parse_consolidated_file(self, filepath: Path, pca_mode: str) -> List[Dict]:
        """
        Parse a consolidated classification_results.json file for SAM-Med3D results.
        
        Args:
            filepath (Path): Path to consolidated JSON file
            pca_mode (str): PCA mode (32, 95, 99, 256)
            
        Returns:
            List[Dict]: List of result dictionaries for each config/classifier
        """
        if not filepath.exists():
            return []
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            results = []
            seen_combinations = set()
            
            # Extract model name from filename - SAM-Med3D uses sam_med3d_turbo
            filename = filepath.name
            if filename.startswith('classification_results_pca_'):
                parts = filename.replace('.json', '').split('_')
                model = '_'.join(parts[4:])
            elif filename.startswith('classification_results_'):
                parts = filename.replace('.json', '').split('_')
                model = '_'.join(parts[2:])
            else:
                return []
            
            # Configuration mapping for SAM-Med3D aggregation methods
            config_mapping = {
                'avg_pool': 'avg_pool',
                'max_pool': 'max_pool',
                'sum_pool': 'sum_pool',
                'flatten': 'flatten'
            }
            
            # Navigate JSON structure: {config: {classifier: results}}
            for config_name, config_data in data.items():
                if isinstance(config_data, dict):
                    for classifier, result in config_data.items():
                        # Only process logistic regression
                        if classifier != 'logistic':
                            continue
                            
                        if isinstance(result, dict) and 'test_metrics' in result:
                            
                            # Apply configuration mapping
                            mapped_config = config_mapping.get(config_name, config_name)
                            
                            # Extract feature dimension
                            if 'data_info' in result:
                                feature_dim = result['data_info']['train_val_shape'][1]
                            else:
                                feature_dim = None
                            
                            # Create unique key to detect duplicates
                            unique_key = (model, mapped_config, pca_mode, classifier)
                            
                            if unique_key in seen_combinations:
                                continue
                            
                            seen_combinations.add(unique_key)
                            
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
                                'cv_stability': cv_metrics.get('cv_stability', None),
                                'strategy': 'sam_med3d'
                            }
                            
                            results.append(parsed_result)
            
            return results
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return []
    
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect all SAM-Med3D classification results from consolidated JSON files.
        
        Returns:
            pd.DataFrame: Complete dataset with all SAM-Med3D experimental results including PCA 99%
        """
        all_results = []
        
        # File patterns for SAM-Med3D results - All PCA configurations including 99%
        file_patterns = [
            ('classification_results_pca_32_sam_med3d_turbo.json', '32'),
            ('classification_results_pca_95_sam_med3d_turbo.json', '95'),
            ('classification_results_pca_99_sam_med3d_turbo.json', '99'),
            ('classification_results_pca_256_sam_med3d_turbo.json', '256')
        ]
        
        # Process consolidated files
        for pattern, pca_mode in file_patterns:
            files = list(self.features_base_path.glob(pattern))
            
            for filepath in files:
                print(f"Processing SAM-Med3D file: {filepath.name} (PCA {pca_mode})")
                results = self._parse_consolidated_file(filepath, pca_mode)
                if results:
                    print(f"  Found {len(results)} results")
                    all_results.extend(results)
                else:
                    print(f"  No results found or file missing")
        
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            print(f"Collected {len(df)} SAM-Med3D experimental results")
            print(f"Models: {df['model'].nunique()}")
            print(f"Configurations: {df['config'].nunique()}")
            print(f"PCA modes: {sorted(df['pca_mode'].unique())}")
        else:
            print("No SAM-Med3D results collected")
        
        return df
    
    def create_complete_table(self, df: pd.DataFrame, metric_type: str = "test") -> plt.Figure:
        """
        Create styled table for complete SAM-Med3D configurations ranking.
        
        Args:
            df (pd.DataFrame): SAM-Med3D dataset to visualize
            metric_type (str): Type of metric used for ranking ("test" or "cv")
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No SAM-Med3D Data Available', ha='center', va='center', fontsize=16)
            ax.set_title(f'Complete SAM-Med3D Configurations - Logistic Regression ({metric_type.upper()})')
            ax.axis('off')
            return fig
        
        # Sort by the specified metric
        sort_column = 'test_roc_auc' if metric_type == "test" else 'cv_roc_auc'
        df_sorted = df.sort_values(sort_column, ascending=False).reset_index(drop=True)
        
        # Create table data
        table_data = []
        for i, row in df_sorted.iterrows():
            model_clean = row['model'].replace('sam_med3d_', '').replace('sam3d_', '').upper()
            config_clean = row['config'].replace('_', ' ').title()
            
            # Handle PCA mode display - Special formatting for 99%
            if row['pca_mode'] == '99':
                pca_clean = "99% Var"  # Special display for 99% variance
            elif row['pca_mode'] == '95':
                pca_clean = "95% Var"
            else:
                pca_clean = f"{row['pca_mode']}D"
            
            gap = f"{row['overfitting_gap']:.3f}" if pd.notna(row['overfitting_gap']) else 'N/A'
            cv_roc_auc = f"{row['cv_roc_auc']:.4f}"
            test_roc_auc = f"{row['test_roc_auc']:.4f}"
            
            # Add feature dimension
            feature_dim = f"{int(row['feature_dim'])}D" if pd.notna(row['feature_dim']) else 'N/A'
            
            table_data.append([
                f"#{i+1}",
                model_clean, 
                config_clean,
                pca_clean,
                feature_dim,
                gap,
                cv_roc_auc,
                test_roc_auc
            ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.axis('off')
        
        # Define column headers
        columns = ['Rank', 'Model', 'Configuration', 'PCA', 'Feature Dim', 'Overfitting Gap', 'CV ROC-AUC', 'Test ROC-AUC']
        
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
        table.set_fontsize(10)
        table.scale(1, 2.0)
        
        # Header styling with SAM-Med3D theme
        for i in range(len(columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')  # Green theme for SAM-Med3D
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.12)
        
        # Row styling with alternating colors
        for i in range(1, len(table_data) + 1):
            row_color = '#E8F5E8' if i % 2 == 1 else '#C8E6C9'  # Green theme variations
            
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
                
                # Special styling for PCA column - Highlight PCA 99%
                elif j == 3:  # PCA column
                    if table_data[i-1][3] == "99% Var":  # Highlight PCA 99%
                        cell.set_facecolor('#81C784')  # Lighter green for 99%
                        cell.set_text_props(weight='bold', color='#1B5E20')
                    else:
                        cell.set_text_props(weight='bold', color='#2E7D32')
                
                # Special styling for feature dimension column
                elif j == 4:  # Feature Dimension column
                    cell.set_text_props(weight='bold', color='#2E7D32')
                elif j == 5:  # Overfitting Gap column
                    cell.set_text_props(weight='bold', color='#D32F2F')
                elif j == 6:  # CV ROC-AUC column
                    if metric_type == "cv":
                        cell.set_text_props(weight='bold', color='#1B5E20')
                    else:
                        cell.set_text_props(weight='bold', color='#1565C0')
                elif j == 7:  # Test ROC-AUC column
                    if metric_type == "test":
                        cell.set_text_props(weight='bold', color='#1B5E20')
                    else:
                        cell.set_text_props(weight='bold', color='#1565C0')
                
                # Border styling
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        # Title
        metric_display = "Test ROC-AUC" if metric_type == "test" else "CV ROC-AUC"
        n_configs = len(df_sorted)
        pca_modes_str = ", ".join(sorted(df['pca_mode'].unique()))
        
        ax.set_title(f'SAM-Med3D - Logistic Regression ({n_configs} Configurations)\n'
                    f'PCA Modes: {pca_modes_str} - Ranked by {metric_display}', 
                    fontsize=16, fontweight='bold', pad=30)
        
        # Add note about PCA 99%
        fig.text(0.5, 0.02, f'SAM-Med3D native 3D processing with PCA 99% variance preservation - Ranked by {metric_display}', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#2E7D32')
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete SAM-Med3D analysis workflow including PCA 99%.
        
        Returns:
            Tuple containing:
                - Complete DataFrame (SAM-Med3D logistic regression results with PCA 99%)
                - List of matplotlib figures (exactly 2 as expected by run_analysis_sam3d.py)
        """
        # Collect all SAM-Med3D results including PCA 99%
        df_all = self.collect_all_results()
        
        # Create visualizations - ONLY 2 figures as expected by run_analysis_sam3d.py
        figures = []
        
        if not df_all.empty:
            # Create complete table ranked by Test ROC-AUC (includes PCA 99%)
            fig_test = self.create_complete_table(df_all, metric_type="test")
            figures.append(fig_test)
            
            # Create complete table ranked by CV ROC-AUC (includes PCA 99%)
            fig_cv = self.create_complete_table(df_all, metric_type="cv")
            figures.append(fig_cv)
        
        return df_all, figures