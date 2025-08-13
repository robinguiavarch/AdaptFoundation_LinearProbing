"""
Analysis script for SC_sylv pooling regression results.

This script analyzes the 8 combinations from feature_extracted_sc_dinov2 directory
and generates comparative tables and CSV files.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class SCAnalyzer:
    """
    Analyzer for SC_sylv pooling regression results.
    
    Processes results from all 8 combinations (2 variants × 4 PCA modes)
    and generates comparative analysis tables and CSV files.
    
    Attributes:
        features_base_path (Path): Path to feature_extracted_sc_dinov2 directory
    """
    
    def __init__(self, features_base_path: str):
        """
        Initialize the SC_sylv analyzer.
        
        Args:
            features_base_path (str): Path to feature_extracted_sc_dinov2 directory
        """
        self.features_base_path = Path(features_base_path)
        
        if not self.features_base_path.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_base_path}")
    
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect all regression results from individual JSON files.
        
        Returns:
            pd.DataFrame: Complete dataset with all experimental results
        """
        all_results = []
        
        variants = [
            'pooling_spatial_without_25d',
            'pooling_spatial_with_25d'
        ]
        
        pca_modes = ['32', '256', '95', '99']
        
        for variant in variants:
            for pca_mode in pca_modes:
                result_file = self.features_base_path / variant / f"PCA_{pca_mode}" / "regression_results.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        if 'aggregated_metrics' in data:
                            result_data = data
                        else:
                            continue
                        
                        aggregation_method = 'pooling'
                        use_25d = 'with_25d' in variant
                        
                        test_r2 = result_data['aggregated_metrics']['mean_test_r2']
                        cv_r2 = result_data['aggregated_metrics']['mean_cv_r2']
                        overfitting_gap = result_data['aggregated_metrics']['mean_test_cv_gap']
                        feature_dim = result_data['data_info']['feature_dimensionality']
                        
                        # Calculate best params from first dimension as representative
                        first_dim_result = result_data['results_per_dimension']['dim_0']
                        best_params = str(first_dim_result['best_params'])
                        cv_stability = first_dim_result['cv_metrics'].get('cv_stability', None)
                        
                        parsed_result = {
                            'model': 'dinov2_vitg14',
                            'variant': variant,
                            'config': f"{aggregation_method}_{'25d' if use_25d else 'standard'}",
                            'aggregation_method': aggregation_method,
                            'use_25d': use_25d,
                            'pca_mode': pca_mode,
                            'classifier': 'elasticnet',
                            'best_params': best_params,
                            'test_r2': test_r2,
                            'cv_r2': cv_r2,
                            'overfitting_gap': overfitting_gap,
                            'feature_dim': feature_dim,
                            'cv_stability': cv_stability,
                            'strategy': 'sc_pooling',
                            'task': 'Isomap_6D_regression',
                            'metric_type': 'r2'
                        }
                        
                        all_results.append(parsed_result)
                        print(f"Loaded: {variant}/PCA_{pca_mode}")
                        
                    except Exception as e:
                        print(f"Error loading {result_file}: {e}")
                        continue
                else:
                    print(f"Missing: {variant}/PCA_{pca_mode}")
        
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            print(f"Collected {len(df)} experimental results")
            print(f"Variants: {df['variant'].nunique()}")
            print(f"PCA modes: {sorted(df['pca_mode'].unique())}")
        else:
            print("No results collected")
        
        return df
    
    def create_complete_table(self, df: pd.DataFrame, metric_type: str = "test") -> plt.Figure:
        """
        Create styled table for complete SC_sylv configurations ranking.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            metric_type (str): Type of metric used for ranking ("test" or "cv")
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No SC_sylv Data Available', ha='center', va='center', fontsize=16)
            ax.set_title(f'SC_sylv Pooling - ElasticNet Regression ({metric_type.upper()})')
            ax.axis('off')
            return fig
        
        sort_column = 'test_r2' if metric_type == "test" else 'cv_r2'
        df_sorted = df.sort_values(sort_column, ascending=False).reset_index(drop=True)
        
        table_data = []
        for i, row in df_sorted.iterrows():
            config_name = row['config'].replace('_', ' ').title()
            
            if row['pca_mode'] == '95':
                pca_clean = "95% Var"
            elif row['pca_mode'] == '99':
                pca_clean = "99% Var"
            else:
                pca_clean = f"{row['pca_mode']}D"
            
            gap = f"{row['overfitting_gap']:.3f}" if pd.notna(row['overfitting_gap']) else 'N/A'
            cv_r2 = f"{row['cv_r2']:.4f}"
            test_r2 = f"{row['test_r2']:.4f}"
            feature_dim = f"{int(row['feature_dim'])}D" if pd.notna(row['feature_dim']) else 'N/A'
            
            table_data.append([
                f"#{i+1}",
                "DINOv2 Giant", 
                config_name,
                pca_clean,
                feature_dim,
                gap,
                cv_r2,
                test_r2
            ])
        
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.axis('off')
        
        columns = ['Rank', 'Model', 'Configuration', 'PCA', 'Feature Dim', 'Overfitting Gap', 'CV R²', 'Test R²']
        
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)
        
        for i in range(len(columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#2196F3')
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.12)
        
        for i in range(1, len(table_data) + 1):
            row_color = '#E3F2FD' if i % 2 == 1 else '#BBDEFB'
            
            for j in range(len(columns)):
                cell = table[(i, j)]
                cell.set_facecolor(row_color)
                cell.set_height(0.08)
                
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
                
                elif j == 2:
                    if '25d' in table_data[i-1][2].lower():
                        cell.set_facecolor('#64B5F6')
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color='#1565C0')
                
                elif j == 3:
                    cell.set_text_props(weight='bold', color='#1565C0')
                
                elif j == 4:
                    cell.set_text_props(weight='bold', color='#1565C0')
                elif j == 5:
                    cell.set_text_props(weight='bold', color='#D32F2F')
                elif j == 6:
                    if metric_type == "cv":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color='#1565C0')
                elif j == 7:
                    if metric_type == "test":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color='#1565C0')
                
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        metric_display = "Test R²" if metric_type == "test" else "CV R²"
        n_configs = len(df_sorted)
        pca_modes_str = ", ".join(sorted(df['pca_mode'].unique()))
        
        ax.set_title(f'SC_sylv Pooling - ElasticNet Regression ({n_configs} Configurations)\n'
                    f'PCA Modes: {pca_modes_str} - Ranked by {metric_display}', 
                    fontsize=16, fontweight='bold', pad=30)
        
        fig.text(0.5, 0.02, f'DINOv2 Giant pooling (16x16 patches) with 2.5D slice grouping - Ranked by {metric_display}', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#1565C0')
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete SC_sylv analysis workflow.
        
        Returns:
            Tuple containing:
                - Complete DataFrame with all results
                - List of matplotlib figures (2 tables)
        """
        df_all = self.collect_all_results()
        
        figures = []
        
        if not df_all.empty:
            fig_test = self.create_complete_table(df_all, metric_type="test")
            figures.append(fig_test)
            
            fig_cv = self.create_complete_table(df_all, metric_type="cv")
            figures.append(fig_cv)
        
        return df_all, figures


def main():
    """
    Main entry point for SC_sylv pooling analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis of SC_sylv pooling regression results"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="feature_extracted_sc_dinov2",
        help="Path to feature_extracted_sc_dinov2 directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results_sc",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"Features directory does not exist: {features_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("SC_sylv Pooling Analysis")
    print(f"Features path: {features_path}")
    print(f"Output directory: {output_dir}")
    
    analyzer = SCAnalyzer(str(features_path))
    
    print("\nExecuting SC_sylv analysis workflow...")
    df_all, figures = analyzer.run_analysis()
    
    if not df_all.empty:
        output_file_elasticnet = output_dir / "analysis_results_sc_elasticnet.csv"
        df_all.to_csv(output_file_elasticnet, index=False)
        print(f"Saved elasticnet results: {output_file_elasticnet}")
    
    figure_names = [
        "complete_sc_table_test",
        "complete_sc_table_cv"
    ]
    
    for i, fig in enumerate(figures):
        if i < len(figure_names):
            filename = f"{figure_names[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
    
    if not df_all.empty:
        print(f"\nSC_sylv Pooling Analysis Summary:")
        print(f"Total experimental results: {len(df_all)}")
        print(f"Variants tested: {df_all['variant'].nunique()}")
        print(f"PCA modes tested: {df_all['pca_mode'].nunique()}")
        print(f"Configurations: {df_all['config'].unique().tolist()}")
        
        best_result_test = df_all.loc[df_all['test_r2'].idxmax()]
        print(f"\nTop performing configuration (Test R²):")
        print(f"Variant: {best_result_test['variant']}")
        print(f"PCA Mode: {best_result_test['pca_mode']}")
        print(f"Test R²: {best_result_test['test_r2']:.4f}")
        print(f"CV R²: {best_result_test['cv_r2']:.4f}")
        print(f"Overfitting Gap: {best_result_test['overfitting_gap']:.4f}")
        
        best_result_cv = df_all.loc[df_all['cv_r2'].idxmax()]
        print(f"\nTop performing configuration (CV R²):")
        print(f"Variant: {best_result_cv['variant']}")
        print(f"PCA Mode: {best_result_cv['pca_mode']}")
        print(f"Test R²: {best_result_cv['test_r2']:.4f}")
        print(f"CV R²: {best_result_cv['cv_r2']:.4f}")
        print(f"Overfitting Gap: {best_result_cv['overfitting_gap']:.4f}")
    
    print(f"\nSC_sylv Pooling analysis completed successfully!")
    print(f"Tables saved as: complete_sc_table_test.png & complete_sc_table_cv.png")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()