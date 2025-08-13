"""
Analysis script for DINOv2 Density-Guided CLS Token variants classification results.

This script analyzes the 18 combinations from density_guided_features directory
and generates comparative tables by aggregation method.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class DensityGuidedAnalyzer:
    """
    Analyzer for DINOv2 Density-Guided CLS Token classification results.
    
    Processes results from all 18 combinations (6 variants × 3 PCA modes)
    and generates comparative analysis tables by aggregation method.
    
    Attributes:
        features_base_path (Path): Path to density_guided_features directory
    """
    
    def __init__(self, features_base_path: str):
        """
        Initialize the DINOv2 Density-Guided analyzer.
        
        Args:
            features_base_path (str): Path to density_guided_features directory
        """
        self.features_base_path = Path(features_base_path)
        
        if not self.features_base_path.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_base_path}")
    
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect all classification results from individual JSON files.
        
        Returns:
            pd.DataFrame: Complete dataset with all experimental results
        """
        all_results = []
        
        variants = [
            'central_uniform_concat',
            'central_uniform_pooling',
            'adaptive_density_concat',
            'adaptive_density_pooling',
            'linear_weighting_concat',
            'linear_weighting_pooling'
        ]
        
        pca_modes = ['32', '256', '99']
        
        for variant in variants:
            for pca_mode in pca_modes:
                result_file = self.features_base_path / variant / f"PCA_{pca_mode}" / "classification_results.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        if 'test_metrics' not in data:
                            continue
                        
                        aggregation_method = 'concat' if 'concat' in variant else 'pooling'
                        
                        if 'central_uniform' in variant:
                            density_approach = 'central_uniform'
                        elif 'adaptive_density' in variant:
                            density_approach = 'adaptive_density'
                        elif 'linear_weighting' in variant:
                            density_approach = 'linear_weighting'
                        else:
                            density_approach = 'unknown'
                        
                        test_roc_auc = data['test_metrics']['roc_auc_weighted']
                        cv_roc_auc = data['best_cv_score']
                        overfitting_gap = data['cv_metrics']['overfitting_gap']
                        feature_dim = data['data_info']['feature_dimensionality']
                        convergence_warning = data['diagnostics']['convergence_warning']
                        best_params = str(data['best_params'])
                        cv_stability = data['cv_metrics'].get('cv_stability', None)
                        
                        parsed_result = {
                            'model': 'dinov2_vitg14',
                            'variant': variant,
                            'config': f"{aggregation_method}_{density_approach}",
                            'aggregation_method': aggregation_method,
                            'density_approach': density_approach,
                            'pca_mode': pca_mode,
                            'classifier': 'logistic',
                            'best_params': best_params,
                            'test_roc_auc': test_roc_auc,
                            'cv_roc_auc': cv_roc_auc,
                            'overfitting_gap': overfitting_gap,
                            'feature_dim': feature_dim,
                            'convergence_ok': not convergence_warning,
                            'cv_stability': cv_stability,
                            'strategy': 'density_guided'
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
    
    def filter_by_aggregation(self, df: pd.DataFrame, aggregation_method: str) -> pd.DataFrame:
        """
        Filter dataframe by aggregation method.
        
        Args:
            df (pd.DataFrame): Complete results dataframe
            aggregation_method (str): Aggregation method ('concat' or 'pooling')
        
        Returns:
            pd.DataFrame: Filtered dataframe with 9 combinations
        """
        return df[df['aggregation_method'] == aggregation_method].reset_index(drop=True)
    
    def create_aggregation_table(self, df_filtered: pd.DataFrame, aggregation_method: str, 
                                metric_type: str = "test") -> plt.Figure:
        """
        Create styled table for aggregation method configurations ranking.
        
        Args:
            df_filtered (pd.DataFrame): Filtered dataset by aggregation method
            aggregation_method (str): Aggregation method ('concat' or 'pooling')
            metric_type (str): Type of metric used for ranking ('test' or 'cv')
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df_filtered.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f'No {aggregation_method.title()} Data Available', 
                   ha='center', va='center', fontsize=16)
            ax.set_title(f'{aggregation_method.title()} Density-Guided - Logistic Regression ({metric_type.upper()})')
            ax.axis('off')
            return fig
        
        sort_column = 'test_roc_auc' if metric_type == "test" else 'cv_roc_auc'
        df_sorted = df_filtered.sort_values(sort_column, ascending=False).reset_index(drop=True)
        
        table_data = []
        for i, row in df_sorted.iterrows():
            density_display = row['density_approach'].replace('_', ' ').title()
            
            if row['pca_mode'] == '99':
                pca_clean = "99% Var"
            else:
                pca_clean = f"{row['pca_mode']}D"
            
            gap = f"{row['overfitting_gap']:.3f}" if pd.notna(row['overfitting_gap']) else 'N/A'
            cv_roc_auc = f"{row['cv_roc_auc']:.4f}"
            test_roc_auc = f"{row['test_roc_auc']:.4f}"
            feature_dim = f"{int(row['feature_dim'])}D" if pd.notna(row['feature_dim']) else 'N/A'
            
            table_data.append([
                f"#{i+1}",
                "DINOv2 Giant", 
                density_display,
                pca_clean,
                feature_dim,
                gap,
                cv_roc_auc,
                test_roc_auc
            ])
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('off')
        
        columns = ['Rank', 'Model', 'Density Approach', 'PCA', 'Feature Dim', 'Overfitting Gap', 'CV ROC-AUC', 'Test ROC-AUC']
        
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
        
        if aggregation_method == 'concat':
            header_color = '#2196F3'
            row_colors = ['#E3F2FD', '#BBDEFB']
            highlight_color = '#1976D2'
        else:
            header_color = '#4CAF50'
            row_colors = ['#E8F5E8', '#C8E6C9']
            highlight_color = '#388E3C'
        
        for i in range(len(columns)):
            cell = table[(0, i)]
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.12)
        
        for i in range(1, len(table_data) + 1):
            row_color = row_colors[0] if i % 2 == 1 else row_colors[1]
            
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
                    if 'Adaptive' in table_data[i-1][2] or 'Linear' in table_data[i-1][2]:
                        cell.set_facecolor(highlight_color)
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                
                elif j == 3:
                    cell.set_text_props(weight='bold', color=highlight_color)
                
                elif j == 4:
                    cell.set_text_props(weight='bold', color=highlight_color)
                elif j == 5:
                    cell.set_text_props(weight='bold', color='#D32F2F')
                elif j == 6:
                    if metric_type == "cv":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                elif j == 7:
                    if metric_type == "test":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        metric_display = "Test ROC-AUC" if metric_type == "test" else "CV ROC-AUC"
        n_configs = len(df_sorted)
        pca_modes_str = ", ".join(sorted(df_filtered['pca_mode'].unique()))
        
        ax.set_title(f'{aggregation_method.title()} Density-Guided - Logistic Regression ({n_configs} Configurations)\n'
                    f'PCA Modes: {pca_modes_str} - Ranked by {metric_display}', 
                    fontsize=16, fontweight='bold', pad=30)
        
        fig.text(0.5, 0.02, f'DINOv2 Giant CLS tokens (1536D) with density-guided variants - {aggregation_method.title()} aggregation - Ranked by {metric_display}', 
                ha='center', va='bottom', fontsize=10, style='italic', color=highlight_color)
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete DINOv2 Density-Guided analysis workflow.
        
        Returns:
            Tuple containing:
                - Complete DataFrame with all results
                - List of matplotlib figures (4 tables)
        """
        df_all = self.collect_all_results()
        
        figures = []
        
        if not df_all.empty:
            df_concat = self.filter_by_aggregation(df_all, 'concat')
            fig_concat_test = self.create_aggregation_table(df_concat, 'concat', metric_type="test")
            fig_concat_cv = self.create_aggregation_table(df_concat, 'concat', metric_type="cv")
            figures.extend([fig_concat_test, fig_concat_cv])
            
            df_pooling = self.filter_by_aggregation(df_all, 'pooling')
            fig_pooling_test = self.create_aggregation_table(df_pooling, 'pooling', metric_type="test")
            fig_pooling_cv = self.create_aggregation_table(df_pooling, 'pooling', metric_type="cv")
            figures.extend([fig_pooling_test, fig_pooling_cv])
        
        return df_all, figures


def main():
    """
    Main entry point for DINOv2 Density-Guided analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis of DINOv2 Density-Guided classification results"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="dinov2_variantes/density_guided/density_guided_features",
        help="Path to density_guided_features directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results_density_guided",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"Features directory does not exist: {features_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("DINOv2 Density-Guided Analysis")
    print(f"Features path: {features_path}")
    print(f"Output directory: {output_dir}")
    
    analyzer = DensityGuidedAnalyzer(str(features_path))
    
    print("\nExecuting DINOv2 Density-Guided analysis workflow...")
    df_all, figures = analyzer.run_analysis()
    
    if not df_all.empty:
        output_file_logistic = output_dir / "analysis_results_density_guided_logistic.csv"
        df_all.to_csv(output_file_logistic, index=False)
        print(f"Saved results: {output_file_logistic}")
    
    figure_names = [
        "concat_density_guided_table_test",
        "concat_density_guided_table_cv", 
        "pooling_density_guided_table_test",
        "pooling_density_guided_table_cv"
    ]
    
    for i, fig in enumerate(figures):
        if i < len(figure_names):
            filename = f"{figure_names[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
    
    if not df_all.empty:
        print(f"\nDensity-Guided Analysis Summary:")
        print(f"Total experimental results: {len(df_all)}")
        print(f"Variants tested: {df_all['variant'].nunique()}")
        print(f"PCA modes tested: {df_all['pca_mode'].nunique()}")
        
        best_result_test = df_all.loc[df_all['test_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (Test ROC-AUC):")
        print(f"Variant: {best_result_test['variant']}")
        print(f"PCA Mode: {best_result_test['pca_mode']}")
        print(f"Test ROC-AUC: {best_result_test['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_test['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_test['overfitting_gap']:.4f}")
        
        best_result_cv = df_all.loc[df_all['cv_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (CV ROC-AUC):")
        print(f"Variant: {best_result_cv['variant']}")
        print(f"PCA Mode: {best_result_cv['pca_mode']}")
        print(f"Test ROC-AUC: {best_result_cv['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_cv['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_cv['overfitting_gap']:.4f}")
    
    print(f"\nDensity-Guided analysis completed successfully!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()