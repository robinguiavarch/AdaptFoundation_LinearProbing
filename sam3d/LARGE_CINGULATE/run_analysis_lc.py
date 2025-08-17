"""
Analysis script for LARGE_CINGULATE SAM-Med3D flatten classification results.

This script analyzes the 4 combinations from feature_extraction_sam3d_lc directory
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


class LCSAMMed3DAnalyzer:
    """
    Analyzer for LARGE_CINGULATE SAM-Med3D flatten classification results.
    
    Processes results from all 4 combinations (1 variant × 4 PCA modes)
    and generates comparative analysis tables and CSV files.
    
    Attributes:
        features_base_path (Path): Path to feature_extraction_sam3d_lc directory
    """
    
    def __init__(self, features_base_path: str):
        """
        Initialize the LARGE_CINGULATE SAM-Med3D analyzer.
        
        Args:
            features_base_path (str): Path to feature_extraction_sam3d_lc directory
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
        
        variants = ['flatten']
        pca_modes = ['32', '256', '95', '995']
        
        for variant in variants:
            for pca_mode in pca_modes:
                result_file = self.features_base_path / variant / f"PCA_{pca_mode}" / "classification_results.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        if 'test_metrics' in data:
                            result_data = data
                        else:
                            continue
                        
                        aggregation_method = 'flatten'
                        use_25d = False
                        
                        test_roc_auc = result_data['test_metrics']['roc_auc']
                        cv_roc_auc = result_data['best_cv_score']
                        overfitting_gap = result_data['cv_metrics']['overfitting_gap']
                        feature_dim = result_data['data_info']['feature_dimensionality']
                        convergence_warning = result_data['diagnostics']['convergence_warning']
                        best_params = str(result_data['best_params'])
                        cv_stability = result_data['cv_metrics'].get('cv_stability', None)
                        
                        parsed_result = {
                            'model': 'sam_med3d_turbo',
                            'variant': variant,
                            'config': 'flatten',
                            'aggregation_method': aggregation_method,
                            'use_25d': use_25d,
                            'pca_mode': pca_mode,
                            'classifier': 'logistic',
                            'best_params': best_params,
                            'test_roc_auc': test_roc_auc,
                            'cv_roc_auc': cv_roc_auc,
                            'overfitting_gap': overfitting_gap,
                            'feature_dim': feature_dim,
                            'convergence_ok': not convergence_warning,
                            'cv_stability': cv_stability,
                            'strategy': 'lc_flatten',
                            'task': 'Right_PCS_binary'
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
        Create styled table for complete LARGE_CINGULATE SAM-Med3D configurations ranking.
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            metric_type (str): Type of metric used for ranking ("test" or "cv")
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No LARGE_CINGULATE SAM-Med3D Data Available', ha='center', va='center', fontsize=16)
            ax.set_title(f'LARGE_CINGULATE SAM-Med3D Flatten - Logistic Regression ({metric_type.upper()})')
            ax.axis('off')
            return fig
        
        sort_column = 'test_roc_auc' if metric_type == "test" else 'cv_roc_auc'
        df_sorted = df.sort_values(sort_column, ascending=False).reset_index(drop=True)
        
        table_data = []
        for i, row in df_sorted.iterrows():
            config_name = row['config'].replace('_', ' ').title()
            
            if row['pca_mode'] == '95':
                pca_clean = "95% Var"
            elif row['pca_mode'] == '995':
                pca_clean = "99.5% Var"
            else:
                pca_clean = f"{row['pca_mode']}D"
            
            gap = f"{row['overfitting_gap']:.3f}" if pd.notna(row['overfitting_gap']) else 'N/A'
            cv_roc_auc = f"{row['cv_roc_auc']:.4f}"
            test_roc_auc = f"{row['test_roc_auc']:.4f}"
            feature_dim = f"{int(row['feature_dim'])}D" if pd.notna(row['feature_dim']) else 'N/A'
            
            table_data.append([
                f"#{i+1}",
                "SAM-Med3D Turbo", 
                config_name,
                pca_clean,
                feature_dim,
                gap,
                cv_roc_auc,
                test_roc_auc
            ])
        
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.axis('off')
        
        columns = ['Rank', 'Model', 'Configuration', 'PCA', 'Feature Dim', 'Overfitting Gap', 'CV ROC-AUC', 'Test ROC-AUC']
        
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
        
        metric_display = "Test ROC-AUC" if metric_type == "test" else "CV ROC-AUC"
        n_configs = len(df_sorted)
        pca_modes_str = ", ".join(sorted(df['pca_mode'].unique()))
        
        ax.set_title(f'LARGE_CINGULATE SAM-Med3D Flatten - Logistic Regression ({n_configs} Configurations)\n'
                    f'PCA Modes: {pca_modes_str} - Ranked by {metric_display}', 
                    fontsize=16, fontweight='bold', pad=30)
        
        fig.text(0.5, 0.02, f'SAM-Med3D turbo flatten aggregation (196608D → PCA reduced) - Ranked by {metric_display}', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#1565C0')
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete LARGE_CINGULATE SAM-Med3D analysis workflow.
        
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
    Main entry point for LARGE_CINGULATE SAM-Med3D flatten analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis of LARGE_CINGULATE SAM-Med3D flatten classification results"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="feature_extraction_sam3d_lc",
        help="Path to feature_extraction_sam3d_lc directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results_lc_sam3d",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"Features directory does not exist: {features_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("LARGE_CINGULATE SAM-Med3D Flatten Analysis")
    print(f"Features path: {features_path}")
    print(f"Output directory: {output_dir}")
    
    analyzer = LCSAMMed3DAnalyzer(str(features_path))
    
    print("\nExecuting LARGE_CINGULATE SAM-Med3D analysis workflow...")
    df_all, figures = analyzer.run_analysis()
    
    if not df_all.empty:
        output_file_logistic = output_dir / "analysis_results_lc_sam3d_logistic.csv"
        df_all.to_csv(output_file_logistic, index=False)
        print(f"Saved logistic results: {output_file_logistic}")
    
    figure_names = [
        "complete_lc_sam3d_table_test",
        "complete_lc_sam3d_table_cv"
    ]
    
    for i, fig in enumerate(figures):
        if i < len(figure_names):
            filename = f"{figure_names[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
    
    if not df_all.empty:
        print(f"\nLARGE_CINGULATE SAM-Med3D Flatten Analysis Summary:")
        print(f"Total experimental results: {len(df_all)}")
        print(f"Variants tested: {df_all['variant'].nunique()}")
        print(f"PCA modes tested: {df_all['pca_mode'].nunique()}")
        print(f"Configurations: {df_all['config'].unique().tolist()}")
        
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
    
    print(f"\nLARGE_CINGULATE SAM-Med3D Flatten analysis completed successfully!")
    print(f"Tables saved as: complete_lc_sam3d_table_test.png & complete_lc_sam3d_table_cv.png")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()