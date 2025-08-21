"""
Analysis script for Point-M2AE preprocessing v1 classification results.

This script analyzes the 3 combinations from point_m2ae directory
and generates comparative tables for preprocessing v1 evaluation.
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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PointM2AEPreprocessingAnalyzer:
    """
    Analyzer for Point-M2AE preprocessing v1 classification results.
    
    Processes results from 3 combinations (1 approach × 3 modes)
    and generates comparative analysis tables for preprocessing v1 evaluation.
    
    Attributes:
        results_base_path (Path): Path to point_m2ae results directory
    """
    
    def __init__(self, results_base_path: str):
        """
        Initialize the Point-M2AE preprocessing analyzer.
        
        Args:
            results_base_path (str): Path to point_m2ae results directory
        """
        self.results_base_path = Path(results_base_path)
        
        if not self.results_base_path.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_base_path}")
    
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect all classification results from individual JSON files.
        
        Returns:
            pd.DataFrame: Complete dataset with all experimental results
        """
        all_results = []
        
        approaches = ['feat_mean_v1']
        modes = ['PCA_32', 'PCA_256', 'raw_features']
        
        for approach in approaches:
            for mode in modes:
                result_file = self.results_base_path / approach / mode / "classification_results.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        # Navigate to nested result structure: {approach: {classifier: result}}
                        if approach in data and 'logistic' in data[approach]:
                            result_data = data[approach]['logistic']
                        else:
                            continue
                        
                        if 'test_metrics' not in result_data:
                            continue
                        
                        # Determine approach type and dimensionality
                        approach_type = 'mean_v1'
                        
                        if mode == 'raw_features':
                            feature_dim = 384
                            mode_clean = mode
                        else:
                            feature_dim = result_data['data_info']['feature_dimensionality']
                            mode_clean = mode
                        
                        test_roc_auc = result_data['test_metrics']['roc_auc_weighted']
                        cv_roc_auc = result_data['best_cv_score']
                        overfitting_gap = result_data['cv_metrics']['overfitting_gap']
                        convergence_warning = result_data['diagnostics']['convergence_warning']
                        best_params = str(result_data['best_params'])
                        cv_stability = result_data['cv_metrics'].get('cv_stability', None)
                        
                        # Extract investigation metrics if available
                        test_vs_cv_gap = 0.0
                        fold_variance = 0.0
                        problematic_folds_count = 0
                        
                        if 'cv_detailed_analysis' in result_data:
                            cv_analysis = result_data['cv_detailed_analysis']
                            test_vs_cv_gap = cv_analysis.get('test_vs_cv_gap', 0.0)
                            fold_variance = cv_analysis.get('fold_variance', 0.0)
                            problematic_folds_count = len(cv_analysis.get('problematic_folds', []))
                        
                        parsed_result = {
                            'model': 'point_m2ae_encoder',
                            'approach': approach,
                            'mode': mode_clean,
                            'approach_type': approach_type,
                            'feature_approach': approach,
                            'classifier': 'logistic',
                            'best_params': best_params,
                            'test_roc_auc': test_roc_auc,
                            'cv_roc_auc': cv_roc_auc,
                            'overfitting_gap': overfitting_gap,
                            'feature_dim': feature_dim,
                            'convergence_ok': not convergence_warning,
                            'cv_stability': cv_stability,
                            'test_vs_cv_gap': test_vs_cv_gap,
                            'fold_variance': fold_variance,
                            'problematic_folds_count': problematic_folds_count,
                            'strategy': 'point_m2ae_preprocessing_v1'
                        }
                        
                        all_results.append(parsed_result)
                        print(f"Loaded: {approach}/{mode}")
                        
                    except Exception as e:
                        print(f"Error loading {result_file}: {e}")
                        continue
                else:
                    print(f"Missing: {approach}/{mode}")
        
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            print(f"Collected {len(df)} experimental results")
            print(f"Approaches: {df['approach'].nunique()}")
            print(f"Modes: {sorted(df['mode'].unique())}")
        else:
            print("No results collected")
        
        return df
    
    def filter_by_approach_type(self, df: pd.DataFrame, approach_type: str = 'all') -> pd.DataFrame:
        """
        Filter dataframe by approach type.
        
        Args:
            df (pd.DataFrame): Complete results dataframe
            approach_type (str): Approach type ('mean_v1' or 'all')
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if approach_type == 'all':
            return df.reset_index(drop=True)
        else:
            return df[df['approach_type'] == approach_type].reset_index(drop=True)
    
    def create_approaches_table(self, df_filtered: pd.DataFrame, approach_type: str, 
                               metric_type: str = "test") -> plt.Figure:
        """
        Create styled table for Point-M2AE preprocessing v1 ranking.
        
        Args:
            df_filtered (pd.DataFrame): Filtered dataset by approach type
            approach_type (str): Approach type ('all' or 'mean_v1')
            metric_type (str): Type of metric used for ranking ('test' or 'cv')
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df_filtered.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f'No {approach_type.title()} Data Available', 
                   ha='center', va='center', fontsize=16)
            ax.set_title(f'Point-M2AE Preprocessing v1 - Logistic Regression ({metric_type.upper()})')
            ax.axis('off')
            return fig
        
        sort_column = 'test_roc_auc' if metric_type == "test" else 'cv_roc_auc'
        df_sorted = df_filtered.sort_values(sort_column, ascending=False).reset_index(drop=True)
        
        # Mapping for display names
        approach_display_mapping = {
            'feat_mean_v1': 'Mean v1'
        }
        
        mode_display_mapping = {
            'PCA_32': '32D',
            'PCA_256': '256D',
            'raw_features': 'Raw'
        }
        
        table_data = []
        for i, row in df_sorted.iterrows():
            approach_display = approach_display_mapping.get(row['feature_approach'], 
                                                          row['feature_approach'].replace('_', ' ').title())
            
            mode_display = mode_display_mapping.get(row['mode'], row['mode'])
            
            gap = f"{row['overfitting_gap']:.3f}" if pd.notna(row['overfitting_gap']) else 'N/A'
            cv_roc_auc = f"{row['cv_roc_auc']:.4f}"
            test_roc_auc = f"{row['test_roc_auc']:.4f}"
            feature_dim = f"{int(row['feature_dim'])}D" if pd.notna(row['feature_dim']) else 'N/A'
            test_cv_gap = f"{row['test_vs_cv_gap']:.3f}" if pd.notna(row['test_vs_cv_gap']) else 'N/A'
            prob_folds = f"{int(row['problematic_folds_count'])}" if pd.notna(row['problematic_folds_count']) else 'N/A'
            
            table_data.append([
                f"#{i+1}",
                "Point-M2AE", 
                approach_display,
                mode_display,
                feature_dim,
                gap,
                test_cv_gap,
                prob_folds,
                cv_roc_auc,
                test_roc_auc
            ])
        
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('off')
        
        columns = ['Rank', 'Model', 'Feature Approach', 'Mode', 'Feature Dim', 
                  'Overfitting Gap', 'Test-CV Gap', 'Prob Folds', 'CV ROC-AUC', 'Test ROC-AUC']
        
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.0)
        
        # Color scheme for Point-M2AE preprocessing v1 (Green for improvements)
        header_color = '#4CAF50'  # Green for preprocessing v1
        row_colors = ['#E8F5E8', '#C8E6C9']
        highlight_color = '#2E7D32'
        
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
                
                if j == 0:  # Rank column
                    if i == 1:
                        cell.set_facecolor('#FFD700')  # Gold
                        cell.set_text_props(weight='bold', color='#8B4513')
                    elif i == 2:
                        cell.set_facecolor('#C0C0C0')  # Silver
                        cell.set_text_props(weight='bold', color='#2F4F4F')
                    elif i == 3:
                        cell.set_facecolor('#CD7F32')  # Bronze
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_text_props(weight='bold')
                
                elif j == 2:  # Feature Approach column
                    cell.set_text_props(weight='bold', color=highlight_color)
                
                elif j == 3:  # Mode column
                    if 'Raw' in table_data[i-1][3]:
                        cell.set_facecolor('#4CAF50')  # Green for raw features
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                
                elif j == 4:  # Feature Dim column
                    cell.set_text_props(weight='bold', color=highlight_color)
                elif j == 5:  # Overfitting Gap column
                    cell.set_text_props(weight='bold', color='#D32F2F')
                elif j == 6:  # Test-CV Gap column
                    cell.set_text_props(weight='bold', color='#FF9800')
                elif j == 7:  # Problematic Folds column
                    cell.set_text_props(weight='bold', color='#9C27B0')
                elif j == 8:  # CV ROC-AUC column
                    if metric_type == "cv":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                elif j == 9:  # Test ROC-AUC column
                    if metric_type == "test":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        metric_display = "Test ROC-AUC" if metric_type == "test" else "CV ROC-AUC"
        n_configs = len(df_sorted)
        modes_str = ", ".join(sorted(df_filtered['mode'].unique()))
        
        approach_type_display = approach_type.replace('_', ' ').title() if approach_type != 'all' else 'All Approaches'
        
        ax.set_title(f'Point-M2AE Preprocessing v1 - Logistic Regression ({n_configs} Configurations)\n'
                    f'Modes: {modes_str} - Ranked by {metric_display} ({approach_type_display})', 
                    fontsize=16, fontweight='bold', pad=30)
        
        fig.text(0.5, 0.02, f'Point-M2AE preprocessing v1 features - Fixed normalization with anatomical preservation - Ranked by {metric_display}', 
                ha='center', va='bottom', fontsize=10, style='italic', color=highlight_color)
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete Point-M2AE preprocessing v1 analysis workflow.
        
        Returns:
            Tuple containing:
                - Complete DataFrame with all results
                - List of matplotlib figures (2 tables)
        """
        df_all = self.collect_all_results()
        
        figures = []
        
        if not df_all.empty:
            df_all_approaches = self.filter_by_approach_type(df_all, 'all')
            fig_all_test = self.create_approaches_table(df_all_approaches, 'all', metric_type="test")
            fig_all_cv = self.create_approaches_table(df_all_approaches, 'all', metric_type="cv")
            figures.extend([fig_all_test, fig_all_cv])
        
        return df_all, figures


def main():
    """
    Main entry point for Point-M2AE preprocessing v1 analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis of Point-M2AE preprocessing v1 classification results"
    )
    
    parser.add_argument(
        "--results-path",
        type=str,
        default="point_m2ae",
        help="Path to point_m2ae results directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results_point_m2ae_pp",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Results directory does not exist: {results_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Point-M2AE Preprocessing v1 Analysis")
    print(f"Results path: {results_path}")
    print(f"Output directory: {output_dir}")
    
    analyzer = PointM2AEPreprocessingAnalyzer(str(results_path))
    
    print("\nExecuting Point-M2AE preprocessing v1 analysis workflow...")
    df_all, figures = analyzer.run_analysis()
    
    if not df_all.empty:
        output_file_logistic = output_dir / "analysis_results_point_m2ae_pp_logistic.csv"
        df_all.to_csv(output_file_logistic, index=False)
        print(f"Saved results: {output_file_logistic}")
    
    figure_names = [
        "point_m2ae_pp_approaches_table_test",
        "point_m2ae_pp_approaches_table_cv"
    ]
    
    for i, fig in enumerate(figures):
        if i < len(figure_names):
            filename = f"{figure_names[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
    
    if not df_all.empty:
        print(f"\nPoint-M2AE Preprocessing v1 Analysis Summary:")
        print(f"Total experimental results: {len(df_all)}")
        print(f"Approaches tested: {df_all['approach'].nunique()}")
        print(f"Modes tested: {df_all['mode'].nunique()}")
        
        best_result_test = df_all.loc[df_all['test_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (Test ROC-AUC):")
        print(f"Approach: {best_result_test['approach']}")
        print(f"Mode: {best_result_test['mode']}")
        print(f"Test ROC-AUC: {best_result_test['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_test['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_test['overfitting_gap']:.4f}")
        print(f"Test-CV Gap: {best_result_test['test_vs_cv_gap']:.4f}")
        print(f"Feature Dimensionality: {int(best_result_test['feature_dim'])}D")
        
        best_result_cv = df_all.loc[df_all['cv_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (CV ROC-AUC):")
        print(f"Approach: {best_result_cv['approach']}")
        print(f"Mode: {best_result_cv['mode']}")
        print(f"Test ROC-AUC: {best_result_cv['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_cv['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_cv['overfitting_gap']:.4f}")
        print(f"Test-CV Gap: {best_result_cv['test_vs_cv_gap']:.4f}")
        print(f"Feature Dimensionality: {int(best_result_cv['feature_dim'])}D")
        
        # Raw vs PCA comparison
        raw_results = df_all[df_all['mode'] == 'raw_features']
        pca_results = df_all[df_all['mode'].isin(['PCA_32', 'PCA_256'])]
        
        if not raw_results.empty and not pca_results.empty:
            print(f"\nRaw vs PCA Features Comparison:")
            print(f"Raw features mean Test ROC-AUC: {raw_results['test_roc_auc'].mean():.4f}")
            print(f"PCA features mean Test ROC-AUC: {pca_results['test_roc_auc'].mean():.4f}")
            print(f"Raw features mean CV ROC-AUC: {raw_results['cv_roc_auc'].mean():.4f}")
            print(f"PCA features mean CV ROC-AUC: {pca_results['cv_roc_auc'].mean():.4f}")
        
        # Dimensionality analysis
        for mode in ['PCA_32', 'PCA_256', 'raw_features']:
            mode_results = df_all[df_all['mode'] == mode]
            if not mode_results.empty:
                print(f"\n{mode} Performance:")
                print(f"Test ROC-AUC: {mode_results['test_roc_auc'].iloc[0]:.4f}")
                print(f"CV ROC-AUC: {mode_results['cv_roc_auc'].iloc[0]:.4f}")
    
    print(f"\nPoint-M2AE preprocessing v1 analysis completed successfully!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()