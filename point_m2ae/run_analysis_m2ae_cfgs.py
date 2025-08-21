"""
Analysis script for Point-M2AE 45 configurations classification results.

This script analyzes the 45 combinations from point_m2ae_cfgs directory
and generates comparative tables for configurations evaluation.
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


class PointM2AEConfigsAnalyzer:
    """
    Analyzer for Point-M2AE 45 configurations classification results.
    
    Processes results from 45 combinations (C1A1-C9A5) and generates
    comparative analysis tables for configurations evaluation.
    
    Attributes:
        results_base_path (Path): Path to point_m2ae_cfgs results directory
    """
    
    def __init__(self, results_base_path: str):
        """
        Initialize the Point-M2AE configurations analyzer.
        
        Args:
            results_base_path (str): Path to point_m2ae_cfgs results directory
        """
        self.results_base_path = Path(results_base_path)
        
        if not self.results_base_path.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_base_path}")
    
    def parse_config_name(self, config_name: str) -> Tuple[str, str]:
        """
        Parse configuration name into config and aggregation components.
        
        Args:
            config_name (str): Configuration name (e.g., 'C1A1', 'C8A2')
            
        Returns:
            Tuple[str, str]: Config key and aggregation key
        """
        if len(config_name) != 4 or not config_name.startswith('C') or 'A' not in config_name:
            raise ValueError(f"Invalid configuration name: {config_name}")
        
        config_key = config_name[:2]  # 'C1', 'C8', etc.
        aggregation_key = config_name[2:]  # 'A1', 'A2', etc.
        
        return config_key, aggregation_key
    
    def get_expected_dim(self, aggregation_key: str) -> int:
        """
        Get expected feature dimension based on aggregation method.
        
        Args:
            aggregation_key (str): Aggregation method key (A1-A5)
            
        Returns:
            int: Expected feature dimension
        """
        dimension_map = {
            'A1': 384,   # mean
            'A2': 1536,  # mean+std+min+max
            'A3': 576,   # multi-level
            'A4': 384,   # adaptive
            'A5': 384    # attention
        }
        
        if aggregation_key not in dimension_map:
            raise ValueError(f"Unknown aggregation method: {aggregation_key}")
        
        return dimension_map[aggregation_key]
    
    def collect_all_results(self) -> pd.DataFrame:
        """
        Collect all classification results from individual JSON files.
        
        Returns:
            pd.DataFrame: Complete dataset with all experimental results
        """
        all_results = []
        
        # Generate all 45 configuration names (C1A1-C9A5)
        configurations = []
        for i in range(1, 10):  # C1-C9
            for j in range(1, 6):  # A1-A5
                configurations.append(f"C{i}A{j}")
        
        mode = 'raw_features'
        
        for config_name in configurations:
            result_file = self.results_base_path / config_name / mode / "classification_results.json"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    # Navigate to nested result structure: {config_name: {classifier: result}}
                    if config_name in data and 'logistic' in data[config_name]:
                        result_data = data[config_name]['logistic']
                    else:
                        continue
                    
                    if 'test_metrics' not in result_data:
                        continue
                    
                    # Parse configuration name
                    config_key, aggregation_key = self.parse_config_name(config_name)
                    expected_dim = self.get_expected_dim(aggregation_key)
                    
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
                        'config_name': config_name,
                        'config': config_key,
                        'aggregation': aggregation_key,
                        'mode': mode,
                        'feature_approach': config_name,
                        'classifier': 'logistic',
                        'best_params': best_params,
                        'test_roc_auc': test_roc_auc,
                        'cv_roc_auc': cv_roc_auc,
                        'overfitting_gap': overfitting_gap,
                        'feature_dim': expected_dim,
                        'convergence_ok': not convergence_warning,
                        'cv_stability': cv_stability,
                        'test_vs_cv_gap': test_vs_cv_gap,
                        'fold_variance': fold_variance,
                        'problematic_folds_count': problematic_folds_count,
                        'strategy': 'point_m2ae_configurations'
                    }
                    
                    all_results.append(parsed_result)
                    print(f"Loaded: {config_name}")
                    
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
                    continue
            else:
                print(f"Missing: {config_name}")
        
        df = pd.DataFrame(all_results)
        
        if not df.empty:
            print(f"Collected {len(df)} experimental results")
            print(f"Configurations: {df['config'].nunique()}")
            print(f"Aggregations: {sorted(df['aggregation'].unique())}")
        else:
            print("No results collected")
        
        return df
    
    def create_configurations_table(self, df: pd.DataFrame, metric_type: str = "test") -> plt.Figure:
        """
        Create styled table for Point-M2AE configurations ranking.
        
        Args:
            df (pd.DataFrame): Complete dataset with all results
            metric_type (str): Type of metric used for ranking ('test' or 'cv')
            
        Returns:
            plt.Figure: Matplotlib figure containing the styled table
        """
        if df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No Configuration Data Available', 
                   ha='center', va='center', fontsize=16)
            ax.set_title(f'Point-M2AE 45 Configurations - Logistic Regression ({metric_type.upper()})')
            ax.axis('off')
            return fig
        
        sort_column = 'test_roc_auc' if metric_type == "test" else 'cv_roc_auc'
        df_sorted = df.sort_values(sort_column, ascending=False).reset_index(drop=True)
        
        table_data = []
        for i, row in df_sorted.iterrows():
            gap = f"{row['overfitting_gap']:.3f}" if pd.notna(row['overfitting_gap']) else 'N/A'
            cv_roc_auc = f"{row['cv_roc_auc']:.4f}"
            test_roc_auc = f"{row['test_roc_auc']:.4f}"
            feature_dim = f"{int(row['feature_dim'])}D" if pd.notna(row['feature_dim']) else 'N/A'
            test_cv_gap = f"{row['test_vs_cv_gap']:.3f}" if pd.notna(row['test_vs_cv_gap']) else 'N/A'
            prob_folds = f"{int(row['problematic_folds_count'])}" if pd.notna(row['problematic_folds_count']) else 'N/A'
            
            table_data.append([
                f"#{i+1}",
                "Point-M2AE",
                row['config'],
                row['aggregation'],
                row['config_name'],
                "Raw",
                feature_dim,
                gap,
                test_cv_gap,
                prob_folds,
                cv_roc_auc,
                test_roc_auc
            ])
        
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis('off')
        
        columns = ['Rank', 'Model', 'Config', 'Aggregation', 'Feature Approach', 'Mode', 'Feature Dim', 
                  'Overfitting Gap', 'Test-CV Gap', 'Prob Folds', 'CV ROC-AUC', 'Test ROC-AUC']
        
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        
        # Color scheme for Point-M2AE configurations
        header_color = '#2196F3'  # Blue for configurations
        row_colors = ['#E3F2FD', '#BBDEFB']
        highlight_color = '#1565C0'
        
        # Config color mapping
        config_colors = {
            'C1': '#4CAF50', 'C2': '#FF9800', 'C3': '#9C27B0', 'C4': '#F44336',
            'C5': '#00BCD4', 'C6': '#795548', 'C7': '#607D8B', 'C8': '#E91E63', 'C9': '#3F51B5'
        }
        
        # Aggregation color mapping
        aggregation_colors = {
            'A1': '#2196F3', 'A2': '#4CAF50', 'A3': '#FF9800', 'A4': '#9C27B0', 'A5': '#F44336'
        }
        
        for i in range(len(columns)):
            cell = table[(0, i)]
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color='white')
            cell.set_height(0.08)
        
        for i in range(1, len(table_data) + 1):
            row_color = row_colors[0] if i % 2 == 1 else row_colors[1]
            
            for j in range(len(columns)):
                cell = table[(i, j)]
                cell.set_facecolor(row_color)
                cell.set_height(0.06)
                
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
                
                elif j == 2:  # Config column
                    config_key = table_data[i-1][2]
                    if config_key in config_colors:
                        cell.set_facecolor(config_colors[config_key])
                        cell.set_text_props(weight='bold', color='white')
                
                elif j == 3:  # Aggregation column
                    agg_key = table_data[i-1][3]
                    if agg_key in aggregation_colors:
                        cell.set_facecolor(aggregation_colors[agg_key])
                        cell.set_text_props(weight='bold', color='white')
                
                elif j == 4:  # Feature Approach column
                    cell.set_text_props(weight='bold', color=highlight_color)
                
                elif j == 5:  # Mode column
                    cell.set_facecolor('#4CAF50')  # Green for raw features
                    cell.set_text_props(weight='bold', color='white')
                
                elif j == 6:  # Feature Dim column
                    cell.set_text_props(weight='bold', color=highlight_color)
                elif j == 7:  # Overfitting Gap column
                    cell.set_text_props(weight='bold', color='#D32F2F')
                elif j == 8:  # Test-CV Gap column
                    cell.set_text_props(weight='bold', color='#FF9800')
                elif j == 9:  # Problematic Folds column
                    cell.set_text_props(weight='bold', color='#9C27B0')
                elif j == 10:  # CV ROC-AUC column
                    if metric_type == "cv":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                elif j == 11:  # Test ROC-AUC column
                    if metric_type == "test":
                        cell.set_text_props(weight='bold', color='#0D47A1')
                    else:
                        cell.set_text_props(weight='bold', color=highlight_color)
                
                cell.set_edgecolor('#CCCCCC')
                cell.set_linewidth(0.5)
        
        metric_display = "Test ROC-AUC" if metric_type == "test" else "CV ROC-AUC"
        n_configs = len(df_sorted)
        
        ax.set_title(f'Point-M2AE 45 Configurations - Logistic Regression ({n_configs} Configurations)\n'
                    f'Configs: C1-C9, Aggregations: A1-A5 - Ranked by {metric_display}', 
                    fontsize=16, fontweight='bold', pad=30)
        
        fig.text(0.5, 0.02, f'Point-M2AE configurations: 10 parameter configs × 5 aggregation methods - Ranked by {metric_display}', 
                ha='center', va='bottom', fontsize=10, style='italic', color=highlight_color)
        
        plt.tight_layout()
        return fig
    
    def run_analysis(self) -> Tuple[pd.DataFrame, List[plt.Figure]]:
        """
        Execute complete Point-M2AE configurations analysis workflow.
        
        Returns:
            Tuple containing:
                - Complete DataFrame with all results
                - List of matplotlib figures (2 tables)
        """
        df_all = self.collect_all_results()
        
        figures = []
        
        if not df_all.empty:
            fig_test = self.create_configurations_table(df_all, metric_type="test")
            fig_cv = self.create_configurations_table(df_all, metric_type="cv")
            figures.extend([fig_test, fig_cv])
        
        return df_all, figures


def main():
    """
    Main entry point for Point-M2AE configurations analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis of Point-M2AE configurations classification results"
    )
    
    parser.add_argument(
        "--results-path",
        type=str,
        default="point_m2ae_cfgs",
        help="Path to point_m2ae_cfgs results directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results_point_m2ae_cfgs",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Results directory does not exist: {results_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Point-M2AE Configurations Analysis")
    print(f"Results path: {results_path}")
    print(f"Output directory: {output_dir}")
    
    analyzer = PointM2AEConfigsAnalyzer(str(results_path))
    
    print("\nExecuting Point-M2AE configurations analysis workflow...")
    df_all, figures = analyzer.run_analysis()
    
    if not df_all.empty:
        output_file_logistic = output_dir / "analysis_results_point_m2ae_cfgs_logistic.csv"
        df_all.to_csv(output_file_logistic, index=False)
        print(f"Saved results: {output_file_logistic}")
    
    figure_names = [
        "point_m2ae_cfgs_configurations_table_test",
        "point_m2ae_cfgs_configurations_table_cv"
    ]
    
    for i, fig in enumerate(figures):
        if i < len(figure_names):
            filename = f"{figure_names[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
    
    if not df_all.empty:
        print(f"\nPoint-M2AE Configurations Analysis Summary:")
        print(f"Total experimental results: {len(df_all)}")
        print(f"Configurations tested: {df_all['config'].nunique()}")
        print(f"Aggregations tested: {df_all['aggregation'].nunique()}")
        
        best_result_test = df_all.loc[df_all['test_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (Test ROC-AUC):")
        print(f"Configuration: {best_result_test['config_name']}")
        print(f"Config: {best_result_test['config']}")
        print(f"Aggregation: {best_result_test['aggregation']}")
        print(f"Test ROC-AUC: {best_result_test['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_test['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_test['overfitting_gap']:.4f}")
        print(f"Test-CV Gap: {best_result_test['test_vs_cv_gap']:.4f}")
        print(f"Feature Dimensionality: {int(best_result_test['feature_dim'])}D")
        
        best_result_cv = df_all.loc[df_all['cv_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (CV ROC-AUC):")
        print(f"Configuration: {best_result_cv['config_name']}")
        print(f"Config: {best_result_cv['config']}")
        print(f"Aggregation: {best_result_cv['aggregation']}")
        print(f"Test ROC-AUC: {best_result_cv['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_cv['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_cv['overfitting_gap']:.4f}")
        print(f"Test-CV Gap: {best_result_cv['test_vs_cv_gap']:.4f}")
        print(f"Feature Dimensionality: {int(best_result_cv['feature_dim'])}D")
        
        # Top configurations analysis
        for config in ['C1', 'C8', 'C9']:
            config_results = df_all[df_all['config'] == config]
            if not config_results.empty:
                best_config = config_results.loc[config_results['test_roc_auc'].idxmax()]
                print(f"\nBest {config} configuration:")
                print(f"Configuration: {best_config['config_name']}")
                print(f"Test ROC-AUC: {best_config['test_roc_auc']:.4f}")
                print(f"CV ROC-AUC: {best_config['cv_roc_auc']:.4f}")
        
        # Top aggregations analysis
        for agg in ['A1', 'A2', 'A3', 'A4', 'A5']:
            agg_results = df_all[df_all['aggregation'] == agg]
            if not agg_results.empty:
                best_agg = agg_results.loc[agg_results['test_roc_auc'].idxmax()]
                print(f"\nBest {agg} aggregation:")
                print(f"Configuration: {best_agg['config_name']}")
                print(f"Test ROC-AUC: {best_agg['test_roc_auc']:.4f}")
                print(f"CV ROC-AUC: {best_agg['cv_roc_auc']:.4f}")
                print(f"Dimension: {int(best_agg['feature_dim'])}D")
    
    print(f"\nPoint-M2AE configurations analysis completed successfully!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()