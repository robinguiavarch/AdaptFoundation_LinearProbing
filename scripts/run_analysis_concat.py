"""
Analysis orchestration script for AdaptFoundation project - CONCATENATION STRATEGY.

This script runs comprehensive analysis of concatenation-based classification results across
all foundation models, configurations, and PCA strategies.

Usage:
python scripts/run_analysis_concat.py --features-path feature_extracted_concat
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.comparison_concat import ComparisonAnalyzerConcat


def main():
    """
    Main entry point for concatenation analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis of AdaptFoundation concatenation classification results"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="feature_extracted_concat",
        help="Path to feature_extracted_concat directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results_concat",
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Validate features path
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"Features directory does not exist: {features_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("AdaptFoundation Phase 11 - Concatenation Strategy Analysis")
    print(f"Features path: {features_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize concatenation analyzer
    analyzer = ComparisonAnalyzerConcat(str(features_path))
    
    # Run complete analysis
    print("\nExecuting concatenation analysis workflow...")
    df_all, df_logistic, df_top10, figures = analyzer.run_analysis()
    
    # Save DataFrames
    if not df_all.empty:
        output_file = output_dir / "analysis_results_concat_all.csv"
        df_all.to_csv(output_file, index=False)
        print(f"Saved complete concatenation results: {output_file}")
    
    if not df_logistic.empty:
        output_file = output_dir / "analysis_results_concat_logistic.csv" 
        df_logistic.to_csv(output_file, index=False)
        print(f"Saved logistic concatenation results: {output_file}")
    
    if not df_top10.empty:
        output_file = output_dir / "analysis_results_concat_top10.csv"
        df_top10.to_csv(output_file, index=False)
        print(f"Saved top 10 concatenation results: {output_file}")
    
    # Save figures
    heatmap_titles = ["all_classifiers_concat", "logistic_only_concat", "top10_configs_concat"]
    
    for i, fig in enumerate(figures):
        if i < len(heatmap_titles):
            filename = f"heatmap_{heatmap_titles[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved concatenation heatmap: {output_file}")
    
    # Display top result for concatenation
    if not df_top10.empty:
        best_result = df_top10.iloc[0]
        print(f"\nTop performing CONCATENATION configuration:")
        print(f"Model: {best_result['model']}")
        print(f"Configuration: {best_result['config']}")
        print(f"PCA Mode: {best_result['pca_mode']}")
        print(f"Test ROC-AUC: {best_result['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result['overfitting_gap']:.4f}")
        
        # Display feature dimensions for concatenation
        if 'feature_dim' in best_result and pd.notna(best_result['feature_dim']):
            print(f"Feature Dimension: {int(best_result['feature_dim'])}D (CONCATENATION)")
    
    print(f"\nConcatenation analysis completed successfully!")
    print(f"Results saved in: {output_dir}")
    
    # Additional concatenation-specific insights
    if not df_all.empty:
        print(f"\n=== CONCATENATION STRATEGY INSIGHTS ===")
        print(f"Total concatenation experiments: {len(df_all)}")
        print(f"Models tested: {df_all['model'].nunique()} ({', '.join(df_all['model'].unique())})")
        print(f"Concatenation configs: {df_all['config'].nunique()} ({', '.join(df_all['config'].unique())})")
        print(f"PCA modes: {df_all['pca_mode'].nunique()} ({', '.join(df_all['pca_mode'].unique())})")
        
        # Dimension analysis for concatenation
        if 'feature_dim' in df_all.columns:
            valid_dims = df_all['feature_dim'].dropna()
            if not valid_dims.empty:
                print(f"Feature dimension range: {int(valid_dims.min())}D - {int(valid_dims.max())}D")
                print(f"Median feature dimension: {int(valid_dims.median())}D")
        
        # Best performance by configuration type
        if not df_logistic.empty:
            config_performance = df_logistic.groupby('config')['test_roc_auc'].agg(['mean', 'max', 'count']).round(4)
            print(f"\nConcatenation Performance by Configuration:")
            for config, stats in config_performance.iterrows():
                print(f"  {config}: Mean={stats['mean']:.4f}, Best={stats['max']:.4f}, N={int(stats['count'])}")


if __name__ == "__main__":
    main()