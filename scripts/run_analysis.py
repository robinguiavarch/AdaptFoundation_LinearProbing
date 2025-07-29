"""
Analysis orchestration script for AdaptFoundation project.

This script runs comprehensive analysis of classification results across
all foundation models, configurations, and PCA strategies.

Usage:
python scripts/run_analysis.py --features-path feature_extracted
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.comparison import ComparisonAnalyzer


def main():
    """
    Main entry point for analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis of AdaptFoundation classification results"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="feature_extracted",
        help="Path to feature_extracted directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
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
    
    print("AdaptFoundation Phase 7 - Comparative Analysis")
    print(f"Features path: {features_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = ComparisonAnalyzer(str(features_path))
    
    # Run complete analysis
    print("\nExecuting analysis workflow...")
    df_all, df_logistic, df_top10, figures = analyzer.run_analysis()
    
    # Save DataFrames
    if not df_all.empty:
        output_file = output_dir / "analysis_results_all.csv"
        df_all.to_csv(output_file, index=False)
        print(f"Saved complete results: {output_file}")
    
    if not df_logistic.empty:
        output_file = output_dir / "analysis_results_logistic.csv" 
        df_logistic.to_csv(output_file, index=False)
        print(f"Saved logistic results: {output_file}")
    
    if not df_top10.empty:
        output_file = output_dir / "analysis_results_top10.csv"
        df_top10.to_csv(output_file, index=False)
        print(f"Saved top 10 results: {output_file}")
    
    # Save figures with updated naming scheme
    figure_names = [
        "comprehensive_pooling_grid",  # Grid 2x3 comprehensive heatmap
        "top10_pooling_table"          # Top 10 configurations table
    ]
    
    for i, fig in enumerate(figures):
        if i < len(figure_names):
            filename = f"{figure_names[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
    
    # Display summary statistics
    if not df_all.empty:
        print(f"\nAnalysis Summary:")
        print(f"Total experimental results: {len(df_all)}")
        print(f"Foundation models tested: {df_all['model'].nunique()}")
        print(f"Configurations tested: {df_all['config'].nunique()}")
        print(f"PCA modes tested: {df_all['pca_mode'].nunique()}")
        print(f"Classifiers tested: {df_all['classifier'].nunique()}")
    
    # Display top result
    if not df_top10.empty:
        best_result = df_top10.iloc[0]
        print(f"\nTop performing configuration (Logistic Regression):")
        print(f"Model: {best_result['model']}")
        print(f"Configuration: {best_result['config']}")
        print(f"PCA Mode: {best_result['pca_mode']}")
        print(f"Test ROC-AUC: {best_result['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result['overfitting_gap']:.4f}")
    
    print(f"\nPooling strategy analysis completed successfully!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()