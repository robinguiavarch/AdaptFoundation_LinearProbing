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
    df_all, df_logistic, df_top10_test, df_top10_cv, figures = analyzer.run_analysis()
    
    # Save DataFrames
    if not df_all.empty:
        output_file = output_dir / "analysis_results_all.csv"
        df_all.to_csv(output_file, index=False)
        print(f"Saved complete results: {output_file}")
    
    if not df_logistic.empty:
        output_file = output_dir / "analysis_results_logistic.csv" 
        df_logistic.to_csv(output_file, index=False)
        print(f"Saved logistic results: {output_file}")
    
    if not df_top10_test.empty:
        output_file = output_dir / "analysis_results_top10_test.csv"
        df_top10_test.to_csv(output_file, index=False)
        print(f"Saved top 10 test results: {output_file}")
    
    if not df_top10_cv.empty:
        output_file = output_dir / "analysis_results_top10_cv.csv"
        df_top10_cv.to_csv(output_file, index=False)
        print(f"Saved top 10 CV results: {output_file}")
    
    # Save figures with updated naming for nested grid
    figure_names = [
        "nested_pooling_grid",          # 2x2 nested grid heatmap (KNN + Logistic)
        "top10_pooling_table_test",     # Top 10 configurations table (Test ROC-AUC)
        "top10_pooling_table_cv"        # Top 10 configurations table (CV ROC-AUC)
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
        print(f"Unique classifiers: {df_all['classifier'].unique().tolist()}")
    
    # Display top results for both metrics
    if not df_top10_test.empty:
        best_result_test = df_top10_test.iloc[0]
        print(f"\nTop performing configuration (Test ROC-AUC):")
        print(f"Model: {best_result_test['model']}")
        print(f"Configuration: {best_result_test['config']}")
        print(f"PCA Mode: {best_result_test['pca_mode']}")
        print(f"Test ROC-AUC: {best_result_test['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_test['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_test['overfitting_gap']:.4f}")
    
    if not df_top10_cv.empty:
        best_result_cv = df_top10_cv.iloc[0]
        print(f"\nTop performing configuration (CV ROC-AUC):")
        print(f"Model: {best_result_cv['model']}")
        print(f"Configuration: {best_result_cv['config']}")
        print(f"PCA Mode: {best_result_cv['pca_mode']}")
        print(f"Test ROC-AUC: {best_result_cv['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_cv['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_cv['overfitting_gap']:.4f}")
    
    print(f"\nPooling strategy analysis completed successfully!")
    print(f"Nested grid visualization saved as: nested_pooling_grid.png")
    print(f"Top 10 tables saved as: top10_pooling_table_test.png & top10_pooling_table_cv.png")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()