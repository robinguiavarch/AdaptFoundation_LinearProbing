"""
Analysis orchestration script for AdaptFoundation project - SAM-MED3D STRATEGY.

This script runs comprehensive analysis of SAM-Med3D classification results across
all aggregation methods and PCA strategies.

Usage:
python scripts/run_analysis_sam3d.py --features-path feature_extracted_sam3d
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.comparison_sam3d import ComparisonAnalyzerSAM3D


def main():
    """
    Main entry point for SAM-Med3D analysis script.
    """
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis of AdaptFoundation SAM-Med3D classification results"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="feature_extracted_sam3d",
        help="Path to feature_extracted_sam3d directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results_sam3d",
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
    
    print("AdaptFoundation Phase 7 - SAM-Med3D Analysis")
    print(f"Features path: {features_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = ComparisonAnalyzerSAM3D(str(features_path))
    
    # Run complete analysis
    print("\nExecuting SAM-Med3D analysis workflow...")
    df_all, figures = analyzer.run_analysis()
    
    # Save DataFrames
    if not df_all.empty:
        output_file = output_dir / "analysis_results_sam3d_all.csv"
        df_all.to_csv(output_file, index=False)
        print(f"Saved complete results: {output_file}")
        
        # Filter logistic only for separate save
        df_logistic = df_all[df_all['classifier'] == 'logistic'].copy()
        output_file_logistic = output_dir / "analysis_results_sam3d_logistic.csv"
        df_logistic.to_csv(output_file_logistic, index=False)
        print(f"Saved logistic results: {output_file_logistic}")
    
    # Save figures
    figure_names = [
        "complete_sam3d_table_test",     # Complete table ranked by Test ROC-AUC
        "complete_sam3d_table_cv"        # Complete table ranked by CV ROC-AUC
    ]
    
    for i, fig in enumerate(figures):
        if i < len(figure_names):
            filename = f"{figure_names[i]}.png"
            output_file = output_dir / filename
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {output_file}")
    
    # Display summary statistics
    if not df_all.empty:
        print(f"\nSAM-Med3D Analysis Summary:")
        print(f"Total experimental results: {len(df_all)}")
        print(f"Foundation models tested: {df_all['model'].nunique()}")
        print(f"Configurations tested: {df_all['config'].nunique()}")
        print(f"PCA modes tested: {df_all['pca_mode'].nunique()}")
        print(f"Unique configurations: {df_all['config'].unique().tolist()}")
        
        # Display best results for both metrics
        best_result_test = df_all.loc[df_all['test_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (Test ROC-AUC):")
        print(f"Model: {best_result_test['model']}")
        print(f"Configuration: {best_result_test['config']}")
        print(f"PCA Mode: {best_result_test['pca_mode']}")
        print(f"Test ROC-AUC: {best_result_test['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_test['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_test['overfitting_gap']:.4f}")
        
        best_result_cv = df_all.loc[df_all['cv_roc_auc'].idxmax()]
        print(f"\nTop performing configuration (CV ROC-AUC):")
        print(f"Model: {best_result_cv['model']}")
        print(f"Configuration: {best_result_cv['config']}")
        print(f"PCA Mode: {best_result_cv['pca_mode']}")
        print(f"Test ROC-AUC: {best_result_cv['test_roc_auc']:.4f}")
        print(f"CV ROC-AUC: {best_result_cv['cv_roc_auc']:.4f}")
        print(f"Overfitting Gap: {best_result_cv['overfitting_gap']:.4f}")
    
    print(f"\nSAM-Med3D analysis completed successfully!")
    print(f"Complete tables saved as: complete_sam3d_table_test.png & complete_sam3d_table_cv.png")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()