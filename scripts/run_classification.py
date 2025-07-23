"""
Classification pipeline script for AdaptFoundation project.

This script runs linear probing classification on extracted features
with comprehensive evaluation and hyperparameter optimization.

YAML Configuration Mode:
python scripts/run_classification.py --config-file configs/classification.yaml

Focus Mode:
python scripts/run_classification.py --config-file configs/classification.yaml --focus-only

Manual Mode:
python scripts/run_classification.py --config multi_axes_average --classifier logistic --model-name dinov2_vits14
"""

import argparse
import json
import time
import yaml
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from classification.linear_probing import LinearProber


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_execution_plan(config: Dict[str, Any], focus_only: bool = False) -> List[Dict[str, str]]:
    """
    Generate execution plan from YAML configuration.
    
    Args:
        config (Dict[str, Any]): Loaded YAML configuration
        focus_only (bool): If True, only run focus combinations
        
    Returns:
        List[Dict[str, str]]: List of execution tasks
    """
    tasks = []
    
    if focus_only:
        focus = config['execution']['focus_combinations']
        task = {
            'model': focus['model'],
            'config': focus['config'],
            'classifier': focus['classifier'],
            'pca_mode': focus['pca_mode']
        }
        tasks.append(task)
        print(f"Focus mode: Running 1 task for {focus['model']}")
    else:
        models = config['models']
        configurations = config['configurations']
        classifiers = config['classifiers']
        pca_modes = config['pca_modes']
        
        for model in models:
            for cfg in configurations:
                for classifier in classifiers:
                    for pca_mode in pca_modes:
                        task = {
                            'model': model,
                            'config': cfg,
                            'classifier': classifier,
                            'pca_mode': pca_mode
                        }
                        tasks.append(task)
        
        print(f"Full mode: Generated {len(tasks)} tasks")
        print(f"Models: {len(models)}, Configs: {len(configurations)}, Classifiers: {len(classifiers)}, PCA modes: {len(pca_modes)}")
    
    return tasks


def validate_task(task: Dict[str, str], features_path: str, validation_config: Dict[str, Any]) -> bool:
    """
    Validate that a task can be executed.
    
    Args:
        task (Dict[str, str]): Task to validate
        features_path (str): Path to feature_extracted directory
        validation_config (Dict[str, Any]): Validation configuration
        
    Returns:
        bool: True if task is valid
    """
    if not validation_config.get('check_feature_files', True):
        return True
    
    config_path = Path(features_path) / task['model'] / task['config']
    
    if not config_path.exists():
        print(f"Skipping {task['model']}/{task['config']}: Directory not found")
        return False
    
    if task['pca_mode'] != 'none':
        pca_path = config_path / f"PCA_{task['pca_mode']}"
        if not pca_path.exists():
            if validation_config.get('check_pca_availability', True):
                print(f"Skipping {task['model']}/{task['config']}/PCA_{task['pca_mode']}: PCA not available")
                return False
    
    if validation_config.get('skip_existing', False):
        results_file = config_path / "classification_results.json"
        if task['pca_mode'] != 'none':
            results_file = config_path / f"PCA_{task['pca_mode']}" / "classification_results.json"
        
        if results_file.exists():
            print(f"Skipping {task['model']}/{task['config']}: Results already exist")
            return False
    
    return True


def execute_task(task: Dict[str, str], features_path: str, reporting_config: Dict[str, Any],
                yaml_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute a single classification task.
    
    Args:
        task (Dict[str, str]): Task to execute
        features_path (str): Path to feature_extracted directory
        reporting_config (Dict[str, Any]): Reporting configuration
        yaml_config (Dict[str, Any], optional): Full YAML configuration for classifier params
        
    Returns:
        Dict[str, Any]: Task results
    """
    model = task['model']
    config = task['config']
    classifier = task['classifier']
    pca_mode = task['pca_mode']
    use_pca = pca_mode != 'none'
    
    if reporting_config.get('verbose', True):
        pca_info = f"PCA_{pca_mode}" if use_pca else "No PCA"
        print(f"Executing: {model} | {config} | {classifier} | {pca_info}")
    
    try:
        # Extract classifier parameters from YAML config
        classifier_params = yaml_config.get('classifier_params', {}) if yaml_config else {}
        
        prober = LinearProber(features_path, model_name=model, classifier_params=classifier_params)
        
        start_time = time.time()
        result = prober.train_classifier(config, classifier, use_pca, int(pca_mode) if pca_mode != 'none' else 95)
        total_time = time.time() - start_time
        
        result['task_metadata'] = {
            'model': model,
            'config': config,
            'classifier': classifier,
            'pca_mode': pca_mode,
            'total_pipeline_time': total_time
        }
        
        if reporting_config.get('verbose', True):
            cv_score = result['best_cv_score']
            test_score = result['test_metrics']['roc_auc_weighted']
            overfitting = result['diagnostics']['overfitting_severity']
            convergence = "OK" if not result['diagnostics']['convergence_warning'] else "WARNING"
            print(f"Completed: CV={cv_score:.4f} | Test={test_score:.4f} | Overfit={overfitting} | Conv={convergence}")
        
        return {'status': 'success', 'result': result}
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        return {'status': 'error', 'error': error_msg, 'task': task}


def save_results(task: Dict[str, str], result_data: Dict[str, Any], 
                features_path: str, output_config: Dict[str, Any]) -> None:
    """
    Save classification results.
    
    Args:
        task (Dict[str, str]): Executed task
        result_data (Dict[str, Any]): Task results
        features_path (str): Path to feature_extracted directory
        output_config (Dict[str, Any]): Output configuration
    """
    if result_data['status'] != 'success':
        return
    
    result = result_data['result']
    model = task['model']
    config = task['config']
    pca_mode = task['pca_mode']
    use_pca = pca_mode != 'none'
    
    if output_config.get('save_individual', True):
        config_dir = Path(features_path) / model / config
        if use_pca:
            output_dir = config_dir / f"PCA_{pca_mode}"
        else:
            output_dir = config_dir
        
        results_file = output_dir / "classification_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({config: {result['classifier_type']: result}}, f, indent=2, default=str)
    
    if output_config.get('save_consolidated', True):
        suffix = f"_pca_{pca_mode}" if use_pca else ""
        consolidated_file = Path(features_path) / f"classification_results{suffix}_{model}.json"
        
        if consolidated_file.exists():
            with open(consolidated_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        if config not in all_results:
            all_results[config] = {}
        all_results[config][result['classifier_type']] = result
        
        with open(consolidated_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)


def run_yaml_classification(config_file: str, focus_only: bool = False) -> None:
    """
    Run classification pipeline using YAML configuration.
    
    Args:
        config_file (str): Path to YAML configuration file
        focus_only (bool): If True, only run focus combinations
    """
    print("AdaptFoundation Classification Pipeline (YAML Mode)")
    print(f"Configuration: {config_file}")
    
    config = load_yaml_config(config_file)
    features_path = config['output']['base_path']
    
    print(f"Features path: {features_path}")
    print(f"Focus mode: {focus_only}")
    
    # Print classifier parameters if available
    if 'classifier_params' in config:
        print("Classifier parameters from YAML:")
        for clf_type, params in config['classifier_params'].items():
            if isinstance(params, dict):
                key_params = {k: v for k, v in params.items() if k in ['max_iter', 'solver', 'penalty']}
                print(f"  {clf_type}: {key_params}")
    
    tasks = get_execution_plan(config, focus_only)
    
    valid_tasks = []
    for task in tasks:
        if validate_task(task, features_path, config['validation']):
            valid_tasks.append(task)
    
    print(f"Execution Plan: {len(valid_tasks)}/{len(tasks)} valid tasks")
    
    if not valid_tasks:
        print("No valid tasks to execute")
        return
    
    results = []
    start_time = time.time()
    
    for i, task in enumerate(valid_tasks, 1):
        print(f"[{i}/{len(valid_tasks)}]")
        
        result_data = execute_task(task, features_path, config['reporting'], config)
        results.append({'task': task, 'result': result_data})
        
        save_results(task, result_data, features_path, config['output'])
    
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if r['result']['status'] == 'success')
    failed = len(results) - successful
    
    print(f"Pipeline Completed")
    print(f"Total time: {total_time:.2f}s")
    print(f"Tasks: {successful} successful, {failed} failed")
    
    if failed > 0:
        print("Failed tasks:")
        for r in results:
            if r['result']['status'] == 'error':
                task = r['task']
                print(f"  {task['model']}/{task['config']}/{task['classifier']}: {r['result']['error']}")


def run_manual_classification(features_path: str, config_name: str = None, 
                             classifier_type: str = None, use_pca: bool = False, 
                             pca_mode: int = 95, model_name: str = 'dinov2_vits14',
                             classifier_params: Dict = None) -> None:
    """
    Run classification pipeline in manual mode.
    
    Args:
        features_path (str): Path to feature_extracted directory
        config_name (str, optional): Specific configuration to evaluate
        classifier_type (str, optional): Specific classifier to train
        use_pca (bool): Whether to use PCA-reduced features
        pca_mode (int): PCA mode (95, 256, 32)
        model_name (str): Foundation model name
        classifier_params (Dict, optional): Custom classifier parameters
    """
    print("AdaptFoundation Classification Pipeline (Manual Mode)")
    print(f"Features path: {features_path}")
    print(f"Model: {model_name}")
    print(f"Use PCA: {use_pca}")
    if use_pca:
        print(f"PCA mode: {pca_mode}")
    
    # Initialize prober with classifier parameters
    prober = LinearProber(features_path, model_name=model_name, classifier_params=classifier_params or {})
    
    available_configs = prober.get_available_configurations()
    print(f"Available configurations: {available_configs}")
    
    if config_name:
        if config_name not in available_configs:
            print(f"Configuration '{config_name}' not found")
            return
        configs_to_evaluate = [config_name]
    else:
        configs_to_evaluate = available_configs
    
    if classifier_type:
        classifiers_to_evaluate = [classifier_type]
    else:
        classifiers_to_evaluate = ['logistic', 'knn', 'svm_linear']
    
    all_results = {}
    
    for config in configs_to_evaluate:
        print(f"Evaluating Configuration: {config}")
        
        pca_available = prober.check_pca_availability(config, pca_mode)
        if use_pca and not pca_available:
            print(f"PCA_{pca_mode} not available for {config}, skipping PCA evaluation")
            continue
        
        config_results = {}
        
        for clf_type in classifiers_to_evaluate:
            print(f"Training {clf_type}")
            start_time = time.time()
            
            try:
                result = prober.train_classifier(config, clf_type, use_pca, pca_mode)
                elapsed_time = time.time() - start_time
                result['total_pipeline_time'] = elapsed_time
                
                config_results[clf_type] = result
                
                print(f"{clf_type} completed in {elapsed_time:.2f}s")
                print(f"Best CV score: {result['best_cv_score']:.4f}")
                print(f"Test ROC-AUC: {result['test_metrics']['roc_auc_weighted']:.4f}")
                print(f"Overfitting: {result['diagnostics']['overfitting_severity']}")
                
            except Exception as e:
                print(f"Error training {clf_type}: {str(e)}")
                config_results[clf_type] = {'error': str(e)}
        
        all_results[config] = config_results
    
    for config in configs_to_evaluate:
        if config in all_results:
            config_dir = Path(features_path) / model_name / config
            if use_pca:
                output_dir = config_dir / f"PCA_{pca_mode}"
            else:
                output_dir = config_dir
            
            results_file = output_dir / "classification_results.json"
            
            with open(results_file, 'w') as f:
                json.dump({config: all_results[config]}, f, indent=2, default=str)
            
            print(f"Results for {config} saved to: {results_file}")
    
    if use_pca:
        suffix = f"_pca_{pca_mode}"
    else:
        suffix = ""
    consolidated_file = Path(features_path) / f"classification_results{suffix}_{model_name}.json"
    
    with open(consolidated_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Classification Pipeline Completed")
    print(f"Consolidated results saved to: {consolidated_file}")
    
    print("Summary:")
    for config, config_results in all_results.items():
        print(f"Configuration: {config}")
        for clf_type, result in config_results.items():
            if 'error' not in result:
                cv_score = result['best_cv_score']
                test_score = result['test_metrics']['roc_auc_weighted']
                overfitting = result['diagnostics']['overfitting_severity']
                convergence = "OK" if not result['diagnostics']['convergence_warning'] else "WARNING"
                print(f"  {clf_type}: CV={cv_score:.4f} | Test={test_score:.4f} | Overfit={overfitting} | Conv={convergence}")
            else:
                print(f"  {clf_type}: ERROR")


def main():
    """
    Main entry point for classification script.
    """
    parser = argparse.ArgumentParser(
        description="Run linear probing classification on extracted features"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--focus-only",
        action="store_true",
        help="Only run focus combinations from YAML config"
    )
    
    parser.add_argument(
        "--features-path",
        type=str,
        default="feature_extracted",
        help="Path to feature_extracted directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Specific configuration to evaluate"
    )
    
    parser.add_argument(
        "--classifier",
        type=str,
        choices=['logistic', 'knn', 'svm_linear'],
        help="Specific classifier to train"
    )
    
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Use PCA-reduced features"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2_vits14",
        help="Foundation model name"
    )
    
    parser.add_argument(
        "--pca-mode",
        type=int,
        choices=[32, 95, 256],
        default=95,
        help="PCA mode: 32, 95, or 256"
    )
    
    args = parser.parse_args()
    
    if args.config_file:
        config_file = Path(args.config_file)
        if not config_file.exists():
            print(f"Configuration file does not exist: {config_file}")
            sys.exit(1)
        
        run_yaml_classification(str(config_file), args.focus_only)
    else:
        features_path = Path(args.features_path)
        if not features_path.exists():
            print(f"Features directory does not exist: {features_path}")
            sys.exit(1)
        
        run_manual_classification(
            str(features_path),
            config_name=args.config,
            classifier_type=args.classifier,
            use_pca=args.use_pca,
            pca_mode=args.pca_mode,
            model_name=args.model_name,
            classifier_params=None  # No YAML config in manual mode
        )


if __name__ == "__main__":
    main()