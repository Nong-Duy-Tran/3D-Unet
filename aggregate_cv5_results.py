"""
Aggregate results from 5-fold cross-validation
Computes mean ± std for all metrics across folds
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_fold_results(results_dir, fold_num):
    """
    Load evaluation results for a single fold
    
    Args:
        results_dir: Base results directory
        fold_num: Fold number (0-4)
    
    Returns:
        Dictionary with metrics or None if not found
    """
    fold_dir = Path(results_dir) / f'cv5_fold_{fold_num}'
    metrics_file = fold_dir / 'metrics.json'
    
    if not metrics_file.exists():
        print(f"Warning: Metrics not found for fold {fold_num}: {metrics_file}")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading metrics for fold {fold_num}: {e}")
        return None


def aggregate_metrics(all_fold_metrics):
    """
    Aggregate metrics across all folds
    
    Args:
        all_fold_metrics: List of metric dictionaries from each fold
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not all_fold_metrics:
        return None
    
    # Collect all metric names
    metric_names = set()
    for metrics in all_fold_metrics:
        if metrics:
            metric_names.update(metrics.keys())
    
    # Aggregate each metric
    aggregated = {}
    
    for metric_name in metric_names:
        values = []
        for metrics in all_fold_metrics:
            if metrics and metric_name in metrics:
                val = metrics[metric_name]
                # Handle both scalar and dict values
                if isinstance(val, (int, float)):
                    values.append(val)
                elif isinstance(val, dict) and 'value' in val:
                    values.append(val['value'])
        
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
    
    return aggregated


def print_results_table(aggregated_metrics, all_fold_metrics):
    """
    Print formatted results table
    """
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    
    # Individual fold results
    print("\nIndividual Fold Results:")
    print("-" * 80)
    
    if all_fold_metrics and any(all_fold_metrics):
        # Get metric names from first valid fold
        metric_names = []
        for metrics in all_fold_metrics:
            if metrics:
                metric_names = list(metrics.keys())
                break
        
        # Header
        header = "Metric".ljust(20)
        for i in range(len(all_fold_metrics)):
            header += f"Fold {i}".rjust(12)
        print(header)
        print("-" * 80)
        
        # Each metric row
        for metric_name in metric_names:
            row = metric_name.ljust(20)
            for metrics in all_fold_metrics:
                if metrics and metric_name in metrics:
                    val = metrics[metric_name]
                    if isinstance(val, (int, float)):
                        row += f"{val:12.4f}"
                    elif isinstance(val, dict) and 'value' in val:
                        row += f"{val['value']:12.4f}"
                    else:
                        row += " " * 12
                else:
                    row += " " * 12
            print(row)
    
    # Aggregated results
    print("\n" + "="*80)
    print("AGGREGATED RESULTS (Mean ± Std)")
    print("="*80)
    
    if aggregated_metrics:
        # Sort metrics by name for consistent display
        sorted_metrics = sorted(aggregated_metrics.items())
        
        print(f"\n{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-" * 80)
        
        for metric_name, stats in sorted_metrics:
            print(f"{metric_name:<20} "
                  f"{stats['mean']:12.4f} "
                  f"{stats['std']:12.4f} "
                  f"{stats['min']:12.4f} "
                  f"{stats['max']:12.4f}")
        
        print("\n" + "="*80)
        print("SUMMARY (for reporting)")
        print("="*80)
        
        # Key metrics for easy copying
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric_name in key_metrics:
            if metric_name in aggregated_metrics:
                stats = aggregated_metrics[metric_name]
                print(f"{metric_name.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\n" + "="*80)


def save_aggregated_results(aggregated_metrics, all_fold_metrics, output_file):
    """
    Save aggregated results to JSON file
    """
    results = {
        'aggregated': aggregated_metrics,
        'individual_folds': all_fold_metrics,
        'n_folds': len(all_fold_metrics)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def create_results_csv(aggregated_metrics, all_fold_metrics, output_file):
    """
    Create CSV file with all results
    """
    # Create DataFrame for individual folds
    rows = []
    for fold_num, metrics in enumerate(all_fold_metrics):
        if metrics:
            row = {'fold': fold_num}
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    row[key] = val
                elif isinstance(val, dict) and 'value' in val:
                    row[key] = val['value']
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        
        # Add aggregated row
        if aggregated_metrics:
            agg_row = {'fold': 'mean'}
            for key, stats in aggregated_metrics.items():
                agg_row[key] = stats['mean']
            df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)
            
            std_row = {'fold': 'std'}
            for key, stats in aggregated_metrics.items():
                std_row[key] = stats['std']
            df = pd.concat([df, pd.DataFrame([std_row])], ignore_index=True)
        
        df.to_csv(output_file, index=False)
        print(f"CSV saved to: {output_file}")


def main(args):
    print("="*80)
    print("Aggregating 5-Fold Cross-Validation Results")
    print("="*80)
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"\nError: Results directory not found: {results_dir}")
        print("Please evaluate all folds first:")
        print("  bash script/evaluate_cv5_all.sh")
        return
    
    print(f"\nResults directory: {results_dir}")
    print(f"Number of folds: {args.n_folds}")
    print()
    
    # Load results from all folds
    all_fold_metrics = []
    missing_folds = []
    
    for fold_num in range(args.n_folds):
        print(f"Loading fold {fold_num}...")
        metrics = load_fold_results(results_dir, fold_num)
        all_fold_metrics.append(metrics)
        
        if metrics is None:
            missing_folds.append(fold_num)
    
    # Check if we have any results
    valid_folds = [m for m in all_fold_metrics if m is not None]
    
    if not valid_folds:
        print("\n" + "="*80)
        print("ERROR: No valid fold results found!")
        print("="*80)
        print("\nPlease evaluate the folds first:")
        print("  bash script/evaluate_cv5_all.sh")
        return
    
    if missing_folds:
        print(f"\nWarning: Missing results for folds: {missing_folds}")
        print(f"Using {len(valid_folds)}/{args.n_folds} available folds for aggregation")
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    aggregated_metrics = aggregate_metrics(all_fold_metrics)
    
    # Print results table
    print_results_table(aggregated_metrics, all_fold_metrics)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_file = output_dir / 'cv5_aggregated_results.json'
    save_aggregated_results(aggregated_metrics, all_fold_metrics, json_file)
    
    # Save CSV
    csv_file = output_dir / 'cv5_results.csv'
    create_results_csv(aggregated_metrics, all_fold_metrics, csv_file)
    
    print("\n" + "="*80)
    print("✓ Aggregation completed!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - {json_file.name}")
    print(f"  - {csv_file.name}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aggregate 5-fold cross-validation results'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./evaluation_results',
        help='Directory containing fold evaluation results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for aggregated results'
    )
    
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of folds'
    )
    
    args = parser.parse_args()
    main(args)
