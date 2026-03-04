"""
Aggregate 5-fold cross-validation results for 2D CNN + Attention models
Combines metrics from all folds and generates summary statistics
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_fold_metrics(eval_dir, fold):
    """Load metrics for a specific fold"""
    fold_dir = Path(eval_dir) / f'fold_{fold}'
    metrics_path = fold_dir / 'metrics.json'
    
    if not metrics_path.exists():
        print(f"Warning: Metrics not found for fold {fold}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def aggregate_cv5_results(eval_dir, checkpoint_dir):
    """
    Aggregate results from all 5 folds
    
    Args:
        eval_dir: Directory containing evaluation results
        checkpoint_dir: Directory containing training checkpoints
    
    Returns:
        Dictionary with aggregated metrics
    """
    print("=" * 70)
    print("Aggregating 5-Fold Cross-Validation Results")
    print("=" * 70)
    
    all_metrics = []
    metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
    
    # Load metrics from each fold
    for fold in range(5):
        print(f"\nLoading fold {fold}...")
        metrics = load_fold_metrics(eval_dir, fold)
        
        if metrics is not None:
            all_metrics.append({
                'fold': fold,
                **{k: metrics[k] for k in metric_names if k in metrics}
            })
            
            # Print fold metrics
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
    
    if len(all_metrics) == 0:
        print("\nError: No valid fold metrics found!")
        return None
    
    # Calculate summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    summary = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            summary[metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(min_val),
                'max': float(max_val),
                'values': values
            }
            
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {mean_val:.4f} ± {std_val:.4f}")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    
    # Add fold results
    summary['fold_results'] = all_metrics
    summary['num_folds'] = len(all_metrics)
    
    # Load training metrics if available
    training_summary = {}
    checkpoint_summary_path = Path(checkpoint_dir) / 'cv5_summary.json'
    if checkpoint_summary_path.exists():
        with open(checkpoint_summary_path, 'r') as f:
            training_summary = json.load(f)
        print("\n" + "=" * 70)
        print("Training Summary")
        print("=" * 70)
        print(f"Average Training Accuracy: {training_summary.get('average_acc', 'N/A')}")
        print(f"Average Training AUC: {training_summary.get('average_auc', 'N/A')}")
    
    return summary, training_summary


def create_comparison_plots(summary, output_dir):
    """Create visualization plots for fold comparison"""
    print("\n" + "=" * 70)
    print("Creating Visualization Plots")
    print("=" * 70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
    
    # 1. Bar plot with error bars for each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metric_names):
        if metric_name in summary:
            values = summary[metric_name]['values']
            mean_val = summary[metric_name]['mean']
            std_val = summary[metric_name]['std']
            
            # Bar plot for each fold
            folds = list(range(len(values)))
            axes[idx].bar(folds, values, alpha=0.7, color='steelblue', edgecolor='black')
            axes[idx].axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            axes[idx].axhline(mean_val + std_val, color='orange', linestyle=':', linewidth=1)
            axes[idx].axhline(mean_val - std_val, color='orange', linestyle=':', linewidth=1)
            
            axes[idx].set_xlabel('Fold', fontsize=12)
            axes[idx].set_ylabel(metric_name.capitalize(), fontsize=12)
            axes[idx].set_title(f'{metric_name.capitalize()}: {mean_val:.3f} ± {std_val:.3f}', fontsize=13, fontweight='bold')
            axes[idx].set_xticks(folds)
            axes[idx].set_ylim([max(0, mean_val - 3*std_val), min(1, mean_val + 3*std_val)])
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fold_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fold_comparison.png")
    
    # 2. Box plot for all metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = []
    labels = []
    for metric_name in metric_names:
        if metric_name in summary:
            data_to_plot.append(summary[metric_name]['values'])
            labels.append(metric_name.capitalize())
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'))
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('5-Fold Cross-Validation Metrics Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: metrics_boxplot.png")
    
    # 3. Radar plot for average metrics
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    values = [summary[m]['mean'] for m in metric_names if m in summary]
    
    # Close the plot
    angles += angles[:1]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue', label='2D Model')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metric_names], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Average Performance Across All Metrics', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: radar_plot.png")
    
    # 4. Create summary table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Metric', 'Mean', 'Std', 'Min', 'Max'])
    
    for metric_name in metric_names:
        if metric_name in summary:
            row = [
                metric_name.capitalize(),
                f"{summary[metric_name]['mean']:.4f}",
                f"{summary[metric_name]['std']:.4f}",
                f"{summary[metric_name]['min']:.4f}",
                f"{summary[metric_name]['max']:.4f}"
            ]
            table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
    
    plt.title('5-Fold Cross-Validation Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: summary_table.png")


def save_results(summary, training_summary, output_dir):
    """Save aggregated results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    json_path = output_dir / 'cv5_aggregated_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'evaluation_summary': summary,
            'training_summary': training_summary
        }, f, indent=2)
    print(f"\n✓ Saved: cv5_aggregated_results.json")
    
    # Save CSV with fold results
    fold_results = summary['fold_results']
    df = pd.DataFrame(fold_results)
    csv_path = output_dir / 'fold_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: fold_results.csv")
    
    # Save summary statistics CSV
    metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
    summary_data = []
    for metric_name in metric_names:
        if metric_name in summary:
            summary_data.append({
                'Metric': metric_name,
                'Mean': summary[metric_name]['mean'],
                'Std': summary[metric_name]['std'],
                'Min': summary[metric_name]['min'],
                'Max': summary[metric_name]['max']
            })
    
    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / 'summary_statistics.csv'
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✓ Saved: summary_statistics.csv")
    
    # Create a formatted text report
    report_path = output_dir / 'cv5_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("5-FOLD CROSS-VALIDATION REPORT\n")
        f.write("2D CNN + Attention Model for Alzheimer's Classification\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 70 + "\n\n")
        
        for metric_name in metric_names:
            if metric_name in summary:
                f.write(f"{metric_name.upper()}\n")
                f.write(f"  Mean:  {summary[metric_name]['mean']:.4f}\n")
                f.write(f"  Std:   {summary[metric_name]['std']:.4f}\n")
                f.write(f"  Range: [{summary[metric_name]['min']:.4f}, {summary[metric_name]['max']:.4f}]\n\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("INDIVIDUAL FOLD RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        for fold_result in fold_results:
            f.write(f"Fold {fold_result['fold']}:\n")
            for metric_name in metric_names:
                if metric_name in fold_result:
                    f.write(f"  {metric_name.capitalize():12s}: {fold_result[metric_name]:.4f}\n")
            f.write("\n")
        
        if training_summary:
            f.write("\n" + "=" * 70 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Average Training Accuracy: {training_summary.get('average_acc', 'N/A')}\n")
            f.write(f"Average Training AUC:      {training_summary.get('average_auc', 'N/A')}\n")
    
    print(f"✓ Saved: cv5_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate 5-fold cross-validation results for 2D models'
    )
    
    parser.add_argument('--eval_dir', type=str, default='./evaluation_results/2d_cv5',
                       help='Directory containing evaluation results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/2d_cv5',
                       help='Directory containing training checkpoints')
    parser.add_argument('--output_dir', type=str, default='./results/2d_cv5',
                       help='Directory to save aggregated results')
    
    args = parser.parse_args()
    
    # Aggregate results
    summary, training_summary = aggregate_cv5_results(args.eval_dir, args.checkpoint_dir)
    
    if summary is None:
        print("\nFailed to aggregate results!")
        return
    
    # Create plots
    create_comparison_plots(summary, args.output_dir)
    
    # Save results
    save_results(summary, training_summary, args.output_dir)
    
    print("\n" + "=" * 70)
    print("Results Aggregation Completed!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
