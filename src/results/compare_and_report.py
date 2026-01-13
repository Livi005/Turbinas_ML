"""Model comparison and reporting utilities."""

import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json


def compare_metrics(
    metrics_list: List[Dict[str, float]],
    model_names: List[str]
) -> Dict[str, Any]:
    """
    Compare multiple models' metrics and determine the best for each metric.
    
    Parameters:
    -----------
    metrics_list : List[Dict[str, float]]
        List of metric dictionaries, each containing metrics like:
        {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.90, 'f1': 0.91, 'roc_auc': 0.94}
    model_names : List[str]
        List of model names corresponding to each metrics dictionary
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'best_by_metric': Dict mapping each metric to best model name and value
        - 'overall_ranking': List of tuples (model_name, overall_score) sorted by score
        - 'detailed_comparison': Dict with all metrics for all models
    """
    if len(metrics_list) != len(model_names):
        raise ValueError("metrics_list and model_names must have the same length")
    
    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")
    
    # Get all metric names from the first dictionary
    metric_names = list(metrics_list[0].keys())
    
    # Initialize comparison structure
    detailed_comparison = {}
    best_by_metric = {}
    
    # For each metric, find the best model
    for metric in metric_names:
        metric_values = []
        for i, metrics in enumerate(metrics_list):
            if metric in metrics:
                metric_values.append((model_names[i], metrics[metric]))
        
        if metric_values:
            # Find best (highest value)
            best_model, best_value = max(metric_values, key=lambda x: x[1])
            best_by_metric[metric] = {
                'model': best_model,
                'value': best_value
            }
            
            # Store all values for detailed comparison
            detailed_comparison[metric] = {
                model_name: metrics.get(metric, 0.0)
                for model_name, metrics in zip(model_names, metrics_list)
            }
    
    # Calculate overall ranking
    # Average of normalized scores across all metrics
    overall_scores = {}
    for i, model_name in enumerate(model_names):
        scores = []
        for metric in metric_names:
            if metric in metrics_list[i]:
                # Normalize by dividing by max value for that metric
                max_val = max(m.get(metric, 0.0) for m in metrics_list if metric in m)
                if max_val > 0:
                    normalized = metrics_list[i][metric] / max_val
                    scores.append(normalized)
        overall_scores[model_name] = sum(scores) / len(scores) if scores else 0.0
    
    # Sort by overall score (descending)
    overall_ranking = sorted(
        overall_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        'best_by_metric': best_by_metric,
        'overall_ranking': overall_ranking,
        'detailed_comparison': detailed_comparison,
        'overall_scores': overall_scores
    }


def generate_markdown_report(
    comparison_results: Dict[str, Any],
    model_names: List[str],
    metrics_list: List[Dict[str, float]],
    output_path: str = None,
    title: str = "Model Comparison Report"
) -> str:
    """
    Generate a markdown report comparing multiple models.
    
    Parameters:
    -----------
    comparison_results : Dict[str, Any]
        Results from compare_metrics function
    model_names : List[str]
        List of model names
    metrics_list : List[Dict[str, float]]
        List of metric dictionaries
    output_path : str, optional
        Path to save the markdown file. If None, generates filename with timestamp.
    title : str, default="Model Comparison Report"
        Title for the report
        
    Returns:
    --------
    str
        Path to the saved markdown file
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.dirname(__file__)
        output_path = os.path.join(
            results_dir,
            f"model_comparison_report_{timestamp}.md"
        )
    else:
        # If output_path is provided but not absolute, make it relative to results directory
        if not os.path.isabs(output_path):
            results_dir = os.path.dirname(__file__)
            output_path = os.path.join(results_dir, output_path)
        # Ensure .md extension
        if not output_path.endswith('.md'):
            output_path += '.md'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate markdown content
    md_content = []
    md_content.append(f"# {title}\n")
    md_content.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"\n**Number of models compared:** {len(model_names)}\n")
    
    # Executive Summary
    md_content.append("\n## Executive Summary\n")
    best_overall = comparison_results['overall_ranking'][0]
    md_content.append(f"\n**Best Overall Model:** `{best_overall[0]}` (Score: {best_overall[1]:.4f})\n")
    
    md_content.append("\n### Best Model by Metric\n")
    md_content.append("| Metric | Best Model | Value |\n")
    md_content.append("|--------|------------|-------|\n")
    for metric, best_info in comparison_results['best_by_metric'].items():
        md_content.append(
            f"| {metric.capitalize()} | `{best_info['model']}` | {best_info['value']:.4f} |\n"
        )
    
    # Overall Ranking
    md_content.append("\n## Overall Ranking\n")
    md_content.append("| Rank | Model Name | Overall Score |\n")
    md_content.append("|------|------------|---------------|\n")
    for rank, (model_name, score) in enumerate(comparison_results['overall_ranking'], 1):
        md_content.append(f"| {rank} | `{model_name}` | {score:.4f} |\n")
    
    # Detailed Metrics Comparison
    md_content.append("\n## Detailed Metrics Comparison\n")
    
    # Create a table with all metrics for all models
    metric_names = list(comparison_results['detailed_comparison'].keys())
    
    # Header row
    header = "| Model | " + " | ".join(m.capitalize() for m in metric_names) + " |\n"
    separator = "|" + "|".join(["------"] * (len(metric_names) + 1)) + "|\n"
    md_content.append(header)
    md_content.append(separator)
    
    # Data rows
    for model_name, metrics in zip(model_names, metrics_list):
        row = f"| `{model_name}` | "
        row += " | ".join(f"{metrics.get(m, 0.0):.4f}" for m in metric_names)
        row += " |\n"
        md_content.append(row)
    
    # Best by Metric Section
    md_content.append("\n## Best Model for Each Metric\n")
    for metric, best_info in comparison_results['best_by_metric'].items():
        md_content.append(f"\n### {metric.capitalize()}\n")
        md_content.append(f"- **Best Model:** `{best_info['model']}`\n")
        md_content.append(f"- **Value:** {best_info['value']:.4f}\n")
        
        # Show all models for this metric
        md_content.append("\n**All Models:**\n")
        md_content.append("| Model | Value |\n")
        md_content.append("|-------|-------|\n")
        metric_values = [
            (name, metrics.get(metric, 0.0))
            for name, metrics in zip(model_names, metrics_list)
        ]
        metric_values.sort(key=lambda x: x[1], reverse=True)
        for name, value in metric_values:
            marker = "⭐" if name == best_info['model'] else ""
            md_content.append(f"| `{name}` {marker} | {value:.4f} |\n")
    
    # Recommendations
    md_content.append("\n## Recommendations\n")
    md_content.append("\n### Best Overall Choice\n")
    md_content.append(f"Based on the overall ranking, **`{best_overall[0]}`** is recommended as the best overall model.\n")
    
    md_content.append("\n### Metric-Specific Recommendations\n")
    for metric, best_info in comparison_results['best_by_metric'].items():
        md_content.append(f"- **For {metric.capitalize()}:** Use `{best_info['model']}` (value: {best_info['value']:.4f})\n")
    
    # Model Descriptions (if available)
    md_content.append("\n## Model Configurations\n")
    md_content.append("\nThe following models were compared:\n")
    for i, model_name in enumerate(model_names, 1):
        md_content.append(f"{i}. **`{model_name}`**\n")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(md_content))
    
    return output_path


def compare_and_report(
    metrics_list: List[Dict[str, float]],
    model_names: List[str],
    output_path: str = None,
    title: str = "Model Comparison Report"
) -> Tuple[Dict[str, Any], str]:
    """
    Compare models and generate a markdown report in one step.
    
    Parameters:
    -----------
    metrics_list : List[Dict[str, float]]
        List of metric dictionaries
    model_names : List[str]
        List of model names
    output_path : str, optional
        Path to save the markdown file
    title : str, default="Model Comparison Report"
        Title for the report
        
    Returns:
    --------
    Tuple[Dict[str, Any], str]
        Tuple of (comparison_results, report_path)
    """
    # Compare metrics
    comparison_results = compare_metrics(metrics_list, model_names)
    
    # Generate report
    report_path = generate_markdown_report(
        comparison_results,
        model_names,
        metrics_list,
        output_path,
        title
    )
    
    return comparison_results, report_path


def print_comparison_summary(comparison_results: Dict[str, Any]) -> None:
    """
    Print a summary of the comparison results to console.
    
    Parameters:
    -----------
    comparison_results : Dict[str, Any]
        Results from compare_metrics function
    """
    print("=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n📊 Best Model by Metric:")
    print("-" * 70)
    for metric, best_info in comparison_results['best_by_metric'].items():
        print(f"  {metric.capitalize():<15}: {best_info['model']:<30} ({best_info['value']:.4f})")
    
    print("\n🏆 Overall Ranking:")
    print("-" * 70)
    for rank, (model_name, score) in enumerate(comparison_results['overall_ranking'], 1):
        marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {marker} Rank {rank}: {model_name:<30} (Score: {score:.4f})")
    
    print("\n" + "=" * 70)

