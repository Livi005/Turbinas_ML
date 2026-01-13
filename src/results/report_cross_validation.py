"""Cross-validation reporting utilities."""

import os
from datetime import datetime
from typing import Dict, Any, List, Tuple


def _ensure_results_path(filename: str) -> str:
    """Return an absolute path inside the results directory."""
    results_dir = os.path.dirname(__file__)
    if os.path.isabs(filename):
        path = filename
    else:
        path = os.path.join(results_dir, filename)
    if not path.endswith('.md'):
        path += '.md'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _metric_order(cv_results: Dict[str, Any]) -> List[str]:
    """Return metric ordering from cv_results."""
    return list(cv_results.get('mean', {}).keys())


def summarize_cv_results(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build helpful aggregates from cross-validation results.

    Parameters
    ----------
    cv_results : dict
        Output dictionary returned by cross_validate_model.

    Returns
    -------
    dict
        Dictionary containing per-metric summaries and per-fold stats.
    """
    if not cv_results:
        raise ValueError("cv_results cannot be empty")

    fold_results = cv_results.get('fold_results', [])
    if not fold_results:
        raise ValueError("cv_results must contain 'fold_results'")

    metric_names = _metric_order(cv_results)
    summaries = {}
    for metric in metric_names:
        values = [fold.get(metric, 0.0) for fold in fold_results]
        summaries[metric] = {
            'mean': cv_results['mean'].get(metric, 0.0),
            'std': cv_results['std'].get(metric, 0.0),
            'min': min(values) if values else 0.0,
            'max': max(values) if values else 0.0,
        }

    best_folds = {}
    worst_folds = {}
    for metric in metric_names:
        fold_values = [
            (fold.get(metric, 0.0), fold.get('fold', idx + 1))
            for idx, fold in enumerate(fold_results)
        ]
        best_value, best_fold = max(fold_values, key=lambda x: x[0])
        worst_value, worst_fold = min(fold_values, key=lambda x: x[0])
        best_folds[metric] = {'fold': best_fold, 'value': best_value}
        worst_folds[metric] = {'fold': worst_fold, 'value': worst_value}

    return {
        'summaries': summaries,
        'best_folds': best_folds,
        'worst_folds': worst_folds,
        'metric_names': metric_names,
        'fold_results': fold_results,
        'cv_folds': cv_results.get('cv_folds'),
    }


def _build_summary_table(summary: Dict[str, Dict[str, float]]) -> str:
    lines = []
    header = "| Metric | Mean | Std | Min | Max |\n"
    divider = "|--------|------|-----|-----|-----|\n"
    lines.append(header)
    lines.append(divider)
    for metric, stats in summary.items():
        lines.append(
            f"| {metric.capitalize()} | "
            f"{stats['mean']:.4f} | "
            f"{stats['std']:.4f} | "
            f"{stats['min']:.4f} | "
            f"{stats['max']:.4f} |\n"
        )
    return "".join(lines)


def _build_fold_table(metric_names: List[str], fold_results: List[Dict[str, Any]]) -> str:
    lines = []
    header = "| Fold | " + " | ".join(m.capitalize() for m in metric_names) + " |\n"
    divider = "|" + "|".join(["------"] * (len(metric_names) + 1)) + "|\n"
    lines.append(header)
    lines.append(divider)
    for fold in fold_results:
        row = f"| {fold.get('fold', '?')} | "
        row += " | ".join(f"{fold.get(metric, 0.0):.4f}" for metric in metric_names)
        row += " |\n"
        lines.append(row)
    return "".join(lines)


def generate_cv_markdown_report(
    cv_results: Dict[str, Any],
    model_name: str = "logistic_regression",
    output_path: str = None,
    title: str = None,
) -> str:
    """
    Generate a markdown report for cross-validation results.

    Parameters
    ----------
    cv_results : dict
        Output dictionary from cross_validate_model.
    model_name : str, default="logistic_regression"
        Name of the model evaluated.
    output_path : str, optional
        Desired output filename (inside results/ if relative). If None, a timestamped
        filename is generated.
    title : str, optional
        Custom report title. Defaults to "Cross-Validation Report - {model_name}".

    Returns
    -------
    str
        Path to the saved markdown report.
    """
    summary_data = summarize_cv_results(cv_results)

    if title is None:
        title = f"Cross-Validation Report - {model_name}"

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_cv_report_{timestamp}.md"
    else:
        filename = output_path
    report_path = _ensure_results_path(filename)

    md = []
    md.append(f"# {title}\n")
    md.append(f"\n**Generated on:** {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    md.append(f"\n**Model:** `{model_name}`\n")
    if summary_data['cv_folds']:
        md.append(f"\n**Number of folds:** {summary_data['cv_folds']}\n")

    md.append("\n## Overall Metrics Summary\n\n")
    md.append(_build_summary_table(summary_data['summaries']))

    md.append("\n## Best & Worst Folds by Metric\n")
    for metric in summary_data['metric_names']:
        best = summary_data['best_folds'][metric]
        worst = summary_data['worst_folds'][metric]
        md.append(f"\n### {metric.capitalize()}\n")
        md.append(f"- **Best Fold:** #{best['fold']} (value: {best['value']:.4f})\n")
        md.append(f"- **Worst Fold:** #{worst['fold']} (value: {worst['value']:.4f})\n")

    md.append("\n## Per-Fold Results\n\n")
    md.append(_build_fold_table(summary_data['metric_names'], summary_data['fold_results']))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(md))

    return report_path


def report_cross_validation(
    cv_results: Dict[str, Any],
    model_name: str = "logistic_regression",
    output_path: str = None,
    title: str = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Convenience function to summarize CV results and generate a report.

    Returns
    -------
    tuple(dict, str)
        Summary dictionary and path to the markdown report.
    """
    summary = summarize_cv_results(cv_results)
    report_path = generate_cv_markdown_report(
        cv_results=cv_results,
        model_name=model_name,
        output_path=output_path,
        title=title,
    )
    return summary, report_path
