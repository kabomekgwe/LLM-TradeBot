"""
Performance report generator with comprehensive visualizations.

Generates:
- Single model reports (equity curves, drawdown, metrics evolution)
- Model comparison reports (BiLSTM vs Transformer vs Ensemble)
- Statistical aggregation across walk-forward folds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Set seaborn style for better-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PerformanceReportGenerator:
    """
    Generate comprehensive performance reports for backtesting results.

    Creates visualizations and text reports for:
    - Single model evaluation (across walk-forward folds)
    - Multi-model comparison (BiLSTM vs Transformer vs Ensemble)
    """

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PerformanceReportGenerator initialized: output={self.output_dir}")

    def generate_single_model_report(
        self,
        model_name: str,
        backtest_results: List[Dict[str, Any]],
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate report for a single model across walk-forward folds.

        Args:
            model_name: Name of the model (e.g., "BiLSTM", "Transformer")
            backtest_results: List of backtest result dicts (one per fold)
            save_path: Optional path to save report (defaults to output_dir/model_name/)

        Returns:
            dict: Summary statistics across folds
                - mean_metrics: Average metrics across folds
                - std_metrics: Standard deviation of metrics
                - best_fold: Fold with highest Sharpe ratio
                - worst_fold: Fold with lowest Sharpe ratio
        """
        if save_path is None:
            save_path = self.output_dir / model_name.lower().replace(' ', '_')

        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Generating report for {model_name}: "
            f"{len(backtest_results)} folds, output={save_path}"
        )

        # Extract metrics from all folds
        metrics_list = [result['financial_metrics'] for result in backtest_results]
        metrics_df = pd.DataFrame(metrics_list)

        # Calculate summary statistics
        mean_metrics = metrics_df.mean().to_dict()
        std_metrics = metrics_df.std().to_dict()

        # Find best and worst folds
        best_fold_idx = metrics_df['sharpe_ratio'].idxmax()
        worst_fold_idx = metrics_df['sharpe_ratio'].idxmin()

        # Generate visualizations
        self._plot_equity_curves(model_name, backtest_results, save_path)
        self._plot_drawdown_distribution(model_name, metrics_df, save_path)
        self._plot_metrics_over_folds(model_name, metrics_df, save_path)

        # Save text report
        self._save_text_report(
            model_name,
            mean_metrics,
            std_metrics,
            metrics_df,
            save_path
        )

        logger.info(
            f"{model_name} report complete: "
            f"Sharpe={mean_metrics['sharpe_ratio']:.2f}±{std_metrics['sharpe_ratio']:.2f}, "
            f"Return={mean_metrics['total_return']:.2%}±{std_metrics['total_return']:.2%}"
        )

        return {
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'best_fold': int(best_fold_idx),
            'worst_fold': int(worst_fold_idx),
            'metrics_df': metrics_df
        }

    def generate_comparison_report(
        self,
        model_results: Dict[str, List[Dict[str, Any]]],
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate comparison report for multiple models.

        Args:
            model_results: Dict mapping model names to backtest results
                          e.g., {'BiLSTM': [...], 'Transformer': [...]}
            save_path: Optional path to save report (defaults to output_dir/comparison/)

        Returns:
            pd.DataFrame: Comparison table with mean±std for each metric
        """
        if save_path is None:
            save_path = self.output_dir / 'comparison'

        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Generating comparison report: "
            f"{len(model_results)} models, output={save_path}"
        )

        # Generate individual reports first
        summaries = {}
        for model_name, results in model_results.items():
            summary = self.generate_single_model_report(
                model_name,
                results,
                save_path=self.output_dir / model_name.lower().replace(' ', '_')
            )
            summaries[model_name] = summary

        # Create comparison DataFrame
        comparison_data = []
        for model_name, summary in summaries.items():
            mean_metrics = summary['mean_metrics']
            std_metrics = summary['std_metrics']

            comparison_data.append({
                'Model': model_name,
                'Sharpe Ratio': f"{mean_metrics['sharpe_ratio']:.2f}±{std_metrics['sharpe_ratio']:.2f}",
                'Sortino Ratio': f"{mean_metrics['sortino_ratio']:.2f}±{std_metrics['sortino_ratio']:.2f}",
                'Total Return': f"{mean_metrics['total_return']:.2%}±{std_metrics['total_return']:.2%}",
                'Annual Return': f"{mean_metrics['annual_return']:.2%}±{std_metrics['annual_return']:.2%}",
                'Max Drawdown': f"{mean_metrics['max_drawdown']:.2%}±{std_metrics['max_drawdown']:.2%}",
                'Win Rate': f"{mean_metrics['win_rate']:.2%}±{std_metrics['win_rate']:.2%}",
                'Calmar Ratio': f"{mean_metrics['calmar_ratio']:.2f}±{std_metrics['calmar_ratio']:.2f}",
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Save comparison table
        comparison_df.to_csv(save_path / 'comparison_table.csv', index=False)

        # Generate comparison visualizations
        self._plot_model_comparison(summaries, save_path)

        logger.info(f"Comparison report complete: {save_path}")

        return comparison_df

    def _plot_equity_curves(
        self,
        model_name: str,
        backtest_results: List[Dict[str, Any]],
        save_path: Path
    ):
        """Plot equity curves for all folds."""
        fig, ax = plt.subplots(figsize=(14, 7))

        for i, result in enumerate(backtest_results):
            returns = result['returns']
            if len(returns) > 0:
                equity = (1 + returns).cumprod()
                fold = result.get('fold', i + 1)
                ax.plot(equity.values, label=f'Fold {fold}', alpha=0.7)

        ax.set_title(f'{model_name} - Equity Curves Across Folds', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Equity (Initial = 1.0)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / 'equity_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved equity curves: {save_path / 'equity_curves.png'}")

    def _plot_drawdown_distribution(
        self,
        model_name: str,
        metrics_df: pd.DataFrame,
        save_path: Path
    ):
        """Plot distribution of maximum drawdowns across folds."""
        fig, ax = plt.subplots(figsize=(10, 6))

        drawdowns = metrics_df['max_drawdown'] * 100  # Convert to percentage
        ax.hist(drawdowns, bins=15, edgecolor='black', alpha=0.7, color='coral')

        ax.set_title(
            f'{model_name} - Maximum Drawdown Distribution',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_ylabel('Frequency')
        ax.axvline(
            drawdowns.mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {drawdowns.mean():.1f}%'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / 'drawdown_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved drawdown distribution: {save_path / 'drawdown_distribution.png'}")

    def _plot_metrics_over_folds(
        self,
        model_name: str,
        metrics_df: pd.DataFrame,
        save_path: Path
    ):
        """Plot evolution of key metrics across folds."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sharpe Ratio
        axes[0, 0].plot(metrics_df.index, metrics_df['sharpe_ratio'], marker='o', color='blue')
        axes[0, 0].axhline(
            metrics_df['sharpe_ratio'].mean(),
            color='red',
            linestyle='--',
            label='Mean'
        )
        axes[0, 0].set_title('Sharpe Ratio', fontweight='bold')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Sortino Ratio
        axes[0, 1].plot(metrics_df.index, metrics_df['sortino_ratio'], marker='o', color='green')
        axes[0, 1].axhline(
            metrics_df['sortino_ratio'].mean(),
            color='red',
            linestyle='--',
            label='Mean'
        )
        axes[0, 1].set_title('Sortino Ratio', fontweight='bold')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Sortino Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Total Return
        axes[1, 0].plot(
            metrics_df.index,
            metrics_df['total_return'] * 100,
            marker='o',
            color='purple'
        )
        axes[1, 0].axhline(
            metrics_df['total_return'].mean() * 100,
            color='red',
            linestyle='--',
            label='Mean'
        )
        axes[1, 0].set_title('Total Return', fontweight='bold')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Win Rate
        axes[1, 1].plot(
            metrics_df.index,
            metrics_df['win_rate'] * 100,
            marker='o',
            color='orange'
        )
        axes[1, 1].axhline(
            metrics_df['win_rate'].mean() * 100,
            color='red',
            linestyle='--',
            label='Mean'
        )
        axes[1, 1].set_title('Win Rate', fontweight='bold')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle(
            f'{model_name} - Metrics Evolution Across Folds',
            fontsize=16,
            fontweight='bold',
            y=1.00
        )
        plt.tight_layout()
        plt.savefig(save_path / 'metrics_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved metrics evolution: {save_path / 'metrics_evolution.png'}")

    def _plot_model_comparison(
        self,
        summaries: Dict[str, Dict[str, Any]],
        save_path: Path
    ):
        """Plot comparison of key metrics across models."""
        models = list(summaries.keys())
        metrics = ['sharpe_ratio', 'sortino_ratio', 'total_return', 'max_drawdown', 'win_rate']

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            means = [summaries[m]['mean_metrics'][metric] for m in models]
            stds = [summaries[m]['std_metrics'][metric] for m in models]

            # Convert to percentage for certain metrics
            if metric in ['total_return', 'max_drawdown', 'win_rate']:
                means = [m * 100 for m in means]
                stds = [s * 100 for s in stds]

            axes[idx].bar(models, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].set_title(metric.replace('_', ' ').title(), fontweight='bold')
            axes[idx].set_ylabel('Value' + (' (%)' if metric in ['total_return', 'max_drawdown', 'win_rate'] else ''))
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].tick_params(axis='x', rotation=45)

        # Hide the last subplot (we have 5 metrics but 6 subplots)
        axes[5].axis('off')

        fig.suptitle('Model Comparison - Mean ± Std', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved model comparison: {save_path / 'model_comparison.png'}")

    def _save_text_report(
        self,
        model_name: str,
        mean_metrics: Dict[str, float],
        std_metrics: Dict[str, float],
        metrics_df: pd.DataFrame,
        save_path: Path
    ):
        """Save text summary report."""
        report_path = save_path / 'report.txt'

        with open(report_path, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{model_name} - Performance Report\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of folds: {len(metrics_df)}\n\n")

            f.write(f"{'='*80}\n")
            f.write(f"Summary Statistics (Mean ± Std)\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Sharpe Ratio:      {mean_metrics['sharpe_ratio']:>8.2f} ± {std_metrics['sharpe_ratio']:>6.2f}\n")
            f.write(f"Sortino Ratio:     {mean_metrics['sortino_ratio']:>8.2f} ± {std_metrics['sortino_ratio']:>6.2f}\n")
            f.write(f"Calmar Ratio:      {mean_metrics['calmar_ratio']:>8.2f} ± {std_metrics['calmar_ratio']:>6.2f}\n\n")

            f.write(f"Total Return:      {mean_metrics['total_return']:>8.2%} ± {std_metrics['total_return']:>6.2%}\n")
            f.write(f"Annual Return:     {mean_metrics['annual_return']:>8.2%} ± {std_metrics['annual_return']:>6.2%}\n")
            f.write(f"Max Drawdown:      {mean_metrics['max_drawdown']:>8.2%} ± {std_metrics['max_drawdown']:>6.2%}\n\n")

            f.write(f"Win Rate:          {mean_metrics['win_rate']:>8.2%} ± {std_metrics['win_rate']:>6.2%}\n")
            f.write(f"Annual Volatility: {mean_metrics['annual_volatility']:>8.2%} ± {std_metrics['annual_volatility']:>6.2%}\n\n")

            f.write(f"{'='*80}\n")
            f.write(f"Per-Fold Results\n")
            f.write(f"{'='*80}\n\n")

            f.write(metrics_df.to_string(float_format=lambda x: f'{x:.4f}'))
            f.write('\n')

        logger.info(f"Saved text report: {report_path}")
