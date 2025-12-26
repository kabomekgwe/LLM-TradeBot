"""Backtest Report Generation - Human-readable performance reports.

Generates markdown and HTML reports from backtest results with charts,
tables, and comprehensive performance analytics.
"""

import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import json

from .engine import BacktestResult


class BacktestReport:
    """Generates human-readable reports from backtest results.

    Creates markdown and HTML reports with:
    - Executive summary
    - Performance metrics table
    - Trade-by-trade breakdown
    - Charts (if matplotlib available)
    - Risk analysis

    Example:
        >>> report = BacktestReport()
        >>> report.generate_markdown(result, output_path)
        >>> report.generate_html(result, output_path)
    """

    def __init__(self):
        """Initialize report generator."""
        self.logger = logging.getLogger(__name__)

    def generate_markdown(
        self,
        result: BacktestResult,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate markdown report from backtest result.

        Args:
            result: BacktestResult to generate report from
            output_path: Optional path to save report (if None, returns string)

        Returns:
            Markdown report as string

        Example:
            >>> report_md = report.generate_markdown(result)
            >>> print(report_md)
        """
        # Build markdown report
        md = self._build_markdown_report(result)

        # Save if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(md)
            self.logger.info(f"Markdown report saved to {output_path}")

        return md

    def generate_html(
        self,
        result: BacktestResult,
        output_path: Optional[Path] = None,
    ) -> str:
        """Generate HTML report from backtest result.

        Args:
            result: BacktestResult to generate report from
            output_path: Optional path to save report

        Returns:
            HTML report as string
        """
        # Build HTML from markdown
        md = self._build_markdown_report(result)
        html = self._markdown_to_html(md, result)

        # Save if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            self.logger.info(f"HTML report saved to {output_path}")

        return html

    def _build_markdown_report(self, result: BacktestResult) -> str:
        """Build comprehensive markdown report."""
        # Header
        md = f"# Backtest Report: {result.symbol}\n\n"
        md += f"**Backtest ID:** `{result.backtest_id}`  \n"
        md += f"**Period:** {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}  \n"
        md += f"**Duration:** {(result.end_date - result.start_date).days} days  \n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n"

        # Executive Summary
        md += "## Executive Summary\n\n"
        md += self._build_executive_summary(result)
        md += "\n\n"

        # Performance Metrics
        md += "## Performance Metrics\n\n"
        md += self._build_metrics_table(result)
        md += "\n\n"

        # Trade Statistics
        md += "## Trade Statistics\n\n"
        md += self._build_trade_stats(result)
        md += "\n\n"

        # Risk Analysis
        md += "## Risk Analysis\n\n"
        md += self._build_risk_analysis(result)
        md += "\n\n"

        # Top Trades
        if result.trades:
            md += "## Top 10 Trades\n\n"
            md += self._build_top_trades_table(result)
            md += "\n\n"

        # Configuration
        md += "## Configuration\n\n"
        md += "```json\n"
        md += json.dumps(result.config, indent=2)
        md += "\n```\n\n"

        # Execution Info
        md += "## Execution Information\n\n"
        md += f"- **Candles Processed:** {result.candles_processed:,}\n"
        md += f"- **Execution Time:** {result.execution_time_seconds:.2f} seconds\n"
        md += f"- **Processing Speed:** {result.candles_processed / result.execution_time_seconds:.0f} candles/sec\n"

        return md

    def _build_executive_summary(self, result: BacktestResult) -> str:
        """Build executive summary section."""
        # Overall verdict
        if result.total_trades == 0:
            verdict = "❌ **No trades executed**"
            verdict_color = "red"
        elif result.total_pnl > 0 and result.sharpe_ratio > 1.0:
            verdict = "✅ **Profitable strategy with good risk-adjusted returns**"
            verdict_color = "green"
        elif result.total_pnl > 0:
            verdict = "⚠️ **Profitable but with moderate risk**"
            verdict_color = "orange"
        else:
            verdict = "❌ **Unprofitable strategy**"
            verdict_color = "red"

        summary = f"{verdict}\n\n"
        summary += f"- **Total P&L:** ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)\n"
        summary += f"- **Win Rate:** {result.win_rate:.1f}% ({result.winning_trades}/{result.total_trades} trades)\n"
        summary += f"- **Sharpe Ratio:** {result.sharpe_ratio:.2f}\n"
        summary += f"- **Max Drawdown:** {result.max_drawdown_pct:.1f}%\n"
        summary += f"- **Profit Factor:** {result.profit_factor:.2f}\n"

        return summary

    def _build_metrics_table(self, result: BacktestResult) -> str:
        """Build performance metrics table."""
        table = "| Metric | Value |\n"
        table += "|--------|-------|\n"

        # Returns
        table += f"| **Total Return** | {result.total_pnl_pct:+.2f}% |\n"
        table += f"| **Total P&L** | ${result.total_pnl:+,.2f} |\n"

        # Trade Stats
        table += f"| **Total Trades** | {result.total_trades} |\n"
        table += f"| **Winning Trades** | {result.winning_trades} |\n"
        table += f"| **Losing Trades** | {result.losing_trades} |\n"
        table += f"| **Win Rate** | {result.win_rate:.1f}% |\n"

        # P&L Distribution
        table += f"| **Average Win** | ${result.avg_win:.2f} |\n"
        table += f"| **Average Loss** | ${result.avg_loss:.2f} |\n"
        table += f"| **Largest Win** | ${result.largest_win:.2f} |\n"
        table += f"| **Largest Loss** | ${result.largest_loss:.2f} |\n"
        table += f"| **Profit Factor** | {result.profit_factor:.2f} |\n"

        # Risk Metrics
        table += f"| **Sharpe Ratio** | {result.sharpe_ratio:.2f} |\n"
        table += f"| **Sortino Ratio** | {result.sortino_ratio:.2f} |\n"
        table += f"| **Max Drawdown** | ${result.max_drawdown:,.2f} |\n"
        table += f"| **Max Drawdown %** | {result.max_drawdown_pct:.1f}% |\n"

        # Trade Duration
        table += f"| **Avg Trade Duration** | {result.avg_trade_duration_hours:.1f} hours |\n"

        return table

    def _build_trade_stats(self, result: BacktestResult) -> str:
        """Build trade statistics breakdown."""
        if not result.trades:
            return "*No trades executed during backtest.*\n"

        closed_trades = [t for t in result.trades if t.closed]

        stats = f"**Total Trades:** {len(result.trades)}  \n"
        stats += f"**Closed Trades:** {len(closed_trades)}  \n\n"

        if not closed_trades:
            return stats + "*No closed trades to analyze.*\n"

        # Winning vs Losing
        winning = [t for t in closed_trades if t.won]
        losing = [t for t in closed_trades if not t.won]

        stats += "### Win/Loss Breakdown\n\n"
        stats += f"- **Winning Trades:** {len(winning)} ({len(winning)/len(closed_trades)*100:.1f}%)\n"
        stats += f"- **Losing Trades:** {len(losing)} ({len(losing)/len(closed_trades)*100:.1f}%)\n"
        stats += f"- **Breakeven Trades:** {len(closed_trades) - len(winning) - len(losing)}\n\n"

        # Side breakdown
        buys = [t for t in closed_trades if t.side == "buy"]
        sells = [t for t in closed_trades if t.side == "sell"]

        stats += "### Side Distribution\n\n"
        stats += f"- **Buy Trades:** {len(buys)} ({len(buys)/len(closed_trades)*100:.1f}%)\n"
        stats += f"- **Sell Trades:** {len(sells)} ({len(sells)/len(closed_trades)*100:.1f}%)\n"

        return stats

    def _build_risk_analysis(self, result: BacktestResult) -> str:
        """Build risk analysis section."""
        analysis = ""

        # Sharpe Ratio interpretation
        if result.sharpe_ratio > 2.0:
            sharpe_verdict = "Excellent risk-adjusted returns"
        elif result.sharpe_ratio > 1.0:
            sharpe_verdict = "Good risk-adjusted returns"
        elif result.sharpe_ratio > 0.5:
            sharpe_verdict = "Moderate risk-adjusted returns"
        else:
            sharpe_verdict = "Poor risk-adjusted returns"

        analysis += f"**Sharpe Ratio:** {result.sharpe_ratio:.2f} - *{sharpe_verdict}*  \n\n"

        # Max Drawdown interpretation
        if result.max_drawdown_pct < 10:
            dd_verdict = "Low drawdown - conservative strategy"
        elif result.max_drawdown_pct < 20:
            dd_verdict = "Moderate drawdown - acceptable risk"
        elif result.max_drawdown_pct < 30:
            dd_verdict = "High drawdown - risky strategy"
        else:
            dd_verdict = "Very high drawdown - dangerous risk level"

        analysis += f"**Max Drawdown:** {result.max_drawdown_pct:.1f}% - *{dd_verdict}*  \n\n"

        # Profit Factor interpretation
        if result.profit_factor > 2.0:
            pf_verdict = "Strong profitability"
        elif result.profit_factor > 1.5:
            pf_verdict = "Good profitability"
        elif result.profit_factor > 1.0:
            pf_verdict = "Marginally profitable"
        else:
            pf_verdict = "Unprofitable"

        analysis += f"**Profit Factor:** {result.profit_factor:.2f} - *{pf_verdict}*\n\n"

        return analysis

    def _build_top_trades_table(self, result: BacktestResult) -> str:
        """Build table of top 10 trades by P&L."""
        closed_trades = [t for t in result.trades if t.closed]

        if not closed_trades:
            return "*No closed trades.*\n"

        # Sort by P&L
        sorted_trades = sorted(closed_trades, key=lambda t: t.realized_pnl, reverse=True)
        top_10 = sorted_trades[:10]

        table = "| # | Symbol | Side | Entry | Exit | P&L | P&L % | Duration |\n"
        table += "|---|--------|------|-------|------|-----|-------|----------|\n"

        for i, trade in enumerate(top_10, 1):
            duration_hours = 0
            if trade.close_timestamp and trade.timestamp:
                duration_hours = (trade.close_timestamp - trade.timestamp) / (1000 * 3600)

            pnl_emoji = "✅" if trade.won else "❌"

            table += (
                f"| {i} | {trade.symbol} | {trade.side} | "
                f"${trade.entry_price:.2f} | ${trade.exit_price:.2f} | "
                f"{pnl_emoji} ${trade.realized_pnl:+.2f} | {trade.pnl_pct:+.1f}% | "
                f"{duration_hours:.1f}h |\n"
            )

        return table

    def _markdown_to_html(self, markdown: str, result: BacktestResult) -> str:
        """Convert markdown to HTML with styling.

        Args:
            markdown: Markdown report content
            result: BacktestResult for additional HTML elements

        Returns:
            Complete HTML document
        """
        # Simple markdown to HTML conversion (basic implementation)
        # In production, use a library like markdown2 or mistune
        html_body = markdown

        # Headers
        html_body = html_body.replace("# ", "<h1>").replace("\n\n", "</h1>\n\n", 1)
        html_body = html_body.replace("## ", "<h2>").replace("\n\n", "</h2>\n\n")
        html_body = html_body.replace("### ", "<h3>").replace("\n\n", "</h3>\n\n")

        # Bold
        import re
        html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)

        # Italic
        html_body = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html_body)

        # Lists
        html_body = html_body.replace("\n- ", "\n<li>")
        html_body = html_body.replace("</li>\n<li>", "</li>\n<li>")

        # Code blocks
        html_body = re.sub(
            r"```json\n(.+?)\n```",
            r"<pre><code>\1</code></pre>",
            html_body,
            flags=re.DOTALL,
        )

        # Wrap in HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {result.symbol}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a202c;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2d3748;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #edf2f7;
            font-weight: 600;
        }}
        pre {{
            background: #2d3748;
            color: #f7fafc;
            padding: 20px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .positive {{
            color: #38a169;
        }}
        .negative {{
            color: #e53e3e;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>"""

        return html

    def generate_json_summary(self, result: BacktestResult) -> dict:
        """Generate JSON summary for programmatic access.

        Args:
            result: BacktestResult to summarize

        Returns:
            Dictionary with key metrics
        """
        return {
            "backtest_id": result.backtest_id,
            "symbol": result.symbol,
            "period": {
                "start": result.start_date.isoformat(),
                "end": result.end_date.isoformat(),
                "days": (result.end_date - result.start_date).days,
            },
            "returns": {
                "total_pnl": result.total_pnl,
                "total_pnl_pct": result.total_pnl_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
            },
            "risk": {
                "max_drawdown": result.max_drawdown,
                "max_drawdown_pct": result.max_drawdown_pct,
            },
            "trades": {
                "total": result.total_trades,
                "winning": result.winning_trades,
                "losing": result.losing_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
            },
        }
