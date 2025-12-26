"""Pattern detection and learning from trade history.

Analyzes trade performance by market regime, agent behavior, and other factors.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from .trade_history import TradeJournal, TradeRecord
from ..models.regime import MarketRegime


@dataclass
class TradingInsight:
    """A discovered pattern or insight from trade analysis."""

    category: str  # "regime", "agent", "symbol", "timing"
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any]
    sample_size: int

    def to_markdown(self) -> str:
        """Convert insight to markdown format."""
        confidence_str = f"{self.confidence * 100:.0f}%"

        md = f"### {self.title} (Confidence: {confidence_str})\n\n"
        md += f"**Category:** {self.category}  \n"
        md += f"**Sample Size:** {self.sample_size} trades  \n\n"
        md += f"{self.description}\n\n"

        if self.supporting_data:
            md += "**Supporting Data:**\n"
            for key, value in self.supporting_data.items():
                if isinstance(value, float):
                    md += f"- {key}: {value:.2f}\n"
                else:
                    md += f"- {key}: {value}\n"
            md += "\n"

        return md


class PatternDetector:
    """Detects patterns and generates insights from trade history."""

    def __init__(self, journal: TradeJournal):
        """Initialize pattern detector.

        Args:
            journal: Trade journal to analyze
        """
        self.journal = journal
        self.logger = logging.getLogger(__name__)

    def analyze_regime_performance(self) -> List[TradingInsight]:
        """Analyze how trading performance varies by market regime.

        Returns:
            List of insights about regime-based performance
        """
        insights = []
        trades = self.journal.get_all_trades()
        closed_trades = [t for t in trades if t.closed]

        if len(closed_trades) < 10:
            return []  # Need minimum sample size

        # Group trades by regime
        regime_trades = defaultdict(list)
        for trade in closed_trades:
            if trade.market_regime:
                regime_trades[trade.market_regime].append(trade)

        # Analyze each regime
        for regime, regime_trade_list in regime_trades.items():
            if len(regime_trade_list) < 5:
                continue  # Need minimum sample per regime

            winning = [t for t in regime_trade_list if t.won]
            win_rate = len(winning) / len(regime_trade_list) * 100
            avg_pnl = sum(t.realized_pnl for t in regime_trade_list) / len(
                regime_trade_list
            )

            # Determine if this regime is favorable
            overall_win_rate = (
                len([t for t in closed_trades if t.won]) / len(closed_trades) * 100
            )

            if win_rate > overall_win_rate + 10:  # Significantly better
                insights.append(
                    TradingInsight(
                        category="regime",
                        title=f"Strong Performance in {regime.upper()} Markets",
                        description=f"Trading strategy shows {win_rate:.1f}% win rate in {regime} market conditions, "
                        f"compared to overall {overall_win_rate:.1f}% win rate. "
                        f"Average P&L per trade: ${avg_pnl:.2f}. "
                        f"Consider increasing position sizes or trading frequency in {regime} markets.",
                        confidence=min(
                            len(regime_trade_list) / 50, 0.95
                        ),  # More trades = higher confidence
                        supporting_data={
                            "regime_win_rate": win_rate,
                            "overall_win_rate": overall_win_rate,
                            "avg_pnl": avg_pnl,
                            "total_trades": len(regime_trade_list),
                        },
                        sample_size=len(regime_trade_list),
                    )
                )
            elif win_rate < overall_win_rate - 10:  # Significantly worse
                insights.append(
                    TradingInsight(
                        category="regime",
                        title=f"Weak Performance in {regime.upper()} Markets",
                        description=f"Trading strategy shows only {win_rate:.1f}% win rate in {regime} market conditions, "
                        f"below overall {overall_win_rate:.1f}% win rate. "
                        f"Average P&L per trade: ${avg_pnl:.2f}. "
                        f"Consider reducing exposure or avoiding trades in {regime} markets.",
                        confidence=min(len(regime_trade_list) / 50, 0.95),
                        supporting_data={
                            "regime_win_rate": win_rate,
                            "overall_win_rate": overall_win_rate,
                            "avg_pnl": avg_pnl,
                            "total_trades": len(regime_trade_list),
                        },
                        sample_size=len(regime_trade_list),
                    )
                )

        return insights

    def analyze_agent_behavior(self) -> List[TradingInsight]:
        """Analyze Bull vs Bear agent performance.

        Returns:
            List of insights about agent decision quality
        """
        insights = []
        trades = self.journal.get_all_trades()
        closed_trades = [t for t in trades if t.closed and t.agent_votes]

        if len(closed_trades) < 10:
            return []

        # Analyze Bull agent accuracy
        bull_correct = 0
        bull_total = 0

        for trade in closed_trades:
            if "bull" in trade.agent_votes:
                bull_total += 1
                bull_conf = trade.agent_votes["bull"].get("confidence", 0)
                bear_conf = trade.agent_votes["bear"].get("confidence", 0)

                # Bull was dominant and trade won (or Bear was dominant and trade lost)
                if (bull_conf > bear_conf and trade.won) or (
                    bear_conf > bull_conf and not trade.won
                ):
                    bull_correct += 1

        if bull_total > 0:
            bull_accuracy = bull_correct / bull_total * 100

            if bull_accuracy > 60:  # Bull agent is doing well
                insights.append(
                    TradingInsight(
                        category="agent",
                        title="Bull Agent Shows Strong Predictive Accuracy",
                        description=f"The Bull agent demonstrates {bull_accuracy:.1f}% accuracy in predicting profitable trades. "
                        f"Consider increasing weight of Bull agent votes in decision threshold.",
                        confidence=min(bull_total / 50, 0.9),
                        supporting_data={
                            "bull_accuracy": bull_accuracy,
                            "correct_predictions": bull_correct,
                            "total_predictions": bull_total,
                        },
                        sample_size=bull_total,
                    )
                )

        return insights

    def analyze_confidence_correlation(self) -> List[TradingInsight]:
        """Analyze correlation between decision confidence and trade success.

        Returns:
            List of insights about confidence thresholds
        """
        insights = []
        trades = self.journal.get_all_trades()
        closed_trades = [t for t in trades if t.closed and t.decision_confidence]

        if len(closed_trades) < 10:
            return []

        # Split trades by confidence level
        high_conf_trades = [t for t in closed_trades if t.decision_confidence > 0.7]
        low_conf_trades = [t for t in closed_trades if t.decision_confidence < 0.5]

        if high_conf_trades and low_conf_trades:
            high_win_rate = (
                len([t for t in high_conf_trades if t.won]) / len(high_conf_trades) * 100
            )
            low_win_rate = (
                len([t for t in low_conf_trades if t.won]) / len(low_conf_trades) * 100
            )

            if high_win_rate > low_win_rate + 15:
                insights.append(
                    TradingInsight(
                        category="confidence",
                        title="High Confidence Trades Significantly Outperform",
                        description=f"Trades with >70% decision confidence show {high_win_rate:.1f}% win rate, "
                        f"compared to {low_win_rate:.1f}% for low confidence (<50%) trades. "
                        f"Recommend raising decision threshold to filter out low confidence signals.",
                        confidence=min(len(high_conf_trades) / 30, 0.85),
                        supporting_data={
                            "high_conf_win_rate": high_win_rate,
                            "low_conf_win_rate": low_win_rate,
                            "high_conf_trades": len(high_conf_trades),
                            "low_conf_trades": len(low_conf_trades),
                        },
                        sample_size=len(high_conf_trades) + len(low_conf_trades),
                    )
                )

        return insights

    def analyze_symbol_performance(self) -> List[TradingInsight]:
        """Analyze performance by trading symbol.

        Returns:
            List of insights about symbol-specific patterns
        """
        insights = []
        trades = self.journal.get_all_trades()
        closed_trades = [t for t in trades if t.closed]

        if len(closed_trades) < 10:
            return []

        # Group by symbol
        symbol_trades = defaultdict(list)
        for trade in closed_trades:
            symbol_trades[trade.symbol].append(trade)

        # Analyze each symbol with enough data
        for symbol, symbol_trade_list in symbol_trades.items():
            if len(symbol_trade_list) < 5:
                continue

            winning = [t for t in symbol_trade_list if t.won]
            win_rate = len(winning) / len(symbol_trade_list) * 100
            avg_pnl = sum(t.realized_pnl for t in symbol_trade_list) / len(
                symbol_trade_list
            )

            if win_rate > 70 and avg_pnl > 0:
                insights.append(
                    TradingInsight(
                        category="symbol",
                        title=f"{symbol} Shows Consistent Profitability",
                        description=f"Trading {symbol} has yielded {win_rate:.1f}% win rate with average P&L of ${avg_pnl:.2f} per trade. "
                        f"Consider focusing more trading activity on this symbol.",
                        confidence=min(len(symbol_trade_list) / 20, 0.8),
                        supporting_data={
                            "win_rate": win_rate,
                            "avg_pnl": avg_pnl,
                            "total_trades": len(symbol_trade_list),
                        },
                        sample_size=len(symbol_trade_list),
                    )
                )

        return insights

    def generate_all_insights(self) -> List[TradingInsight]:
        """Run all analysis and generate comprehensive insights.

        Returns:
            List of all discovered insights
        """
        all_insights = []

        all_insights.extend(self.analyze_regime_performance())
        all_insights.extend(self.analyze_agent_behavior())
        all_insights.extend(self.analyze_confidence_correlation())
        all_insights.extend(self.analyze_symbol_performance())

        # Sort by confidence
        all_insights.sort(key=lambda x: x.confidence, reverse=True)

        return all_insights

    def save_insights_to_markdown(self, output_path: Path):
        """Generate and save insights as markdown file.

        Args:
            output_path: Path to save markdown file
        """
        insights = self.generate_all_insights()

        if not insights:
            self.logger.info("No insights generated (insufficient data)")
            return

        # Generate markdown
        md_content = "# Trading Insights\n\n"
        md_content += f"*Generated from {len(self.journal.get_all_trades())} total trades*\n\n"
        md_content += "---\n\n"

        for insight in insights:
            md_content += insight.to_markdown()
            md_content += "---\n\n"

        # Save to file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(md_content)

            self.logger.info(f"Saved {len(insights)} insights to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save insights: {e}")
