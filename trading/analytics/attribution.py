"""Performance Attribution - Analyze agent and strategy contributions.

Analyzes which agents, market regimes, and decision factors contribute most
to trading performance. Helps identify what's working and what needs improvement.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from ..memory.trade_history import TradeRecord
from ..models.regime import MarketRegime


@dataclass
class AgentAttribution:
    """Performance attribution for a specific agent or factor."""

    name: str  # Agent name or factor
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    contribution_pct: float  # % of total P&L

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "contribution_pct": self.contribution_pct,
        }


@dataclass
class RegimeAttribution:
    """Performance by market regime."""

    regime: str
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    best_performing: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "regime": self.regime,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "best_performing": self.best_performing,
        }


class AttributionAnalyzer:
    """Performance attribution analyzer.

    Analyzes trading performance by:
    - Individual agent decisions (bull vs bear)
    - Market regime (bullish, bearish, sideways)
    - Decision confidence levels
    - Time of day / day of week
    - Trade duration

    Features:
    - Identify best-performing agents
    - Find optimal market regimes
    - Discover confidence sweet spots
    - Time-based performance patterns

    Example:
        >>> analyzer = AttributionAnalyzer()
        >>> regime_perf = analyzer.analyze_by_regime(trades)
        >>> for regime in regime_perf:
        ...     print(f"{regime.regime}: {regime.win_rate:.1f}% win rate")
    """

    def __init__(self):
        """Initialize attribution analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_by_regime(
        self,
        trades: List[TradeRecord],
    ) -> List[RegimeAttribution]:
        """Analyze performance by market regime.

        Args:
            trades: List of trades to analyze

        Returns:
            List of RegimeAttribution sorted by total P&L

        Example:
            >>> attributions = analyzer.analyze_by_regime(trades)
            >>> best = attributions[0]
            >>> print(f"Best regime: {best.regime} ({best.total_pnl:+.2f})")
        """
        closed_trades = [t for t in trades if t.closed and t.market_regime]

        if not closed_trades:
            return []

        # Group by regime
        by_regime: Dict[str, List[TradeRecord]] = defaultdict(list)
        for trade in closed_trades:
            by_regime[trade.market_regime].append(trade)

        # Calculate attribution for each regime
        attributions = []
        total_pnl_all = sum(t.realized_pnl for t in closed_trades)

        for regime, regime_trades in by_regime.items():
            winning = [t for t in regime_trades if t.won]
            total_pnl = sum(t.realized_pnl for t in regime_trades)

            # Calculate Sharpe ratio
            returns = [t.pnl_pct / 100 for t in regime_trades]
            sharpe = self._calculate_sharpe(returns)

            attribution = RegimeAttribution(
                regime=regime,
                total_trades=len(regime_trades),
                win_rate=(len(winning) / len(regime_trades) * 100),
                total_pnl=total_pnl,
                sharpe_ratio=sharpe,
            )

            attributions.append(attribution)

        # Sort by total P&L and mark best
        attributions.sort(key=lambda a: a.total_pnl, reverse=True)
        if attributions:
            attributions[0].best_performing = True

        return attributions

    def analyze_by_agent_confidence(
        self,
        trades: List[TradeRecord],
        bins: List[float] = [0.0, 0.5, 0.7, 0.85, 1.0],
    ) -> List[AgentAttribution]:
        """Analyze performance by decision confidence levels.

        Args:
            trades: List of trades
            bins: Confidence level bins (e.g., [0.0, 0.5, 0.7, 1.0])

        Returns:
            List of AgentAttribution for each confidence bin

        Example:
            >>> by_confidence = analyzer.analyze_by_agent_confidence(trades)
            >>> for attr in by_confidence:
            ...     print(f"{attr.name}: {attr.win_rate:.1f}% win rate")
        """
        closed_trades = [t for t in trades if t.closed and t.decision_confidence is not None]

        if not closed_trades:
            return []

        # Create bins
        bin_labels = []
        for i in range(len(bins) - 1):
            bin_labels.append(f"{bins[i]:.0%}-{bins[i+1]:.0%}")

        # Group by confidence bin
        by_bin: Dict[str, List[TradeRecord]] = {label: [] for label in bin_labels}

        for trade in closed_trades:
            confidence = trade.decision_confidence or 0.0

            # Find bin
            for i in range(len(bins) - 1):
                if bins[i] <= confidence < bins[i + 1]:
                    by_bin[bin_labels[i]].append(trade)
                    break

        # Calculate attribution
        attributions = []
        total_pnl_all = sum(t.realized_pnl for t in closed_trades)

        for bin_label, bin_trades in by_bin.items():
            if not bin_trades:
                continue

            winning = [t for t in bin_trades if t.won]
            total_pnl = sum(t.realized_pnl for t in bin_trades)

            attribution = AgentAttribution(
                name=f"Confidence {bin_label}",
                total_trades=len(bin_trades),
                winning_trades=len(winning),
                win_rate=(len(winning) / len(bin_trades) * 100),
                total_pnl=total_pnl,
                avg_pnl=total_pnl / len(bin_trades),
                contribution_pct=(total_pnl / total_pnl_all * 100) if total_pnl_all != 0 else 0.0,
            )

            attributions.append(attribution)

        return attributions

    def analyze_bull_vs_bear(
        self,
        trades: List[TradeRecord],
    ) -> Dict[str, AgentAttribution]:
        """Compare bull agent vs bear agent performance.

        Args:
            trades: List of trades

        Returns:
            Dictionary with "bull" and "bear" attributions

        Example:
            >>> bull_bear = analyzer.analyze_bull_vs_bear(trades)
            >>> print(f"Bull: {bull_bear['bull'].win_rate:.1f}%")
            >>> print(f"Bear: {bull_bear['bear'].win_rate:.1f}%")
        """
        closed_trades = [
            t for t in trades
            if t.closed and t.bull_confidence is not None and t.bear_confidence is not None
        ]

        if not closed_trades:
            return {}

        # Classify trades by which agent was more confident
        bull_trades = [t for t in closed_trades if (t.bull_confidence or 0) > (t.bear_confidence or 0)]
        bear_trades = [t for t in closed_trades if (t.bear_confidence or 0) > (t.bull_confidence or 0)]

        total_pnl_all = sum(t.realized_pnl for t in closed_trades)

        # Bull attribution
        bull_winning = [t for t in bull_trades if t.won]
        bull_pnl = sum(t.realized_pnl for t in bull_trades)

        bull_attr = AgentAttribution(
            name="Bull Agent",
            total_trades=len(bull_trades),
            winning_trades=len(bull_winning),
            win_rate=(len(bull_winning) / len(bull_trades) * 100) if bull_trades else 0.0,
            total_pnl=bull_pnl,
            avg_pnl=bull_pnl / len(bull_trades) if bull_trades else 0.0,
            contribution_pct=(bull_pnl / total_pnl_all * 100) if total_pnl_all != 0 else 0.0,
        )

        # Bear attribution
        bear_winning = [t for t in bear_trades if t.won]
        bear_pnl = sum(t.realized_pnl for t in bear_trades)

        bear_attr = AgentAttribution(
            name="Bear Agent",
            total_trades=len(bear_trades),
            winning_trades=len(bear_winning),
            win_rate=(len(bear_winning) / len(bear_trades) * 100) if bear_trades else 0.0,
            total_pnl=bear_pnl,
            avg_pnl=bear_pnl / len(bear_trades) if bear_trades else 0.0,
            contribution_pct=(bear_pnl / total_pnl_all * 100) if total_pnl_all != 0 else 0.0,
        )

        return {
            "bull": bull_attr,
            "bear": bear_attr,
        }

    def analyze_by_trade_duration(
        self,
        trades: List[TradeRecord],
        bins_hours: List[float] = [0, 1, 4, 12, 24, 72],
    ) -> List[AgentAttribution]:
        """Analyze performance by trade duration (holding time).

        Args:
            trades: List of trades
            bins_hours: Duration bins in hours

        Returns:
            List of AgentAttribution for each duration bin

        Example:
            >>> by_duration = analyzer.analyze_by_trade_duration(trades)
            >>> for attr in by_duration:
            ...     print(f"{attr.name}: ${attr.avg_pnl:.2f} avg P&L")
        """
        closed_trades = [
            t for t in trades
            if t.closed and t.timestamp and t.close_timestamp
        ]

        if not closed_trades:
            return []

        # Create bin labels
        bin_labels = []
        for i in range(len(bins_hours) - 1):
            bin_labels.append(f"{bins_hours[i]:.0f}-{bins_hours[i+1]:.0f}h")
        bin_labels.append(f"{bins_hours[-1]:.0f}h+")

        # Group by duration
        by_duration: Dict[str, List[TradeRecord]] = {label: [] for label in bin_labels}

        for trade in closed_trades:
            duration_ms = (trade.close_timestamp or 0) - trade.timestamp
            duration_hours = duration_ms / (1000 * 3600)

            # Find bin
            for i in range(len(bins_hours) - 1):
                if bins_hours[i] <= duration_hours < bins_hours[i + 1]:
                    by_duration[bin_labels[i]].append(trade)
                    break
            else:
                # Longer than all bins
                by_duration[bin_labels[-1]].append(trade)

        # Calculate attribution
        attributions = []
        total_pnl_all = sum(t.realized_pnl for t in closed_trades)

        for duration_label, duration_trades in by_duration.items():
            if not duration_trades:
                continue

            winning = [t for t in duration_trades if t.won]
            total_pnl = sum(t.realized_pnl for t in duration_trades)

            attribution = AgentAttribution(
                name=f"Duration {duration_label}",
                total_trades=len(duration_trades),
                winning_trades=len(winning),
                win_rate=(len(winning) / len(duration_trades) * 100),
                total_pnl=total_pnl,
                avg_pnl=total_pnl / len(duration_trades),
                contribution_pct=(total_pnl / total_pnl_all * 100) if total_pnl_all != 0 else 0.0,
            )

            attributions.append(attribution)

        return attributions

    def get_best_performing_patterns(
        self,
        trades: List[TradeRecord],
    ) -> Dict[str, Any]:
        """Identify best-performing patterns across all dimensions.

        Args:
            trades: List of trades

        Returns:
            Dictionary with best patterns by regime, confidence, etc.

        Example:
            >>> patterns = analyzer.get_best_performing_patterns(trades)
            >>> print(f"Best regime: {patterns['best_regime']['regime']}")
            >>> print(f"Optimal confidence: {patterns['best_confidence']['name']}")
        """
        # Analyze all dimensions
        by_regime = self.analyze_by_regime(trades)
        by_confidence = self.analyze_by_agent_confidence(trades)
        bull_bear = self.analyze_bull_vs_bear(trades)

        result = {}

        # Best regime
        if by_regime:
            best_regime = max(by_regime, key=lambda r: r.total_pnl)
            result["best_regime"] = best_regime.to_dict()

        # Best confidence level
        if by_confidence:
            best_confidence = max(by_confidence, key=lambda c: c.win_rate)
            result["best_confidence"] = best_confidence.to_dict()

        # Bull vs Bear winner
        if bull_bear:
            if bull_bear["bull"].total_pnl > bull_bear["bear"].total_pnl:
                result["best_agent"] = bull_bear["bull"].to_dict()
            else:
                result["best_agent"] = bull_bear["bear"].to_dict()

        return result

    def generate_attribution_report(
        self,
        trades: List[TradeRecord],
    ) -> str:
        """Generate comprehensive attribution report.

        Args:
            trades: List of trades

        Returns:
            Formatted attribution report string

        Example:
            >>> print(analyzer.generate_attribution_report(trades))
        """
        report = "=== Performance Attribution Report ===\n\n"

        # By regime
        by_regime = self.analyze_by_regime(trades)
        if by_regime:
            report += "📊 By Market Regime:\n"
            for regime_attr in by_regime:
                report += f"  {regime_attr.regime}:\n"
                report += f"    Trades: {regime_attr.total_trades}\n"
                report += f"    Win Rate: {regime_attr.win_rate:.1f}%\n"
                report += f"    P&L: ${regime_attr.total_pnl:+,.2f}\n"
                report += f"    Sharpe: {regime_attr.sharpe_ratio:.2f}\n"
            report += "\n"

        # Bull vs Bear
        bull_bear = self.analyze_bull_vs_bear(trades)
        if bull_bear:
            report += "🐂 vs 🐻 Bull vs Bear Agents:\n"
            for agent_name, attr in bull_bear.items():
                report += f"  {attr.name}:\n"
                report += f"    Trades: {attr.total_trades}\n"
                report += f"    Win Rate: {attr.win_rate:.1f}%\n"
                report += f"    P&L: ${attr.total_pnl:+,.2f}\n"
                report += f"    Contribution: {attr.contribution_pct:.1f}%\n"
            report += "\n"

        # By confidence
        by_confidence = self.analyze_by_agent_confidence(trades)
        if by_confidence:
            report += "🎯 By Decision Confidence:\n"
            for conf_attr in by_confidence:
                report += f"  {conf_attr.name}:\n"
                report += f"    Trades: {conf_attr.total_trades}\n"
                report += f"    Win Rate: {conf_attr.win_rate:.1f}%\n"
                report += f"    Avg P&L: ${conf_attr.avg_pnl:+.2f}\n"
            report += "\n"

        # Best patterns
        patterns = self.get_best_performing_patterns(trades)
        if patterns:
            report += "✨ Best Performing Patterns:\n"
            if "best_regime" in patterns:
                report += f"  Regime: {patterns['best_regime']['regime']}\n"
            if "best_confidence" in patterns:
                report += f"  Confidence: {patterns['best_confidence']['name']}\n"
            if "best_agent" in patterns:
                report += f"  Agent: {patterns['best_agent']['name']}\n"

        return report

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if not returns or len(returns) < 2:
            return 0.0

        import math
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        sharpe = (mean_return / std_dev) * math.sqrt(252)
        return sharpe
