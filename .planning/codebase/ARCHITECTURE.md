# Architecture

**Analysis Date:** 2025-12-26

## Pattern Overview

**Overall:** Layered Multi-Agent System with Adversarial Decision Framework

**Key Characteristics:**
- 8-agent orchestrator pattern with central coordinator
- Adversarial analysis (Bull vs Bear agents)
- Event-driven async architecture
- File-based state persistence (JSON)
- Multi-provider abstraction (6 exchanges via unified interface)

## Layers

**Entry Point Layer:**
- Purpose: User interface and system initialization
- Contains: CLI commands, TradingManager orchestrator
- Implementation: `trading/cli.py`, `trading/manager.py`
- Used by: Users, external UI (Electron app)

**Agent Layer (8-agent system):**
- Purpose: Autonomous trading decision pipeline
- Contains: Data fetching, analysis, prediction, decision, risk, execution agents
- Implementation: `trading/agents/` directory (8 agents)
- Depends on: Provider layer, ML models, sentiment services
- Flow: DataSync → QuantAnalyst → Predict → Bull/Bear → DecisionCore → RiskAudit → Execution

**Provider Layer:**
- Purpose: Unified interface to 6 different exchanges
- Contains: Base abstract provider, 6 exchange implementations, factory pattern
- Implementation: `trading/providers/` directory
- Depends on: CCXT library, exchange APIs
- Used by: Agent layer

**Data Model Layer:**
- Purpose: Type-safe data structures
- Contains: Market data (OHLCV, Ticker, OrderBook), positions (Order, Position, Trade), decisions (TradingDecision, MarketRegime)
- Implementation: `trading/models/` directory
- Pattern: Python dataclasses with validation

**Support Layers:**
- Memory: Trade journaling, pattern detection - `trading/memory/`
- ML: Model training, ensemble predictions - `trading/ml/`
- Analytics: Performance metrics, risk calculations - `trading/analytics/`
- Sentiment: News/Twitter/OnChain aggregation - `trading/sentiment/`
- Notifications: Multi-channel alerts - `trading/notifications/`
- Backtest: Historical simulation - `trading/backtest/`
- Web: Dashboard API - `trading/web/`

## Data Flow

**Primary Trading Loop:**

1. **CLI/Manager** invokes `run_trading_loop(symbol)`
2. **DataSyncAgent** → Fetches multi-timeframe OHLCV (5m, 15m, 1h), ticker, orderbook
3. **QuantAnalystAgent** → Calculates technical indicators (RSI, MACD, Bollinger Bands) *[TODO: incomplete]*
4. **PredictAgent** → ML forecasting via LightGBM *[TODO: incomplete]*
5. **BullAgent + BearAgent** (parallel) → Adversarial analysis with confidence scores
6. **DecisionCoreAgent** → Weighted voting with regime awareness
7. **RiskAuditAgent** → Safety validation (can veto decisions)
8. **ExecutionEngine** → Places orders, manages positions
9. **TradeJournal** → Records trade with full context
10. **State Persistence** → Updates `.trading_state.json`

**Context Dictionary Pattern:**
- Each agent receives `context: dict[str, Any]`
- Adds outputs to context
- Passes enriched context to next agent
- Final context contains complete decision chain

**State Management:**
- File-based: All state in `.trading_state.json`
- No persistent database
- Each trading session is independent

## Key Abstractions

**BaseAgent:**
- Purpose: Template for all 8 agents
- Implementation: `trading/agents/base_agent.py`
- Pattern: Abstract base class with `execute(context)` method

**BaseExchangeProvider:**
- Purpose: Unified interface for exchanges
- Implementation: `trading/providers/base.py`
- Pattern: Strategy pattern + Factory pattern
- Implementations: Binance Futures/Spot, Kraken, Coinbase, Alpaca, Paper

**TradingManager:**
- Purpose: Central orchestrator
- Implementation: `trading/manager.py`
- Pattern: Coordinator pattern
- Responsibilities: Agent lifecycle, error handling, state management

**TradingConfig:**
- Purpose: Configuration management
- Implementation: `trading/config.py`
- Pattern: Configuration object from environment

**TradeJournal:**
- Purpose: Trade history and learning
- Implementation: `trading/memory/trade_history.py`
- Pattern: Repository pattern (file-based)

## Entry Points

**CLI Entry:**
- Location: `trading/cli.py`
- Triggers: User runs CLI commands
- Commands: `status`, `positions`, `run`, `history`, `cancel`, `close`, `insights`
- Responsibilities: JSON-based IPC for Electron UI

**Manager Entry:**
- Location: `trading/manager.py`
- Triggers: Programmatic invocation
- Method: `await manager.run_trading_loop(symbol)`
- Responsibilities: Orchestrate 8-agent pipeline

**Web Dashboard:**
- Location: `trading/web/server.py`
- Triggers: HTTP/WebSocket requests
- Port: 5173 (configurable)
- Responsibilities: Real-time monitoring, REST API

## Error Handling

**Strategy:** Broad exception catching with logging (needs improvement)

**Patterns:**
- Try/catch at agent level with context preservation
- Generic `Exception` catches throughout (50+ occurrences - technical debt)
- Circuit breaker in RiskAudit agent (veto power)

## Cross-Cutting Concerns

**Logging:**
- Python logging module
- File output: `logs/trading.log`
- Level: Configurable via `LOG_LEVEL` env var
- Mixed with print() statements (79 occurrences - technical debt)

**Validation:**
- Pydantic v2 for configuration
- Dataclass validation in models
- Risk limit validation in RiskAudit agent

**Configuration:**
- Environment variables via python-dotenv
- TradingConfig dataclass loads and validates on startup

---

*Architecture analysis: 2025-12-26*
*Update when major patterns change*
