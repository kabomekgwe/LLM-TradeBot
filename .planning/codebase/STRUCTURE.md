# Codebase Structure

**Analysis Date:** 2025-12-26

## Directory Layout

```
LLM-TradeBot/
├── trading/                 # Main Python package
│   ├── agents/             # 8-agent decision system
│   ├── providers/          # Exchange integrations (6 platforms)
│   ├── models/             # Data models (OHLCV, Position, Order, etc.)
│   ├── memory/             # Trade history & pattern learning
│   ├── ml/                 # Machine learning models
│   ├── analytics/          # Performance metrics
│   ├── sentiment/          # News/Twitter/OnChain sentiment
│   ├── notifications/      # Discord/Slack/Telegram/Email alerts
│   ├── orders/             # Advanced order types
│   ├── portfolio/          # Portfolio management
│   ├── risk/               # Risk calculations
│   ├── backtest/           # Historical simulation
│   ├── web/                # FastAPI dashboard
│   ├── tests/              # Test suite
│   ├── cli.py              # CLI entry point
│   ├── manager.py          # Trading orchestrator
│   ├── config.py           # Configuration
│   └── state.py            # State persistence
├── config/                  # Configuration templates
│   └── .env.example        # Environment variable template
├── examples/                # Usage examples
├── docs/                    # Documentation
├── .planning/               # GSD project planning
├── requirements.txt         # Python dependencies
└── README.md               # Main documentation
```

## Directory Purposes

**trading/agents/**
- Purpose: 8-agent adversarial trading system
- Contains: Agent implementations (*.py files)
- Key files:
  - `base_agent.py` - Abstract base class
  - `data_sync.py` - Market data fetching
  - `quant_analyst.py` - Technical indicators
  - `predict.py` - ML predictions
  - `bull.py` - Bullish analysis
  - `bear.py` - Bearish analysis
  - `decision_core.py` - Weighted voting
  - `risk_audit.py` - Safety validation
  - `execution.py` - Order placement

**trading/providers/**
- Purpose: Unified exchange interface
- Contains: Provider implementations, factory, base class
- Key files:
  - `base.py` - Abstract provider interface
  - `factory.py` - Provider creation
  - `binance_futures.py`, `binance_spot.py`, `kraken.py`, `coinbase.py`, `alpaca.py`, `paper.py` - Exchange implementations

**trading/models/**
- Purpose: Type-safe data structures
- Contains: Dataclass definitions
- Key files:
  - `market_data.py` - OHLCV, Ticker, OrderBook, Balance
  - `positions.py` - Order, Position, Trade
  - `signals.py` - TradingSignal classes
  - `decision.py` - TradingDecision, Action enums
  - `regime.py` - MarketRegime detection

**trading/memory/**
- Purpose: Learning and trade history
- Contains: Trade journal, pattern detection
- Key files:
  - `trade_history.py` - TradeJournal, TradeRecord
  - `patterns.py` - PatternDetector, Insight
  - `graphiti_trading.py` - Optional Graphiti integration

**trading/ml/**
- Purpose: Machine learning predictions
- Contains: Models, ensemble, training
- Subdirectories: `models/` (lightgbm, xgboost, lstm)
- Key files:
  - `ensemble.py` - Model ensemble
  - `feature_engineering.py` - Feature creation
  - `training.py` - Training pipeline

**trading/sentiment/**
- Purpose: Sentiment aggregation
- Contains: News, Twitter, OnChain integrations
- Key files:
  - `aggregator.py` - Combines all sources
  - `news.py` - News sentiment
  - `twitter.py` - Twitter/X sentiment
  - `onchain.py` - Blockchain metrics
  - `fear_greed.py` - Fear & Greed Index

**trading/notifications/**
- Purpose: Alert delivery
- Contains: Multi-channel notifiers
- Key files:
  - `manager.py` - Notification orchestration
  - `discord.py`, `slack.py`, `telegram.py`, `email.py` - Channel implementations

**trading/web/**
- Purpose: Real-time dashboard
- Contains: FastAPI server, WebSocket handler
- Key files:
  - `server.py` - DashboardServer
  - `websocket.py` - WebSocket manager

**trading/tests/**
- Purpose: Test suite
- Contains: pytest tests
- Key files:
  - `test_providers.py` - Provider interface tests
  - *Additional test files needed (low coverage)*

## Key File Locations

**Entry Points:**
- `trading/cli.py` - CLI interface (JSON output for IPC)
- `trading/manager.py` - TradingManager orchestrator
- `trading/web/server.py` - FastAPI dashboard

**Configuration:**
- `trading/config.py` - TradingConfig dataclass
- `config/.env.example` - Environment template
- `.env` - Active configuration (gitignored)

**Core Logic:**
- `trading/agents/` - 8-agent decision pipeline
- `trading/providers/` - Exchange integrations
- `trading/models/` - Data structures

**Testing:**
- `trading/tests/test_providers.py` - Provider tests
- *Missing: Agent tests, integration tests*

**Documentation:**
- `README.md` - Main documentation
- `docs/` - Additional documentation
- `.planning/` - GSD project planning

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `base_agent.py`, `quant_analyst.py`)
- Directories: `snake_case/` (e.g., `agents/`, `providers/`)
- Config/docs: `PascalCase` or `UPPERCASE` (e.g., `README.md`, `.env.example`)

**Directories:**
- Lowercase with underscores: `trading/`, `agents/`, `ml/`
- Plural for collections: `agents/`, `providers/`, `models/`

**Special Patterns:**
- `__init__.py` - Package initialization
- `test_*.py` - pytest test files
- `base*.py` - Abstract base classes

## Where to Add New Code

**New Agent:**
- Primary code: `trading/agents/new_agent.py`
- Base class: Extend `BaseAgent` from `trading/agents/base_agent.py`
- Tests: `trading/tests/test_agents.py`

**New Exchange Provider:**
- Implementation: `trading/providers/exchange_name.py`
- Base class: Extend `BaseExchangeProvider` from `trading/providers/base.py`
- Registration: Add to `factory.py` provider map
- Tests: Add to `trading/tests/test_providers.py`

**New Notification Channel:**
- Implementation: `trading/notifications/channel_name.py`
- Integration: Update `trading/notifications/manager.py`
- Config: Add env vars to `trading/config.py`

**New ML Model:**
- Implementation: `trading/ml/models/model_name.py`
- Integration: Update `trading/ml/ensemble.py`
- Training: Add to `trading/ml/training.py`

**Utilities:**
- Not applicable (no utils/ directory exists)

## Special Directories

**.planning/**
- Purpose: GSD project planning
- Source: Created by /gsd commands
- Committed: Yes

**.claude/**
- Purpose: Claude Code workspace
- Source: User configuration
- Committed: No (user-specific)

**logs/**
- Purpose: Application logs
- Source: Generated at runtime
- Committed: No (should be gitignored - **missing .gitignore**)

---

*Structure analysis: 2025-12-26*
*Update when directory structure changes*
