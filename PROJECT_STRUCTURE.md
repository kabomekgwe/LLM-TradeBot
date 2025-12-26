# LLM-TradeBot Project Structure

## Overview

```
/Users/kabo/Desktop/LLM-TradeBot/
├── README.md                     # Main documentation
├── requirements.txt              # Python dependencies
├── setup.sh                      # Automated setup script
├── .env.example                  # Configuration template (in config/)
│
├── trading/                      # Main package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration management
│   ├── state.py                 # State persistence
│   ├── manager.py               # Trading manager (orchestrator)
│   ├── cli.py                   # Command-line interface
│   │
│   ├── providers/               # Exchange integrations
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract provider interface
│   │   ├── factory.py          # Provider factory
│   │   ├── normalizer.py       # Symbol/timeframe conversion
│   │   ├── binance_futures.py  # Binance Futures
│   │   ├── binance_spot.py     # Binance Spot
│   │   ├── kraken.py           # Kraken exchange
│   │   ├── coinbase.py         # Coinbase Advanced Trade
│   │   ├── alpaca.py           # Alpaca stocks
│   │   └── paper.py            # Paper trading simulator
│   │
│   ├── agents/                  # 8-Agent System
│   │   ├── __init__.py
│   │   ├── base_agent.py       # Base agent interface
│   │   ├── data_sync.py        # Market data fetcher
│   │   ├── quant_analyst.py    # Technical analysis
│   │   ├── predict.py          # ML predictions
│   │   ├── bull.py             # Bullish analysis
│   │   ├── bear.py             # Bearish analysis (adversarial)
│   │   ├── decision_core.py    # Weighted voting
│   │   ├── risk_audit.py       # Risk veto
│   │   └── execution.py        # Order execution
│   │
│   ├── models/                  # Data Models
│   │   ├── __init__.py
│   │   ├── market_data.py      # OHLCV, Ticker, OrderBook, Balance
│   │   ├── positions.py        # Order, Position, Trade
│   │   ├── signals.py          # Trading signals
│   │   ├── decision.py         # Trading decisions
│   │   └── regime.py           # Market regime detection
│   │
│   ├── memory/                  # Learning & Memory
│   │   ├── __init__.py
│   │   ├── trade_history.py    # Trade journaling
│   │   ├── patterns.py         # Pattern detection
│   │   ├── graphiti_trading.py # Optional Graphiti integration
│   │   └── USAGE_EXAMPLE.md    # Memory system documentation
│   │
│   ├── risk/                    # Risk Management
│   │   ├── __init__.py
│   │   ├── calculator.py       # Risk calculations
│   │   ├── limits.py           # Position/drawdown limits
│   │   └── veto_rules.py       # Veto conditions
│   │
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── indicators.py       # TA-Lib wrappers
│   │   ├── regime_detector.py  # Regime classification
│   │   └── ml_model.py         # ML model management
│   │
│   └── tests/                   # Test Suite
│       ├── __init__.py
│       ├── test_providers.py   # Provider tests
│       ├── test_agents.py      # Agent tests
│       └── test_memory.py      # Memory system tests
│
├── config/                      # Configuration
│   └── .env.example            # Environment template
│
├── examples/                    # Usage Examples
│   ├── basic_trading.py        # Basic usage
│   ├── pattern_analysis.py     # Pattern detection
│   └── advanced_strategies.py  # Advanced examples
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # System architecture
│   ├── AGENTS.md               # Agent documentation
│   ├── PROVIDERS.md            # Provider documentation
│   └── API_REFERENCE.md        # API documentation
│
└── data/                        # Runtime Data (gitignored)
    └── trading/                # Trade history storage
        └── memory/
            └── trades/         # JSON trade records
```

## Component Descriptions

### Core Components

**config.py** - Configuration Management
- `TradingConfig` dataclass
- Loads from environment variables
- Validates API credentials and risk parameters

**state.py** - State Management
- `TradingState` dataclass
- Persists to `.trading_state.json`
- Tracks positions, trades, circuit breaker status

**manager.py** - Trading Manager
- Orchestrates 8-agent decision loop
- Coordinates data flow between agents
- Executes trades via providers

**cli.py** - Command Line Interface
- Commands: status, positions, run, history, insights, etc.
- JSON output for IPC integration

### Providers (Exchange Integrations)

All providers implement `BaseExchangeProvider` interface:
- `fetch_ohlcv()` - Historical candle data
- `fetch_ticker()` - Current price
- `fetch_orderbook()` - Order book
- `fetch_balance()` - Account balance
- `fetch_positions()` - Open positions
- `create_order()` - Place order
- `cancel_order()` - Cancel order

### 8-Agent System

**Dataflow:**
```
DataSync → QuantAnalyst → Predict → Bull & Bear → DecisionCore → RiskAudit → Execution
```

1. **DataSync**: Fetches multi-timeframe OHLCV data
2. **QuantAnalyst**: Calculates technical indicators
3. **Predict**: ML-based price forecasting
4. **Bull/Bear**: Adversarial analysis (opposing views)
5. **DecisionCore**: Weighted voting with regime awareness
6. **RiskAudit**: Safety checks (can veto)
7. **Execution**: Order placement

### Memory System

**TradeJournal** - File-based storage
- Stores each trade as JSON
- Human-readable format
- Fast index-based lookups

**PatternDetector** - Analysis
- Regime performance
- Agent accuracy
- Confidence correlation
- Symbol patterns

**TradingMemory** - Optional Graphiti
- Semantic search
- Cross-session learning
- Contextual recommendations

## File Sizes

| Component | Files | Lines of Code |
|-----------|-------|---------------|
| Providers | 8 files | ~2,000 lines |
| Agents | 9 files | ~2,500 lines |
| Models | 5 files | ~1,500 lines |
| Memory | 3 files | ~1,000 lines |
| Risk | 3 files | ~500 lines |
| Tests | 3+ files | ~800 lines |
| **Total** | **30+ files** | **~8,300 lines** |

## Key Features by Directory

### `/trading/providers/`
- Multi-platform support (6 exchanges)
- Unified API via factory pattern
- Symbol normalization
- Paper trading simulator

### `/trading/agents/`
- 8-agent adversarial system
- Market regime awareness
- ML predictions (LightGBM)
- Risk veto mechanism

### `/trading/models/`
- Exchange-agnostic data structures
- Type-safe with validation
- CCXT conversion methods

### `/trading/memory/`
- Trade journaling
- Performance analytics
- Pattern detection
- Optional semantic search

## Getting Started

1. **Setup**:
   ```bash
   ./setup.sh
   ```

2. **Configure**:
   ```bash
   cp config/.env.example .env
   nano .env  # Edit configuration
   ```

3. **Run**:
   ```bash
   python -m trading.cli status
   python -m trading.cli run --symbol BTC/USDT
   ```

4. **Analyze**:
   ```bash
   python -m trading.cli insights
   ```

## Development

### Adding a New Provider

1. Create `trading/providers/your_exchange.py`
2. Inherit from `BaseExchangeProvider`
3. Implement required methods
4. Add to `factory.py`
5. Add tests to `tests/test_providers.py`

### Adding a New Agent

1. Create `trading/agents/your_agent.py`
2. Inherit from `BaseAgent`
3. Implement `execute()` method
4. Add to `manager.py` pipeline
5. Add tests

### Running Tests

```bash
pytest tests/ -v
pytest tests/test_providers.py::TestBinanceFutures -v
```

## Dependencies

See `requirements.txt` for full list:
- **ccxt**: Exchange API
- **alpaca-py**: Stock trading
- **ta-lib**: Technical analysis
- **lightgbm**: ML predictions
- **pandas/numpy**: Data manipulation

## License

MIT License
