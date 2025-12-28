# LLM-TradeBot 🤖📈

**Production-ready autonomous cryptocurrency trading bot powered by LLM agents, ensemble ML models, and real-time risk management**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

> **⚡ Quick Start:** See [TUTORIAL.md](TUTORIAL.md) for a complete step-by-step guide

Multi-platform algorithmic trading system powered by an 8-agent adversarial decision framework.

## 📚 Documentation

- **[TUTORIAL.md](TUTORIAL.md)** - Complete step-by-step guide (START HERE!)
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment guide
- **[PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)** - Pre-deployment checklist
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and architecture

## ⚡ Quick Start with Docker

The fastest way to get started is using Docker:

```bash
# Clone repository
git clone https://github.com/kabomekgwe/LLM-TradeBot.git
cd LLM-TradeBot

# Copy environment template
cp .env.production.template .env

# Edit .env and add your API keys (or use TRADING_PROVIDER=paper for testing)
nano .env

# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Access dashboard
open http://localhost:5173
```

**👉 For detailed instructions, see [TUTORIAL.md](TUTORIAL.md)**

## ✨ Key Features

### 🤖 LLM-Powered Trading Agents
- **Bull & Bear Consensus** - Adversarial agents analyze markets from opposing perspectives
- **GPT-4 Integration** - Advanced reasoning for market analysis
- **Regime-Aware Decisions** - Adapts to trending, choppy, and volatile markets

### 🧠 Ensemble ML Models
- **XGBoost & LightGBM** - Gradient boosting for price predictions
- **LSTM Networks** - Deep learning for sequential pattern recognition
- **Transformer Models** - Attention mechanisms for market analysis
- **86 Technical Indicators** - Comprehensive feature engineering

### 🛡️ Production-Grade Safety
- **Kill Switch API** - Emergency stop with HMAC authentication
- **Circuit Breaker** - Automatic trading halt on losses
- **4-Layer Position Limits** - Per-symbol, per-strategy, portfolio, and max positions
- **Real-time Monitoring** - Sharpe ratio, drawdown, P&L tracking

### 📊 Real-Time Dashboard
- **Live Metrics** - Performance, risk, and health monitoring
- **Multi-Channel Alerts** - Slack, Email, Telegram notifications
- **Trade History** - TimescaleDB with time-series optimization
- **WebSocket Updates** - Real-time data streaming

### 🐳 Docker Deployment
- **Multi-Stage Builds** - Optimized image size (<1GB)
- **Health Checks** - Automatic container restart on failures
- **Graceful Shutdown** - Safe position closure on stop
- **Docker Secrets** - Secure API key management

### 🔐 Security & Observability
- **Structured JSON Logging** - Queryable logs with correlation IDs
- **Docker Secrets** - Production-grade secrets management
- **Request Tracing** - X-Request-ID headers for debugging
- **Compliance Ready** - Audit trail and trade history persistence

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd LLM-TradeBot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example configuration:

```bash
cp config/.env.example .env
```

Edit `.env` and configure your trading provider:

```bash
# Choose your provider
TRADING_PROVIDER=paper  # Start with paper trading (safe!)

# For real trading, add API keys:
# BINANCE_FUTURES_API_KEY=your-key
# BINANCE_FUTURES_API_SECRET=your-secret

# Risk parameters
TRADING_TESTNET=true
TRADING_MAX_POSITION_SIZE_USD=1000.0
TRADING_MAX_DAILY_DRAWDOWN_PCT=5.0
TRADING_MAX_OPEN_POSITIONS=3
```

### 3. Run Trading System

```bash
# Check system status
python -m trading.cli status

# Get current positions
python -m trading.cli positions

# Execute trading loop (paper trading)
python -m trading.cli run --symbol BTC/USDT

# Generate insights from trade history
python -m trading.cli insights
```

## Architecture

### 8-Agent System

1. **DataSync Agent**: Fetches multi-timeframe market data (5m, 15m, 1h)
2. **QuantAnalyst Agent**: Calculates technical indicators and signals
3. **Predict Agent**: LightGBM ML forecasting
4. **Bull Agent**: Bullish analysis and reasoning
5. **Bear Agent**: Bearish analysis and reasoning (adversarial)
6. **DecisionCore Agent**: Weighted voting with regime awareness
7. **RiskAudit Agent**: Safety checks with veto power
8. **Execution Agent**: Order placement and management

### Providers

```python
from trading.providers.factory import create_provider
from trading.config import TradingConfig

# Paper trading (no API keys)
config = TradingConfig(provider="paper", testnet=True)
provider = create_provider(config)

# Real exchange
config = TradingConfig.from_env()  # Loads from .env
provider = create_provider(config)
```

### Memory & Learning

```python
from trading.memory import TradeJournal, PatternDetector

# Load trade history
journal = TradeJournal(spec_dir)

# Analyze patterns
detector = PatternDetector(journal)
insights = detector.generate_all_insights()

# Insights include:
# - Regime performance analysis
# - Agent prediction accuracy
# - Confidence correlation
# - Symbol-specific patterns
```

## Docker Deployment

### Local Development

```bash
# Quick start
./scripts/deploy-local.sh

# Or manually
docker-compose build
docker-compose up -d
docker-compose logs -f
```

### Production Deployment

```bash
# On production server
./scripts/deploy-production.sh

# Or manually
docker-compose build
docker-compose up -d
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete deployment guide and [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md) for pre-deployment checklist.

## Directory Structure

```
LLM-TradeBot/
├── trading/                   # Main package
│   ├── providers/             # Exchange integrations
│   │   ├── binance_futures.py
│   │   ├── binance_spot.py
│   │   ├── kraken.py
│   │   ├── coinbase.py
│   │   ├── alpaca.py
│   │   └── paper.py
│   ├── agents/                # 8-agent system
│   ├── models/                # Data models
│   ├── memory/                # Trade journaling & learning
│   ├── risk/                  # Risk management
│   ├── web/                   # Dashboard server
│   ├── utils/                 # Utilities (shutdown handler, etc.)
│   └── cli.py                 # Command-line interface
├── config/                    # Configuration files
│   └── .env.example
├── scripts/                   # Deployment scripts
│   ├── docker-build.sh
│   ├── deploy-local.sh
│   └── deploy-production.sh
├── docs/                      # Documentation
│   ├── DEPLOYMENT.md
│   └── PRODUCTION_CHECKLIST.md
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Container orchestration
├── examples/                  # Usage examples
└── tests/                     # Test suite
```

## Usage Examples

### Basic Trading

```python
import asyncio
from trading.config import TradingConfig
from trading.manager import TradingManager

async def main():
    # Load configuration
    config = TradingConfig.from_env()

    # Initialize manager
    manager = TradingManager(config)

    # Run trading loop
    result = await manager.run_trading_loop(symbol="BTC/USDT")

    if result["success"]:
        print(f"Decision: {result['decision']['action']}")
        print(f"Confidence: {result['decision']['confidence']}")

asyncio.run(main())
```

### Pattern Analysis

```python
from pathlib import Path
from trading.memory import TradeJournal, PatternDetector

# Load journal
journal = TradeJournal(Path("data/trading"))

# Calculate metrics
metrics = journal.calculate_metrics()
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")

# Detect patterns
detector = PatternDetector(journal)
insights = detector.generate_all_insights()

for insight in insights:
    print(f"\n{insight.title}")
    print(insight.description)
```

## Safety Features

- **Testnet Default**: Always starts in testnet mode
- **Circuit Breakers**: Auto-halt on excessive losses
- **Position Limits**: Max size and count restrictions
- **Risk Veto**: Risk agent can block dangerous trades
- **Paper Trading**: Test strategies without real money

## CLI Commands

```bash
# System status
python -m trading.cli status

# View positions
python -m trading.cli positions

# Execute trade
python -m trading.cli run --symbol BTC/USDT [--dry-run]

# Trade history
python -m trading.cli history --limit 50

# Close position
python -m trading.cli close --symbol BTC/USDT

# Cancel order
python -m trading.cli cancel --order-id ORDER_ID --symbol BTC/USDT

# Generate insights
python -m trading.cli insights
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_PROVIDER` | - | Exchange provider (binance_futures, paper, etc.) |
| `TRADING_TESTNET` | `true` | Use testnet/sandbox mode |
| `TRADING_MAX_POSITION_SIZE_USD` | `1000.0` | Max position size |
| `TRADING_MAX_DAILY_DRAWDOWN_PCT` | `5.0` | Daily loss limit (%) |
| `TRADING_MAX_OPEN_POSITIONS` | `3` | Max concurrent positions |
| `TRADING_DECISION_THRESHOLD` | `0.6` | Min confidence (0.0-1.0) |

## Supported Exchanges

| Exchange | Type | Testnet | API Keys Required |
|----------|------|---------|-------------------|
| Binance Futures | Crypto | ✅ | ✅ |
| Binance Spot | Crypto | ✅ | ✅ |
| Kraken | Crypto | ❌ | ✅ |
| Coinbase | Crypto | ✅ | ✅ |
| Alpaca | Stocks | ✅ | ✅ |
| Paper Trading | Simulator | ✅ | ❌ |

## Dependencies

- Python 3.9+
- ccxt >= 4.0.0 (exchange API)
- ta-lib >= 0.4.0 (technical indicators)
- lightgbm >= 4.0.0 (ML predictions)
- pandas, numpy
- alpaca-py (for stock trading)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_providers.py -v

# Test with coverage
pytest tests/ --cov=trading --cov-report=html
```

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Disclaimer

**This software is for educational purposes only. Trading cryptocurrencies and stocks involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.**

## 📖 Learning Resources

- **[TUTORIAL.md](TUTORIAL.md)** - Complete beginner-friendly guide
- **Architecture Overview** - See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Examples** - Check `examples/` directory
- **API Documentation** - See `docs/API.md`

## 🤝 Support

- **Documentation**: `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/kabomekgwe/LLM-TradeBot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kabomekgwe/LLM-TradeBot/discussions)

## 🗺️ Roadmap

- [x] Phase 9: Emergency Safety Controls ✅
- [x] Phase 10: Real-Time Monitoring Infrastructure ✅
- [x] Phase 11: Dockerized Production Deployment ✅
- [x] Phase 12: Model Serving & Data Infrastructure ✅
- [ ] Phase 13: Advanced ML (Model versioning, A/B testing)
- [ ] Phase 14: Portfolio Optimization
- [ ] Phase 15: Multi-Exchange Arbitrage

See [.planning/ROADMAP.md](.planning/ROADMAP.md) for full roadmap.

## 🙏 Acknowledgments

- Built with [Claude Code](https://claude.com/claude-code)
- Powered by [OpenAI GPT-4](https://openai.com)
- Exchange integration via [CCXT](https://github.com/ccxt/ccxt)
- Time-series database: [TimescaleDB](https://www.timescale.com)

---

**⭐ Star this repo if you find it useful!**

Built with ❤️ using Claude Code
