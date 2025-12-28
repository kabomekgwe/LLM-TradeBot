# LLM-TradeBot Tutorial: Complete Guide to Autonomous Crypto Trading

Welcome to LLM-TradeBot! This tutorial will guide you through setting up and running your own production-ready autonomous cryptocurrency trading system.

## Table of Contents

1. [What is LLM-TradeBot?](#what-is-llm-tradebot)
2. [Quick Start (5 minutes)](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Detailed Setup Guide](#detailed-setup-guide)
5. [Running Your First Trade](#running-your-first-trade)
6. [Dashboard & Monitoring](#dashboard--monitoring)
7. [Safety Controls](#safety-controls)
8. [Production Deployment](#production-deployment)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## What is LLM-TradeBot?

LLM-TradeBot is a **production-ready autonomous cryptocurrency trading bot** that combines:

- 🤖 **LLM-powered trading agents** (Bull & Bear consensus)
- 🧠 **Ensemble ML models** (XGBoost, LightGBM, LSTM, Transformers)
- 📊 **86 technical indicators** for feature engineering
- 🛡️ **Multi-layer safety controls** (kill switch, circuit breaker, position limits)
- 📈 **Real-time monitoring** (Sharpe ratio, drawdown, P&L tracking)
- 🐳 **Docker deployment** with health checks and graceful shutdown
- 🗄️ **TimescaleDB** for time-series trade history
- 🔐 **Docker secrets** for secure API key management

### Key Features

✅ **Fully Autonomous** - Analyzes markets, makes decisions, executes trades
✅ **Risk Management** - Position limits, circuit breakers, kill switch API
✅ **Real-time Monitoring** - Live dashboard with metrics and alerts
✅ **Production Ready** - Docker deployment, structured logging, graceful shutdown
✅ **Backtesting** - Evaluate strategies before live trading
✅ **Multi-Exchange** - Supports 100+ exchanges via CCXT

---

## Quick Start

Get the bot running in **5 minutes**:

```bash
# 1. Clone repository
git clone https://github.com/kabomekgwe/LLM-TradeBot.git
cd LLM-TradeBot

# 2. Create environment file
cp .env.example .env

# 3. Add your API keys to .env
# EXCHANGE_API_KEY=your_binance_key
# EXCHANGE_API_SECRET=your_binance_secret
# OPENAI_API_KEY=your_openai_key

# 4. Start with Docker Compose
docker-compose up -d

# 5. Initialize secrets (production)
./scripts/init-secrets.sh

# 6. Open dashboard
open http://localhost:5173
```

That's it! The bot is now running in **paper trading mode** (testnet).

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM-TradeBot Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐        ┌──────────────┐                   │
│  │ Bull Agent   │◄──────►│ Bear Agent   │                   │
│  │ (LLM)        │        │ (LLM)        │                   │
│  └──────┬───────┘        └──────┬───────┘                   │
│         │                       │                            │
│         └───────────┬───────────┘                            │
│                     ▼                                        │
│         ┌──────────────────────┐                            │
│         │  Decision Engine     │                            │
│         │  (Consensus Voting)  │                            │
│         └──────────┬───────────┘                            │
│                    ▼                                         │
│         ┌──────────────────────┐                            │
│         │  Ensemble Predictor  │                            │
│         │  XGBoost│LightGBM│   │                            │
│         │  LSTM│Transformer    │                            │
│         └──────────┬───────────┘                            │
│                    ▼                                         │
│         ┌──────────────────────┐                            │
│         │  Safety Layer        │                            │
│         │  Kill│Circuit│Limits │                            │
│         └──────────┬───────────┘                            │
│                    ▼                                         │
│         ┌──────────────────────┐                            │
│         │  Exchange Executor   │                            │
│         │  (CCXT Integration)  │                            │
│         └──────────────────────┘                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Market Data** → Feature Engineering (86 indicators)
2. **Features** → Ensemble Models (XGBoost, LightGBM, LSTM)
3. **Predictions** → LLM Agents (Bull & Bear analysis)
4. **Agent Votes** → Consensus Decision
5. **Decision** → Safety Checks (kill switch, limits)
6. **Approved** → Exchange Execution
7. **Execution** → Monitoring & Logging

---

## Detailed Setup Guide

### Prerequisites

- **Docker** (v20.10+) & Docker Compose (v2.0+)
- **Python** 3.13+ (for local development)
- **Exchange Account** (Binance, Coinbase, etc.)
- **OpenAI API Key** (for LLM agents)

### Step 1: Environment Configuration

Create `.env` file with your credentials:

```bash
# Exchange Configuration
EXCHANGE=binance                    # Exchange name (binance, coinbase, kraken, etc.)
EXCHANGE_TESTNET=true              # Use testnet for paper trading
EXCHANGE_API_KEY=your_api_key      # Your exchange API key
EXCHANGE_API_SECRET=your_secret    # Your exchange API secret

# LLM Configuration
OPENAI_API_KEY=your_openai_key     # OpenAI API key for agents
LLM_MODEL=gpt-4                    # Model for trading agents

# Trading Configuration
SYMBOLS=BTC/USDT,ETH/USDT          # Trading pairs (comma-separated)
TIMEFRAME=1h                        # Candlestick timeframe
POSITION_SIZE=0.01                  # Position size (in quote currency)

# Safety Limits
MAX_POSITION_SIZE=0.1               # Maximum position per trade
MAX_DAILY_LOSS=100                  # Maximum daily loss (USDT)
MAX_POSITIONS=5                     # Maximum concurrent positions

# Kill Switch
KILL_SWITCH_SECRET=your_secret_key  # Secret for kill switch API
```

### Step 2: Docker Secrets (Production)

For production deployment, use Docker secrets instead of `.env`:

```bash
# Initialize secrets with automated script
./scripts/init-secrets.sh

# This creates:
# - secrets/exchange_api_key
# - secrets/exchange_api_secret
# - secrets/kill_switch_secret
# - secrets/db_password

# Secrets are automatically mounted at /run/secrets/ in containers
```

### Step 3: Database Setup

TimescaleDB is automatically configured via Docker Compose:

```bash
# Start database service
docker-compose up -d postgres

# Run migrations
docker-compose run --rm trading-bot alembic upgrade head

# Verify hypertable creation
docker exec -it llm-tradebot-postgres-1 psql -U tradingbot -d tradingbot \
  -c "SELECT * FROM timescaledb_information.hypertables;"
```

### Step 4: Model Training (Optional)

Train your own ML models before trading:

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train ensemble models
python -m trading.ml.training --symbol BTC/USDT --timeframe 1h --days 365

# Models saved to: models/
# - xgboost_model.pkl
# - lightgbm_model.pkl
# - lstm_model.h5
# - transformer_model.pth
```

---

## Running Your First Trade

### Step 1: Start in Paper Trading Mode

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Expected output:
# ✓ Loaded models: XGBoost, LightGBM, LSTM
# ✓ Connected to Binance Testnet
# ✓ Kill switch: INACTIVE
# ✓ Circuit breaker: OPEN
# → Trading loop started...
```

### Step 2: Monitor the Dashboard

Open http://localhost:5173 to see:

- **Real-time Metrics**: Sharpe ratio, drawdown, win rate, P&L
- **System Health**: Kill switch status, circuit breaker, position limits
- **Active Positions**: Current trades and their performance
- **Trade History**: Past trades with profit/loss

### Step 3: Understanding Agent Decisions

Watch the logs to see how agents make decisions:

```
[BULL AGENT] BTC/USDT Analysis:
  Confidence: 0.78
  Reasoning: Strong upward momentum, RSI oversold, positive MACD crossover
  Vote: BUY

[BEAR AGENT] BTC/USDT Analysis:
  Confidence: 0.45
  Reasoning: Approaching resistance, volume declining
  Vote: HOLD

[DECISION ENGINE] Consensus reached:
  Action: BUY
  Final Confidence: 0.615
  Executing trade...

[SAFETY CHECK] ✓ Position limits OK (4/5 positions)
[SAFETY CHECK] ✓ Circuit breaker OPEN
[SAFETY CHECK] ✓ Kill switch INACTIVE

[EXECUTION] BUY 0.01 BTC/USDT @ $45,230.50
[TRADE] Order filled: trade_20231228_001
```

### Step 4: Test Safety Controls

Trigger the kill switch via API:

```bash
# Generate HMAC signature
TIMESTAMP=$(date +%s)
MESSAGE="activate:emergency_test"
SIGNATURE=$(echo -n "$MESSAGE" | openssl dgst -sha256 -hmac "$(cat secrets/kill_switch_secret)" -binary | base64)

# Activate kill switch
curl -X POST http://localhost:5173/api/v1/safety/kill-switch/activate \
  -H "Content-Type: application/json" \
  -H "X-Kill-Switch-Signature: $SIGNATURE" \
  -d '{
    "reason": "Emergency test",
    "close_positions": true
  }'

# Response:
# {
#   "status": "activated",
#   "reason": "Emergency test",
#   "timestamp": "2025-12-28T10:30:00Z",
#   "positions_closed": 4
# }
```

Bot immediately stops trading and closes all positions!

---

## Dashboard & Monitoring

### Real-Time Metrics

The dashboard displays:

**Performance Metrics:**
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 = good, >2.0 = excellent)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Current Drawdown**: Current decline from peak
- **Win Rate**: Percentage of winning trades
- **Consecutive Losses**: Early warning system

**P&L Tracking:**
- **Total P&L**: All-time profit/loss
- **Daily P&L**: Today's performance
- **Weekly P&L**: Rolling 7-day performance
- **Current Equity**: Live account value
- **Peak Equity**: Highest account value reached

**System Health:**
- **Kill Switch**: Active/Inactive status
- **Circuit Breaker**: Open/Closed (halts trading on losses)
- **Position Limits**: Current vs. maximum positions
- **Exchange Connection**: API connectivity status

### Multi-Channel Alerts

Configure alerts in `.env`:

```bash
# Slack Integration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Email Alerts
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_FROM=bot@yourdomain.com
EMAIL_TO=your@email.com
EMAIL_PASSWORD=your_app_password

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Alerts trigger for:
- ⚠️ Circuit breaker activation
- 🛑 Kill switch activation
- 📉 Drawdown threshold exceeded
- 📊 Daily loss limit reached
- 🔌 Exchange connection lost

### Structured Logging

All logs are in JSON format for easy querying:

```bash
# View all logs with correlation ID
docker-compose logs trading-bot | jq 'select(.correlation_id == "abc-123")'

# Filter by log level
docker-compose logs trading-bot | jq 'select(.level == "ERROR")'

# Search for trade executions
docker-compose logs trading-bot | jq 'select(.message | contains("Executing trade"))'

# View logs for specific symbol
docker-compose logs trading-bot | jq 'select(.symbol == "BTC/USDT")'
```

Each log entry includes:
- `timestamp`: ISO 8601 timestamp
- `level`: DEBUG, INFO, WARNING, ERROR
- `logger`: Module name
- `message`: Log message
- `correlation_id`: Request/trade correlation ID
- `symbol`: Trading pair (if applicable)
- `trade_id`: Trade identifier (if applicable)

---

## Safety Controls

### 1. Kill Switch API

Emergency stop for all trading:

```bash
# Activate (closes all positions)
curl -X POST http://localhost:5173/api/v1/safety/kill-switch/activate \
  -H "X-Kill-Switch-Signature: $SIGNATURE" \
  -d '{"reason": "Market volatility", "close_positions": true}'

# Deactivate (resume trading)
curl -X POST http://localhost:5173/api/v1/safety/kill-switch/deactivate \
  -H "X-Kill-Switch-Signature: $SIGNATURE"

# Check status
curl http://localhost:5173/api/v1/safety/kill-switch/status
```

### 2. Circuit Breaker

Automatically halts trading on losses:

**Thresholds:**
- **Daily loss** > $100 (configurable)
- **Drawdown** > 10% (configurable)
- **Consecutive losses** > 5 trades
- **Win rate** < 30% (last 20 trades)
- **Sharpe ratio** < 0.0 (negative risk-adjusted returns)

When triggered:
1. Stops opening new positions
2. Maintains existing positions
3. Sends alerts to all channels
4. Logs detailed reason

Reset:
- Automatically after 24 hours
- Manually via API: `POST /api/v1/safety/circuit-breaker/reset`

### 3. Position Limits

Four-layer protection:

```python
# 1. Per-Symbol Limit
MAX_POSITION_PER_SYMBOL = 0.1 BTC  # Max 0.1 BTC in BTC/USDT

# 2. Per-Strategy Limit
MAX_POSITION_PER_STRATEGY = 0.5 BTC  # Max 0.5 BTC in "momentum" strategy

# 3. Portfolio Limit
MAX_PORTFOLIO_EXPOSURE = 1.0 BTC  # Max 1.0 BTC across all positions

# 4. Max Concurrent Positions
MAX_POSITIONS = 5  # Maximum 5 open positions
```

Configure in `.env`:

```bash
MAX_POSITION_SIZE=0.1
MAX_DAILY_LOSS=100
MAX_POSITIONS=5
MAX_DRAWDOWN=0.10  # 10%
```

---

## Production Deployment

### Local Development Deployment

```bash
# Build and start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production VPS Deployment

```bash
# 1. Copy project to VPS
scp -r LLM-TradeBot user@your-vps:/opt/

# 2. SSH into VPS
ssh user@your-vps

# 3. Navigate to project
cd /opt/LLM-TradeBot

# 4. Initialize secrets
./scripts/init-secrets.sh

# 5. Update production environment
cp .env.production.template .env.production
nano .env.production  # Edit with production values

# 6. Deploy with production config
docker-compose -f docker-compose.yml --env-file .env.production up -d

# 7. Verify health
curl http://localhost:5173/health
# Response: {"status": "healthy", "timestamp": "..."}

# 8. Set up reverse proxy (Nginx)
# See: docs/DEPLOYMENT.md for Nginx configuration
```

### Health Checks

Docker health checks run every 30 seconds:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5173/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

If health check fails:
1. Docker automatically restarts container
2. Graceful shutdown handler runs
3. Positions closed safely
4. State saved to database

### Graceful Shutdown

When stopping the bot:

```bash
docker-compose down
```

**Shutdown sequence:**
1. Receives SIGTERM signal
2. Cancels all pending orders
3. Closes all open positions
4. Saves state to database
5. Sends shutdown notification
6. Exits cleanly (30s timeout)

---

## Advanced Features

### 1. Backtesting Strategies

Test strategies on historical data:

```bash
# Run backtest
python -m trading.ml.evaluation.backtester \
  --symbol BTC/USDT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --strategy ensemble \
  --initial-capital 10000

# Output:
# ===== Backtest Results =====
# Total Return: 42.5%
# Sharpe Ratio: 1.85
# Max Drawdown: -12.3%
# Win Rate: 58.2%
# Total Trades: 342
# Avg Trade: +$124.50
```

### 2. Custom Strategies

Create your own trading strategy:

```python
# trading/strategies/my_strategy.py
from trading.models.decision import TradingDecision, Action

class MyStrategy:
    def analyze(self, symbol: str, data: pd.DataFrame) -> TradingDecision:
        # Your custom logic
        if data['rsi'].iloc[-1] < 30:
            return TradingDecision(
                symbol=symbol,
                action=Action.BUY,
                confidence=0.8,
                reasoning="RSI oversold"
            )
        return TradingDecision(symbol=symbol, action=Action.HOLD)
```

Register in `trading/manager.py`:

```python
from trading.strategies.my_strategy import MyStrategy

self.strategies = [
    MyStrategy(),
    # ... other strategies
]
```

### 3. ML Model Serving API

Serve predictions via REST API:

```bash
# Get available models
curl http://localhost:5173/api/v1/ml/models

# Response:
# {
#   "models": [
#     {"name": "xgboost_model.pkl", "size_mb": 2.4},
#     {"name": "lightgbm_model.pkl", "size_mb": 1.8},
#     {"name": "lstm_model.h5", "size_mb": 5.2}
#   ]
# }

# Get prediction
curl -X POST http://localhost:5173/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgboost_model.pkl",
    "model_type": "xgboost",
    "features": [[1.5, 2.3, 0.8, 1.2, 0.5, ...]]
  }'

# Response:
# {
#   "predictions": [0.78],
#   "model_name": "xgboost_model.pkl",
#   "model_type": "xgboost"
# }
```

### 4. Database Queries

Query trade history with SQL:

```sql
-- Get recent trades
SELECT trade_id, symbol, timestamp, side, realized_pnl
FROM trade_history
ORDER BY timestamp DESC
LIMIT 10;

-- Get performance by symbol
SELECT
  symbol,
  COUNT(*) as total_trades,
  SUM(CASE WHEN won THEN 1 ELSE 0 END) as winning_trades,
  AVG(realized_pnl) as avg_pnl,
  SUM(realized_pnl) as total_pnl
FROM trade_history
GROUP BY symbol;

-- Get trades by time bucket (TimescaleDB)
SELECT
  time_bucket('1 day', timestamp) as day,
  COUNT(*) as trades,
  SUM(realized_pnl) as daily_pnl
FROM trade_history
GROUP BY day
ORDER BY day DESC;
```

### 5. Custom Alerts

Add custom alert conditions:

```python
# trading/monitoring/alerts.py
from trading.notifications import AlertManager

alert_manager = AlertManager()

# Custom alert condition
if portfolio_value > 100000:
    alert_manager.send_alert(
        title="🎉 Portfolio Milestone",
        message=f"Portfolio reached ${portfolio_value:,.2f}",
        severity="info",
        channels=["slack", "email"]
    )
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Error

**Error:**
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution:**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Restart database
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

#### 2. Model Loading Failed

**Error:**
```
FileNotFoundError: Model file not found: models/xgboost_model.pkl
```

**Solution:**
```bash
# Train models first
python -m trading.ml.training --symbol BTC/USDT --timeframe 1h

# Or download pre-trained models
wget https://github.com/kabomekgwe/LLM-TradeBot/releases/download/v1.2/models.zip
unzip models.zip
```

#### 3. Exchange Authentication Failed

**Error:**
```
ccxt.AuthenticationError: Invalid API key
```

**Solution:**
```bash
# Verify API keys in .env
cat .env | grep EXCHANGE

# Test connection
python -c "
import ccxt
exchange = ccxt.binance({
    'apiKey': 'YOUR_KEY',
    'secret': 'YOUR_SECRET',
    'enableRateLimit': True
})
print(exchange.fetch_balance())
"
```

#### 4. Docker Health Check Failing

**Error:**
```
health: starting
health: unhealthy
```

**Solution:**
```bash
# Check health endpoint manually
docker exec llm-tradebot-trading-bot-1 curl http://localhost:5173/health

# View detailed logs
docker-compose logs trading-bot | tail -50

# Restart with fresh state
docker-compose down
docker-compose up -d
```

#### 5. Kill Switch Not Working

**Error:**
```
401 Unauthorized: Invalid signature
```

**Solution:**
```bash
# Verify secret is correct
cat secrets/kill_switch_secret

# Generate signature correctly
MESSAGE="activate:test"
SIGNATURE=$(echo -n "$MESSAGE" | openssl dgst -sha256 -hmac "$(cat secrets/kill_switch_secret)" -binary | base64)

# Include timestamp in message
TIMESTAMP=$(date +%s)
MESSAGE="activate:${TIMESTAMP}"
```

### Debug Mode

Enable verbose logging:

```bash
# In .env
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart

# View debug logs
docker-compose logs -f trading-bot | grep DEBUG
```

### Performance Tuning

Optimize for better performance:

```python
# In docker-compose.yml, add resource limits
services:
  trading-bot:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

---

## Next Steps

Now that you have the bot running:

1. **📚 Read the Documentation**
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design
   - [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Production deployment
   - [PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md) - Pre-deployment checklist

2. **🧪 Test in Paper Trading**
   - Run for at least 1 week in testnet mode
   - Monitor metrics and adjust parameters
   - Verify safety controls work correctly

3. **📊 Optimize Strategies**
   - Backtest different timeframes
   - Tune position sizes and limits
   - Train models on more data

4. **🚀 Deploy to Production**
   - Switch to live exchange (testnet=false)
   - Start with small position sizes
   - Monitor closely for first 24 hours

5. **🔧 Customize & Extend**
   - Add custom strategies
   - Integrate new exchanges
   - Build custom dashboards

---

## Support & Community

- **GitHub Issues**: https://github.com/kabomekgwe/LLM-TradeBot/issues
- **Documentation**: [README.md](README.md)
- **License**: MIT

---

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK**

Cryptocurrency trading involves substantial risk of loss. This software is provided "as is" without warranty of any kind. The authors are not responsible for any losses incurred through use of this bot.

**Always:**
- Start with paper trading (testnet)
- Use only funds you can afford to lose
- Monitor the bot regularly
- Understand the code before deploying
- Comply with local regulations

---

**Happy Trading! 🚀**

*Built with ❤️ using Claude Code*
