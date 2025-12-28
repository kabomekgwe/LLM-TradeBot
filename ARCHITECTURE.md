# LLM-TradeBot Architecture

Complete system architecture and design documentation.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Technology Stack](#technology-stack)
5. [Database Schema](#database-schema)
6. [Security Architecture](#security-architecture)
7. [Deployment Architecture](#deployment-architecture)

---

## System Overview

LLM-TradeBot is a production-ready autonomous trading system built with a multi-layer architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM-TradeBot System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Presentation Layer                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │ │
│  │  │  Dashboard   │  │  REST API    │  │  WebSocket   │        │ │
│  │  │  (React)     │  │  (FastAPI)   │  │  (Real-time) │        │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Decision Layer                              │ │
│  │  ┌──────────────┐        ┌──────────────┐                     │ │
│  │  │ Bull Agent   │◄──────►│ Bear Agent   │                     │ │
│  │  │ (GPT-4)      │        │ (GPT-4)      │                     │ │
│  │  └──────┬───────┘        └──────┬───────┘                     │ │
│  │         │                       │                              │ │
│  │         └───────────┬───────────┘                              │ │
│  │                     ▼                                          │ │
│  │         ┌──────────────────────┐                              │ │
│  │         │  Consensus Engine    │                              │ │
│  │         │  (Weighted Voting)   │                              │ │
│  │         └──────────┬───────────┘                              │ │
│  └────────────────────┼────────────────────────────────────────┘ │
│                       ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Intelligence Layer                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │ │
│  │  │  XGBoost     │  │  LightGBM    │  │   LSTM       │        │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │ │
│  │  ┌──────────────┐  ┌──────────────────────────────┐          │ │
│  │  │ Transformer  │  │  Feature Engineering (86)    │          │ │
│  │  └──────────────┘  └──────────────────────────────┘          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                       ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Safety Layer                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │ │
│  │  │ Kill Switch  │  │  Circuit     │  │  Position    │        │ │
│  │  │ (HMAC Auth)  │  │  Breaker     │  │  Limits      │        │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                       ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Execution Layer                             │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │ │
│  │  │  Exchange    │  │  Order       │  │  Position    │        │ │
│  │  │  Client      │  │  Manager     │  │  Tracker     │        │ │
│  │  │  (CCXT)      │  │              │  │              │        │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                       ▼                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     Data Layer                                  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │ │
│  │  │ TimescaleDB  │  │  Model       │  │  JSON        │        │ │
│  │  │ (Trades)     │  │  Cache       │  │  Logs        │        │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Presentation Layer

**Dashboard (React + WebSocket)**
- Real-time metrics visualization
- System health monitoring
- Trade history display
- Position management UI

**REST API (FastAPI)**
- `/api/v1/metrics/*` - Performance metrics
- `/api/v1/health/*` - System health
- `/api/v1/safety/*` - Safety controls (kill switch, circuit breaker)
- `/api/v1/ml/*` - Model serving endpoints

**WebSocket Server**
- Real-time metric updates
- Health status broadcasts
- Position updates
- Alert notifications

### 2. Decision Layer

**Bull Agent (LLM)**
- Analyzes market from bullish perspective
- Evaluates technical indicators
- Assesses market regime
- Votes BUY, SELL, or HOLD with confidence score

**Bear Agent (LLM)**
- Analyzes market from bearish perspective
- Provides adversarial viewpoint
- Challenges bull agent assumptions
- Votes BUY, SELL, or HOLD with confidence score

**Consensus Engine**
- Weighted voting algorithm
- Regime-aware decision making
- Confidence threshold enforcement
- Final action determination

### 3. Intelligence Layer

**Ensemble ML Models**
- **XGBoost**: Gradient boosting for price predictions
- **LightGBM**: Fast gradient boosting with categorical features
- **LSTM**: Sequential pattern recognition
- **Transformer**: Attention-based market analysis

**Feature Engineering**
- **86 technical indicators**: RSI, MACD, Bollinger Bands, ATR, Volume indicators, etc.
- **Market microstructure**: Order book imbalance, spread, depth
- **Time features**: Hour, day of week, market session
- **Regime features**: Volatility regime, trend strength

### 4. Safety Layer

**Kill Switch**
- HMAC-SHA256 authenticated API
- Emergency stop all trading
- Automatic position closure
- State persistence

**Circuit Breaker**
- Monitors 5 threshold types:
  - Daily loss limit
  - Maximum drawdown
  - Consecutive losses
  - Win rate floor
  - Sharpe ratio floor
- Automatic trading halt
- 24-hour cooldown period

**Position Limits**
- 4-layer protection:
  1. Per-symbol limit
  2. Per-strategy limit
  3. Portfolio exposure limit
  4. Max concurrent positions

### 5. Execution Layer

**Exchange Client (CCXT)**
- Unified interface for 100+ exchanges
- Order execution (market, limit)
- Position management
- Balance tracking

**Order Manager**
- Order lifecycle management
- Fill tracking
- Partial fill handling
- Order cancellation

**Position Tracker**
- Open position monitoring
- P&L calculation
- Risk exposure tracking
- Automatic position closure on kill switch

### 6. Data Layer

**TimescaleDB**
- Time-series optimized PostgreSQL
- Trade history hypertable
- Performance metrics storage
- Automatic data retention

**Model Cache**
- Singleton model loader
- LRU cache eviction
- Memory-mapped model files
- Lazy loading

**Structured Logs**
- JSON format
- Correlation IDs
- Request tracing
- Queryable via `jq`

---

## Data Flow

### Trading Loop Flow

```
1. Market Data Collection
   ↓
2. Feature Engineering (86 indicators)
   ↓
3. ML Model Predictions (XGBoost, LightGBM, LSTM, Transformer)
   ↓
4. LLM Agent Analysis (Bull & Bear)
   ↓
5. Consensus Decision (Weighted Voting)
   ↓
6. Safety Checks (Kill Switch, Circuit Breaker, Position Limits)
   ↓
7. Order Execution (Exchange API)
   ↓
8. Position Tracking & Monitoring
   ↓
9. Trade History Persistence (TimescaleDB)
   ↓
10. Metrics Update & Alerts
```

### Request Flow Example

```
User clicks "Activate Kill Switch" on Dashboard
   ↓
Dashboard sends HMAC-signed POST to /api/v1/safety/kill-switch/activate
   ↓
FastAPI validates HMAC signature
   ↓
KillSwitch.activate() called
   ↓
All pending orders cancelled
   ↓
All open positions closed
   ↓
State saved to database
   ↓
Alert sent via Slack/Email/Telegram
   ↓
WebSocket broadcasts status update to Dashboard
   ↓
Dashboard shows "Kill Switch: ACTIVE"
```

---

## Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI 0.104+ | REST API, WebSocket, async support |
| ML Framework | Scikit-learn, XGBoost, LightGBM | Ensemble models |
| Deep Learning | PyTorch 2.1+ | LSTM, Transformer models |
| Exchange API | CCXT 4.0+ | Multi-exchange support |
| LLM Integration | OpenAI API (GPT-4) | Trading agents |
| Database ORM | SQLAlchemy 2.0+ | Database abstraction |
| Migrations | Alembic 1.13+ | Schema management |
| Logging | python-json-logger | Structured JSON logs |

### Database

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Primary DB | PostgreSQL 16 | Relational data |
| Time-Series | TimescaleDB 2.13+ | Trade history optimization |
| Caching | In-memory (LRU) | Model caching |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker 24.0+ | Service isolation |
| Orchestration | Docker Compose | Multi-container management |
| Secrets | Docker Secrets | Secure API key storage |
| Health Checks | Docker HEALTHCHECK | Auto-restart on failure |
| Monitoring | Prometheus-compatible | Metrics export |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dashboard | HTML/CSS/JS | Real-time visualization |
| WebSocket | FastAPI WebSocket | Live updates |
| Charting | Chart.js / Plotly | Metrics visualization |

---

## Database Schema

### TradeHistory Table (Hypertable)

```sql
CREATE TABLE trade_history (
    -- Primary key (composite for hypertable partitioning)
    timestamp TIMESTAMPTZ NOT NULL,
    trade_id VARCHAR(50) NOT NULL,

    -- Trade details
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,        -- buy/sell
    order_type VARCHAR(10) NOT NULL,  -- market/limit
    amount FLOAT NOT NULL,
    entry_price FLOAT NOT NULL,
    exit_price FLOAT,

    -- Performance
    realized_pnl FLOAT DEFAULT 0.0,
    pnl_pct FLOAT DEFAULT 0.0,
    fees FLOAT DEFAULT 0.0,

    -- Context
    market_regime VARCHAR(20),
    bull_confidence FLOAT,
    bear_confidence FLOAT,
    decision_confidence FLOAT,

    -- Outcome
    won BOOLEAN DEFAULT FALSE,
    closed BOOLEAN DEFAULT FALSE,
    close_timestamp TIMESTAMPTZ,

    -- Agent insights (JSONB for flexibility)
    agent_votes JSONB,
    signals JSONB,

    -- Composite primary key
    PRIMARY KEY (timestamp, trade_id)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trade_history', 'timestamp', if_not_exists => TRUE);

-- Indexes for common queries
CREATE INDEX idx_symbol ON trade_history(symbol);
CREATE INDEX idx_won ON trade_history(won);
CREATE INDEX idx_market_regime ON trade_history(market_regime);
```

### Queries

**Performance by Symbol:**
```sql
SELECT
    symbol,
    COUNT(*) as trades,
    SUM(CASE WHEN won THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(realized_pnl), 2) as avg_pnl,
    ROUND(SUM(realized_pnl), 2) as total_pnl,
    ROUND(100.0 * SUM(CASE WHEN won THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
FROM trade_history
GROUP BY symbol
ORDER BY total_pnl DESC;
```

**Daily P&L (TimescaleDB time_bucket):**
```sql
SELECT
    time_bucket('1 day', timestamp) as day,
    COUNT(*) as trades,
    ROUND(SUM(realized_pnl), 2) as daily_pnl,
    ROUND(AVG(realized_pnl), 2) as avg_trade
FROM trade_history
GROUP BY day
ORDER BY day DESC
LIMIT 30;
```

---

## Security Architecture

### Secrets Management

```
┌─────────────────────────────────────────┐
│         Docker Secrets Flow              │
├─────────────────────────────────────────┤
│                                           │
│  Host: ./secrets/exchange_api_key        │
│         (600 permissions, gitignored)    │
│              ↓                            │
│  Docker: /run/secrets/exchange_api_key   │
│         (mounted read-only)              │
│              ↓                            │
│  App: SecretsManager.get_secret()        │
│       (loads from /run/secrets/)         │
│              ↓                            │
│  Fallback: os.getenv() for local dev    │
│                                           │
└─────────────────────────────────────────┘
```

**Key Security Features:**
- **HMAC-SHA256** for kill switch authentication
- **Docker secrets** for API key storage (never in .env in production)
- **600 permissions** on secret files
- **Gitignore** all secrets/*
- **Read-only mounts** in containers
- **Correlation IDs** for request tracing
- **Structured logging** (no secrets in logs)

### Authentication Flow

```
Kill Switch Request
   ↓
Extract timestamp and action from request
   ↓
Generate HMAC signature:
  HMAC-SHA256(secret, "action:timestamp")
   ↓
Compare with X-Kill-Switch-Signature header
   ↓
If valid: Process request
If invalid: Return 401 Unauthorized
```

---

## Deployment Architecture

### Local Development

```
┌────────────────────────────────────────────────┐
│              docker-compose.yml                 │
├────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────────┐  ┌──────────────────┐   │
│  │  trading-bot     │  │  dashboard       │   │
│  │  Port: -         │  │  Port: 5173      │   │
│  │  Env: .env       │  │  Env: .env       │   │
│  └──────────────────┘  └──────────────────┘   │
│                                                  │
│  ┌──────────────────┐                          │
│  │  postgres        │                          │
│  │  Port: 5437      │                          │
│  │  Volume: db_data │                          │
│  └──────────────────┘                          │
│                                                  │
└────────────────────────────────────────────────┘
```

### Production VPS

```
┌─────────────────────────────────────────────────┐
│                Production Server                 │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌───────────────────────────────────────────┐  │
│  │            Nginx Reverse Proxy            │  │
│  │         (SSL, Rate Limiting)              │  │
│  └───────────────────┬───────────────────────┘  │
│                      ↓                            │
│  ┌───────────────────────────────────────────┐  │
│  │         Docker Compose Services           │  │
│  │  ┌─────────────┐  ┌─────────────┐        │  │
│  │  │ trading-bot │  │  dashboard  │        │  │
│  │  │ (internal)  │  │ (internal)  │        │  │
│  │  └─────────────┘  └─────────────┘        │  │
│  │  ┌─────────────┐                         │  │
│  │  │  postgres   │                         │  │
│  │  │ (internal)  │                         │  │
│  │  └─────────────┘                         │  │
│  └───────────────────────────────────────────┘  │
│                                                   │
│  Volumes:                                        │
│  - postgres_data (persistent)                   │
│  - ./secrets (600 permissions)                  │
│  - ./models (ML model files)                    │
│  - ./logs (structured JSON logs)                │
│                                                   │
└─────────────────────────────────────────────────┘
```

### Health Check Strategy

```
Docker HEALTHCHECK
   ↓
Every 30 seconds: curl http://localhost:5173/health
   ↓
Checks:
  - Kill switch status (not active)
  - System health (not critical)
  - Database connection
  - Model loading status
   ↓
Returns 200 (healthy) or 503 (unhealthy)
   ↓
If unhealthy for 3 consecutive checks:
  - Docker restarts container
  - Graceful shutdown handler runs
  - Positions closed safely
  - State saved to database
```

---

## Monitoring & Observability

### Metrics Pipeline

```
Trading Loop Execution
   ↓
MetricsTracker.update()
   ↓
Calculate:
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Current Drawdown
  - Win Rate
  - P&L (total, daily, weekly)
   ↓
Broadcast via WebSocket
   ↓
Dashboard updates in real-time
   ↓
Persist to TimescaleDB
```

### Alert Pipeline

```
Trigger Condition Met
  (e.g., Drawdown > 10%)
   ↓
AlertManager.send_alert()
   ↓
Debounce Check (prevent spam)
   ↓
Format alert message
   ↓
Send to all configured channels:
  - Slack (webhook)
  - Email (SMTP)
  - Telegram (bot API)
   ↓
Log alert event
```

### Logging Strategy

```
Event Occurs
   ↓
Logger.info/error/warning()
   ↓
CorrelationFilter injects correlation_id
   ↓
CustomJsonFormatter formats as:
  {
    "timestamp": "2025-12-28T10:30:00Z",
    "level": "INFO",
    "logger": "trading.manager",
    "message": "Executing trade",
    "correlation_id": "abc-123-xyz",
    "symbol": "BTC/USDT",
    "trade_id": "trade_20251228_001"
  }
   ↓
Output to:
  - stdout (Docker logs)
  - File (logs/trading.json.log)
   ↓
Queryable with jq:
  docker-compose logs | jq 'select(.correlation_id == "abc-123")'
```

---

## Design Principles

1. **Separation of Concerns**: Each layer has distinct responsibilities
2. **Fail-Safe Defaults**: Kill switch inactive, circuit breaker open, testnet mode
3. **Defense in Depth**: Multiple safety layers (kill switch, circuit breaker, limits)
4. **Observability First**: Structured logging, metrics, correlation IDs
5. **Graceful Degradation**: File-based fallback if database unavailable
6. **Stateless Services**: All state in database, services can restart freely
7. **Idempotent Operations**: Safe to retry failed operations
8. **Least Privilege**: Docker secrets, non-root user, read-only mounts

---

## Performance Considerations

**Model Loading:**
- Singleton pattern with LRU cache
- Lazy loading (on first prediction)
- Memory-mapped files for large models

**Database Queries:**
- Hypertable partitioning by timestamp
- Indexes on symbol, won, market_regime
- Connection pooling (5 connections)

**API Response Time:**
- Health check: <50ms
- Metrics endpoint: <100ms
- Prediction endpoint: <200ms (with cached model)

**WebSocket:**
- Broadcast only on metric changes
- Debounce updates (max 1/second)
- Compression enabled

---

## Scalability

**Horizontal Scaling:**
- Stateless services (can run multiple instances)
- Database connection pooling
- Load balancer distributes requests

**Vertical Scaling:**
- Docker resource limits configurable
- Model cache size tunable
- TimescaleDB chunk size adjustable

**Future Enhancements:**
- Redis for distributed caching
- Message queue for async processing
- Multi-region deployment
- Kubernetes orchestration

---

## Further Reading

- [TUTORIAL.md](TUTORIAL.md) - Complete usage guide
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Deployment instructions
- [QUICKREF.md](QUICKREF.md) - Command reference
- [PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md) - Pre-deployment checklist

---

**Built with ❤️ using Claude Code**
