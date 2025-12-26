# Technology Stack

**Analysis Date:** 2025-12-26

## Languages

**Primary:**
- Python 3.7+ - All application code

**Secondary:**
- Not detected

## Runtime

**Environment:**
- Python 3.7+ (asyncio-based async framework)
- No .python-version file detected

**Package Manager:**
- pip
- Lockfile: No lock file detected (no Pipfile.lock or poetry.lock)
- Dependencies: `requirements.txt`

## Frameworks

**Core:**
- FastAPI 0.104.0+ - REST API and WebSocket server for dashboard - `trading/web/server.py`
- asyncio - Async/await framework (Python standard library)

**Testing:**
- pytest 7.4.0+ - Testing framework
- pytest-asyncio 0.21.0+ - Async test support
- pytest-cov 4.1.0+ - Coverage reporting

**Build/Dev:**
- Uvicorn 0.24.0+ - ASGI server for FastAPI - `trading/web/server.py`

## Key Dependencies

**Critical:**
- CCXT 4.0.0+ - Unified exchange API for 100+ exchanges - `trading/providers/binance_futures.py`, `trading/providers/binance_spot.py`, `trading/providers/kraken.py`, `trading/providers/coinbase.py`
- Alpaca-py 0.9.0+ - Alpaca stock trading API - `trading/providers/alpaca.py`
- Pandas 2.0.0+ - Data manipulation and analysis
- NumPy 1.24.0+ - Numerical computing
- TA-Lib 0.4.0+ - Technical indicators - `trading/agents/quant_analyst.py`

**Machine Learning:**
- LightGBM 4.0.0+ - Gradient boosting for predictions - `trading/ml/models/lightgbm_model.py`
- XGBoost 2.0.0+ - Extreme gradient boosting - `trading/ml/models/xgboost_model.py`
- PyTorch 2.0.0+ - Deep learning for LSTM - `trading/ml/models/lstm_model.py`
- Scikit-learn 1.3.0+ - ML utilities - `trading/ml/training.py`, `trading/ml/ensemble.py`

**Sentiment Analysis:**
- Tweepy 4.14.0+ - Twitter/X API client - `trading/sentiment/twitter.py`
- NewsAPI 0.2.7+ - News API client - `trading/sentiment/news.py`
- VADER Sentiment 3.3.2+ - Sentiment analysis - `trading/sentiment/twitter.py`, `trading/sentiment/news.py`
- Transformers 4.30.0+ - FinBERT for financial sentiment (optional)

**Infrastructure:**
- aiohttp 3.9.0+ - Async HTTP client - `trading/notifications/`, `trading/sentiment/`
- python-telegram-bot 20.0+ - Telegram bot - `trading/notifications/telegram.py`
- aiosmtplib 2.0+ - Async SMTP - `trading/notifications/email.py`
- WebSockets 11.0+ - WebSocket support - `trading/web/websocket.py`
- python-dotenv 1.0.0+ - Environment variables - `trading/config.py`
- Pydantic 2.0.0+ - Data validation

**Optional:**
- Graphiti Core 0.3.0+ - Semantic learning and memory (disabled by default) - `trading/memory/graphiti_trading.py`

## Configuration

**Environment:**
- Configuration via `.env` files (copied from `config/.env.example`)
- Environment variable prefix: `TRADING_*`, `TELEGRAM_*`, `DISCORD_*`, `SLACK_*`, `SMTP_*`, `GRAPHITI_*`
- Key variables: `TRADING_PROVIDER`, `TRADING_TESTNET`, `TRADING_MAX_POSITION_SIZE_USD`, `TRADING_MAX_DAILY_DRAWDOWN_PCT`

**Build:**
- No build configuration (pure Python)

## Platform Requirements

**Development:**
- macOS/Linux/Windows (any platform with Python 3.7+)
- No external dependencies besides Python packages

**Production:**
- Python 3.7+ runtime
- Exchange API keys (Binance, Kraken, Coinbase, Alpaca)
- Optional: Twitter API, News API, Glassnode API for sentiment

---

*Stack analysis: 2025-12-26*
*Update after major dependency changes*
