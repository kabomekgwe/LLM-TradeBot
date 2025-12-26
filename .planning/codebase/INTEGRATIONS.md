# External Integrations

**Analysis Date:** 2025-12-26

## APIs & External Services

**Trading Platforms:**
- Binance Futures - Crypto derivatives trading
  - SDK/Client: CCXT v4.0.0+
  - Auth: `BINANCE_FUTURES_API_KEY`, `BINANCE_FUTURES_API_SECRET` env vars
  - Implementation: `trading/providers/binance_futures.py`

- Binance Spot - Crypto spot trading
  - SDK/Client: CCXT v4.0.0+
  - Auth: `BINANCE_SPOT_API_KEY`, `BINANCE_SPOT_API_SECRET` env vars
  - Implementation: `trading/providers/binance_spot.py`

- Kraken - Crypto exchange
  - SDK/Client: CCXT v4.0.0+
  - Auth: `KRAKEN_API_KEY`, `KRAKEN_API_SECRET` env vars
  - Implementation: `trading/providers/kraken.py`

- Coinbase Advanced Trade - Crypto exchange
  - SDK/Client: CCXT v4.0.0+
  - Auth: `COINBASE_API_KEY`, `COINBASE_API_SECRET` env vars
  - Implementation: `trading/providers/coinbase.py`

- Alpaca - US stock trading
  - SDK/Client: alpaca-py v0.9.0+
  - Auth: `ALPACA_API_KEY`, `ALPACA_API_SECRET` env vars
  - Implementation: `trading/providers/alpaca.py`

- Paper Trading - Simulated trading (no API keys needed)
  - Implementation: `trading/providers/paper.py`

**Sentiment Analysis:**
- Twitter/X API v2 - Social media sentiment
  - Integration method: tweepy v4.14.0+
  - Auth: `TWITTER_BEARER_TOKEN` env var
  - Implementation: `trading/sentiment/twitter.py`

- News API - Financial news sentiment
  - Integration method: newsapi-python v0.2.7+
  - Auth: `NEWS_API_KEY` env var
  - Implementation: `trading/sentiment/news.py`

- Glassnode - On-chain blockchain metrics
  - Integration method: REST API via aiohttp
  - Auth: `GLASSNODE_API_KEY` env var
  - Implementation: `trading/sentiment/onchain.py`

- Alternative.me - Fear & Greed Index
  - Integration method: Free API (no auth)
  - Implementation: `trading/sentiment/fear_greed.py`

## Data Storage

**Databases:**
- Not detected (uses JSON file-based state)

**File Storage:**
- Local file system - State persistence
  - Format: JSON files in `.trading_state.json`
  - Implementation: `trading/state.py`

- Local file system - Trade history
  - Format: JSON files per trade
  - Implementation: `trading/memory/trade_history.py`

**Caching:**
- Not detected

## Authentication & Identity

**Auth Provider:**
- Not applicable (trading bot, not user-facing app)

**API Authentication:**
- Exchange API keys stored in environment variables
- Managed via `trading/config.py` TradingConfig dataclass

## Monitoring & Observability

**Error Tracking:**
- Not detected

**Analytics:**
- Not detected

**Logs:**
- File-based logging to `logs/trading.log`
- Log level configurable via `LOG_LEVEL` env var
- Implementation: Python logging module

## CI/CD & Deployment

**Hosting:**
- Not detected (runs locally or on user infrastructure)

**CI Pipeline:**
- Not detected

## Environment Configuration

**Development:**
- Required env vars: `TRADING_PROVIDER`, `TRADING_TESTNET=true`
- Optional: Exchange API keys (paper trading works without)
- Secrets location: `.env` file (gitignored)
- Template: `config/.env.example`

**Staging:**
- Not applicable (same as development with `TRADING_TESTNET=true`)

**Production:**
- Secrets management: `.env` file with real exchange credentials
- Required: Exchange API keys, notification credentials (optional)

## Webhooks & Callbacks

**Incoming:**
- Not detected

**Outgoing:**
- Discord - `/api/webhooks/discord` (webhook URL)
  - Implementation: `trading/notifications/discord.py`
  - Auth: `DISCORD_WEBHOOK` env var

- Slack - `/api/webhooks/slack` (webhook URL)
  - Implementation: `trading/notifications/slack.py`
  - Auth: `SLACK_WEBHOOK` env var

- Telegram - Bot notifications
  - Implementation: `trading/notifications/telegram.py`
  - Auth: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` env vars

- Email - SMTP notifications
  - Implementation: `trading/notifications/email.py`
  - Auth: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_TO`, `EMAIL_FROM` env vars

---

*Integration audit: 2025-12-26*
*Update when adding/removing external services*
