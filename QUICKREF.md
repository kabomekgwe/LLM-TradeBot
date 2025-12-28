# LLM-TradeBot Quick Reference Card

One-page reference for common commands and operations.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/kabomekgwe/LLM-TradeBot.git
cd LLM-TradeBot
cp .env.example .env

# 2. Start with Docker
docker-compose up -d

# 3. View dashboard
open http://localhost:5173
```

## 📦 Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f trading-bot
docker-compose logs -f postgres
docker-compose logs -f dashboard

# Restart service
docker-compose restart trading-bot

# Check service status
docker-compose ps

# Execute command in container
docker exec -it llm-tradebot-trading-bot-1 bash

# View resource usage
docker stats
```

## 🗄️ Database Commands

```bash
# Connect to PostgreSQL
docker exec -it llm-tradebot-postgres-1 psql -U tradingbot -d tradingbot

# Run migrations
docker-compose run --rm trading-bot alembic upgrade head

# Rollback migration
docker-compose run --rm trading-bot alembic downgrade -1

# View trade history
docker exec -it llm-tradebot-postgres-1 psql -U tradingbot -d tradingbot \
  -c "SELECT * FROM trade_history ORDER BY timestamp DESC LIMIT 10;"

# Check hypertable status
docker exec -it llm-tradebot-postgres-1 psql -U tradingbot -d tradingbot \
  -c "SELECT * FROM timescaledb_information.hypertables;"
```

## 🔐 Secrets Management

```bash
# Initialize secrets
./scripts/init-secrets.sh

# View secrets (be careful!)
cat secrets/exchange_api_key
cat secrets/kill_switch_secret

# Test secrets loading
docker exec llm-tradebot-trading-bot-1 cat /run/secrets/exchange_api_key

# Regenerate kill switch secret
openssl rand -base64 32 > secrets/kill_switch_secret
chmod 600 secrets/kill_switch_secret
```

## 🛡️ Safety Controls

```bash
# Activate kill switch
SIGNATURE=$(echo -n "activate:emergency" | openssl dgst -sha256 \
  -hmac "$(cat secrets/kill_switch_secret)" -binary | base64)

curl -X POST http://localhost:5173/api/v1/safety/kill-switch/activate \
  -H "X-Kill-Switch-Signature: $SIGNATURE" \
  -d '{"reason": "Emergency stop", "close_positions": true}'

# Deactivate kill switch
SIGNATURE=$(echo -n "deactivate" | openssl dgst -sha256 \
  -hmac "$(cat secrets/kill_switch_secret)" -binary | base64)

curl -X POST http://localhost:5173/api/v1/safety/kill-switch/deactivate \
  -H "X-Kill-Switch-Signature: $SIGNATURE"

# Check kill switch status
curl http://localhost:5173/api/v1/safety/kill-switch/status

# Check circuit breaker status
curl http://localhost:5173/api/v1/safety/circuit-breaker/status

# Reset circuit breaker
curl -X POST http://localhost:5173/api/v1/safety/circuit-breaker/reset
```

## 📊 Monitoring & Metrics

```bash
# Get real-time metrics
curl http://localhost:5173/api/v1/metrics/realtime

# Get system health
curl http://localhost:5173/api/v1/health/status

# Get safety status
curl http://localhost:5173/api/v1/health/safety

# Test alerts
curl -X POST http://localhost:5173/api/v1/alerts/test \
  -H "Content-Type: application/json" \
  -d '{"channel": "slack", "message": "Test alert"}'
```

## 🤖 ML Model Serving

```bash
# List available models
curl http://localhost:5173/api/v1/ml/models

# Get prediction
curl -X POST http://localhost:5173/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgboost_model.pkl",
    "model_type": "xgboost",
    "features": [[1.5, 2.3, 0.8, 1.2, 0.5]]
  }'

# Check model cache
curl http://localhost:5173/api/v1/ml/cache

# Clear model cache
curl -X DELETE http://localhost:5173/api/v1/ml/cache
```

## 📝 Logging

```bash
# View JSON logs
docker-compose logs trading-bot | tail -50

# Filter by correlation ID
docker-compose logs trading-bot | jq 'select(.correlation_id == "abc-123")'

# Filter by log level
docker-compose logs trading-bot | jq 'select(.level == "ERROR")'

# Search for trade executions
docker-compose logs trading-bot | jq 'select(.message | contains("Executing trade"))'

# View logs for specific symbol
docker-compose logs trading-bot | jq 'select(.symbol == "BTC/USDT")'

# Save logs to file
docker-compose logs trading-bot > logs/trading-$(date +%Y%m%d).log
```

## 🧪 Testing & Debugging

```bash
# Run health check
curl http://localhost:5173/health

# Test database connection
docker exec llm-tradebot-trading-bot-1 python -c "
from trading.database.connection import engine
with engine.connect() as conn:
    print('Database connected!')
"

# Test exchange connection
docker exec llm-tradebot-trading-bot-1 python -c "
from trading.config import TradingConfig
from trading.providers.factory import create_provider
config = TradingConfig.from_env()
provider = create_provider(config)
print(provider.fetch_balance())
"

# Enable debug logging
# Edit .env: LOG_LEVEL=DEBUG
docker-compose restart trading-bot
```

## 🔧 Maintenance

```bash
# Backup database
docker exec llm-tradebot-postgres-1 pg_dump -U tradingbot tradingbot \
  > backups/tradingbot-$(date +%Y%m%d).sql

# Restore database
docker exec -i llm-tradebot-postgres-1 psql -U tradingbot tradingbot \
  < backups/tradingbot-20251228.sql

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Prune Docker images
docker system prune -a

# Update to latest code
git pull origin main
docker-compose build
docker-compose up -d
```

## 📈 Common SQL Queries

```sql
-- Get trade performance summary
SELECT
  symbol,
  COUNT(*) as total_trades,
  SUM(CASE WHEN won THEN 1 ELSE 0 END) as wins,
  ROUND(AVG(realized_pnl), 2) as avg_pnl,
  ROUND(SUM(realized_pnl), 2) as total_pnl
FROM trade_history
GROUP BY symbol
ORDER BY total_pnl DESC;

-- Get daily P&L
SELECT
  time_bucket('1 day', timestamp) as day,
  COUNT(*) as trades,
  ROUND(SUM(realized_pnl), 2) as daily_pnl
FROM trade_history
GROUP BY day
ORDER BY day DESC
LIMIT 30;

-- Get recent winning trades
SELECT trade_id, symbol, timestamp, side, realized_pnl
FROM trade_history
WHERE won = true
ORDER BY timestamp DESC
LIMIT 10;

-- Get worst drawdown periods
SELECT
  time_bucket('1 hour', timestamp) as hour,
  MIN(realized_pnl) as worst_loss,
  COUNT(*) as trades
FROM trade_history
WHERE realized_pnl < 0
GROUP BY hour
ORDER BY worst_loss ASC
LIMIT 10;
```

## 🌐 Environment Variables

```bash
# Exchange Configuration
EXCHANGE=binance
EXCHANGE_TESTNET=true
EXCHANGE_API_KEY=your_key
EXCHANGE_API_SECRET=your_secret

# Trading Settings
SYMBOLS=BTC/USDT,ETH/USDT
TIMEFRAME=1h
POSITION_SIZE=0.01

# Safety Limits
MAX_POSITION_SIZE=0.1
MAX_DAILY_LOSS=100
MAX_POSITIONS=5
MAX_DRAWDOWN=0.10

# LLM Configuration
OPENAI_API_KEY=your_key
LLM_MODEL=gpt-4

# Database
DATABASE_URL=postgresql://tradingbot:password@postgres:5432/tradingbot

# Logging
LOG_LEVEL=INFO

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
EMAIL_SMTP_HOST=smtp.gmail.com
TELEGRAM_BOT_TOKEN=your_token
```

## 🆘 Emergency Procedures

### Trading Bot Misbehaving
```bash
# 1. Activate kill switch (closes all positions)
curl -X POST http://localhost:5173/api/v1/safety/kill-switch/activate \
  -H "X-Kill-Switch-Signature: $SIGNATURE" \
  -d '{"reason": "Emergency", "close_positions": true}'

# 2. Stop containers
docker-compose down

# 3. Check logs for errors
docker-compose logs trading-bot | tail -100

# 4. Verify positions closed on exchange
# (login to exchange web interface)
```

### Database Issues
```bash
# 1. Check database is running
docker-compose ps postgres

# 2. Restart database
docker-compose restart postgres

# 3. Check database logs
docker-compose logs postgres | tail -50

# 4. Restore from backup if corrupted
docker-compose down
docker volume rm llm-tradebot_postgres_data
docker-compose up -d postgres
# Then restore backup (see Maintenance section)
```

### High CPU/Memory Usage
```bash
# 1. Check resource usage
docker stats

# 2. Restart specific service
docker-compose restart trading-bot

# 3. Clear model cache
curl -X DELETE http://localhost:5173/api/v1/ml/cache

# 4. Reduce position size
# Edit .env: POSITION_SIZE=0.001
docker-compose restart trading-bot
```

## 📚 Further Reading

- **Full Tutorial**: [TUTORIAL.md](TUTORIAL.md)
- **Deployment Guide**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Production Checklist**: [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)

---

**⚠️ Remember:** Always test in paper trading mode (testnet) before deploying to production!
