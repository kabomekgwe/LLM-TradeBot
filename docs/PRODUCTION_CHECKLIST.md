# Production Deployment Checklist

Complete pre-flight checklist before deploying LLM-TradeBot to production with real money.

## Pre-Deployment (Before Building)

### Code Quality

- [ ] All tests passing
  ```bash
  pytest tests/ -v
  ```
- [ ] No critical/high severity security issues
  ```bash
  # Run security scan if available
  docker scan llm-tradebot:latest
  ```
- [ ] Code reviewed and approved
- [ ] Latest changes committed to main branch
- [ ] Git tag created for this release
  ```bash
  git tag -a v1.0.0 -m "Production release 1.0.0"
  git push origin v1.0.0
  ```

### Phase 10 Monitoring Verified

- [ ] Metrics tracking tested (Sharpe, drawdown, win rate)
  ```bash
  # Verify metrics endpoint
  curl http://localhost:5173/api/v1/metrics/realtime
  ```
- [ ] Health monitoring tested (kill switch, circuit breaker)
  ```bash
  # Verify health endpoint
  curl http://localhost:5173/api/v1/health/status
  ```
- [ ] Alert system tested (Telegram, Email, Slack)
  ```bash
  # Send test alert
  curl -X POST http://localhost:5173/api/v1/alerts/test \
    -H "X-HMAC-Signature: <signature>" \
    -H "Content-Type: application/json" \
    -d '{"channel": "all"}'
  ```
- [ ] Alert thresholds configured appropriately
  - High drawdown alert threshold: `____%`
  - Win rate alert threshold: `____%`
  - Loss streak alert threshold: `___` consecutive losses

### Phase 9 Safety Controls Verified

- [ ] Kill switch tested
  - Trigger endpoint tested
  - HMAC authentication working
  - Trading stops immediately when triggered
  - Reset functionality working
- [ ] Circuit breaker tested
  - Trips on excessive drawdown
  - Trips on API error threshold
  - Trips on consecutive losses
  - Resets correctly after cooldown
- [ ] Position limits tested
  - Max position size enforced: `$____`
  - Max open positions enforced: `___`
  - Max daily drawdown enforced: `____%`

### Configuration

- [ ] `.env` file created from `.env.production.template`
- [ ] **CRITICAL**: `TRADING_TESTNET` is set correctly
  - [ ] `TRADING_TESTNET=true` for testnet (SAFE)
  - [ ] `TRADING_TESTNET=false` for production (REAL MONEY!)
- [ ] Trading provider configured
  - Provider: `_______________` (binance_futures, kraken, etc.)
  - Exchange account verified
- [ ] API keys valid and tested
  - [ ] API keys have correct permissions (trading only, no withdrawal)
  - [ ] IP whitelisting enabled on exchange
  - [ ] Keys tested with small test order on testnet
- [ ] Risk parameters set appropriately
  - Max position size: `$____` (appropriate for account size)
  - Max daily drawdown: `____%` (conservative for production)
  - Max open positions: `___` (limit concurrent exposure)
  - Decision threshold: `0.__` (confidence level 0.0-1.0)
- [ ] Kill switch secret generated
  ```bash
  openssl rand -hex 32
  # Add to .env: KILL_SWITCH_SECRET=<generated-secret>
  ```
- [ ] Notification services configured
  - [ ] Telegram bot token and chat ID
  - [ ] Email SMTP credentials (if using email)
  - [ ] Discord/Slack webhooks (if using)
  - [ ] Test notifications sent successfully
- [ ] Dashboard configuration
  - [ ] `DASHBOARD_ENABLED=true`
  - [ ] `DASHBOARD_HOST=0.0.0.0` (for Docker)
  - [ ] `DASHBOARD_PORT=5173`

### Environment Security

- [ ] `.env` file permissions set to 600 (owner read/write only)
  ```bash
  chmod 600 .env
  ls -la .env  # Should show: -rw-------
  ```
- [ ] `.env` file backed up securely
  ```bash
  # Encrypted backup
  gpg -c .env
  # Store .env.gpg in secure location (not in git!)
  ```
- [ ] `.env` file NOT in git
  ```bash
  git status .env  # Should be ignored
  ```

### Server Preparation

- [ ] Production server provisioned
  - OS: `_______________` (Ubuntu 22.04 recommended)
  - RAM: `____ GB` (minimum 4GB, recommended 8GB)
  - Storage: `____ GB` free (minimum 20GB)
  - CPU: `____` cores (minimum 2, recommended 4)
- [ ] Docker installed and running
  ```bash
  docker --version  # Should be 24.0+
  ```
- [ ] Docker Compose installed
  ```bash
  docker compose version  # Should be 2.20+
  ```
- [ ] Firewall configured
  ```bash
  sudo ufw status
  # Should allow SSH (22), block 5173 (use SSH tunnel instead)
  ```
- [ ] SSH key authentication enabled
- [ ] Root SSH disabled
- [ ] System updated
  ```bash
  sudo apt-get update && sudo apt-get upgrade -y
  ```

### Backup Plan

- [ ] Current production state backed up (if updating existing deployment)
  ```bash
  # Backup data directory
  tar -czf backup-$(date +%Y%m%d).tar.gz data/ models/ .env
  ```
- [ ] Rollback procedure documented
- [ ] Last known good commit hash recorded: `__________`
- [ ] Downtime window planned (if applicable): `__________`

---

## During Deployment

### Build and Start

- [ ] Code transferred to production server
  - [ ] Via git clone (recommended), or
  - [ ] Via rsync from local machine
- [ ] Required directories created
  ```bash
  mkdir -p models data logs
  ```
- [ ] Docker images built successfully
  ```bash
  ./scripts/docker-build.sh production
  # Or: docker-compose build
  ```
- [ ] Build completed without errors
- [ ] Image size acceptable (<1.5GB)
  ```bash
  docker images llm-tradebot:latest
  ```

### Container Startup

- [ ] Containers started
  ```bash
  docker-compose up -d
  ```
- [ ] All containers show "Up" status
  ```bash
  docker-compose ps
  ```
- [ ] Dashboard container healthy
  ```bash
  docker inspect llm-tradebot-dashboard --format='{{.State.Health.Status}}'
  # Should show: healthy
  ```
- [ ] Trading bot container healthy
  ```bash
  docker inspect llm-tradebot-trading --format='{{.State.Health.Status}}'
  # Should show: healthy
  ```

### Health Verification

- [ ] Health endpoint responding
  ```bash
  curl http://localhost:5173/health
  # Expected: {"status":"healthy","timestamp":"..."}
  ```
- [ ] Dashboard accessible
  - [ ] Via SSH tunnel (secure): `ssh -L 5173:localhost:5173 user@server`
  - [ ] Then access: http://localhost:5173
- [ ] System health status checked
  ```bash
  curl http://localhost:5173/api/v1/health/status
  ```
- [ ] Safety controls status checked
  ```bash
  curl http://localhost:5173/api/v1/health/safety
  ```

### Log Verification

- [ ] No critical errors in logs
  ```bash
  docker-compose logs --tail=100 | grep -i critical
  # Should be empty or only expected warnings
  ```
- [ ] No API errors in logs
  ```bash
  docker-compose logs --tail=100 | grep -i "api_error"
  ```
- [ ] Agents initializing correctly
  ```bash
  docker-compose logs | grep "agent_start"
  ```
- [ ] Provider connected
  ```bash
  docker-compose logs | grep "provider_initialized"
  ```

---

## Post-Deployment

### Monitoring (First 15 Minutes)

- [ ] Monitor container status
  ```bash
  watch -n 10 docker-compose ps
  ```
- [ ] Watch logs for errors
  ```bash
  docker-compose logs -f trading-bot
  ```
- [ ] Verify trading activity (if applicable)
  ```bash
  docker-compose logs | grep "decision_executed"
  ```
- [ ] Check resource usage
  ```bash
  docker stats
  ```

### Functional Testing

- [ ] Positions endpoint working
  ```bash
  docker-compose exec trading-bot python -m trading.cli positions
  ```
- [ ] Status endpoint working
  ```bash
  docker-compose exec trading-bot python -m trading.cli status
  ```
- [ ] Alert notifications received (send test alert)
- [ ] Dashboard shows correct data
  - Positions displayed
  - Metrics updating
  - Health indicators correct

### Data Persistence

- [ ] Restart container and verify state preserved
  ```bash
  docker-compose restart trading-bot
  # Wait for restart
  docker-compose exec trading-bot python -m trading.cli status
  # Verify state preserved
  ```
- [ ] Check data directory has files
  ```bash
  ls -la data/
  # Should contain state files, logs
  ```
- [ ] Check models directory accessible
  ```bash
  ls -la models/
  # Should contain .pth/.pkl files if applicable
  ```

### Graceful Shutdown Test

- [ ] Test graceful shutdown
  ```bash
  docker-compose stop
  ```
- [ ] Check logs for shutdown messages
  ```bash
  docker logs llm-tradebot-trading | grep -i "shutdown\|graceful"
  ```
- [ ] Verify positions closed (if any were open)
- [ ] Restart services
  ```bash
  docker-compose up -d
  ```

### Long-Term Monitoring Setup

- [ ] Auto-start on reboot configured
  ```bash
  sudo systemctl enable llm-tradebot
  sudo systemctl status llm-tradebot
  ```
- [ ] Log rotation configured (Docker handles this by default)
- [ ] Monitoring alerts verified
  - High drawdown alert
  - Win rate drop alert
  - Kill switch activation alert
  - Circuit breaker trip alert
- [ ] Daily health check scheduled (optional)
  ```bash
  # Add to crontab: 0 9 * * * /path/to/health-check-script.sh
  ```

### Documentation

- [ ] Deployment details recorded
  - Date deployed: `__________`
  - Version deployed: `__________`
  - Commit hash: `__________`
  - Server details: `__________`
  - Initial account balance: `$__________`
- [ ] Team notified of deployment
  - Slack/Discord message sent
  - Or email sent
- [ ] Runbook updated with any changes

---

## Final Verification

### Security Check

- [ ] `.env` file secured (chmod 600)
- [ ] No secrets in logs
  ```bash
  docker-compose logs | grep -i "api_key\|secret"
  # Should NOT show actual keys
  ```
- [ ] Firewall properly configured
- [ ] SSH access secure (key-based, root disabled)
- [ ] Server access limited to authorized personnel

### Risk Management

- [ ] Position limits appropriate for account size
- [ ] Stop-loss configured (if applicable)
- [ ] Daily drawdown limit set conservatively
- [ ] Kill switch accessible and tested
- [ ] Manual intervention plan documented

### Readiness Confirmation

- [ ] **I confirm all checklist items are complete**
- [ ] **I understand this trades with real money (if testnet=false)**
- [ ] **I have tested thoroughly on testnet first**
- [ ] **I accept the risk of financial loss**
- [ ] **I have backup and rollback plans ready**

---

## Sign-Off

**Deployed by:** `_______________`
**Date:** `_______________`
**Time:** `_______________`
**Version:** `_______________`
**Commit hash:** `_______________`

**Approved by:** `_______________`
**Date:** `_______________`

---

## Emergency Contacts

**On-Call Engineer:** `_______________`
**Phone:** `_______________`
**Email:** `_______________`

**Backup Contact:** `_______________`
**Phone:** `_______________`
**Email:** `_______________`

---

## Notes

(Space for any additional notes, observations, or special considerations for this deployment)

```
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
```

---

**CRITICAL REMINDERS:**

1. **Never skip testnet testing** - Always test with testnet=true first
2. **Start small** - Use conservative position sizes initially
3. **Monitor closely** - Watch first 24-48 hours continuously
4. **Have kill switch ready** - Know how to emergency stop trading
5. **Document everything** - Record all decisions and changes

**Trading involves substantial risk of loss. This checklist does not guarantee profitable trading or eliminate all risks.**
