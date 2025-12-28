# LLM-TradeBot Production Deployment Guide

Complete guide for deploying LLM-TradeBot to production using Docker and Docker Compose.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Secrets Management](#docker-secrets-management)
3. [Local Development Deployment](#local-development-deployment)
4. [Production VPS Deployment](#production-vps-deployment)
5. [Updating Deployment](#updating-deployment)
6. [Rollback Procedure](#rollback-procedure)
7. [Monitoring and Logs](#monitoring-and-logs)
8. [Troubleshooting](#troubleshooting)
9. [Security Best Practices](#security-best-practices)

---

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 22.04 LTS (recommended), Debian 11+, or any Linux with Docker support
- **RAM**: Minimum 4GB, recommended 8GB (PyTorch models are memory-intensive)
- **Storage**: Minimum 20GB free space (Docker images + models + data)
- **CPU**: Minimum 2 cores, recommended 4 cores
- **Network**: Stable internet connection with low latency to exchange APIs

### Required Software

1. **Docker** (version 24.0+)
   ```bash
   # Install Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   # Log out and back in for group changes to take effect
   ```

2. **Docker Compose** (version 2.20+)
   ```bash
   # Docker Compose is included with Docker Desktop
   # For Linux servers, install separately:
   sudo apt-get update
   sudo apt-get install docker-compose-plugin

   # Verify installation
   docker compose version
   ```

### Required Credentials

- **Exchange API Keys**: From your trading platform (Binance, Kraken, etc.)
  - Create API keys with trading permissions
  - Enable IP whitelisting for security
  - Start with testnet keys for testing
- **Notification Services** (optional but recommended):
  - Telegram bot token and chat ID
  - Email SMTP credentials (Gmail app password)
  - Discord/Slack webhooks

---

## Docker Secrets Management

LLM-TradeBot uses Docker secrets for secure credential management in production. Secrets are mounted at `/run/secrets/<secret_name>` inside containers with 600 permissions, ensuring API keys and passwords are never exposed in environment variables or logs.

### Secrets Overview

The system uses four Docker secrets:

| Secret | Description | Auto-Generated |
|--------|-------------|----------------|
| `exchange_api_key` | Exchange API key for trading | No (manual input) |
| `exchange_api_secret` | Exchange API secret for trading | No (manual input) |
| `kill_switch_secret` | HMAC secret for kill switch authentication | Yes (32-byte random) |
| `db_password` | PostgreSQL database password | Yes (32-byte random) |

### Initialize Secrets

Use the initialization script to create secrets with secure permissions:

```bash
# Run secrets initialization script
chmod +x scripts/init-secrets.sh
./scripts/init-secrets.sh
```

The script will:
1. Prompt for exchange API credentials (if not already set)
2. Auto-generate kill switch secret using OpenSSL (if not already set)
3. Auto-generate database password using OpenSSL (if not already set)
4. Set file permissions to 600 (owner read/write only)
5. Display generated secrets for backup

**Example output:**
```
=== Docker Secrets Initialization ===

Enter Exchange API Key: your_api_key_here
✓ Created exchange_api_key
Enter Exchange API Secret: your_api_secret_here
✓ Created exchange_api_secret
✓ Generated kill_switch_secret: Xy7Bq9...
✓ Generated db_password: Kl3mP8...

=== Secrets initialized successfully ===
Secret files created in: ./secrets
File permissions set to 600 (owner read/write only)

IMPORTANT: Never commit secrets/* to version control!
```

### Secrets Files Location

All secrets are stored in the `./secrets/` directory:

```
secrets/
├── .gitkeep                 # Keeps directory in git
├── exchange_api_key         # Exchange API key (600 permissions)
├── exchange_api_secret      # Exchange API secret (600 permissions)
├── kill_switch_secret       # Kill switch HMAC secret (600 permissions)
└── db_password              # Database password (600 permissions)
```

**IMPORTANT:** The `secrets/` directory is in `.gitignore` (except `.gitkeep`). Never commit actual secret files to version control.

### Verify Secrets Configuration

After initializing secrets, verify they're properly configured:

```bash
# Check secret files exist with correct permissions
ls -lah secrets/
# Should show 4 files with -rw------- (600) permissions

# Start containers
docker-compose up -d

# Verify secrets mounted in containers
docker exec llm-tradebot-trading-bot-1 ls -la /run/secrets/
# Should show all 4 secrets mounted

# Test secret loading (check logs)
docker-compose logs trading-bot | grep "Loaded secret"
# Should show: "Loaded secret from Docker secrets: exchange_api_key"

# Verify secrets not in git
git status | grep secrets/
# Should only show secrets/.gitkeep (if directory is tracked)
```

### Backup Secrets

**CRITICAL:** Backup your secrets securely before production deployment:

```bash
# Create encrypted backup
tar -czf secrets-backup-$(date +%Y%m%d).tar.gz secrets/
gpg -c secrets-backup-$(date +%Y%m%d).tar.gz  # Enter strong passphrase
rm secrets-backup-$(date +%Y%m%d).tar.gz

# Store encrypted .gpg file in secure location (NOT in git)
# Store passphrase separately (password manager, offline storage)
```

### Restore Secrets

To restore secrets on a new server or after data loss:

```bash
# Decrypt backup
gpg secrets-backup-YYYYMMDD.tar.gz.gpg
# Enter passphrase

# Extract secrets
tar -xzf secrets-backup-YYYYMMDD.tar.gz

# Verify permissions
chmod 600 secrets/*
ls -lah secrets/
```

### Update Secrets

To rotate API keys or update secrets:

```bash
# Update individual secret file
echo -n "new_api_key_value" > secrets/exchange_api_key
chmod 600 secrets/exchange_api_key

# Restart containers to load new secrets
docker-compose restart trading-bot

# Verify new secret loaded
docker-compose logs trading-bot | grep "Loaded secret"
```

### Fallback to Environment Variables

The system supports fallback to `.env` file for local development:

- **Production**: Uses Docker secrets from `/run/secrets/` (recommended)
- **Development**: Falls back to `.env` file if secrets not available

Priority: Docker secrets > Environment variables

This allows local development without secrets setup while maintaining security in production.

---

## Local Development Deployment

Use this for testing on your local machine before production deployment.

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd LLM-TradeBot
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.production.template .env

# Edit configuration
nano .env  # or use your preferred editor
```

**Required configuration:**
```bash
# Set your trading provider
TRADING_PROVIDER=binance_futures  # or paper for testing

# IMPORTANT: Keep testnet enabled for development
TRADING_TESTNET=true

# Add your API keys
BINANCE_FUTURES_API_KEY=your_testnet_key_here
BINANCE_FUTURES_API_SECRET=your_testnet_secret_here

# Configure risk limits
TRADING_MAX_POSITION_SIZE_USD=1000.0
TRADING_MAX_DAILY_DRAWDOWN_PCT=5.0
TRADING_MAX_OPEN_POSITIONS=3

# Enable dashboard
DASHBOARD_ENABLED=true
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5173
```

### Step 3: Create Required Directories

```bash
# Create directories for persistent data
mkdir -p models data logs
```

### Step 4: Build Docker Images

```bash
# Build with versioning
./scripts/docker-build.sh v1.0.0

# Or use Docker Compose
docker-compose build
```

### Step 5: Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Verify containers started
docker-compose ps
```

Expected output:
```
NAME                       STATUS         PORTS
llm-tradebot-dashboard     Up (healthy)   0.0.0.0:5173->5173/tcp
llm-tradebot-trading       Up (healthy)
```

### Step 6: Verify Deployment

```bash
# Check health endpoint
curl http://localhost:5173/health

# Expected response:
# {"status":"healthy","timestamp":"2025-12-28T10:30:00.000000"}

# Access dashboard
open http://localhost:5173  # macOS
# or visit http://localhost:5173 in browser

# View logs
docker-compose logs -f trading-bot
docker-compose logs -f dashboard
```

### Step 7: Test Trading (Paper Trading)

```bash
# Containers should already be running trading loop
# Monitor logs for agent decisions
docker-compose logs -f trading-bot | grep "decision"

# Check positions
docker-compose exec trading-bot python -m trading.cli positions

# Check system status
docker-compose exec trading-bot python -m trading.cli status
```

### Step 8: Stop Services

```bash
# Graceful shutdown (sends SIGTERM, triggers graceful shutdown handler)
docker-compose stop

# Remove containers (preserves data in volumes)
docker-compose down

# Remove everything including volumes (WARNING: deletes all data)
docker-compose down -v
```

---

## Production VPS Deployment

Follow these steps to deploy to a production server (VPS, dedicated server, cloud VM).

### Step 1: Prepare Production Server

```bash
# SSH to your production server
ssh user@your-server-ip

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Step 2: Transfer Code to Server

**Option A: Git Clone (Recommended)**
```bash
# On production server
cd /opt  # or your preferred location
sudo mkdir -p tradebot
sudo chown $USER:$USER tradebot
cd tradebot

git clone <repository-url> .
```

**Option B: rsync from Local Machine**
```bash
# On local machine
rsync -avz --exclude='.git' --exclude='venv' --exclude='data' \
  ./ user@your-server-ip:/opt/tradebot/
```

### Step 3: Configure Production Environment

```bash
# On production server
cd /opt/tradebot

# Copy environment template
cp .env.production.template .env

# Edit with production credentials
nano .env
```

**CRITICAL Production Settings:**
```bash
# PRODUCTION CONFIGURATION

# Use REAL exchange (not paper trading)
TRADING_PROVIDER=binance_futures

# DISABLE testnet for real trading
# WARNING: This trades with REAL MONEY!
TRADING_TESTNET=false

# Add PRODUCTION API keys (not testnet)
BINANCE_FUTURES_API_KEY=your_production_key_here
BINANCE_FUTURES_API_SECRET=your_production_secret_here

# Set appropriate position limits based on account size
TRADING_MAX_POSITION_SIZE_USD=500.0  # Adjust based on account
TRADING_MAX_DAILY_DRAWDOWN_PCT=3.0   # Conservative for production
TRADING_MAX_OPEN_POSITIONS=2         # Limit concurrent exposure

# Enable notifications for production monitoring
NOTIFICATIONS_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Generate strong kill switch secret
KILL_SWITCH_SECRET=$(openssl rand -hex 32)  # or manually generate

# Production logging
LOG_LEVEL=INFO

# Dashboard (bind to all interfaces for Docker)
DASHBOARD_ENABLED=true
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5173
```

### Step 4: Secure Environment File

```bash
# Set restrictive permissions
chmod 600 .env

# Verify permissions
ls -la .env
# Should show: -rw------- (only owner can read/write)

# Create encrypted backup
tar -czf env-backup-$(date +%Y%m%d).tar.gz .env
gpg -c env-backup-$(date +%Y%m%d).tar.gz  # Enter passphrase
rm env-backup-$(date +%Y%m%d).tar.gz
```

### Step 5: Create Directories

```bash
mkdir -p models data logs
chmod 755 models data logs
```

### Step 6: Build Production Images

```bash
# Build with production tag
./scripts/docker-build.sh production

# Or use Docker Compose
docker-compose build
```

### Step 7: Start Production Services

```bash
# Start services
docker-compose up -d

# Wait for health checks to pass (may take 60-90 seconds)
sleep 90

# Verify services are healthy
docker-compose ps
```

Expected output:
```
NAME                       STATUS         PORTS
llm-tradebot-dashboard     Up (healthy)   0.0.0.0:5173->5173/tcp
llm-tradebot-trading       Up (healthy)
```

### Step 8: Verify Production Deployment

```bash
# Check health endpoint
curl http://localhost:5173/health

# Check logs for errors
docker-compose logs --tail=100 trading-bot | grep -i error
docker-compose logs --tail=100 dashboard | grep -i error

# Monitor first few trading cycles
docker-compose logs -f trading-bot | grep "decision_executed"
```

### Step 9: Configure Firewall (Security)

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow dashboard (optional, only if external access needed)
# WARNING: Dashboard has no authentication - use VPN or SSH tunnel instead
# sudo ufw allow 5173/tcp

# Enable firewall
sudo ufw enable
```

**Recommended: Access dashboard via SSH tunnel**
```bash
# On local machine, create SSH tunnel
ssh -L 5173:localhost:5173 user@your-server-ip

# Then access dashboard at http://localhost:5173 on local machine
```

### Step 10: Set Up Auto-Start on Reboot

```bash
# Create systemd service
sudo nano /etc/systemd/system/llm-tradebot.service
```

Add this content:
```ini
[Unit]
Description=LLM-TradeBot Docker Compose
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/tradebot
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose stop
User=your-username

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable llm-tradebot
sudo systemctl start llm-tradebot

# Check status
sudo systemctl status llm-tradebot
```

---

## Updating Deployment

Rolling out new code versions without downtime.

### Step 1: Pull Latest Code

```bash
cd /opt/tradebot
git pull origin main
```

### Step 2: Rebuild Images

```bash
# Build new images
docker-compose build

# Or with versioning
./scripts/docker-build.sh v1.1.0
```

### Step 3: Graceful Restart

```bash
# This will recreate containers with new code
# Graceful shutdown handler ensures positions are closed
docker-compose up -d

# Monitor logs during restart
docker-compose logs -f
```

### Step 4: Verify Update

```bash
# Check health
curl http://localhost:5173/health

# Verify version (if you added version endpoint)
# curl http://localhost:5173/api/version

# Monitor for errors
docker-compose logs --tail=50 trading-bot | grep -i error
```

---

## Rollback Procedure

If deployment fails or introduces bugs, rollback to previous version.

### Step 1: Identify Last Working Version

```bash
# Check git history
git log --oneline -10

# Example output:
# abc123f feat: Add new indicator
# def456g fix: API timeout handling  <- Last known good version
```

### Step 2: Checkout Previous Version

```bash
# Checkout last working commit
git checkout def456g

# Or checkout previous tag
git checkout v1.0.0
```

### Step 3: Rebuild and Restart

```bash
# Rebuild images with old code
docker-compose build

# Restart services
docker-compose up -d
```

### Step 4: Verify Rollback

```bash
# Check health
curl http://localhost:5173/health

# Monitor logs
docker-compose logs -f trading-bot

# Verify positions are correct
docker-compose exec trading-bot python -m trading.cli positions
```

### Step 5: Return to Latest (After Fix)

```bash
# Return to main branch when ready
git checkout main
docker-compose build
docker-compose up -d
```

---

## Monitoring and Logs

### View Live Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f trading-bot
docker-compose logs -f dashboard

# With timestamps
docker-compose logs -f --timestamps

# Tail last N lines
docker-compose logs --tail=100 trading-bot
```

### Search Logs for Errors

```bash
# Critical errors
docker-compose logs | grep -i critical

# API errors
docker-compose logs | grep -i "api_error"

# Risk vetoes
docker-compose logs | grep -i "veto"

# Kill switch activations
docker-compose logs | grep -i "kill_switch"
```

### Export Logs

```bash
# Export to file
docker-compose logs > logs-$(date +%Y%m%d-%H%M%S).txt

# Export last 24 hours
docker-compose logs --since 24h > logs-recent.txt
```

### Monitor Container Resources

```bash
# Real-time resource usage
docker stats

# Specific containers
docker stats llm-tradebot-trading llm-tradebot-dashboard
```

### Monitor Health Status

```bash
# Check health status
docker inspect llm-tradebot-dashboard --format='{{.State.Health.Status}}'

# View health check history
docker inspect llm-tradebot-trading --format='{{json .State.Health}}' | python3 -m json.tool
```

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker-compose logs trading-bot
```

**Common issues:**
- Missing `.env` file → Copy from `.env.production.template`
- Invalid API keys → Verify keys in exchange account
- Insufficient memory → Increase Docker memory limit or upgrade server
- Port conflict → Change `DASHBOARD_PORT` in `.env`

### Health Check Failing

**Manually test health endpoint:**
```bash
# From inside container
docker-compose exec dashboard curl http://localhost:5173/health

# From host
curl http://localhost:5173/health
```

**Common issues:**
- Dashboard not binding to 0.0.0.0 → Check `DASHBOARD_HOST=0.0.0.0` in `.env`
- Kill switch active → Check logs, reset if needed
- System health critical → Check monitoring logs for specific issue

### Dashboard Not Accessible

**Check if dashboard is running:**
```bash
docker-compose ps dashboard
```

**Test port locally:**
```bash
curl http://localhost:5173/health
```

**If accessible locally but not remotely:**
- Firewall blocking port → Configure UFW/iptables
- Use SSH tunnel instead (more secure)

### Models Not Loading

**Check volume mounts:**
```bash
docker-compose config | grep -A5 volumes
```

**Check models directory:**
```bash
ls -la models/
docker-compose exec trading-bot ls -la /app/models/
```

**Common issues:**
- Models directory empty → Download/train models first
- Permission errors → Fix with `chown -R 1000:1000 models/`

### Out of Memory

**Check memory usage:**
```bash
docker stats --no-stream
```

**Solutions:**
- Reduce `TRADING_MAX_OPEN_POSITIONS`
- Increase server RAM
- Add swap space
- Reduce Docker memory limit if using Mac/Windows

### API Errors

**Check API connectivity:**
```bash
docker-compose exec trading-bot python -m trading.cli status
```

**Common issues:**
- Invalid API keys → Verify in exchange account
- IP not whitelisted → Add server IP to exchange whitelist
- Rate limiting → Reduce request frequency
- Exchange maintenance → Check exchange status page

---

## Security Best Practices

### API Key Security

1. **Never commit `.env` to version control**
   - Already in `.gitignore`, verify with `git status`

2. **Use minimum required permissions**
   - Trading only (no withdrawal permissions)
   - Enable IP whitelisting

3. **Rotate keys regularly**
   - Every 90 days minimum
   - After any suspected compromise

### Server Security

1. **Keep system updated**
   ```bash
   sudo apt-get update
   sudo apt-get upgrade -y
   ```

2. **Configure firewall**
   ```bash
   sudo ufw enable
   sudo ufw allow 22/tcp
   # Don't expose 5173 publicly
   ```

3. **Disable root SSH**
   ```bash
   sudo nano /etc/ssh/sshd_config
   # Set: PermitRootLogin no
   sudo systemctl restart sshd
   ```

4. **Use SSH keys (not passwords)**
   ```bash
   # On local machine
   ssh-copy-id user@your-server-ip
   ```

### Application Security

1. **Set strong kill switch secret**
   ```bash
   # Generate random secret
   openssl rand -hex 32
   # Add to .env: KILL_SWITCH_SECRET=<generated-secret>
   ```

2. **Enable notifications**
   - Get alerted on critical events
   - Monitor for unauthorized access attempts

3. **Monitor logs regularly**
   - Check for suspicious activity
   - Review failed authentication attempts

4. **Backup `.env` securely**
   ```bash
   # Encrypted backup
   gpg -c .env
   # Store gpg file in secure location
   ```

### Docker Security

1. **Run as non-root user** (already configured in Dockerfile)

2. **Keep Docker updated**
   ```bash
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

3. **Scan images for vulnerabilities**
   ```bash
   docker scan llm-tradebot:latest
   ```

---

## Support and Resources

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **Issues**: GitHub Issues
- **Logs**: `docker-compose logs`

---

**CRITICAL REMINDERS:**

1. **Test with testnet FIRST** - Never deploy to production without thorough testnet testing
2. **Start with small positions** - Use conservative `TRADING_MAX_POSITION_SIZE_USD`
3. **Monitor closely** - Check logs and positions regularly, especially first 24-48 hours
4. **Enable kill switch** - Keep `KILL_SWITCH_SECRET` in `.env` for emergency shutdown
5. **Backup everything** - `.env` file, state files, model files

**Trading involves substantial risk of loss. Use at your own risk.**
