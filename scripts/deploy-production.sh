#!/bin/bash
# Production Deployment Script
# Deploy to production server with safety checks
#
# Usage:
#   ./scripts/deploy-production.sh
#
# IMPORTANT: Run this ON the production server, not locally
# Transfer code to server first using git clone or rsync

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${RED}[WARNING]${NC} $1"
}

# Header
print_header "LLM-TradeBot Production Deployment"

print_warning "This script will deploy trading bot to PRODUCTION"
print_warning "Ensure you have completed the pre-deployment checklist:"
print_warning "  - See docs/PRODUCTION_CHECKLIST.md"
echo ""
read -p "Have you completed the pre-deployment checklist? (yes/no): " checklist_done
if [ "$checklist_done" != "yes" ]; then
    print_error "Please complete the checklist first."
    print_info "See docs/PRODUCTION_CHECKLIST.md"
    exit 1
fi

# Safety check: .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found."
    print_info "Copy .env.production.template to .env and configure it:"
    echo "  cp .env.production.template .env"
    echo "  nano .env"
    exit 1
fi

# Safety check: verify environment file permissions
ENV_PERMS=$(stat -c "%a" .env 2>/dev/null || stat -f "%A" .env 2>/dev/null)
if [ "$ENV_PERMS" != "600" ]; then
    print_warning ".env file permissions are not secure (currently $ENV_PERMS)"
    print_info "Setting permissions to 600 (owner read/write only)..."
    chmod 600 .env
    print_success "Permissions updated"
fi

# Safety check: testnet mode
if ! grep -q "TRADING_TESTNET" .env; then
    print_error "TRADING_TESTNET not found in .env"
    print_error "Please add TRADING_TESTNET=true or TRADING_TESTNET=false"
    exit 1
fi

if grep -q "TRADING_TESTNET=true" .env; then
    print_info "TRADING_TESTNET is set to true (testnet mode)"
    print_info "This is safe for testing but will not trade on real markets"
    echo ""
    read -p "Continue with testnet mode? (yes/no): " continue_testnet
    if [ "$continue_testnet" != "yes" ]; then
        print_info "Deployment cancelled."
        exit 0
    fi
elif grep -q "TRADING_TESTNET=false" .env; then
    print_warning "╔════════════════════════════════════════════════════════╗"
    print_warning "║  WARNING: TRADING_TESTNET IS SET TO FALSE            ║"
    print_warning "║                                                        ║"
    print_warning "║  THIS WILL TRADE WITH REAL MONEY!                     ║"
    print_warning "║                                                        ║"
    print_warning "║  Financial losses are possible.                       ║"
    print_warning "║  Make sure you have tested on testnet first.          ║"
    print_warning "╚════════════════════════════════════════════════════════╝"
    echo ""
    read -p "Are you ABSOLUTELY SURE you want to continue? (type 'I ACCEPT THE RISK'): " confirm_risk
    if [ "$confirm_risk" != "I ACCEPT THE RISK" ]; then
        print_info "Deployment cancelled (correct decision for safety)."
        exit 0
    fi
fi

# Safety check: provider configured
if ! grep -q "TRADING_PROVIDER" .env; then
    print_error "TRADING_PROVIDER not configured in .env"
    exit 1
fi

PROVIDER=$(grep "TRADING_PROVIDER" .env | cut -d'=' -f2)
print_info "Trading provider: $PROVIDER"

# Safety check: API keys present
if [ "$PROVIDER" != "paper" ]; then
    if ! grep -q "${PROVIDER^^}_API_KEY" .env 2>/dev/null; then
        print_error "API keys for provider '$PROVIDER' not found in .env"
        exit 1
    fi
fi

# Create required directories
print_info "Creating required directories..."
mkdir -p models data logs
print_success "Directories created"

# Build Docker images
print_header "Building Production Images"
print_info "This may take several minutes..."

# Tag with production and timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
print_info "Building with tags: production, production-$TIMESTAMP"

docker-compose build

if [ $? -eq 0 ]; then
    print_success "Docker images built successfully"

    # Tag with timestamp for rollback capability
    docker tag llm-tradebot:latest llm-tradebot:production-$TIMESTAMP
    print_success "Tagged as llm-tradebot:production-$TIMESTAMP"
else
    print_error "Docker build failed"
    exit 1
fi

# Stop existing services (if running)
if docker-compose ps | grep -q "Up"; then
    print_info "Stopping existing services..."
    docker-compose stop
    print_success "Existing services stopped"
fi

# Start production services
print_header "Starting Production Services"

docker-compose up -d

if [ $? -eq 0 ]; then
    print_success "Services started"
else
    print_error "Failed to start services"
    print_info "Check logs: docker-compose logs"
    exit 1
fi

# Wait for health checks
print_info "Waiting for health checks to pass..."
print_info "Dashboard: 60 seconds max"
print_info "Trading bot: 90 seconds max (model loading)"

sleep 10

# Function to wait for container to be healthy
wait_for_healthy() {
    local container=$1
    local timeout=$2
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if docker inspect --format='{{.State.Health.Status}}' $container 2>/dev/null | grep -q "healthy"; then
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    echo ""
    return 1
}

# Check dashboard health
echo -n "Dashboard"
if wait_for_healthy "llm-tradebot-dashboard" 60; then
    print_success "Dashboard healthy"
else
    print_error "Dashboard health check failed"
    print_info "Checking logs..."
    docker-compose logs --tail=50 dashboard
    print_error "Deployment failed - services may be unhealthy"
    exit 1
fi

# Check trading bot health
echo -n "Trading bot"
if wait_for_healthy "llm-tradebot-trading" 90; then
    print_success "Trading bot healthy"
else
    print_error "Trading bot health check failed"
    print_info "Checking logs..."
    docker-compose logs --tail=50 trading-bot
    print_error "Deployment failed - services may be unhealthy"
    exit 1
fi

# Verify services
print_header "Verifying Deployment"

# Check health endpoint
print_info "Testing health endpoint..."
if curl -f http://localhost:5173/health &> /dev/null; then
    print_success "Health endpoint responding"
    HEALTH_RESPONSE=$(curl -s http://localhost:5173/health)
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    print_error "Health endpoint not responding"
    print_warning "Deployment may have issues"
fi

# Check for errors in logs
print_info "Checking logs for critical errors..."
ERROR_COUNT=$(docker-compose logs | grep -i "critical\|fatal" | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    print_warning "Found $ERROR_COUNT critical/fatal log entries"
    print_info "Review logs: docker-compose logs | grep -i 'critical\|fatal'"
else
    print_success "No critical errors in logs"
fi

# Display container status
print_info "Container status:"
docker-compose ps

# Deployment complete
print_header "Production Deployment Complete"

print_success "LLM-TradeBot is now running in production!"
echo ""
print_info "Deployment details:"
echo "  Timestamp: $TIMESTAMP"
echo "  Provider: $PROVIDER"
echo "  Testnet: $(grep TRADING_TESTNET .env | cut -d'=' -f2)"
echo ""
print_info "Monitoring:"
echo "  Dashboard: http://localhost:5173 (use SSH tunnel for remote access)"
echo "  Health: curl http://localhost:5173/health"
echo "  Logs: docker-compose logs -f"
echo ""
print_info "Management commands:"
echo "  View positions:  docker-compose exec trading-bot python -m trading.cli positions"
echo "  Check status:    docker-compose exec trading-bot python -m trading.cli status"
echo "  Stop services:   docker-compose stop"
echo "  View logs:       docker-compose logs -f trading-bot"
echo ""

# Post-deployment checklist reminder
print_header "Post-Deployment Tasks"
echo "[ ] Monitor logs for first 15 minutes"
echo "[ ] Verify alert notifications working"
echo "[ ] Check positions endpoint"
echo "[ ] Test graceful shutdown (when appropriate)"
echo "[ ] Document deployment in team chat/log"
echo ""
print_info "See docs/PRODUCTION_CHECKLIST.md for complete post-deployment checklist"
echo ""

# Critical reminders
print_warning "IMPORTANT REMINDERS:"
echo "1. Monitor logs continuously for first 24-48 hours"
echo "2. Keep kill switch endpoint accessible for emergencies"
echo "3. Verify notifications are being received"
echo "4. Check positions and metrics regularly"
echo "5. Have rollback plan ready"
echo ""
print_info "For rollback: docker-compose stop && git checkout <previous-commit> && docker-compose build && docker-compose up -d"
echo ""
print_success "Deployment script completed successfully"
