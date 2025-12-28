#!/bin/bash
# Local Development Deployment Script
# Quick deployment for local testing
#
# Usage:
#   ./scripts/deploy-local.sh

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

# Header
print_header "LLM-TradeBot Local Deployment"

# Check prerequisites
print_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Prerequisites met"

# Check for .env file
if [ ! -f ".env" ]; then
    print_info ".env file not found. Creating from template..."
    cp .env.production.template .env

    print_info "Please edit .env and configure your settings:"
    echo "  - Add API keys (or use TRADING_PROVIDER=paper for testing)"
    echo "  - Keep TRADING_TESTNET=true for safety"
    echo "  - Configure risk parameters"
    echo ""
    read -p "Press Enter after editing .env to continue..."
fi

# Verify testnet mode
if grep -q "TRADING_TESTNET=false" .env; then
    print_error "WARNING: TRADING_TESTNET is set to false!"
    print_error "This will trade with REAL MONEY!"
    echo ""
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm
    if [ "$confirm" != "yes" ]; then
        print_info "Deployment cancelled."
        exit 0
    fi
fi

# Create required directories
print_info "Creating required directories..."
mkdir -p models data logs
print_success "Directories created"

# Build Docker images
print_header "Building Docker Images"
print_info "This may take several minutes on first build..."

docker-compose build

if [ $? -eq 0 ]; then
    print_success "Docker images built successfully"
else
    print_error "Docker build failed"
    exit 1
fi

# Start services
print_header "Starting Services"

docker-compose up -d

if [ $? -eq 0 ]; then
    print_success "Services started"
else
    print_error "Failed to start services"
    exit 1
fi

# Wait for health checks
print_info "Waiting for health checks to pass..."
print_info "This may take up to 90 seconds for model loading..."

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
    print_info "Check logs: docker-compose logs dashboard"
    exit 1
fi

# Check trading bot health
echo -n "Trading bot"
if wait_for_healthy "llm-tradebot-trading" 90; then
    print_success "Trading bot healthy"
else
    print_error "Trading bot health check failed"
    print_info "Check logs: docker-compose logs trading-bot"
    exit 1
fi

# Display status
print_header "Deployment Complete"

docker-compose ps

echo ""
print_success "LLM-TradeBot is running!"
echo ""
print_info "Dashboard: http://localhost:5173"
print_info "Health endpoint: http://localhost:5173/health"
echo ""
print_info "Useful commands:"
echo "  View logs:       docker-compose logs -f"
echo "  View positions:  docker-compose exec trading-bot python -m trading.cli positions"
echo "  Check status:    docker-compose exec trading-bot python -m trading.cli status"
echo "  Stop services:   docker-compose stop"
echo "  Remove all:      docker-compose down"
echo ""

# Test health endpoint
print_info "Testing health endpoint..."
sleep 2
if curl -f http://localhost:5173/health &> /dev/null; then
    print_success "Health endpoint responding"
    curl -s http://localhost:5173/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:5173/health
else
    print_error "Health endpoint not responding"
fi

echo ""
print_header "Next Steps"
echo "1. Monitor logs: docker-compose logs -f trading-bot"
echo "2. Open dashboard: http://localhost:5173"
echo "3. Check positions: docker-compose exec trading-bot python -m trading.cli positions"
echo "4. When done: docker-compose stop"
echo ""
print_info "For production deployment, see docs/DEPLOYMENT.md"
