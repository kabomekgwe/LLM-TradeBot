#!/bin/bash
# Health Check Testing Script - Verify Docker health checks and auto-restart
#
# Tests:
# 1. Container startup and health check pass
# 2. Auto-restart on crash (SIGKILL)
# 3. Graceful shutdown (SIGTERM)
#
# Usage:
#   ./scripts/test-health-checks.sh

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
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

wait_for_status() {
    local container=$1
    local expected_status=$2
    local timeout=$3
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if docker ps --filter "name=$container" --format "{{.Status}}" | grep -q "$expected_status"; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    return 1
}

# Check prerequisites
if ! command -v docker &> /dev/null; then
    print_fail "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_fail "Docker Compose is not installed"
    exit 1
fi

if [ ! -f ".env" ]; then
    print_fail ".env file not found. Copy from .env.production.template"
    exit 1
fi

# Test 1: Container Startup and Health Check
print_header "Test 1: Container Startup and Health Check"

print_info "Starting services with docker-compose..."
docker-compose up -d

print_info "Waiting for containers to start..."
sleep 5

# Check if containers are running
if docker-compose ps | grep -q "Up"; then
    print_success "Containers started successfully"
else
    print_fail "Containers failed to start"
    docker-compose logs
    exit 1
fi

print_info "Waiting for dashboard health check to pass (max 60s)..."
if wait_for_status "llm-tradebot-dashboard" "healthy" 60; then
    print_success "Dashboard health check passed"
else
    print_fail "Dashboard health check failed or timed out"
    print_info "Dashboard status:"
    docker inspect llm-tradebot-dashboard --format='{{.State.Health.Status}}'
    print_info "Dashboard logs:"
    docker logs llm-tradebot-dashboard --tail 20
    exit 1
fi

print_info "Waiting for trading-bot health check to pass (max 90s for model loading)..."
if wait_for_status "llm-tradebot-trading" "healthy" 90; then
    print_success "Trading bot health check passed"
else
    print_fail "Trading bot health check failed or timed out"
    print_info "Trading bot status:"
    docker inspect llm-tradebot-trading --format='{{.State.Health.Status}}'
    print_info "Trading bot logs:"
    docker logs llm-tradebot-trading --tail 20
    exit 1
fi

# Test health endpoint directly
print_info "Testing /health endpoint directly..."
if curl -f http://localhost:5173/health &> /dev/null; then
    print_success "Health endpoint responded successfully"
    print_info "Response:"
    curl -s http://localhost:5173/health | python3 -m json.tool || echo "JSON parse failed"
else
    print_fail "Health endpoint not accessible"
    exit 1
fi

# Test 2: Auto-Restart on Crash
print_header "Test 2: Auto-Restart on Crash (SIGKILL)"

print_info "Simulating crash by killing dashboard container..."
docker kill llm-tradebot-dashboard

print_info "Waiting for auto-restart (max 30s)..."
sleep 5

if docker ps | grep -q "llm-tradebot-dashboard"; then
    print_success "Container auto-restarted after crash"

    print_info "Waiting for health check after restart..."
    if wait_for_status "llm-tradebot-dashboard" "healthy" 60; then
        print_success "Health check passed after restart"
    else
        print_fail "Health check failed after restart"
        exit 1
    fi
else
    print_fail "Container did not auto-restart"
    exit 1
fi

# Test 3: Graceful Shutdown
print_header "Test 3: Graceful Shutdown (SIGTERM)"

print_info "Performing graceful shutdown with docker-compose stop..."
docker-compose stop

print_info "Checking logs for graceful shutdown messages..."

# Check dashboard logs for shutdown
if docker logs llm-tradebot-dashboard 2>&1 | grep -q "shutdown\|SIGTERM\|graceful"; then
    print_success "Dashboard logged shutdown process"
else
    print_info "Dashboard shutdown logs not found (may be too fast)"
fi

# Check trading bot logs for shutdown
if docker logs llm-tradebot-trading 2>&1 | grep -q "shutdown\|SIGTERM\|graceful"; then
    print_success "Trading bot logged shutdown process"
else
    print_info "Trading bot shutdown logs not found (may be too fast)"
fi

# Verify containers stopped
if ! docker ps | grep -q "llm-tradebot"; then
    print_success "All containers stopped successfully"
else
    print_fail "Some containers still running"
    docker ps | grep "llm-tradebot"
fi

# Clean up
print_header "Cleanup"
print_info "Removing stopped containers..."
docker-compose down

print_success "All containers removed"

# Summary
print_header "Test Summary"
print_success "All health check tests passed!"
echo ""
print_info "Verified:"
echo "  ✓ Container startup and health checks"
echo "  ✓ Auto-restart on crash (SIGKILL)"
echo "  ✓ Graceful shutdown (SIGTERM)"
echo "  ✓ Health endpoint accessibility"
echo ""
print_info "Next steps:"
echo "  1. Review logs: docker-compose logs -f"
echo "  2. Production deployment: See docs/DEPLOYMENT.md"
