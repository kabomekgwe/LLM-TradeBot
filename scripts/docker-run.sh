#!/bin/bash
# Docker Run Script - Quick local testing of Docker container
#
# Usage:
#   ./scripts/docker-run.sh [command]
#
# Examples:
#   ./scripts/docker-run.sh                      # Run default command (trading bot)
#   ./scripts/docker-run.sh status               # Check bot status
#   ./scripts/docker-run.sh bash                 # Interactive shell

set -e  # Exit on error

# Configuration
IMAGE_NAME="llm-tradebot:latest"
CONTAINER_NAME="llm-tradebot-test"
COMMAND=${1:-run}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if image exists
if ! docker image inspect $IMAGE_NAME &> /dev/null; then
    print_error "Image $IMAGE_NAME not found. Build it first:"
    echo "  ./scripts/docker-build.sh"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Copying from .env.production.template..."
    cp .env.production.template .env
    print_warning "Please edit .env and add your API keys before running."
    exit 1
fi

# Create required directories
mkdir -p models data logs

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    print_info "Stopping existing container..."
    docker stop $CONTAINER_NAME &> /dev/null || true
    docker rm $CONTAINER_NAME &> /dev/null || true
fi

# Run container based on command
case $COMMAND in
    run)
        print_info "Starting trading bot container..."
        docker run -d \
            --name $CONTAINER_NAME \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/logs:/app/logs \
            -v $(pwd)/.env:/app/.env:ro \
            -p 5173:5173 \
            $IMAGE_NAME \
            python -m trading.cli run --symbol BTC/USDT

        print_info "Container started: $CONTAINER_NAME"
        print_info "Dashboard: http://localhost:5173"
        print_info "View logs: docker logs -f $CONTAINER_NAME"
        ;;

    status)
        print_info "Checking bot status..."
        docker run --rm \
            -v $(pwd)/.env:/app/.env:ro \
            $IMAGE_NAME \
            python -m trading.cli status
        ;;

    bash)
        print_info "Starting interactive shell..."
        docker run -it --rm \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/logs:/app/logs \
            -v $(pwd)/.env:/app/.env:ro \
            $IMAGE_NAME \
            /bin/bash
        ;;

    test)
        print_info "Running health check test..."
        docker run -d \
            --name $CONTAINER_NAME \
            -v $(pwd)/models:/app/models \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/logs:/app/logs \
            -v $(pwd)/.env:/app/.env:ro \
            -p 5173:5173 \
            $IMAGE_NAME \
            python -m trading.web.server

        print_info "Waiting for container to start..."
        sleep 5

        print_info "Testing health endpoint..."
        if curl -f http://localhost:5173/health &> /dev/null; then
            print_info "Health check passed!"
        else
            print_warning "Health check failed. Check logs:"
            docker logs $CONTAINER_NAME
        fi

        print_info "Stopping test container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        ;;

    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  run       Start trading bot (default)"
        echo "  status    Check bot status"
        echo "  bash      Interactive shell"
        echo "  test      Test health endpoint"
        exit 1
        ;;
esac
