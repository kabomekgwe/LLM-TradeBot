#!/bin/bash
# Docker Build Script - Build LLM-TradeBot images with versioning
#
# Usage:
#   ./scripts/docker-build.sh [version] [export]
#
# Examples:
#   ./scripts/docker-build.sh                    # Build with 'latest' tag
#   ./scripts/docker-build.sh v1.0.0             # Build with version tag
#   ./scripts/docker-build.sh v1.0.0 export      # Build and export to tar

set -e  # Exit on error

# Configuration
IMAGE_NAME="llm-tradebot"
VERSION=${1:-latest}
EXPORT_FLAG=${2:-}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile not found. Run this script from project root."
    exit 1
fi

# Start build
print_header "Building Docker Image"
print_info "Image name: $IMAGE_NAME"
print_info "Version: $VERSION"

# Build image with both version and latest tags
print_info "Building image with tags: $IMAGE_NAME:$VERSION and $IMAGE_NAME:latest"
docker build \
    -t $IMAGE_NAME:$VERSION \
    -t $IMAGE_NAME:latest \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VERSION=$VERSION \
    .

# Check build status
if [ $? -eq 0 ]; then
    print_info "Build successful!"
else
    print_error "Build failed!"
    exit 1
fi

# Display image information
print_header "Image Information"
docker images $IMAGE_NAME:$VERSION --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Check image size
IMAGE_SIZE=$(docker images $IMAGE_NAME:$VERSION --format "{{.Size}}")
print_info "Final image size: $IMAGE_SIZE"

# Warn if image is too large (>1.5GB)
SIZE_MB=$(docker inspect -f "{{ .Size }}" $IMAGE_NAME:$VERSION | awk '{print int($1/1024/1024)}')
if [ $SIZE_MB -gt 1500 ]; then
    print_warning "Image size is larger than expected (>1.5GB). Consider optimization."
fi

# Export image to tar file if requested
if [ "$EXPORT_FLAG" == "export" ]; then
    print_header "Exporting Image"
    EXPORT_FILE="${IMAGE_NAME}-${VERSION}.tar"
    print_info "Exporting to $EXPORT_FILE..."

    docker save $IMAGE_NAME:$VERSION -o $EXPORT_FILE

    if [ $? -eq 0 ]; then
        EXPORT_SIZE=$(du -h $EXPORT_FILE | cut -f1)
        print_info "Export successful! File: $EXPORT_FILE ($EXPORT_SIZE)"
        print_info "To load on another machine: docker load -i $EXPORT_FILE"
    else
        print_error "Export failed!"
        exit 1
    fi
fi

# Test image (basic smoke test)
print_header "Testing Image"
print_info "Running smoke test: python --version"
docker run --rm $IMAGE_NAME:$VERSION python --version

print_info "Running smoke test: python -m trading.cli --version"
docker run --rm $IMAGE_NAME:$VERSION python -m trading.cli --version || print_warning "CLI version test failed (expected for Phase 11)"

print_header "Build Complete"
print_info "Image: $IMAGE_NAME:$VERSION"
print_info "Image: $IMAGE_NAME:latest"
echo ""
print_info "Next steps:"
echo "  1. Test locally: docker-compose up -d"
echo "  2. View logs: docker-compose logs -f"
echo "  3. Stop services: docker-compose down"
if [ "$EXPORT_FLAG" == "export" ]; then
    echo "  4. Transfer tar: scp $EXPORT_FILE user@production-server:~/"
    echo "  5. Load on server: docker load -i $EXPORT_FILE"
fi
