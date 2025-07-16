#!/bin/bash
set -e

# Portfolio Rebalancer Docker Build Script
# This script builds all service containers with proper tagging

# Configuration
IMAGE_PREFIX="portfolio-rebalancer"
VERSION=${VERSION:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=${GITHUB_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build arguments
BUILD_ARGS="--build-arg BUILD_DATE=${BUILD_DATE} --build-arg VCS_REF=${VCS_REF} --build-arg VERSION=${VERSION}"

# Services to build
SERVICES=("fetcher" "optimizer" "executor" "scheduler")

# Function to build a service
build_service() {
    local service=$1
    local dockerfile="dockerfiles/Dockerfile.${service}"
    local image_name="${IMAGE_PREFIX}-${service}:${VERSION}"
    
    log_info "Building ${service} service..."
    
    if [ ! -f "$dockerfile" ]; then
        log_error "Dockerfile not found: $dockerfile"
        return 1
    fi
    
    # Build the image
    if docker build ${BUILD_ARGS} -f "$dockerfile" -t "$image_name" .; then
        log_info "Successfully built ${image_name}"
        
        # Tag as latest if version is not latest
        if [ "$VERSION" != "latest" ]; then
            docker tag "$image_name" "${IMAGE_PREFIX}-${service}:latest"
            log_info "Tagged as ${IMAGE_PREFIX}-${service}:latest"
        fi
        
        return 0
    else
        log_error "Failed to build ${image_name}"
        return 1
    fi
}

# Function to build all services
build_all() {
    local failed_services=()
    
    log_info "Starting build process for all services..."
    log_info "Build date: ${BUILD_DATE}"
    log_info "VCS ref: ${VCS_REF}"
    log_info "Version: ${VERSION}"
    
    for service in "${SERVICES[@]}"; do
        if ! build_service "$service"; then
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log_info "All services built successfully!"
        return 0
    else
        log_error "Failed to build services: ${failed_services[*]}"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [SERVICE]"
    echo ""
    echo "Build Docker images for Portfolio Rebalancer services"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --version  Set version tag (default: latest)"
    echo "  --no-cache     Build without using cache"
    echo ""
    echo "SERVICE:"
    echo "  fetcher        Build only the data fetcher service"
    echo "  optimizer      Build only the portfolio optimizer service"
    echo "  executor       Build only the trade executor service"
    echo "  scheduler      Build only the scheduler service"
    echo "  all            Build all services (default)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                           # Build all services with latest tag"
    echo "  $0 -v 1.0.0 fetcher         # Build fetcher service with version 1.0.0"
    echo "  $0 --no-cache all           # Build all services without cache"
}

# Parse command line arguments
NO_CACHE=""
SERVICE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            BUILD_ARGS="${BUILD_ARGS} --no-cache"
            shift
            ;;
        fetcher|optimizer|executor|scheduler|all)
            SERVICE="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "src" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Main execution
case $SERVICE in
    all)
        build_all
        ;;
    fetcher|optimizer|executor|scheduler)
        build_service "$SERVICE"
        ;;
    *)
        log_error "Invalid service: $SERVICE"
        show_usage
        exit 1
        ;;
esac

exit $?