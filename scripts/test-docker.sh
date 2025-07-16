#!/bin/bash
set -e

# Portfolio Rebalancer Docker Test Script
# This script tests Docker containers for health and functionality

# Configuration
IMAGE_PREFIX="portfolio-rebalancer"
VERSION=${VERSION:-"latest"}
NETWORK_NAME="portfolio-test-network"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Services to test
SERVICES=("fetcher" "optimizer" "executor" "scheduler")
PORTS=("8080" "8081" "8082" "8083")

# Cleanup function
cleanup() {
    log_info "Cleaning up test containers and network..."
    
    for service in "${SERVICES[@]}"; do
        container_name="test-${service}"
        if docker ps -q -f name="$container_name" | grep -q .; then
            docker stop "$container_name" >/dev/null 2>&1 || true
            docker rm "$container_name" >/dev/null 2>&1 || true
        fi
    done
    
    # Remove test network
    if docker network ls | grep -q "$NETWORK_NAME"; then
        docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
    fi
}

# Setup test environment
setup_test_env() {
    log_info "Setting up test environment..."
    
    # Create test network
    if ! docker network ls | grep -q "$NETWORK_NAME"; then
        docker network create "$NETWORK_NAME"
        log_info "Created test network: $NETWORK_NAME"
    fi
    
    # Create test data directory
    mkdir -p test-data
    
    # Create minimal test configuration
    cat > test-data/.env << EOF
# Test configuration
TICKERS=AAPL,GOOGL,MSFT
STORAGE_TYPE=parquet
STORAGE_PATH=/app/data
USER_AGE=35
LOOKBACK_DAYS=252
REBALANCE_THRESHOLD=0.05
EXECUTION_TIME=16:00
TIMEZONE=America/New_York
LOG_LEVEL=INFO
EOF
}

# Test container health
test_container_health() {
    local service=$1
    local port=$2
    local container_name="test-${service}"
    local image_name="${IMAGE_PREFIX}-${service}:${VERSION}"
    
    log_info "Testing ${service} container health..."
    
    # Check if image exists
    if ! docker images | grep -q "${IMAGE_PREFIX}-${service}"; then
        log_error "Image not found: ${image_name}"
        return 1
    fi
    
    # Start container in server mode for health checks
    log_debug "Starting ${service} container..."
    docker run -d \
        --name "$container_name" \
        --network "$NETWORK_NAME" \
        -p "${port}:${port}" \
        -v "$(pwd)/test-data:/app/test-data" \
        -e "ENV_FILE=/app/test-data/.env" \
        "$image_name" \
        python -m "src.portfolio_rebalancer.services.${service}_service" --mode server --port "$port" \
        >/dev/null
    
    # Wait for container to start
    log_debug "Waiting for ${service} to start..."
    sleep 10
    
    # Check if container is running
    if ! docker ps | grep -q "$container_name"; then
        log_error "${service} container failed to start"
        docker logs "$container_name" 2>/dev/null || true
        return 1
    fi
    
    # Test health endpoint
    log_debug "Testing health endpoint for ${service}..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:${port}/health" >/dev/null 2>&1; then
            log_info "${service} health check passed"
            
            # Get health status
            local health_response=$(curl -s "http://localhost:${port}/health" 2>/dev/null || echo "{}")
            log_debug "Health response: $health_response"
            
            return 0
        fi
        
        log_debug "Health check attempt ${attempt}/${max_attempts} failed, retrying..."
        sleep 2
        ((attempt++))
    done
    
    log_error "${service} health check failed after ${max_attempts} attempts"
    docker logs "$container_name" 2>/dev/null || true
    return 1
}

# Test container functionality
test_container_functionality() {
    local service=$1
    local container_name="test-${service}"
    local image_name="${IMAGE_PREFIX}-${service}:${VERSION}"
    
    log_info "Testing ${service} container functionality..."
    
    # Test running in 'once' mode
    log_debug "Testing ${service} in 'once' mode..."
    
    # Note: This is a basic smoke test. In a real scenario, you'd want to:
    # - Set up test data
    # - Mock external APIs
    # - Verify expected outputs
    
    local exit_code=0
    docker run --rm \
        --network "$NETWORK_NAME" \
        -v "$(pwd)/test-data:/app/test-data" \
        -e "ENV_FILE=/app/test-data/.env" \
        "$image_name" \
        python -c "from src.portfolio_rebalancer.services.${service}_service import ${service^}Service; print('${service} service can be imported')" \
        >/dev/null 2>&1 || exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_info "${service} functionality test passed"
        return 0
    else
        log_error "${service} functionality test failed"
        return 1
    fi
}

# Test all services
test_all_services() {
    local failed_services=()
    
    log_info "Starting comprehensive container tests..."
    
    for i in "${!SERVICES[@]}"; do
        local service="${SERVICES[$i]}"
        local port="${PORTS[$i]}"
        
        log_info "Testing ${service} service..."
        
        # Test health
        if test_container_health "$service" "$port"; then
            log_info "${service} health test passed"
        else
            log_error "${service} health test failed"
            failed_services+=("${service}-health")
        fi
        
        # Test functionality
        if test_container_functionality "$service"; then
            log_info "${service} functionality test passed"
        else
            log_error "${service} functionality test failed"
            failed_services+=("${service}-functionality")
        fi
        
        # Stop the container
        docker stop "test-${service}" >/dev/null 2>&1 || true
        docker rm "test-${service}" >/dev/null 2>&1 || true
        
        log_info "Completed testing ${service} service"
        echo ""
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log_info "All container tests passed successfully!"
        return 0
    else
        log_error "Failed tests: ${failed_services[*]}"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [SERVICE]"
    echo ""
    echo "Test Docker containers for Portfolio Rebalancer services"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --version  Set version tag to test (default: latest)"
    echo "  --cleanup      Only run cleanup"
    echo ""
    echo "SERVICE:"
    echo "  fetcher        Test only the data fetcher service"
    echo "  optimizer      Test only the portfolio optimizer service"
    echo "  executor       Test only the trade executor service"
    echo "  scheduler      Test only the scheduler service"
    echo "  all            Test all services (default)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                           # Test all services"
    echo "  $0 -v 1.0.0 fetcher         # Test fetcher service with version 1.0.0"
    echo "  $0 --cleanup                 # Clean up test containers"
}

# Parse command line arguments
SERVICE="all"
CLEANUP_ONLY=false

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
        --cleanup)
            CLEANUP_ONLY=true
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

# Check if curl is available
if ! command -v curl &> /dev/null; then
    log_error "curl is not installed or not in PATH"
    exit 1
fi

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run cleanup if requested
if [ "$CLEANUP_ONLY" = true ]; then
    cleanup
    log_info "Cleanup completed"
    exit 0
fi

# Setup test environment
setup_test_env

# Main execution
case $SERVICE in
    all)
        test_all_services
        ;;
    fetcher|optimizer|executor|scheduler)
        i=0
        for s in "${SERVICES[@]}"; do
            if [ "$s" = "$SERVICE" ]; then
                break
            fi
            ((i++))
        done
        
        if test_container_health "$SERVICE" "${PORTS[$i]}" && test_container_functionality "$SERVICE"; then
            log_info "${SERVICE} tests passed"
            exit 0
        else
            log_error "${SERVICE} tests failed"
            exit 1
        fi
        ;;
    *)
        log_error "Invalid service: $SERVICE"
        show_usage
        exit 1
        ;;
esac

exit $?