#!/bin/bash
set -e

# Portfolio Rebalancer Docker Compose Integration Test Script
# This script tests Docker Compose deployments with different profiles

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.test"
VERSION=${VERSION:-"latest"}
TEST_NETWORK="portfolio-test-network"
TEST_PREFIX="test-portfolio"

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

# Test profiles to validate
TEST_PROFILES=("default" "manual" "monitoring" "cache" "database" "backup")

# Cleanup function
cleanup() {
    log_info "Cleaning up test environment..."
    
    # Stop all test containers
    for profile in "${TEST_PROFILES[@]}"; do
        log_debug "Stopping $profile profile containers..."
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" \
            --profile "$profile" down --volumes --remove-orphans 2>/dev/null || true
    done
    
    # Remove test network
    docker network rm "$TEST_NETWORK" 2>/dev/null || true
    
    # Remove test environment file
    rm -f "$ENV_FILE"
    
    # Remove test data directories
    rm -rf test-data test-logs
    
    log_info "Cleanup completed"
}

# Setup test environment
setup_test_env() {
    log_info "Setting up test environment..."
    
    # Create test directories
    mkdir -p test-data test-logs
    
    # Create test environment file
    cat > "$ENV_FILE" << EOF
# Test Environment Configuration
VERSION=$VERSION
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=test

# Data Configuration
TICKERS=AAPL,GOOGL,MSFT
STORAGE_TYPE=parquet
STORAGE_PATH=/app/data
BACKFILL_DAYS=30

# Volume Configuration
DATA_PATH=./test-data
LOGS_PATH=./test-logs

# Optimization Configuration
USER_AGE=35
RISK_FREE_RATE=0.02
LOOKBACK_DAYS=30
MIN_WEIGHT=0.0
MAX_WEIGHT=0.4
SAFE_PORTFOLIO_BONDS=0.8

# Executor Configuration
REBALANCE_THRESHOLD=0.05
ORDER_TYPE=market
BROKER_TYPE=alpaca

# Broker Configuration (Test/Paper Trading)
ALPACA_API_KEY=test_api_key
ALPACA_SECRET_KEY=test_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# Scheduler Configuration
EXECUTION_TIME=16:30
TIMEZONE=America/New_York
RETRY_ATTEMPTS=2
RETRY_DELAY=60

# Logging Configuration
LOG_LEVEL=DEBUG
LOG_FORMAT=json

# Monitoring Configuration
GRAFANA_PASSWORD=test123

# Database Configuration
POSTGRES_DB=test_portfolio
POSTGRES_USER=test_user
POSTGRES_PASSWORD=test_pass
EOF
    
    # Create test network
    if ! docker network ls | grep -q "$TEST_NETWORK"; then
        docker network create "$TEST_NETWORK"
        log_info "Created test network: $TEST_NETWORK"
    fi
    
    log_info "Test environment setup completed"
}

# Test service health
test_service_health() {
    local service=$1
    local port=$2
    local max_attempts=${3:-30}
    
    log_debug "Testing health for $service on port $port..."
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
            log_info "$service health check passed"
            return 0
        fi
        
        log_debug "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 2
        ((attempt++))
    done
    
    log_error "$service health check failed after $max_attempts attempts"
    return 1
}

# Test profile deployment
test_profile() {
    local profile=$1
    log_info "Testing $profile profile deployment..."
    
    # Get profile-specific compose arguments
    local compose_args=""
    case $profile in
        "default")
            compose_args=""
            ;;
        *)
            compose_args="--profile $profile"
            ;;
    esac
    
    # Start services
    log_debug "Starting $profile services..."
    if ! docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $compose_args up -d --build; then
        log_error "Failed to start $profile services"
        return 1
    fi
    
    # Wait for services to start
    sleep 15
    
    # Check if containers are running
    local running_containers=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $compose_args ps -q)
    if [ -z "$running_containers" ]; then
        log_error "No containers running for $profile profile"
        return 1
    fi
    
    # Test health endpoints based on profile
    local health_tests_passed=true
    case $profile in
        "default")
            test_service_health "scheduler" "8083" 30 || health_tests_passed=false
            ;;
        "manual")
            test_service_health "fetcher" "8080" 30 || health_tests_passed=false
            test_service_health "optimizer" "8081" 30 || health_tests_passed=false
            test_service_health "executor" "8082" 30 || health_tests_passed=false
            ;;
        "monitoring")
            # Test Prometheus
            if curl -s -f "http://localhost:9090/-/healthy" >/dev/null 2>&1; then
                log_info "Prometheus health check passed"
            else
                log_error "Prometheus health check failed"
                health_tests_passed=false
            fi
            
            # Test Grafana
            if curl -s -f "http://localhost:3000/api/health" >/dev/null 2>&1; then
                log_info "Grafana health check passed"
            else
                log_error "Grafana health check failed"
                health_tests_passed=false
            fi
            ;;
        "cache")
            # Test Redis
            if docker exec $(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $compose_args ps -q redis) redis-cli ping | grep -q "PONG"; then
                log_info "Redis health check passed"
            else
                log_error "Redis health check failed"
                health_tests_passed=false
            fi
            ;;
        "database")
            # Test PostgreSQL
            if docker exec $(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $compose_args ps -q postgres) pg_isready -U test_user >/dev/null 2>&1; then
                log_info "PostgreSQL health check passed"
            else
                log_error "PostgreSQL health check failed"
                health_tests_passed=false
            fi
            ;;
        "backup")
            # Check if backup container is running
            local backup_container=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $compose_args ps -q backup)
            if [ -n "$backup_container" ] && docker ps | grep -q "$backup_container"; then
                log_info "Backup service health check passed"
            else
                log_error "Backup service health check failed"
                health_tests_passed=false
            fi
            ;;
    esac
    
    # Stop services
    log_debug "Stopping $profile services..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $compose_args down --volumes 2>/dev/null || true
    
    if [ "$health_tests_passed" = true ]; then
        log_info "$profile profile test passed"
        return 0
    else
        log_error "$profile profile test failed"
        return 1
    fi
}

# Test volume persistence
test_volume_persistence() {
    log_info "Testing volume persistence..."
    
    # Start default profile to create volumes
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build
    sleep 10
    
    # Create test data
    echo "test data" > test-data/test-file.txt
    echo "test log" > test-logs/test-log.txt
    
    # Stop services
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    
    # Restart services
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    sleep 10
    
    # Check if data persisted
    if [ -f "test-data/test-file.txt" ] && [ -f "test-logs/test-log.txt" ]; then
        log_info "Volume persistence test passed"
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down --volumes
        return 0
    else
        log_error "Volume persistence test failed"
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down --volumes
        return 1
    fi
}

# Test environment variable loading
test_env_loading() {
    log_info "Testing environment variable loading..."
    
    # Start scheduler service
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d scheduler
    sleep 10
    
    # Check if environment variables are loaded correctly
    local scheduler_container=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps -q scheduler)
    
    if [ -n "$scheduler_container" ]; then
        # Test a few key environment variables
        local tickers=$(docker exec "$scheduler_container" printenv TICKERS)
        local user_age=$(docker exec "$scheduler_container" printenv USER_AGE)
        local log_level=$(docker exec "$scheduler_container" printenv LOG_LEVEL)
        
        if [ "$tickers" = "AAPL,GOOGL,MSFT" ] && [ "$user_age" = "35" ] && [ "$log_level" = "DEBUG" ]; then
            log_info "Environment variable loading test passed"
            docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
            return 0
        else
            log_error "Environment variable loading test failed"
            log_debug "TICKERS: $tickers, USER_AGE: $user_age, LOG_LEVEL: $log_level"
            docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
            return 1
        fi
    else
        log_error "Scheduler container not found"
        return 1
    fi
}

# Run all tests
run_all_tests() {
    local failed_tests=()
    
    log_info "Starting comprehensive Docker Compose tests..."
    
    # Test environment variable loading
    if test_env_loading; then
        log_info "Environment variable test passed"
    else
        failed_tests+=("env-loading")
    fi
    
    # Test volume persistence
    if test_volume_persistence; then
        log_info "Volume persistence test passed"
    else
        failed_tests+=("volume-persistence")
    fi
    
    # Test each profile
    for profile in "${TEST_PROFILES[@]}"; do
        if test_profile "$profile"; then
            log_info "$profile profile test passed"
        else
            failed_tests+=("$profile-profile")
        fi
    done
    
    # Summary
    echo ""
    if [ ${#failed_tests[@]} -eq 0 ]; then
        log_info "All Docker Compose tests passed successfully!"
        return 0
    else
        log_error "Failed tests: ${failed_tests[*]}"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [TEST]"
    echo ""
    echo "Test Docker Compose configurations for Portfolio Rebalancer"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help         Show this help message"
    echo "  -v, --version      Set version tag to test (default: latest)"
    echo "  --cleanup          Only run cleanup"
    echo ""
    echo "TESTS:"
    echo "  all                Run all tests (default)"
    echo "  env                Test environment variable loading"
    echo "  volumes            Test volume persistence"
    echo "  default            Test default profile"
    echo "  manual             Test manual profile"
    echo "  monitoring         Test monitoring profile"
    echo "  cache              Test cache profile"
    echo "  database           Test database profile"
    echo "  backup             Test backup profile"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                           # Run all tests"
    echo "  $0 -v 1.0.0 manual          # Test manual profile with version 1.0.0"
    echo "  $0 --cleanup                 # Clean up test environment"
}

# Parse command line arguments
TEST="all"
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
        all|env|volumes|default|manual|monitoring|cache|database|backup)
            TEST="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check prerequisites
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed or not available"
    exit 1
fi

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

# Execute tests
case $TEST in
    "all")
        run_all_tests
        ;;
    "env")
        test_env_loading
        ;;
    "volumes")
        test_volume_persistence
        ;;
    "default"|"manual"|"monitoring"|"cache"|"database"|"backup")
        test_profile "$TEST"
        ;;
    *)
        log_error "Invalid test: $TEST"
        show_usage
        exit 1
        ;;
esac

exit $?