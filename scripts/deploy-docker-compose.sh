#!/bin/bash
set -e

# Portfolio Rebalancer Docker Compose Deployment Script
# This script manages Docker Compose deployments with different profiles

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
VERSION=${VERSION:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=${GITHUB_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")}

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

# Available deployment profiles
PROFILES=(
    "default"      # Scheduler only (automated daily execution)
    "manual"       # All services for manual execution
    "monitoring"   # Include Prometheus and Grafana
    "cache"        # Include Redis for caching
    "database"     # Include PostgreSQL for advanced storage
    "backup"       # Include data backup service
    "full"         # All services and monitoring
)

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Manage Portfolio Rebalancer Docker Compose deployments"
    echo ""
    echo "COMMANDS:"
    echo "  up         Start services"
    echo "  down       Stop and remove services"
    echo "  restart    Restart services"
    echo "  logs       Show service logs"
    echo "  status     Show service status"
    echo "  build      Build all images"
    echo "  pull       Pull latest images"
    echo "  clean      Clean up containers, images, and volumes"
    echo "  init       Initialize environment and create .env file"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --profile PROFILE    Deployment profile (default: default)"
    echo "  -v, --version VERSION    Image version tag (default: latest)"
    echo "  -f, --file FILE         Docker Compose file (default: docker-compose.yml)"
    echo "  -e, --env-file FILE     Environment file (default: .env)"
    echo "  --build                 Build images before starting"
    echo "  --force-recreate        Force recreate containers"
    echo "  --no-deps              Don't start dependent services"
    echo "  -d, --detach           Run in background"
    echo "  --follow               Follow log output"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "PROFILES:"
    for profile in "${PROFILES[@]}"; do
        case $profile in
            "default")
                echo "  default      Scheduler service only (automated execution)"
                ;;
            "manual")
                echo "  manual       All core services for manual execution"
                ;;
            "monitoring")
                echo "  monitoring   Include Prometheus and Grafana monitoring"
                ;;
            "cache")
                echo "  cache        Include Redis for caching"
                ;;
            "database")
                echo "  database     Include PostgreSQL for advanced storage"
                ;;
            "backup")
                echo "  backup       Include data backup service"
                ;;
            "full")
                echo "  full         All services with monitoring and storage"
                ;;
        esac
    done
    echo ""
    echo "EXAMPLES:"
    echo "  $0 up                           # Start scheduler service only"
    echo "  $0 up -p manual                 # Start all services for manual execution"
    echo "  $0 up -p full --build           # Build and start all services"
    echo "  $0 logs -p manual --follow      # Follow logs for manual services"
    echo "  $0 down -p full                 # Stop all services"
    echo "  $0 clean                        # Clean up everything"
}

# Function to validate profile
validate_profile() {
    local profile=$1
    for p in "${PROFILES[@]}"; do
        if [ "$p" = "$profile" ]; then
            return 0
        fi
    done
    return 1
}

# Function to get compose profiles for deployment profile
get_compose_profiles() {
    local profile=$1
    case $profile in
        "default")
            echo ""
            ;;
        "manual")
            echo "--profile manual"
            ;;
        "monitoring")
            echo "--profile monitoring"
            ;;
        "cache")
            echo "--profile cache"
            ;;
        "database")
            echo "--profile database"
            ;;
        "backup")
            echo "--profile backup"
            ;;
        "full")
            echo "--profile manual --profile monitoring --profile cache --profile database --profile backup"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to check prerequisites
check_prerequisites() {
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker Compose is available
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed or not available"
        exit 1
    fi

    # Check if we're in the right directory
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        log_error "Please run this script from the project root directory"
        exit 1
    fi
}

# Function to initialize environment
init_environment() {
    log_info "Initializing Portfolio Rebalancer environment..."
    
    # Create directories
    mkdir -p data logs
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example "$ENV_FILE"
            log_info "Created $ENV_FILE from .env.example"
            log_warn "Please edit $ENV_FILE with your configuration before deployment"
        else
            log_error ".env.example file not found"
            return 1
        fi
    else
        log_info "$ENV_FILE already exists"
    fi
    
    # Set permissions
    chmod 755 data logs
    
    log_info "Environment initialization complete"
    log_info "Next steps:"
    log_info "1. Edit $ENV_FILE with your configuration"
    log_info "2. Run: $0 up -p [profile] to start services"
}

# Function to build images
build_images() {
    local profiles=$1
    local build_args="--build-arg BUILD_DATE=$BUILD_DATE --build-arg VCS_REF=$VCS_REF --build-arg VERSION=$VERSION"
    
    log_info "Building Docker images..."
    log_info "Build date: $BUILD_DATE"
    log_info "VCS ref: $VCS_REF"
    log_info "Version: $VERSION"
    
    # Export build args as environment variables for docker-compose
    export BUILD_DATE VCS_REF VERSION
    
    if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profiles build; then
        log_info "Successfully built all images"
        return 0
    else
        log_error "Failed to build images"
        return 1
    fi
}

# Function to start services
start_services() {
    local profiles=$1
    local options=$2
    
    log_info "Starting Portfolio Rebalancer services..."
    log_debug "Profiles: $profiles"
    log_debug "Options: $options"
    
    # Export build args as environment variables
    export BUILD_DATE VCS_REF VERSION
    
    if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profiles up $options; then
        log_info "Services started successfully"
        return 0
    else
        log_error "Failed to start services"
        return 1
    fi
}

# Function to stop services
stop_services() {
    local profiles=$1
    
    log_info "Stopping Portfolio Rebalancer services..."
    
    if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profiles down; then
        log_info "Services stopped successfully"
        return 0
    else
        log_error "Failed to stop services"
        return 1
    fi
}

# Function to restart services
restart_services() {
    local profiles=$1
    local options=$2
    
    log_info "Restarting Portfolio Rebalancer services..."
    
    stop_services "$profiles"
    start_services "$profiles" "$options"
}

# Function to show logs
show_logs() {
    local profiles=$1
    local options=$2
    
    log_info "Showing service logs..."
    
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profiles logs $options
}

# Function to show status
show_status() {
    local profiles=$1
    
    log_info "Portfolio Rebalancer service status:"
    echo ""
    
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profiles ps
    
    echo ""
    log_info "Docker images:"
    docker images | grep portfolio-rebalancer || log_warn "No portfolio-rebalancer images found"
    
    echo ""
    log_info "Docker volumes:"
    docker volume ls | grep portfolio || log_warn "No portfolio volumes found"
}

# Function to pull images
pull_images() {
    local profiles=$1
    
    log_info "Pulling latest images..."
    
    if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" $profiles pull; then
        log_info "Successfully pulled images"
        return 0
    else
        log_error "Failed to pull images"
        return 1
    fi
}

# Function to clean up
cleanup() {
    log_info "Cleaning up Portfolio Rebalancer deployment..."
    
    # Stop and remove all containers
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile manual --profile monitoring --profile cache --profile database down --volumes --remove-orphans 2>/dev/null || true
    
    # Remove images
    docker images | grep portfolio-rebalancer | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
    
    # Remove volumes
    docker volume ls | grep portfolio | awk '{print $2}' | xargs -r docker volume rm 2>/dev/null || true
    
    # Remove network
    docker network rm portfolio-rebalancer-network 2>/dev/null || true
    
    log_info "Cleanup complete"
}

# Parse command line arguments
COMMAND=""
PROFILE="default"
OPTIONS=""
BUILD_FLAG=""
DETACH_FLAG=""
FOLLOW_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        up|down|restart|logs|status|build|pull|clean|init)
            COMMAND="$1"
            shift
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        -e|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --force-recreate)
            OPTIONS="$OPTIONS --force-recreate"
            shift
            ;;
        --no-deps)
            OPTIONS="$OPTIONS --no-deps"
            shift
            ;;
        -d|--detach)
            DETACH_FLAG="-d"
            shift
            ;;
        --follow)
            FOLLOW_FLAG="-f"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate command
if [ -z "$COMMAND" ]; then
    log_error "No command specified"
    show_usage
    exit 1
fi

# Validate profile
if ! validate_profile "$PROFILE"; then
    log_error "Invalid profile: $PROFILE"
    log_error "Available profiles: ${PROFILES[*]}"
    exit 1
fi

# Check prerequisites (except for init command)
if [ "$COMMAND" != "init" ]; then
    check_prerequisites
fi

# Get compose profiles
COMPOSE_PROFILES=$(get_compose_profiles "$PROFILE")

# Execute command
case $COMMAND in
    "init")
        init_environment
        ;;
    "up")
        if [ -n "$BUILD_FLAG" ]; then
            build_images "$COMPOSE_PROFILES"
        fi
        start_services "$COMPOSE_PROFILES" "$OPTIONS $DETACH_FLAG"
        ;;
    "down")
        stop_services "$COMPOSE_PROFILES"
        ;;
    "restart")
        restart_services "$COMPOSE_PROFILES" "$OPTIONS $DETACH_FLAG"
        ;;
    "logs")
        show_logs "$COMPOSE_PROFILES" "$FOLLOW_FLAG"
        ;;
    "status")
        show_status "$COMPOSE_PROFILES"
        ;;
    "build")
        build_images "$COMPOSE_PROFILES"
        ;;
    "pull")
        pull_images "$COMPOSE_PROFILES"
        ;;
    "clean")
        cleanup
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

exit $?