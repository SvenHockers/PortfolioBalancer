#!/bin/bash
set -e

# Configuration
REGISTRY="docker.io"
IMAGE_PREFIX="svenhockers/portfoliobalancer"
SERVICES=("fetcher" "optimizer" "executor" "scheduler")
VERSION="${VERSION:-latest}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

help() {
    cat << EOF
Portfolio Rebalancer Management Script

USAGE:
    ./portfolio.sh [COMMAND] [OPTIONS]

COMMANDS:
    (no args)       Download/pull all Docker images from registry
    up              Start all services (scheduler only by default)
    down            Stop all services
    restart         Restart all services
    logs            Show logs for all services
    status          Show status of all services
    pull            Pull latest images from registry
    clean           Remove stopped containers and unused images
    run SERVICE     Run a specific service manually (fetcher, optimizer, executor)
    monitoring      Start with monitoring (Prometheus + Grafana)
    full            Start all services including optional ones

OPTIONS:
    -v, --version   Specify image version (default: latest)
    -h, --help      Show this help message

EXAMPLES:
    ./portfolio.sh              # Pull all images
    ./portfolio.sh up           # Start scheduler service
    ./portfolio.sh up manual    # Start all manual services
    ./portfolio.sh logs         # Show all service logs
    ./portfolio.sh -v v1.0.1 up # Start with specific version

SERVICES:
    - scheduler: Automated daily portfolio rebalancing
    - fetcher: Manual data fetching
    - optimizer: Manual portfolio optimization
    - executor: Manual trade execution

EOF
}

pull_images() {
    check_docker
    log_info "Pulling Docker images from registry..."
    
    for service in "${SERVICES[@]}"; do
        local image="${REGISTRY}/${IMAGE_PREFIX}:${service}"
        log_info "Pulling ${image}..."
        
        if docker pull "${image}"; then
            log_success "Successfully pulled ${service} image"
        else
            log_error "Failed to pull ${service} image"
            return 1
        fi
    done
    
    log_success "All images pulled successfully!"
}

start_services() {
    local profile=""
    local extra_args=""
    
    case "${1:-}" in
        "manual")
            profile="--profile manual"
            log_info "Starting manual services (fetcher, optimizer, executor)..."
            ;;
        "monitoring")
            profile="--profile monitoring"
            log_info "Starting with monitoring services..."
            ;;
        "full")
            profile="--profile manual --profile monitoring --profile cache --profile database"
            log_info "Starting all services..."
            ;;
        *)
            log_info "Starting scheduler service..."
            ;;
    esac
    
    export COMPOSE_PROJECT_NAME="portfolio-rebalancer"
    # Override image names to use registry images
    for service in "${SERVICES[@]}"; do
        export "$(echo ${service} | tr '[:lower:]' '[:upper:]')_IMAGE=${REGISTRY}/${IMAGE_PREFIX}:${service}"
    done
    
    if docker-compose ${profile} up -d ${extra_args}; then
        log_success "Services started successfully!"
        log_info "Use './portfolio.sh logs' to view logs"
        log_info "Use './portfolio.sh status' to check service status"
    else
        log_error "Failed to start services"
        return 1
    fi
}

stop_services() {
    log_info "Stopping all services..."
    
    if docker-compose down; then
        log_success "All services stopped successfully!"
    else
        log_error "Failed to stop some services"
        return 1
    fi
}

restart_services() {
    log_info "Restarting services..."
    stop_services
    sleep 2
    start_services "$@"
}

show_logs() {
    local service="${1:-}"
    
    if [[ -n "$service" ]]; then
        log_info "Showing logs for ${service}..."
        docker-compose logs -f "$service"
    else
        log_info "Showing logs for all services..."
        docker-compose logs -f
    fi
}

show_status() {
    log_info "Service status:"
    docker-compose ps
    
    echo
    log_info "Docker images:"
    for service in "${SERVICES[@]}"; do
        local image="${REGISTRY}/${IMAGE_PREFIX}:${service}"
        if docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | grep -q "${IMAGE_PREFIX}:${service}"; then
            echo "✓ ${image}"
        else
            echo "✗ ${image} (not found locally)"
        fi
    done
}

run_service() {
    local service="$1"
    
    if [[ ! " ${SERVICES[@]} " =~ " ${service} " ]]; then
        log_error "Invalid service: ${service}"
        log_info "Available services: ${SERVICES[*]}"
        exit 1
    fi
    
    log_info "Running ${service} service manually..."
    
    case "${service}" in
        "fetcher")
            docker-compose --profile manual run --rm data-fetcher
            ;;
        "optimizer")
            docker-compose --profile manual run --rm optimizer
            ;;
        "executor")
            docker-compose --profile manual run --rm executor
            ;;
        "scheduler")
            log_warning "Scheduler runs automatically. Use 'up' command to start it."
            ;;
    esac
}

clean_up() {
    check_docker
    log_info "Cleaning up portfolio rebalancer containers and images..."
    
    log_info "Stopping and removing portfolio rebalancer containers..."
    docker-compose down --remove-orphans 2>/dev/null || true
    
    log_info "Removing stopped portfolio containers..."
    docker container ls -a --filter "name=portfolio" --format "{{.ID}}" | xargs -r docker container rm 2>/dev/null || true
    
    log_info "Removing portfolio rebalancer images..."
    for service in "${SERVICES[@]}"; do
        local image="${REGISTRY}/${IMAGE_PREFIX}:${service}"
        
        if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMAGE_PREFIX}:${service}"; then
            log_info "Removing image: ${image}"
            docker rmi "${image}" 2>/dev/null || log_warning "Could not remove ${image} (may be in use)"
        fi
    done
    
    # Remove dangling images from our builds
    log_info "Removing dangling images from portfolio builds..."
    docker images --filter "dangling=true" --filter "label=project=portfolio-rebalancer" --format "{{.ID}}" | xargs -r docker rmi 2>/dev/null || true
    
    # Clean up project-specific volumes
    read -p "Remove portfolio rebalancer volumes? This will delete your data (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing portfolio volumes..."
        docker volume ls --filter "name=portfolio" --format "{{.Name}}" | xargs -r docker volume rm 2>/dev/null || true
        log_warning "Portfolio data volumes removed!"
    fi
    
    # Clean up project-specific networks
    log_info "Removing portfolio networks..."
    docker network ls --filter "name=portfolio" --format "{{.Name}}" | xargs -r docker network rm 2>/dev/null || true
    
    log_success "Portfolio rebalancer cleanup completed!"
    log_info "Other Docker resources left untouched"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            help
            exit 0
            ;;
        up)
            COMMAND="up"
            shift
            SUBCOMMAND="$1"
            shift || true
            ;;
        down)
            COMMAND="down"
            shift
            ;;
        restart)
            COMMAND="restart"
            shift
            SUBCOMMAND="$1"
            shift || true
            ;;
        logs)
            COMMAND="logs"
            shift
            SERVICE="$1"
            shift || true
            ;;
        status)
            COMMAND="status"
            shift
            ;;
        pull)
            COMMAND="pull"
            shift
            ;;
        clean)
            COMMAND="clean"
            shift
            ;;
        run)
            COMMAND="run"
            shift
            SERVICE="$1"
            shift || true
            ;;
        *)
            log_error "Unknown option: $1"
            help
            exit 1
            ;;
    esac
done

# Execute commands
case "${COMMAND:-pull}" in
    "up")
        start_services "$SUBCOMMAND"
        ;;
    "down")
        stop_services
        ;;
    "restart")
        restart_services "$SUBCOMMAND"
        ;;
    "logs")
        show_logs "$SERVICE"
        ;;
    "status")
        show_status
        ;;
    "pull")
        pull_images
        ;;
    "delete")
        clean_up
        ;;
    "run")
        if [[ -z "$SERVICE" ]]; then
            log_error "Please specify a service to run"
            log_info "Available services: ${SERVICES[*]}"
            exit 1
        fi
        run_service "$SERVICE"
        ;;
    *)
        pull_images
        ;;
esac