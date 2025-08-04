#!/bin/bash

# Service startup script with retry logic and delays
# This script provides robust startup handling for portfolio rebalancer services

set -e

# Configuration from environment variables
SERVICE_NAME=${SERVICE_NAME:-"unknown"}
SERVICE_STARTUP_DELAY=${SERVICE_STARTUP_DELAY:-30}
SERVICE_RETRY_ATTEMPTS=${SERVICE_RETRY_ATTEMPTS:-5}
SERVICE_RETRY_DELAY=${SERVICE_RETRY_DELAY:-10}
SERVICE_HEALTH_CHECK_URL=${SERVICE_HEALTH_CHECK_URL:-"http://localhost:8080/health"}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$SERVICE_NAME] $1"
}

# Wait for dependencies function
wait_for_dependencies() {
    local dependencies=("$@")
    
    for dep in "${dependencies[@]}"; do
        log "Waiting for dependency: $dep"
        
        local attempts=0
        while [ $attempts -lt $SERVICE_RETRY_ATTEMPTS ]; do
            if curl -f -s "$dep" > /dev/null 2>&1; then
                log "Dependency $dep is ready"
                break
            fi
            
            attempts=$((attempts + 1))
            if [ $attempts -lt $SERVICE_RETRY_ATTEMPTS ]; then
                log "Dependency $dep not ready, attempt $attempts/$SERVICE_RETRY_ATTEMPTS. Retrying in ${SERVICE_RETRY_DELAY}s..."
                sleep $SERVICE_RETRY_DELAY
            else
                log "WARNING: Dependency $dep not ready after $SERVICE_RETRY_ATTEMPTS attempts. Continuing anyway..."
            fi
        done
    done
}

# Service initialization with retry logic
initialize_service() {
    local init_command="$1"
    
    log "Starting service initialization with ${SERVICE_STARTUP_DELAY}s delay..."
    sleep $SERVICE_STARTUP_DELAY
    
    local attempts=0
    while [ $attempts -lt $SERVICE_RETRY_ATTEMPTS ]; do
        attempts=$((attempts + 1))
        log "Service initialization attempt $attempts/$SERVICE_RETRY_ATTEMPTS"
        
        if eval "$init_command"; then
            log "Service initialization successful"
            return 0
        else
            if [ $attempts -lt $SERVICE_RETRY_ATTEMPTS ]; then
                log "Service initialization failed, retrying in ${SERVICE_RETRY_DELAY}s..."
                sleep $SERVICE_RETRY_DELAY
            else
                log "ERROR: Service initialization failed after $SERVICE_RETRY_ATTEMPTS attempts"
                return 1
            fi
        fi
    done
}

# Health check function
wait_for_health() {
    local health_url="$1"
    local max_wait=${2:-300}  # 5 minutes default
    local wait_time=0
    
    log "Waiting for service health check at $health_url"
    
    while [ $wait_time -lt $max_wait ]; do
        if curl -f -s "$health_url" > /dev/null 2>&1; then
            log "Service health check passed"
            return 0
        fi
        
        sleep 5
        wait_time=$((wait_time + 5))
        
        if [ $((wait_time % 30)) -eq 0 ]; then
            log "Still waiting for health check... (${wait_time}s elapsed)"
        fi
    done
    
    log "WARNING: Health check did not pass within ${max_wait}s"
    return 1
}

# Graceful shutdown handler
shutdown_handler() {
    log "Received shutdown signal, performing graceful shutdown..."
    # Add any cleanup logic here
    exit 0
}

# Set up signal handlers
trap shutdown_handler SIGTERM SIGINT

# Main startup function
main() {
    log "Starting $SERVICE_NAME service startup sequence"
    
    # Parse command line arguments
    case "$1" in
        "fetcher")
            SERVICE_NAME="fetcher"
            SERVICE_HEALTH_CHECK_URL="http://localhost:8080/health"
            ;;
        "optimizer")
            SERVICE_NAME="optimizer"
            SERVICE_HEALTH_CHECK_URL="http://localhost:8081/health"
            ;;
        "executor")
            SERVICE_NAME="executor"
            SERVICE_HEALTH_CHECK_URL="http://localhost:8082/health"
            ;;
        "scheduler")
            SERVICE_NAME="scheduler"
            SERVICE_HEALTH_CHECK_URL="http://localhost:8083/health"
            ;;
        *)
            log "Usage: $0 {fetcher|optimizer|executor|scheduler} [command...]"
            exit 1
            ;;
    esac
    
    # Shift to get the actual service command
    shift
    
    # Initialize service with retry logic
    if initialize_service "$*"; then
        log "Service startup completed successfully"
        
        # Keep the script running to maintain the container
        while true; do
            sleep 30
            # Optional: perform periodic health checks or maintenance
        done
    else
        log "Service startup failed"
        exit 1
    fi
}

# Run main function if script is executed directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi