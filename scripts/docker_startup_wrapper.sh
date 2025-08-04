#!/bin/bash

# Docker startup wrapper with built-in delays and retry logic
# This script can be used as an entrypoint in Dockerfiles

set -e

# Configuration from environment variables
SERVICE_STARTUP_DELAY=${SERVICE_STARTUP_DELAY:-30}
SERVICE_RETRY_ATTEMPTS=${SERVICE_RETRY_ATTEMPTS:-5}
SERVICE_RETRY_DELAY=${SERVICE_RETRY_DELAY:-10}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STARTUP] $1"
}

# Startup delay
if [ "$SERVICE_STARTUP_DELAY" -gt 0 ]; then
    log "Applying startup delay of ${SERVICE_STARTUP_DELAY} seconds..."
    sleep $SERVICE_STARTUP_DELAY
fi

# Execute the main command with retry logic
attempts=0
while [ $attempts -lt $SERVICE_RETRY_ATTEMPTS ]; do
    attempts=$((attempts + 1))
    log "Starting service, attempt $attempts/$SERVICE_RETRY_ATTEMPTS"
    
    # Execute the original command
    if "$@"; then
        log "Service started successfully"
        exit 0
    else
        exit_code=$?
        if [ $attempts -lt $SERVICE_RETRY_ATTEMPTS ]; then
            log "Service startup failed (exit code: $exit_code), retrying in ${SERVICE_RETRY_DELAY}s..."
            sleep $SERVICE_RETRY_DELAY
        else
            log "Service startup failed after $SERVICE_RETRY_ATTEMPTS attempts (exit code: $exit_code)"
            exit $exit_code
        fi
    fi
done