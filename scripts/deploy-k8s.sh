#!/bin/bash

# Kubernetes Deployment Script for Portfolio Rebalancer
# This script deploys the portfolio rebalancer system to a Kubernetes cluster

set -e

# Configuration
NAMESPACE="portfolio-rebalancer"
K8S_DIR="k8s"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig"
        exit 1
    fi
    
    log_success "kubectl is available and connected to cluster"
}

# Function to validate required files
validate_files() {
    local required_files=(
        "$K8S_DIR/namespace.yaml"
        "$K8S_DIR/configmap.yaml"
        "$K8S_DIR/secret.yaml"
        "$K8S_DIR/pvc.yaml"
        "$K8S_DIR/deployments.yaml"
        "$K8S_DIR/services.yaml"
        "$K8S_DIR/cronjob.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_success "All required Kubernetes manifest files found"
}

# Function to create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    kubectl apply -f "$PROJECT_ROOT/$K8S_DIR/namespace.yaml"
    log_success "Namespace created/updated"
}

# Function to deploy configuration
deploy_config() {
    log_info "Deploying ConfigMap and Secret"
    kubectl apply -f "$PROJECT_ROOT/$K8S_DIR/configmap.yaml"
    kubectl apply -f "$PROJECT_ROOT/$K8S_DIR/secret.yaml"
    log_success "Configuration deployed"
}

# Function to create persistent volumes
create_storage() {
    log_info "Creating PersistentVolumeClaims"
    kubectl apply -f "$PROJECT_ROOT/$K8S_DIR/pvc.yaml"
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc/portfolio-data-pvc -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=Bound pvc/portfolio-logs-pvc -n $NAMESPACE --timeout=300s
    log_success "Storage created and bound"
}

# Function to deploy services
deploy_services() {
    log_info "Deploying Services"
    kubectl apply -f "$PROJECT_ROOT/$K8S_DIR/services.yaml"
    log_success "Services deployed"
}

# Function to deploy applications
deploy_apps() {
    log_info "Deploying application Deployments"
    kubectl apply -f "$PROJECT_ROOT/$K8S_DIR/deployments.yaml"
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=Available deployment/portfolio-fetcher -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=Available deployment/portfolio-optimizer -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=Available deployment/portfolio-executor -n $NAMESPACE --timeout=300s
    log_success "Applications deployed and ready"
}

# Function to deploy cronjob
deploy_cronjob() {
    log_info "Deploying CronJob for scheduled execution"
    kubectl apply -f "$PROJECT_ROOT/$K8S_DIR/cronjob.yaml"
    log_success "CronJob deployed"
}

# Function to show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    echo "Namespace:"
    kubectl get namespace $NAMESPACE
    echo
    echo "ConfigMaps and Secrets:"
    kubectl get configmap,secret -n $NAMESPACE
    echo
    echo "Storage:"
    kubectl get pvc -n $NAMESPACE
    echo
    echo "Services:"
    kubectl get services -n $NAMESPACE
    echo
    echo "Deployments:"
    kubectl get deployments -n $NAMESPACE
    echo
    echo "Pods:"
    kubectl get pods -n $NAMESPACE
    echo
    echo "CronJob:"
    kubectl get cronjob -n $NAMESPACE
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --dry-run    Show what would be deployed without actually deploying"
    echo "  --status     Show current deployment status"
    echo "  --help       Show this help message"
    echo
    echo "Examples:"
    echo "  $0                 # Deploy everything"
    echo "  $0 --dry-run       # Show what would be deployed"
    echo "  $0 --status        # Show current status"
}

# Main deployment function
main() {
    local dry_run=false
    local show_status_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            --status)
                show_status_only=true
                shift
                ;;
            --help)
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
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    log_info "Starting Kubernetes deployment for Portfolio Rebalancer"
    
    # Check prerequisites
    check_kubectl
    validate_files
    
    if [[ "$show_status_only" == true ]]; then
        show_status
        exit 0
    fi
    
    if [[ "$dry_run" == true ]]; then
        log_info "DRY RUN MODE - No changes will be made"
        log_info "Would deploy the following resources:"
        echo "- Namespace: $NAMESPACE"
        echo "- ConfigMap: portfolio-config"
        echo "- Secret: broker-credentials"
        echo "- PVC: portfolio-data-pvc, portfolio-logs-pvc"
        echo "- Services: portfolio-fetcher-service, portfolio-optimizer-service, portfolio-executor-service"
        echo "- Deployments: portfolio-fetcher, portfolio-optimizer, portfolio-executor"
        echo "- CronJob: portfolio-rebalancer-daily"
        exit 0
    fi
    
    # Deploy components in order
    create_namespace
    deploy_config
    create_storage
    deploy_services
    deploy_apps
    deploy_cronjob
    
    log_success "Deployment completed successfully!"
    echo
    show_status
    
    log_info "Next steps:"
    echo "1. Update the Secret with your actual broker API credentials:"
    echo "   kubectl edit secret broker-credentials -n $NAMESPACE"
    echo "2. Monitor the CronJob execution:"
    echo "   kubectl get cronjob -n $NAMESPACE"
    echo "3. Check logs:"
    echo "   kubectl logs -f deployment/portfolio-fetcher -n $NAMESPACE"
    echo "4. Access services (if needed):"
    echo "   kubectl port-forward service/portfolio-fetcher-service 8080:8080 -n $NAMESPACE"
}

# Run main function
main "$@"