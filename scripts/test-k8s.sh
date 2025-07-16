#!/bin/bash

# Kubernetes Testing Script for Portfolio Rebalancer
# This script tests the deployed portfolio rebalancer system in Kubernetes

set -e

# Configuration
NAMESPACE="portfolio-rebalancer"
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
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "kubectl is available and connected"
}

# Function to check namespace exists
check_namespace() {
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist. Please deploy first."
        exit 1
    fi
    log_success "Namespace $NAMESPACE exists"
}

# Function to test resource deployment
test_resources() {
    log_info "Testing resource deployment..."
    
    # Check ConfigMap
    if kubectl get configmap portfolio-config -n $NAMESPACE &> /dev/null; then
        log_success "ConfigMap exists"
    else
        log_error "ConfigMap not found"
        return 1
    fi
    
    # Check Secret
    if kubectl get secret broker-credentials -n $NAMESPACE &> /dev/null; then
        log_success "Secret exists"
    else
        log_error "Secret not found"
        return 1
    fi
    
    # Check PVCs
    local pvc_status
    pvc_status=$(kubectl get pvc portfolio-data-pvc -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
    if [[ "$pvc_status" == "Bound" ]]; then
        log_success "Data PVC is bound"
    else
        log_error "Data PVC status: $pvc_status"
        return 1
    fi
    
    pvc_status=$(kubectl get pvc portfolio-logs-pvc -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
    if [[ "$pvc_status" == "Bound" ]]; then
        log_success "Logs PVC is bound"
    else
        log_error "Logs PVC status: $pvc_status"
        return 1
    fi
    
    # Check Services
    local services=("portfolio-fetcher-service" "portfolio-optimizer-service" "portfolio-executor-service")
    for service in "${services[@]}"; do
        if kubectl get service $service -n $NAMESPACE &> /dev/null; then
            log_success "Service $service exists"
        else
            log_error "Service $service not found"
            return 1
        fi
    done
    
    # Check Deployments
    local deployments=("portfolio-fetcher" "portfolio-optimizer" "portfolio-executor")
    for deployment in "${deployments[@]}"; do
        local ready_replicas
        ready_replicas=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired_replicas
        desired_replicas=$(kubectl get deployment $deployment -n $NAMESPACE -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")
        
        if [[ "$ready_replicas" == "$desired_replicas" ]]; then
            log_success "Deployment $deployment is ready ($ready_replicas/$desired_replicas)"
        else
            log_error "Deployment $deployment not ready ($ready_replicas/$desired_replicas)"
            return 1
        fi
    done
    
    # Check CronJob
    if kubectl get cronjob portfolio-rebalancer-daily -n $NAMESPACE &> /dev/null; then
        log_success "CronJob exists"
    else
        log_error "CronJob not found"
        return 1
    fi
    
    log_success "All resources deployed correctly"
}

# Function to test pod health
test_pod_health() {
    log_info "Testing pod health..."
    
    local deployments=("portfolio-fetcher" "portfolio-optimizer" "portfolio-executor")
    for deployment in "${deployments[@]}"; do
        local pod_name
        pod_name=$(kubectl get pods -n $NAMESPACE -l app=portfolio-rebalancer,component=${deployment#portfolio-} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        
        if [[ -z "$pod_name" ]]; then
            log_error "No pod found for deployment $deployment"
            continue
        fi
        
        local pod_status
        pod_status=$(kubectl get pod $pod_name -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        
        if [[ "$pod_status" == "Running" ]]; then
            log_success "Pod $pod_name is running"
            
            # Test readiness probe if available
            local ready_condition
            ready_condition=$(kubectl get pod $pod_name -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")
            if [[ "$ready_condition" == "True" ]]; then
                log_success "Pod $pod_name is ready"
            else
                log_warning "Pod $pod_name is not ready"
            fi
        else
            log_error "Pod $pod_name status: $pod_status"
        fi
    done
}

# Function to test service connectivity
test_service_connectivity() {
    log_info "Testing service connectivity..."
    
    local services=(
        "portfolio-fetcher-service:8080"
        "portfolio-optimizer-service:8081"
        "portfolio-executor-service:8082"
    )
    
    for service_port in "${services[@]}"; do
        local service_name="${service_port%:*}"
        local port="${service_port#*:}"
        
        # Test if service endpoint exists
        local endpoints
        endpoints=$(kubectl get endpoints $service_name -n $NAMESPACE -o jsonpath='{.subsets[0].addresses[0].ip}' 2>/dev/null || echo "")
        
        if [[ -n "$endpoints" ]]; then
            log_success "Service $service_name has endpoints"
            
            # Test connectivity using a temporary pod
            log_info "Testing connectivity to $service_name:$port..."
            if kubectl run test-connectivity-$RANDOM --rm -i --restart=Never --image=curlimages/curl:latest -n $NAMESPACE -- curl -s --connect-timeout 10 http://$service_name:$port/health &> /dev/null; then
                log_success "Service $service_name is reachable"
            else
                log_warning "Service $service_name health check failed (this may be expected if health endpoints are not implemented)"
            fi
        else
            log_error "Service $service_name has no endpoints"
        fi
    done
}

# Function to test configuration
test_configuration() {
    log_info "Testing configuration..."
    
    # Check ConfigMap data
    local config_keys
    config_keys=$(kubectl get configmap portfolio-config -n $NAMESPACE -o jsonpath='{.data}' 2>/dev/null || echo "{}")
    
    if [[ "$config_keys" != "{}" ]]; then
        log_success "ConfigMap contains configuration data"
        
        # Check for required keys
        local required_keys=("TICKERS" "STORAGE_TYPE" "USER_AGE" "REBALANCE_THRESHOLD")
        for key in "${required_keys[@]}"; do
            local value
            value=$(kubectl get configmap portfolio-config -n $NAMESPACE -o jsonpath="{.data.$key}" 2>/dev/null || echo "")
            if [[ -n "$value" ]]; then
                log_success "ConfigMap has $key: $value"
            else
                log_error "ConfigMap missing required key: $key"
            fi
        done
    else
        log_error "ConfigMap has no data"
    fi
    
    # Check Secret data (without revealing values)
    local secret_keys
    secret_keys=$(kubectl get secret broker-credentials -n $NAMESPACE -o jsonpath='{.data}' 2>/dev/null || echo "{}")
    
    if [[ "$secret_keys" != "{}" ]]; then
        log_success "Secret contains credential data"
    else
        log_error "Secret has no data"
    fi
}

# Function to test storage
test_storage() {
    log_info "Testing storage..."
    
    # Test if PVCs are mounted in pods
    local deployments=("portfolio-fetcher" "portfolio-optimizer" "portfolio-executor")
    for deployment in "${deployments[@]}"; do
        local pod_name
        pod_name=$(kubectl get pods -n $NAMESPACE -l app=portfolio-rebalancer,component=${deployment#portfolio-} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        
        if [[ -n "$pod_name" ]]; then
            # Check if data volume is mounted
            if kubectl exec $pod_name -n $NAMESPACE -- ls /app/data &> /dev/null; then
                log_success "Data volume mounted in $pod_name"
            else
                log_error "Data volume not accessible in $pod_name"
            fi
            
            # Check if logs volume is mounted
            if kubectl exec $pod_name -n $NAMESPACE -- ls /app/logs &> /dev/null; then
                log_success "Logs volume mounted in $pod_name"
            else
                log_error "Logs volume not accessible in $pod_name"
            fi
        fi
    done
}

# Function to test cronjob
test_cronjob() {
    log_info "Testing CronJob..."
    
    local cronjob_name="portfolio-rebalancer-daily"
    
    # Check if CronJob is scheduled correctly
    local schedule
    schedule=$(kubectl get cronjob $cronjob_name -n $NAMESPACE -o jsonpath='{.spec.schedule}' 2>/dev/null || echo "")
    
    if [[ -n "$schedule" ]]; then
        log_success "CronJob scheduled: $schedule"
        
        # Check last schedule time
        local last_schedule
        last_schedule=$(kubectl get cronjob $cronjob_name -n $NAMESPACE -o jsonpath='{.status.lastScheduleTime}' 2>/dev/null || echo "Never")
        log_info "Last scheduled: $last_schedule"
        
        # Check active jobs
        local active_jobs
        active_jobs=$(kubectl get cronjob $cronjob_name -n $NAMESPACE -o jsonpath='{.status.active}' 2>/dev/null || echo "0")
        log_info "Active jobs: $active_jobs"
        
    else
        log_error "CronJob schedule not found"
    fi
}

# Function to show logs
show_logs() {
    log_info "Recent logs from deployments:"
    
    local deployments=("portfolio-fetcher" "portfolio-optimizer" "portfolio-executor")
    for deployment in "${deployments[@]}"; do
        echo
        log_info "Logs from $deployment:"
        kubectl logs deployment/$deployment -n $NAMESPACE --tail=10 --since=1h 2>/dev/null || log_warning "No logs available for $deployment"
    done
}

# Function to run manual test job
run_manual_test() {
    log_info "Running manual test job..."
    
    local test_job_name="portfolio-test-$(date +%s)"
    
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: $test_job_name
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: test-runner
        image: portfolio-rebalancer/scheduler:latest
        command: ["python", "-c", "print('Test job completed successfully')"]
        envFrom:
        - configMapRef:
            name: portfolio-config
        - secretRef:
            name: broker-credentials
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
        - name: logs-storage
          mountPath: /app/logs
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: portfolio-data-pvc
      - name: logs-storage
        persistentVolumeClaim:
          claimName: portfolio-logs-pvc
EOF

    # Wait for job completion
    log_info "Waiting for test job to complete..."
    kubectl wait --for=condition=complete job/$test_job_name -n $NAMESPACE --timeout=300s
    
    # Show job logs
    kubectl logs job/$test_job_name -n $NAMESPACE
    
    # Clean up test job
    kubectl delete job $test_job_name -n $NAMESPACE
    
    log_success "Manual test job completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --basic      Run basic deployment tests (default)"
    echo "  --full       Run comprehensive tests including connectivity"
    echo "  --logs       Show recent logs from all services"
    echo "  --manual     Run a manual test job"
    echo "  --help       Show this help message"
    echo
    echo "Examples:"
    echo "  $0           # Run basic tests"
    echo "  $0 --full    # Run comprehensive tests"
    echo "  $0 --logs    # Show logs"
    echo "  $0 --manual  # Run manual test"
}

# Main function
main() {
    local test_type="basic"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --basic)
                test_type="basic"
                shift
                ;;
            --full)
                test_type="full"
                shift
                ;;
            --logs)
                test_type="logs"
                shift
                ;;
            --manual)
                test_type="manual"
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
    
    log_info "Starting Kubernetes tests for Portfolio Rebalancer"
    
    # Check prerequisites
    check_kubectl
    check_namespace
    
    case $test_type in
        "basic")
            test_resources
            test_pod_health
            test_configuration
            test_storage
            test_cronjob
            log_success "Basic tests completed"
            ;;
        "full")
            test_resources
            test_pod_health
            test_configuration
            test_storage
            test_cronjob
            test_service_connectivity
            log_success "Full tests completed"
            ;;
        "logs")
            show_logs
            ;;
        "manual")
            run_manual_test
            ;;
    esac
    
    log_info "Testing completed"
}

# Run main function
main "$@"