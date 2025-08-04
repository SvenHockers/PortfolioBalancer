"""Analytics-specific monitoring and metrics collection."""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

# Import psutil conditionally
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create dummy psutil functions
    class DummyPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 0.0
        
        @staticmethod
        def virtual_memory():
            class Memory:
                percent = 0.0
                available = 1024**3  # 1GB
            return Memory()
        
        @staticmethod
        def disk_usage(path):
            class Disk:
                used = 0
                total = 1024**3  # 1GB
            return Disk()
    
    psutil = DummyPsutil()

from ..common.metrics import (
    metrics_registry, timed, record_portfolio_metrics,
    is_prometheus_available
)
from .config import get_analytics_config
from .exceptions import AnalyticsError, ErrorSeverity, ErrorCategory
from .logging import get_analytics_logger

# Import prometheus_client conditionally
if is_prometheus_available():
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info
else:
    # Use dummy classes if prometheus not available
    class DummyMetric:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def info(self, *args, **kwargs): pass
    
    Counter = Gauge = Histogram = Summary = Info = DummyMetric


@dataclass
class OperationMetrics:
    """Metrics for a specific operation."""
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_count == 0:
            return 1.0
        return self.success_count / self.total_count
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate
    
    @property
    def average_duration(self) -> float:
        """Calculate average duration."""
        if self.total_count == 0:
            return 0.0
        return self.total_duration / self.total_count


class AnalyticsMonitor:
    """Analytics-specific monitoring and metrics collection."""
    
    def __init__(self):
        """Initialize analytics monitor."""
        self.config = get_analytics_config()
        self.logger = get_analytics_logger("monitor")
        self._operation_metrics: Dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self._resource_monitor_thread = None
        self._monitoring_active = False
        
        # Initialize Prometheus metrics if available
        if is_prometheus_available():
            self._init_prometheus_metrics()
        
        # Start resource monitoring if enabled
        if self.config.monitoring.enable_resource_monitoring:
            self.start_resource_monitoring()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for analytics."""
        # Analytics operation metrics
        self.analytics_operations_total = Counter(
            'portfolio_analytics_operations_total',
            'Total number of analytics operations',
            ['operation', 'status']
        )
        
        self.analytics_operation_duration = Histogram(
            'portfolio_analytics_operation_duration_seconds',
            'Duration of analytics operations in seconds',
            ['operation'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, float('inf')]
        )
        
        # Backtest metrics
        self.backtest_results = Gauge(
            'portfolio_analytics_backtest_results',
            'Backtest result metrics',
            ['portfolio_id', 'metric']
        )
        
        # Monte Carlo metrics
        self.monte_carlo_results = Gauge(
            'portfolio_analytics_monte_carlo_results',
            'Monte Carlo simulation results',
            ['portfolio_id', 'metric']
        )
        
        # Risk analysis metrics
        self.risk_analysis_results = Gauge(
            'portfolio_analytics_risk_analysis_results',
            'Risk analysis results',
            ['portfolio_id', 'metric']
        )
        
        # Performance metrics
        self.performance_metrics = Gauge(
            'portfolio_analytics_performance_metrics',
            'Portfolio performance metrics',
            ['portfolio_id', 'metric']
        )
        
        # System resource metrics
        self.system_resources = Gauge(
            'portfolio_analytics_system_resources',
            'System resource usage',
            ['resource']
        )
        
        # Database metrics
        self.database_operations = Counter(
            'portfolio_analytics_database_operations_total',
            'Database operations count',
            ['operation', 'table', 'status']
        )
        
        self.database_operation_duration = Histogram(
            'portfolio_analytics_database_operation_duration_seconds',
            'Database operation duration',
            ['operation', 'table']
        )
        
        # Cache metrics
        self.cache_operations = Counter(
            'portfolio_analytics_cache_operations_total',
            'Cache operations count',
            ['operation', 'result']
        )
        
        # Error metrics
        self.error_count = Counter(
            'portfolio_analytics_errors_total',
            'Total number of errors',
            ['error_type', 'severity', 'category']
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'portfolio_analytics_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['circuit_name']
        )
        
        # Degradation level
        self.degradation_level = Gauge(
            'portfolio_analytics_degradation_level',
            'Service degradation level (0=none, 1=partial, 2=minimal, 3=emergency)'
        )
    
    def record_operation(self, operation: str, duration: float, success: bool, error: Exception = None):
        """Record metrics for an analytics operation."""
        # Update internal metrics
        metrics = self._operation_metrics[operation]
        metrics.total_count += 1
        metrics.total_duration += duration
        metrics.min_duration = min(metrics.min_duration, duration)
        metrics.max_duration = max(metrics.max_duration, duration)
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
            if error:
                metrics.recent_errors.append({
                    'timestamp': datetime.now(timezone.utc),
                    'error': str(error),
                    'type': type(error).__name__
                })
        
        # Update Prometheus metrics
        if is_prometheus_available():
            status = 'success' if success else 'failure'
            self.analytics_operations_total.labels(operation=operation, status=status).inc()
            self.analytics_operation_duration.labels(operation=operation).observe(duration)
        
        # Check for alerts
        self._check_operation_alerts(operation, duration, success, error)
    
    def record_backtest_results(self, portfolio_id: str, results: Dict[str, Any]):
        """Record backtest results metrics."""
        if not is_prometheus_available():
            return
        
        metrics_to_record = {
            'total_return': results.get('total_return', 0),
            'annualized_return': results.get('annualized_return', 0),
            'volatility': results.get('volatility', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'calmar_ratio': results.get('calmar_ratio', 0)
        }
        
        for metric_name, value in metrics_to_record.items():
            if value is not None:
                self.backtest_results.labels(
                    portfolio_id=portfolio_id,
                    metric=metric_name
                ).set(value)
    
    def record_monte_carlo_results(self, portfolio_id: str, results: Dict[str, Any]):
        """Record Monte Carlo simulation results."""
        if not is_prometheus_available():
            return
        
        metrics_to_record = {
            'expected_value': results.get('expected_value', 0),
            'probability_of_loss': results.get('probability_of_loss', 0),
            'var_95': results.get('value_at_risk_95', 0),
            'cvar_95': results.get('conditional_var_95', 0)
        }
        
        for metric_name, value in metrics_to_record.items():
            if value is not None:
                self.monte_carlo_results.labels(
                    portfolio_id=portfolio_id,
                    metric=metric_name
                ).set(value)
    
    def record_risk_analysis_results(self, portfolio_id: str, results: Dict[str, Any]):
        """Record risk analysis results."""
        if not is_prometheus_available():
            return
        
        metrics_to_record = {
            'portfolio_beta': results.get('portfolio_beta', 0),
            'tracking_error': results.get('tracking_error', 0),
            'information_ratio': results.get('information_ratio', 0),
            'var_95': results.get('var_95', 0),
            'cvar_95': results.get('cvar_95', 0),
            'concentration_risk': results.get('concentration_risk', 0)
        }
        
        for metric_name, value in metrics_to_record.items():
            if value is not None:
                self.risk_analysis_results.labels(
                    portfolio_id=portfolio_id,
                    metric=metric_name
                ).set(value)
    
    def record_performance_metrics(self, portfolio_id: str, metrics: Dict[str, Any]):
        """Record performance metrics."""
        if not is_prometheus_available():
            return
        
        metrics_to_record = {
            'total_return': metrics.get('total_return', 0),
            'annualized_return': metrics.get('annualized_return', 0),
            'volatility': metrics.get('volatility', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'alpha': metrics.get('alpha', 0),
            'beta': metrics.get('beta', 0),
            'tracking_error': metrics.get('tracking_error', 0)
        }
        
        for metric_name, value in metrics_to_record.items():
            if value is not None:
                self.performance_metrics.labels(
                    portfolio_id=portfolio_id,
                    metric=metric_name
                ).set(value)
    
    def record_database_operation(self, operation: str, table: str, duration: float, success: bool):
        """Record database operation metrics."""
        if not is_prometheus_available():
            return
        
        status = 'success' if success else 'failure'
        self.database_operations.labels(
            operation=operation,
            table=table,
            status=status
        ).inc()
        
        self.database_operation_duration.labels(
            operation=operation,
            table=table
        ).observe(duration)
    
    def record_cache_operation(self, operation: str, hit: bool = None):
        """Record cache operation metrics."""
        if not is_prometheus_available():
            return
        
        if hit is not None:
            result = 'hit' if hit else 'miss'
        else:
            result = 'operation'
        
        self.cache_operations.labels(
            operation=operation,
            result=result
        ).inc()
    
    def record_error(self, error: Exception):
        """Record error metrics."""
        if not is_prometheus_available():
            return
        
        error_type = type(error).__name__
        
        if isinstance(error, AnalyticsError):
            severity = error.severity.value
            category = error.category.value
        else:
            severity = 'medium'
            category = 'unknown'
        
        self.error_count.labels(
            error_type=error_type,
            severity=severity,
            category=category
        ).inc()
    
    def record_circuit_breaker_state(self, circuit_name: str, state: str):
        """Record circuit breaker state."""
        if not is_prometheus_available():
            return
        
        state_mapping = {
            'closed': 0,
            'open': 1,
            'half_open': 2
        }
        
        state_value = state_mapping.get(state, 0)
        self.circuit_breaker_state.labels(circuit_name=circuit_name).set(state_value)
    
    def record_degradation_level(self, level: str):
        """Record service degradation level."""
        if not is_prometheus_available():
            return
        
        level_mapping = {
            'none': 0,
            'partial': 1,
            'minimal': 2,
            'emergency': 3
        }
        
        level_value = level_mapping.get(level, 0)
        self.degradation_level.set(level_value)
    
    def start_resource_monitoring(self):
        """Start background resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_loop,
            daemon=True
        )
        self._resource_monitor_thread.start()
        self.logger.log_operation_start("resource_monitoring")
    
    def stop_resource_monitoring(self):
        """Stop background resource monitoring."""
        self._monitoring_active = False
        if self._resource_monitor_thread:
            self._resource_monitor_thread.join(timeout=5)
    
    def _resource_monitor_loop(self):
        """Background loop for resource monitoring."""
        while self._monitoring_active:
            try:
                self._collect_resource_metrics()
                time.sleep(self.config.monitoring.metric_collection_interval)
            except Exception as e:
                self.logger.log_operation_failure("resource_monitoring", e, 0)
                time.sleep(30)  # Wait longer on error
    
    def _collect_resource_metrics(self):
        """Collect system resource metrics."""
        if not is_prometheus_available() or not PSUTIL_AVAILABLE:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_resources.labels(resource='cpu_percent').set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_resources.labels(resource='memory_percent').set(memory.percent)
            self.system_resources.labels(resource='memory_available_gb').set(memory.available / (1024**3))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_resources.labels(resource='disk_percent').set(disk_percent)
            
            # Check thresholds and alert if necessary
            self._check_resource_alerts(cpu_percent, memory.percent, disk_percent)
            
        except Exception as e:
            self.logger.log_operation_failure("resource_collection", e, 0)
    
    def _check_operation_alerts(self, operation: str, duration: float, success: bool, error: Exception = None):
        """Check for operation-specific alerts."""
        config = self.config.monitoring
        
        # Check duration thresholds
        duration_thresholds = {
            'backtest': config.backtest_duration_alert_threshold,
            'monte_carlo': config.monte_carlo_duration_alert_threshold,
            'risk_analysis': config.risk_analysis_duration_alert_threshold
        }
        
        threshold = duration_thresholds.get(operation)
        if threshold and duration > threshold:
            self.logger.log_operation_failure(
                "duration_alert",
                Exception(f"Operation {operation} exceeded duration threshold"),
                duration,
                operation=operation,
                threshold=threshold
            )
        
        # Check error rates
        if not success and error:
            metrics = self._operation_metrics[operation]
            if metrics.total_count >= 10:  # Only check after sufficient samples
                if metrics.failure_rate > config.error_rate_threshold:
                    self.logger.log_operation_failure(
                        "error_rate_alert",
                        Exception(f"Operation {operation} error rate too high"),
                        0,
                        operation=operation,
                        error_rate=metrics.failure_rate,
                        threshold=config.error_rate_threshold
                    )
    
    def _check_resource_alerts(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Check resource usage alerts."""
        config = self.config.monitoring
        
        if cpu_percent > config.cpu_usage_threshold * 100:
            self.logger.log_operation_failure(
                "cpu_alert",
                Exception("CPU usage too high"),
                0,
                cpu_percent=cpu_percent,
                threshold=config.cpu_usage_threshold * 100
            )
        
        if memory_percent > config.memory_usage_threshold * 100:
            self.logger.log_operation_failure(
                "memory_alert",
                Exception("Memory usage too high"),
                0,
                memory_percent=memory_percent,
                threshold=config.memory_usage_threshold * 100
            )
        
        if disk_percent > config.disk_usage_threshold * 100:
            self.logger.log_operation_failure(
                "disk_alert",
                Exception("Disk usage too high"),
                0,
                disk_percent=disk_percent,
                threshold=config.disk_usage_threshold * 100
            )
    
    def get_operation_metrics(self, operation: str = None) -> Dict[str, Any]:
        """Get operation metrics summary."""
        if operation:
            metrics = self._operation_metrics.get(operation, OperationMetrics())
            return {
                'operation': operation,
                'total_count': metrics.total_count,
                'success_count': metrics.success_count,
                'failure_count': metrics.failure_count,
                'success_rate': metrics.success_rate,
                'failure_rate': metrics.failure_rate,
                'average_duration': metrics.average_duration,
                'min_duration': metrics.min_duration if metrics.min_duration != float('inf') else 0,
                'max_duration': metrics.max_duration,
                'recent_errors': list(metrics.recent_errors)
            }
        else:
            return {
                op: self.get_operation_metrics(op)
                for op in self._operation_metrics.keys()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': {}
        }
        
        try:
            # Check system resources if psutil is available
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                health_status['checks']['cpu'] = {
                    'status': 'healthy' if cpu_percent < self.config.monitoring.cpu_usage_threshold * 100 else 'unhealthy',
                    'value': cpu_percent,
                    'threshold': self.config.monitoring.cpu_usage_threshold * 100
                }
                
                health_status['checks']['memory'] = {
                    'status': 'healthy' if memory.percent < self.config.monitoring.memory_usage_threshold * 100 else 'unhealthy',
                    'value': memory.percent,
                    'threshold': self.config.monitoring.memory_usage_threshold * 100
                }
                
                health_status['checks']['disk'] = {
                    'status': 'healthy' if (disk.used / disk.total) < self.config.monitoring.disk_usage_threshold else 'unhealthy',
                    'value': (disk.used / disk.total) * 100,
                    'threshold': self.config.monitoring.disk_usage_threshold * 100
                }
            else:
                health_status['checks']['system_monitoring'] = {
                    'status': 'warning',
                    'message': 'psutil not available - system resource monitoring disabled'
                }
            
            # Check operation error rates
            for operation, metrics in self._operation_metrics.items():
                if metrics.total_count >= 10:  # Only check with sufficient samples
                    health_status['checks'][f'{operation}_error_rate'] = {
                        'status': 'healthy' if metrics.failure_rate <= self.config.monitoring.error_rate_threshold else 'unhealthy',
                        'value': metrics.failure_rate,
                        'threshold': self.config.monitoring.error_rate_threshold
                    }
            
            # Overall health status
            unhealthy_checks = [
                check for check in health_status['checks'].values()
                if check['status'] == 'unhealthy'
            ]
            
            if unhealthy_checks:
                health_status['status'] = 'unhealthy'
                health_status['unhealthy_checks'] = len(unhealthy_checks)
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status


@contextmanager
def monitor_operation(operation: str, portfolio_id: str = None):
    """Context manager to monitor analytics operations."""
    monitor = get_analytics_monitor()
    start_time = time.time()
    success = False
    error = None
    
    try:
        yield monitor
        success = True
    except Exception as e:
        error = e
        monitor.record_error(e)
        raise
    finally:
        duration = time.time() - start_time
        monitor.record_operation(operation, duration, success, error)


def monitored_operation(operation: str):
    """Decorator to monitor analytics operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with monitor_operation(operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global analytics monitor instance
_analytics_monitor = None


def get_analytics_monitor() -> AnalyticsMonitor:
    """Get the global analytics monitor instance."""
    global _analytics_monitor
    if _analytics_monitor is None:
        _analytics_monitor = AnalyticsMonitor()
    return _analytics_monitor


def start_analytics_monitoring():
    """Start analytics monitoring."""
    monitor = get_analytics_monitor()
    if monitor.config.monitoring.enable_metrics:
        # Start metrics server if not already running
        from ..common.metrics import start_metrics_server
        start_metrics_server(monitor.config.monitoring.metrics_port)


# Initialize monitoring on module import
if get_analytics_config().monitoring.enable_metrics:
    start_analytics_monitoring()