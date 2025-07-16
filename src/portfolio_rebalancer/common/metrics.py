"""
Metrics collection module for Portfolio Rebalancer.

This module provides Prometheus metrics collection and export functionality
for monitoring execution time, success rates, and portfolio performance.
"""

import time
import logging
import os
import socket
from typing import Dict, Any, Optional, Callable, List, Union
from functools import wraps
import threading
from contextlib import contextmanager

# Import prometheus_client conditionally to allow the system to work
# even if prometheus_client is not installed
# Create dummy metric classes for graceful degradation
class DummyMetric:
    def __init__(self, *args, **kwargs):
        pass
    
    def inc(self, *args, **kwargs):
        pass
        
    def dec(self, *args, **kwargs):
        pass
        
    def set(self, *args, **kwargs):
        pass
        
    def observe(self, *args, **kwargs):
        pass
        
    def time(self):
        class DummyTimer:
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return DummyTimer()
    
    def labels(self, *args, **kwargs):
        return self
    
    def info(self, *args, **kwargs):
        pass

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary, Info
    from prometheus_client.exposition import start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = Summary = Info = DummyMetric

logger = logging.getLogger(__name__)

# Singleton pattern for metrics registry
class MetricsRegistry:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._metrics = {}
        self._server_started = False
        self._initialized = True
        
        # Create default metrics
        if PROMETHEUS_AVAILABLE:
            # System information
            self.register_metric(
                "info",
                Info,
                "Portfolio Rebalancer system information"
            )
            
            # Execution time metrics
            self.register_metric(
                "execution_time_seconds",
                Histogram,
                "Execution time in seconds",
                ["component", "operation"]
            )
            
            # Success rate metrics
            self.register_metric(
                "operation_total",
                Counter,
                "Total number of operations",
                ["component", "operation", "status"]
            )
            
            # Portfolio performance metrics
            self.register_metric(
                "portfolio_metrics",
                Gauge,
                "Portfolio performance metrics",
                ["metric_name"]
            )
            
            # Portfolio allocation metrics
            self.register_metric(
                "portfolio_allocation",
                Gauge,
                "Portfolio allocation percentages",
                ["symbol", "asset_class"]
            )
            
            # Optimization convergence metrics
            self.register_metric(
                "optimization_iterations",
                Histogram,
                "Number of iterations for optimization convergence",
                ["strategy"]
            )
            
            # Optimization performance metrics
            self.register_metric(
                "optimization_performance",
                Gauge,
                "Optimization algorithm performance metrics",
                ["strategy", "metric_name"]
            )
            
            # Trade execution metrics
            self.register_metric(
                "trade_execution",
                Counter,
                "Trade execution counts",
                ["broker", "order_type", "status"]
            )
            
            # Trade execution latency
            self.register_metric(
                "trade_execution_latency_seconds",
                Histogram,
                "Trade execution latency in seconds",
                ["broker", "order_type"]
            )
            
            # Portfolio drift metrics
            self.register_metric(
                "portfolio_drift",
                Gauge,
                "Portfolio drift from target allocation",
                ["symbol"]
            )
            
            # Portfolio drift summary
            self.register_metric(
                "portfolio_drift_summary",
                Gauge,
                "Portfolio drift summary statistics",
                ["statistic"]
            )
            
            # System health metrics
            self.register_metric(
                "system_health",
                Gauge,
                "System health status (1=healthy, 0=unhealthy)",
                ["component"]
            )
            
            # Pipeline execution metrics
            self.register_metric(
                "pipeline_execution",
                Counter,
                "Pipeline execution counts",
                ["status"]
            )
            
            # Pipeline step duration
            self.register_metric(
                "pipeline_step_duration_seconds",
                Histogram,
                "Pipeline step duration in seconds",
                ["step"]
            )
            
            # Data quality metrics
            self.register_metric(
                "data_quality",
                Gauge,
                "Data quality metrics",
                ["metric_name"]
            )
            
            # Set system info
            info_metric = self.get_metric("info")
            if info_metric:
                hostname = socket.gethostname()
                info_metric.info({
                    "hostname": hostname,
                    "version": os.environ.get("APP_VERSION", "dev"),
                    "environment": os.environ.get("ENVIRONMENT", "development")
                })
            
            logger.info("Prometheus metrics initialized")
        else:
            logger.warning("prometheus_client not available, metrics collection disabled")
    
    def register_metric(self, name: str, metric_type: Any, description: str, labels: list = None) -> Any:
        """
        Register a new metric in the registry.
        
        Args:
            name: Metric name
            metric_type: Prometheus metric type (Counter, Gauge, etc.)
            description: Metric description
            labels: List of label names
            
        Returns:
            The created metric object
        """
        if not PROMETHEUS_AVAILABLE:
            return DummyMetric()
            
        full_name = f"portfolio_rebalancer_{name}"
        
        if full_name in self._metrics:
            return self._metrics[full_name]
            
        if labels is None:
            labels = []
            
        metric = metric_type(full_name, description, labels)
        self._metrics[full_name] = metric
        return metric
    
    def get_metric(self, name: str) -> Any:
        """
        Get a registered metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            The metric object or None if not found
        """
        full_name = f"portfolio_rebalancer_{name}"
        return self._metrics.get(full_name)
    
    def start_server(self, port: int = 8000) -> bool:
        """
        Start the metrics HTTP server for Prometheus scraping.
        
        Args:
            port: HTTP port to listen on
            
        Returns:
            True if server started successfully, False otherwise
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Cannot start metrics server: prometheus_client not available")
            return False
            
        if self._server_started:
            logger.info(f"Metrics server already running on port {port}")
            return True
            
        try:
            start_http_server(port)
            self._server_started = True
            logger.info(f"Started Prometheus metrics server on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
            return False


# Global metrics registry instance
metrics_registry = MetricsRegistry()


def timed(component: str, operation: str) -> Callable:
    """
    Decorator to measure execution time of a function.
    
    Args:
        component: Component name (e.g., 'fetcher', 'optimizer')
        operation: Operation name (e.g., 'fetch_data', 'optimize_portfolio')
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            histogram = metrics_registry.get_metric("execution_time_seconds")
            counter = metrics_registry.get_metric("operation_total")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                # Record successful operation
                if counter:
                    counter.labels(component=component, operation=operation, status="success").inc()
                return result
            except Exception as e:
                # Record failed operation
                if counter:
                    counter.labels(component=component, operation=operation, status="failure").inc()
                raise
            finally:
                # Record execution time
                execution_time = time.time() - start_time
                if histogram:
                    histogram.labels(component=component, operation=operation).observe(execution_time)
                
        return wrapper
    return decorator


def record_portfolio_metrics(expected_return: float, volatility: float, sharpe_ratio: float) -> None:
    """
    Record portfolio performance metrics.
    
    Args:
        expected_return: Expected portfolio return
        volatility: Portfolio volatility
        sharpe_ratio: Portfolio Sharpe ratio
    """
    gauge = metrics_registry.get_metric("portfolio_metrics")
    if gauge:
        gauge.labels(metric_name="expected_return").set(expected_return)
        gauge.labels(metric_name="volatility").set(volatility)
        gauge.labels(metric_name="sharpe_ratio").set(sharpe_ratio)


def record_optimization_convergence(strategy: str, iterations: int) -> None:
    """
    Record optimization convergence metrics.
    
    Args:
        strategy: Optimization strategy name
        iterations: Number of iterations to convergence
    """
    histogram = metrics_registry.get_metric("optimization_iterations")
    if histogram:
        histogram.labels(strategy=strategy).observe(iterations)


def record_trade_execution(broker: str, order_type: str, status: str) -> None:
    """
    Record trade execution metrics.
    
    Args:
        broker: Broker name (e.g., 'alpaca', 'ib')
        order_type: Order type (e.g., 'market', 'limit')
        status: Execution status (e.g., 'success', 'failure')
    """
    counter = metrics_registry.get_metric("trade_execution")
    if counter:
        counter.labels(broker=broker, order_type=order_type, status=status).inc()


def update_system_health(component: str, is_healthy: bool) -> None:
    """
    Update system health status.
    
    Args:
        component: Component name
        is_healthy: Health status (True=healthy, False=unhealthy)
    """
    gauge = metrics_registry.get_metric("system_health")
    if gauge:
        gauge.labels(component=component).set(1 if is_healthy else 0)


def record_portfolio_allocation(allocations: Dict[str, float], asset_classes: Dict[str, str] = None) -> None:
    """
    Record portfolio allocation percentages.
    
    Args:
        allocations: Dictionary mapping symbols to allocation percentages
        asset_classes: Optional dictionary mapping symbols to asset classes
    """
    gauge = metrics_registry.get_metric("portfolio_allocation")
    if gauge:
        # Reset all previous allocations
        for symbol, allocation in allocations.items():
            asset_class = asset_classes.get(symbol, "unknown") if asset_classes else "unknown"
            gauge.labels(symbol=symbol, asset_class=asset_class).set(allocation)


def record_portfolio_drift(drift_values: Dict[str, float]) -> None:
    """
    Record portfolio drift from target allocation.
    
    Args:
        drift_values: Dictionary mapping symbols to drift percentages
    """
    gauge = metrics_registry.get_metric("portfolio_drift")
    summary = metrics_registry.get_metric("portfolio_drift_summary")
    
    if gauge and drift_values:
        # Record individual symbol drifts
        for symbol, drift in drift_values.items():
            gauge.labels(symbol=symbol).set(drift)
        
        # Record summary statistics
        if summary:
            drift_values_list = list(drift_values.values())
            summary.labels(statistic="max").set(max(abs(d) for d in drift_values_list))
            summary.labels(statistic="mean").set(sum(abs(d) for d in drift_values_list) / len(drift_values_list))
            summary.labels(statistic="total").set(sum(abs(d) for d in drift_values_list))


def record_optimization_performance(strategy: str, metrics: Dict[str, float]) -> None:
    """
    Record optimization algorithm performance metrics.
    
    Args:
        strategy: Optimization strategy name
        metrics: Dictionary of performance metrics
    """
    gauge = metrics_registry.get_metric("optimization_performance")
    if gauge:
        for metric_name, value in metrics.items():
            gauge.labels(strategy=strategy, metric_name=metric_name).set(value)


def record_data_quality(metrics: Dict[str, float]) -> None:
    """
    Record data quality metrics.
    
    Args:
        metrics: Dictionary of data quality metrics
    """
    gauge = metrics_registry.get_metric("data_quality")
    if gauge:
        for metric_name, value in metrics.items():
            gauge.labels(metric_name=metric_name).set(value)


def record_pipeline_execution(status: str) -> None:
    """
    Record pipeline execution count.
    
    Args:
        status: Execution status (e.g., 'success', 'failure', 'partial')
    """
    counter = metrics_registry.get_metric("pipeline_execution")
    if counter:
        counter.labels(status=status).inc()


def record_pipeline_step_duration(step: str, duration_seconds: float) -> None:
    """
    Record pipeline step duration.
    
    Args:
        step: Pipeline step name
        duration_seconds: Duration in seconds
    """
    histogram = metrics_registry.get_metric("pipeline_step_duration_seconds")
    if histogram:
        histogram.labels(step=step).observe(duration_seconds)


def record_trade_execution_latency(broker: str, order_type: str, latency_seconds: float) -> None:
    """
    Record trade execution latency.
    
    Args:
        broker: Broker name
        order_type: Order type
        latency_seconds: Latency in seconds
    """
    histogram = metrics_registry.get_metric("trade_execution_latency_seconds")
    if histogram:
        histogram.labels(broker=broker, order_type=order_type).observe(latency_seconds)


@contextmanager
def measure_latency(broker: str, order_type: str):
    """
    Context manager to measure trade execution latency.
    
    Args:
        broker: Broker name
        order_type: Order type
    """
    start_time = time.time()
    try:
        yield
    finally:
        latency = time.time() - start_time
        record_trade_execution_latency(broker, order_type, latency)


def start_metrics_server(port: int = 8000) -> bool:
    """
    Start the metrics HTTP server for Prometheus scraping.
    
    Args:
        port: HTTP port to listen on
        
    Returns:
        True if server started successfully, False otherwise
    """
    return metrics_registry.start_server(port)


def get_metrics_registry() -> MetricsRegistry:
    """
    Get the global metrics registry instance.
    
    Returns:
        The global MetricsRegistry instance
    """
    return metrics_registry


def is_prometheus_available() -> bool:
    """
    Check if Prometheus client is available.
    
    Returns:
        True if prometheus_client is available, False otherwise
    """
    return PROMETHEUS_AVAILABLE