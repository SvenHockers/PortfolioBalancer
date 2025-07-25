"""Portfolio analytics module."""

from .analytics_service import AnalyticsService
from .models import (
    BacktestConfig,
    BacktestResult,
    MonteCarloConfig,
    MonteCarloResult,
    RiskAnalysis,
    PerformanceMetrics,
    DividendAnalysis
)
from .exceptions import (
    AnalyticsError,
    BacktestError,
    InsufficientDataError,
    SimulationError,
    RiskAnalysisError,
    DataError,
    DataQualityError,
    StorageError,
    DatabaseConnectionError,
    CacheError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    TimeoutError,
    AuthenticationError,
    AuthorizationError
)
from .storage import AnalyticsStorage, PostgreSQLAnalyticsStorage
from .config import AnalyticsConfig, get_analytics_config
from .error_recovery import (
    ErrorRecoveryManager,
    recovery_manager,
    with_retry,
    with_circuit_breaker,
    with_fallback,
    with_degradation_check,
    error_context
)
from .logging import (
    AnalyticsLogger,
    get_analytics_logger,
    analytics_operation_context,
    log_analytics_operation,
    setup_analytics_logging
)
from .monitoring import (
    AnalyticsMonitor,
    get_analytics_monitor,
    monitor_operation,
    monitored_operation,
    start_analytics_monitoring
)

__all__ = [
    # Core service
    'AnalyticsService',
    
    # Data models
    'BacktestConfig',
    'BacktestResult',
    'MonteCarloConfig',
    'MonteCarloResult',
    'RiskAnalysis',
    'PerformanceMetrics',
    'DividendAnalysis',
    
    # Exceptions
    'AnalyticsError',
    'BacktestError',
    'InsufficientDataError',
    'SimulationError',
    'RiskAnalysisError',
    'DataError',
    'DataQualityError',
    'StorageError',
    'DatabaseConnectionError',
    'CacheError',
    'ValidationError',
    'ConfigurationError',
    'ResourceError',
    'TimeoutError',
    'AuthenticationError',
    'AuthorizationError',
    
    # Storage
    'AnalyticsStorage',
    'PostgreSQLAnalyticsStorage',
    
    # Configuration
    'AnalyticsConfig',
    'get_analytics_config',
    
    # Error recovery
    'ErrorRecoveryManager',
    'recovery_manager',
    'with_retry',
    'with_circuit_breaker',
    'with_fallback',
    'with_degradation_check',
    'error_context',
    
    # Logging
    'AnalyticsLogger',
    'get_analytics_logger',
    'analytics_operation_context',
    'log_analytics_operation',
    'setup_analytics_logging',
    
    # Monitoring
    'AnalyticsMonitor',
    'get_analytics_monitor',
    'monitor_operation',
    'monitored_operation',
    'start_analytics_monitoring'
]