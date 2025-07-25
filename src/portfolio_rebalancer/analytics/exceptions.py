"""Enhanced exception hierarchy and error handling for analytics operations."""

import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA = "data"
    COMPUTATION = "computation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    STORAGE = "storage"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"


class AnalyticsError(Exception):
    """Base exception for analytics operations with enhanced context."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize analytics error.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            severity: Error severity level
            category: Error category for classification
            context: Additional context information
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
        
        # Add stack trace to context
        self.context['stack_trace'] = traceback.format_exc()
        
        # Add cause information if available
        if cause:
            self.context['cause_type'] = type(cause).__name__
            self.context['cause_message'] = str(cause)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'recoverable': self.recoverable,
            'retry_after': self.retry_after,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        return f"[{self.error_code}] {self.message}"


class DataError(AnalyticsError):
    """Errors related to data issues."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA)
        kwargs.setdefault('error_code', 'DATA_ERROR')
        super().__init__(message, **kwargs)


class InsufficientDataError(DataError):
    """Insufficient historical data for analysis."""
    
    def __init__(
        self,
        message: str,
        required_points: Optional[int] = None,
        available_points: Optional[int] = None,
        missing_tickers: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if required_points is not None:
            context['required_points'] = required_points
        if available_points is not None:
            context['available_points'] = available_points
        if missing_tickers:
            context['missing_tickers'] = missing_tickers
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'INSUFFICIENT_DATA')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class DataQualityError(DataError):
    """Data quality issues that affect analysis."""
    
    def __init__(
        self,
        message: str,
        quality_issues: Optional[List[str]] = None,
        affected_tickers: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if quality_issues:
            context['quality_issues'] = quality_issues
        if affected_tickers:
            context['affected_tickers'] = affected_tickers
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'DATA_QUALITY')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class BacktestError(AnalyticsError):
    """Errors during backtesting operations."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('error_code', 'BACKTEST_ERROR')
        super().__init__(message, **kwargs)


class BacktestTimeoutError(BacktestError):
    """Backtest operation timed out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'BACKTEST_TIMEOUT')
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('retry_after', 60)  # Suggest 1 minute retry
        super().__init__(message, **kwargs)


class BacktestConfigurationError(BacktestError):
    """Invalid backtest configuration."""
    
    def __init__(
        self,
        message: str,
        invalid_fields: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if invalid_fields:
            context['invalid_fields'] = invalid_fields
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'BACKTEST_CONFIG')
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recoverable', False)  # Config errors need manual fix
        super().__init__(message, **kwargs)


class SimulationError(AnalyticsError):
    """Errors during Monte Carlo simulation."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('error_code', 'SIMULATION_ERROR')
        super().__init__(message, **kwargs)


class SimulationTimeoutError(SimulationError):
    """Monte Carlo simulation timed out."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[int] = None,
        completed_simulations: Optional[int] = None,
        total_simulations: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        if completed_simulations is not None:
            context['completed_simulations'] = completed_simulations
        if total_simulations is not None:
            context['total_simulations'] = total_simulations
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'SIMULATION_TIMEOUT')
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('retry_after', 120)  # Suggest 2 minute retry
        super().__init__(message, **kwargs)


class SimulationConvergenceError(SimulationError):
    """Monte Carlo simulation failed to converge."""
    
    def __init__(
        self,
        message: str,
        convergence_metric: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if convergence_metric is not None:
            context['convergence_metric'] = convergence_metric
        if threshold is not None:
            context['threshold'] = threshold
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'SIMULATION_CONVERGENCE')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class RiskAnalysisError(AnalyticsError):
    """Errors during risk analysis."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('error_code', 'RISK_ANALYSIS_ERROR')
        super().__init__(message, **kwargs)


class CorrelationCalculationError(RiskAnalysisError):
    """Error calculating correlation matrix."""
    
    def __init__(
        self,
        message: str,
        matrix_size: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if matrix_size is not None:
            context['matrix_size'] = matrix_size
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'CORRELATION_CALC')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class VaRCalculationError(RiskAnalysisError):
    """Error calculating Value at Risk."""
    
    def __init__(
        self,
        message: str,
        confidence_level: Optional[float] = None,
        method: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if confidence_level is not None:
            context['confidence_level'] = confidence_level
        if method is not None:
            context['method'] = method
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'VAR_CALCULATION')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class PerformanceTrackingError(AnalyticsError):
    """Errors during performance tracking."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('error_code', 'PERFORMANCE_ERROR')
        super().__init__(message, **kwargs)


class BenchmarkDataError(PerformanceTrackingError):
    """Error retrieving benchmark data."""
    
    def __init__(
        self,
        message: str,
        benchmark_symbol: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if benchmark_symbol is not None:
            context['benchmark_symbol'] = benchmark_symbol
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'BENCHMARK_DATA')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class AttributionCalculationError(PerformanceTrackingError):
    """Error calculating performance attribution."""
    
    def __init__(
        self,
        message: str,
        attribution_method: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if attribution_method is not None:
            context['attribution_method'] = attribution_method
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'ATTRIBUTION_CALC')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class DividendAnalysisError(AnalyticsError):
    """Errors during dividend analysis."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('error_code', 'DIVIDEND_ERROR')
        super().__init__(message, **kwargs)


class DividendDataError(DividendAnalysisError):
    """Error retrieving dividend data."""
    
    def __init__(
        self,
        message: str,
        ticker: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if ticker is not None:
            context['ticker'] = ticker
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'DIVIDEND_DATA')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class StorageError(AnalyticsError):
    """Errors related to data storage operations."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.STORAGE)
        kwargs.setdefault('error_code', 'STORAGE_ERROR')
        super().__init__(message, **kwargs)


class DatabaseConnectionError(StorageError):
    """Database connection error."""
    
    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if host is not None:
            context['host'] = host
        if port is not None:
            context['port'] = port
        if database is not None:
            context['database'] = database
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'DB_CONNECTION')
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('retry_after', 30)  # Suggest 30 second retry
        super().__init__(message, **kwargs)


class DatabaseQueryError(StorageError):
    """Database query execution error."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        query_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if query is not None:
            # Truncate long queries for logging
            context['query'] = query[:500] + '...' if len(query) > 500 else query
        if query_params is not None:
            context['query_params'] = query_params
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'DB_QUERY')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class CacheError(AnalyticsError):
    """Errors related to caching operations."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.STORAGE)
        kwargs.setdefault('error_code', 'CACHE_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.LOW)  # Cache errors are usually not critical
        super().__init__(message, **kwargs)


class CacheConnectionError(CacheError):
    """Cache connection error."""
    
    def __init__(
        self,
        message: str,
        cache_backend: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if cache_backend is not None:
            context['cache_backend'] = cache_backend
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'CACHE_CONNECTION')
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ValidationError(AnalyticsError):
    """Input validation errors."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if field_errors:
            context['field_errors'] = field_errors
        
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('error_code', 'VALIDATION_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recoverable', False)  # Validation errors need input correction
        super().__init__(message, **kwargs)


class ConfigurationError(AnalyticsError):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_section is not None:
            context['config_section'] = config_section
        
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('error_code', 'CONFIG_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)  # Config errors need manual fix
        super().__init__(message, **kwargs)


class ResourceError(AnalyticsError):
    """Resource-related errors (memory, CPU, etc.)."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if resource_type is not None:
            context['resource_type'] = resource_type
        if current_usage is not None:
            context['current_usage'] = current_usage
        if limit is not None:
            context['limit'] = limit
        
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.RESOURCE)
        kwargs.setdefault('error_code', 'RESOURCE_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('retry_after', 300)  # Suggest 5 minute retry
        super().__init__(message, **kwargs)


class MemoryError(ResourceError):
    """Memory-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('resource_type', 'memory')
        kwargs.setdefault('error_code', 'MEMORY_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class TimeoutError(AnalyticsError):
    """Generic timeout errors."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if operation is not None:
            context['operation'] = operation
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('error_code', 'TIMEOUT_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class AuthenticationError(AnalyticsError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION)
        kwargs.setdefault('error_code', 'AUTH_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)  # Auth errors need credential fix
        super().__init__(message, **kwargs)


class AuthorizationError(AnalyticsError):
    """Authorization-related errors."""
    
    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if required_permission is not None:
            context['required_permission'] = required_permission
        
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.AUTHORIZATION)
        kwargs.setdefault('error_code', 'AUTHZ_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)  # Authz errors need permission fix
        super().__init__(message, **kwargs)


class RateLimitError(AnalyticsError):
    """Rate limiting errors."""
    
    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if limit is not None:
            context['limit'] = limit
        if window_seconds is not None:
            context['window_seconds'] = window_seconds
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', 'RATE_LIMIT')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('retry_after', retry_after or 60)
        super().__init__(message, **kwargs)


class VisualizationError(AnalyticsError):
    """Errors during visualization data generation."""
    
    def __init__(
        self,
        message: str,
        chart_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if chart_type is not None:
            context['chart_type'] = chart_type
        
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('error_code', 'VISUALIZATION_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ExportError(AnalyticsError):
    """Errors during data export operations."""
    
    def __init__(
        self,
        message: str,
        export_format: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if export_format is not None:
            context['export_format'] = export_format
        
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.COMPUTATION)
        kwargs.setdefault('error_code', 'EXPORT_ERROR')
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


# Error mapping for converting generic exceptions to analytics exceptions
ERROR_MAPPING = {
    ConnectionError: DatabaseConnectionError,
    TimeoutError: TimeoutError,
    MemoryError: MemoryError,
    ValueError: ValidationError,
    KeyError: ValidationError,
    TypeError: ValidationError,
}


def wrap_exception(
    exc: Exception,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AnalyticsError:
    """
    Wrap a generic exception in an appropriate AnalyticsError.
    
    Args:
        exc: Original exception
        message: Custom message (uses original if not provided)
        context: Additional context
        **kwargs: Additional error parameters
        
    Returns:
        Wrapped analytics error
    """
    exc_type = type(exc)
    error_class = ERROR_MAPPING.get(exc_type, AnalyticsError)
    
    error_message = message or str(exc)
    error_context = context or {}
    
    return error_class(
        message=error_message,
        context=error_context,
        cause=exc,
        **kwargs
    )


def is_recoverable_error(error: Exception) -> bool:
    """
    Check if an error is recoverable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is recoverable
    """
    if isinstance(error, AnalyticsError):
        return error.recoverable
    
    # Default recovery rules for non-analytics errors
    non_recoverable_types = (
        ValueError, TypeError, KeyError, AttributeError,
        ImportError, SyntaxError, NameError
    )
    
    return not isinstance(error, non_recoverable_types)


def get_retry_delay(error: Exception) -> Optional[int]:
    """
    Get suggested retry delay for an error.
    
    Args:
        error: Exception to check
        
    Returns:
        Retry delay in seconds, or None if no retry suggested
    """
    if isinstance(error, AnalyticsError):
        return error.retry_after
    
    # Default retry delays for common error types
    if isinstance(error, ConnectionError):
        return 30
    elif isinstance(error, TimeoutError):
        return 60
    elif isinstance(error, MemoryError):
        return 300
    
    return None


def categorize_error(error: Exception) -> ErrorCategory:
    """
    Categorize an error for monitoring and alerting.
    
    Args:
        error: Exception to categorize
        
    Returns:
        Error category
    """
    if isinstance(error, AnalyticsError):
        return error.category
    
    # Default categorization for non-analytics errors
    if isinstance(error, (ConnectionError, OSError)):
        return ErrorCategory.NETWORK
    elif isinstance(error, (ValueError, TypeError, KeyError)):
        return ErrorCategory.VALIDATION
    elif isinstance(error, MemoryError):
        return ErrorCategory.RESOURCE
    elif isinstance(error, TimeoutError):
        return ErrorCategory.TIMEOUT
    else:
        return ErrorCategory.COMPUTATION