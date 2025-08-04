"""Error recovery strategies and graceful degradation logic for analytics operations."""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Union, List
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from .exceptions import (
    AnalyticsError, ErrorSeverity, ErrorCategory,
    DatabaseConnectionError, CacheConnectionError, TimeoutError,
    InsufficientDataError, DataQualityError, ResourceError,
    is_recoverable_error, get_retry_delay, wrap_exception
)

T = TypeVar('T')
logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAKER = "circuit_breaker"


class DegradationLevel(Enum):
    """Service degradation levels."""
    NONE = "none"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise AnalyticsError(
                    "Circuit breaker is OPEN",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.RESOURCE,
                    recoverable=True,
                    retry_after=self.config.recovery_timeout
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Require multiple successes
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker reset to CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ErrorRecoveryManager:
    """Manages error recovery strategies and graceful degradation."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.degradation_level = DegradationLevel.NONE
        self.fallback_handlers: Dict[str, Callable] = {}
        self.recovery_metrics: Dict[str, Any] = {
            'total_retries': 0,
            'successful_recoveries': 0,
            'fallback_activations': 0,
            'circuit_breaker_trips': 0
        }
    
    def register_fallback(self, operation: str, handler: Callable):
        """Register a fallback handler for an operation."""
        self.fallback_handlers[operation] = handler
        logger.info(f"Registered fallback handler for operation: {operation}")
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(config)
        return self.circuit_breakers[name]
    
    def set_degradation_level(self, level: DegradationLevel):
        """Set service degradation level."""
        if level != self.degradation_level:
            logger.warning(f"Service degradation level changed: {self.degradation_level.value} -> {level.value}")
            self.degradation_level = level
    
    def should_degrade_operation(self, operation: str) -> bool:
        """Check if operation should be degraded based on current level."""
        if self.degradation_level == DegradationLevel.NONE:
            return False
        
        # Define which operations to degrade at each level
        degradation_rules = {
            DegradationLevel.PARTIAL: [
                'monte_carlo_simulation',
                'detailed_risk_analysis',
                'dividend_projections'
            ],
            DegradationLevel.MINIMAL: [
                'monte_carlo_simulation',
                'detailed_risk_analysis',
                'dividend_projections',
                'performance_attribution',
                'stress_testing'
            ],
            DegradationLevel.EMERGENCY: [
                'monte_carlo_simulation',
                'detailed_risk_analysis',
                'dividend_projections',
                'performance_attribution',
                'stress_testing',
                'backtesting',
                'correlation_analysis'
            ]
        }
        
        operations_to_degrade = degradation_rules.get(self.degradation_level, [])
        return operation in operations_to_degrade
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics for monitoring."""
        return self.recovery_metrics.copy()


# Global error recovery manager
recovery_manager = ErrorRecoveryManager()


def with_retry(
    config: Optional[RetryConfig] = None,
    operation_name: Optional[str] = None
):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        config: Retry configuration
        operation_name: Name for logging and metrics
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            op_name = operation_name or func.__name__
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        recovery_manager.recovery_metrics['successful_recoveries'] += 1
                        logger.info(f"Operation {op_name} succeeded on attempt {attempt + 1}")
                    return result
                
                except Exception as e:
                    last_exception = e
                    recovery_manager.recovery_metrics['total_retries'] += 1
                    
                    # Check if error is recoverable
                    if not is_recoverable_error(e):
                        logger.error(f"Non-recoverable error in {op_name}: {e}")
                        raise
                    
                    # Check if we should retry
                    if attempt == config.max_attempts - 1:
                        logger.error(f"Max retry attempts ({config.max_attempts}) exceeded for {op_name}")
                        break
                    
                    # Calculate delay
                    delay = _calculate_retry_delay(config, attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {op_name}, retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            # All retries exhausted
            if isinstance(last_exception, AnalyticsError):
                raise last_exception
            else:
                raise wrap_exception(
                    last_exception,
                    message=f"Operation {op_name} failed after {config.max_attempts} attempts",
                    context={'operation': op_name, 'attempts': config.max_attempts}
                )
        
        return wrapper
    return decorator


def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator for adding circuit breaker protection.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
    """
    if config is None:
        config = CircuitBreakerConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            circuit_breaker = recovery_manager.get_circuit_breaker(name, config)
            try:
                return circuit_breaker.call(func, *args, **kwargs)
            except AnalyticsError as e:
                if e.error_code == "CIRCUIT_BREAKER_OPEN":
                    recovery_manager.recovery_metrics['circuit_breaker_trips'] += 1
                raise
        
        return wrapper
    return decorator


def with_fallback(
    fallback_func: Optional[Callable] = None,
    operation_name: Optional[str] = None
):
    """
    Decorator for adding fallback functionality.
    
    Args:
        fallback_func: Fallback function to call on error
        operation_name: Operation name for registered fallbacks
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or func.__name__
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary operation {op_name} failed, attempting fallback: {e}")
                
                # Try registered fallback first
                if op_name in recovery_manager.fallback_handlers:
                    try:
                        result = recovery_manager.fallback_handlers[op_name](*args, **kwargs)
                        recovery_manager.recovery_metrics['fallback_activations'] += 1
                        logger.info(f"Fallback succeeded for operation {op_name}")
                        return result
                    except Exception as fallback_error:
                        logger.error(f"Registered fallback failed for {op_name}: {fallback_error}")
                
                # Try provided fallback function
                if fallback_func:
                    try:
                        result = fallback_func(*args, **kwargs)
                        recovery_manager.recovery_metrics['fallback_activations'] += 1
                        logger.info(f"Provided fallback succeeded for operation {op_name}")
                        return result
                    except Exception as fallback_error:
                        logger.error(f"Provided fallback failed for {op_name}: {fallback_error}")
                
                # No fallback available or all fallbacks failed
                raise
        
        return wrapper
    return decorator


def with_degradation_check(operation_name: str):
    """
    Decorator to check if operation should be degraded.
    
    Args:
        operation_name: Name of the operation to check
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if recovery_manager.should_degrade_operation(operation_name):
                raise AnalyticsError(
                    f"Operation {operation_name} is currently degraded",
                    error_code="OPERATION_DEGRADED",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.RESOURCE,
                    context={
                        'operation': operation_name,
                        'degradation_level': recovery_manager.degradation_level.value
                    },
                    recoverable=True,
                    retry_after=300  # Suggest 5 minute retry
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, **context):
    """
    Context manager for enhanced error handling with operation context.
    
    Args:
        operation: Operation name
        **context: Additional context information
    """
    start_time = time.time()
    try:
        logger.debug(f"Starting operation: {operation}", extra=context)
        yield
        duration = time.time() - start_time
        logger.info(f"Operation {operation} completed successfully in {duration:.2f}s", extra=context)
    except Exception as e:
        duration = time.time() - start_time
        error_context_data = {
            'operation': operation,
            'duration': duration,
            **context
        }
        
        if isinstance(e, AnalyticsError):
            e.context.update(error_context_data)
            logger.error(f"Operation {operation} failed: {e}", extra=error_context_data)
            raise
        else:
            wrapped_error = wrap_exception(
                e,
                message=f"Operation {operation} failed",
                context=error_context_data
            )
            logger.error(f"Operation {operation} failed: {wrapped_error}", extra=error_context_data)
            raise wrapped_error


def _calculate_retry_delay(config: RetryConfig, attempt: int) -> float:
    """Calculate retry delay based on configuration."""
    if config.backoff_strategy == "exponential":
        delay = config.base_delay * (config.exponential_base ** attempt)
    elif config.backoff_strategy == "linear":
        delay = config.base_delay * (attempt + 1)
    else:  # fixed
        delay = config.base_delay
    
    # Apply maximum delay limit
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        import random
        delay *= (0.5 + random.random() * 0.5)
    
    return delay


# Predefined fallback functions for common operations
def basic_backtest_fallback(*args, **kwargs):
    """Basic fallback for backtesting - returns simplified results."""
    logger.info("Using basic backtest fallback")
    # Return minimal backtest result with warning
    from .models import BacktestResult, BacktestConfig
    
    config = args[0] if args and isinstance(args[0], BacktestConfig) else BacktestConfig(
        tickers=['SPY'], start_date='2023-01-01', end_date='2023-12-31', strategy='equal_weight'
    )
    
    return BacktestResult(
        config=config,
        total_return=0.0,
        annualized_return=0.0,
        volatility=0.15,
        sharpe_ratio=0.0,
        max_drawdown=-0.05,
        calmar_ratio=0.0,
        transaction_costs=0.0,
        num_rebalances=0,
        final_value=config.initial_capital,
        returns_data={'warning': 'Fallback data - not actual results'},
        allocation_data={'warning': 'Fallback data - not actual results'}
    )


def basic_risk_analysis_fallback(*args, **kwargs):
    """Basic fallback for risk analysis - returns conservative estimates."""
    logger.info("Using basic risk analysis fallback")
    from .models import RiskAnalysis
    from datetime import date
    
    portfolio_id = args[0] if args else 'unknown'
    
    return RiskAnalysis(
        portfolio_id=portfolio_id,
        analysis_date=date.today(),
        portfolio_beta=1.0,
        tracking_error=0.05,
        information_ratio=0.0,
        var_95=-0.05,  # Conservative 5% VaR
        cvar_95=-0.08,  # Conservative 8% CVaR
        max_drawdown=-0.20,  # Conservative 20% max drawdown
        concentration_risk=0.3,
        correlation_data={'warning': 'Fallback data - not actual correlations'},
        factor_exposures={'market': 1.0},
        sector_exposures={'diversified': 1.0}
    )


# Register default fallback handlers
recovery_manager.register_fallback('backtest', basic_backtest_fallback)
recovery_manager.register_fallback('risk_analysis', basic_risk_analysis_fallback)


def configure_error_recovery():
    """Configure error recovery with default settings."""
    # Set up circuit breakers for critical operations
    db_circuit_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=DatabaseConnectionError
    )
    recovery_manager.get_circuit_breaker('database', db_circuit_config)
    
    cache_circuit_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=CacheConnectionError
    )
    recovery_manager.get_circuit_breaker('cache', cache_circuit_config)
    
    logger.info("Error recovery configured with default settings")


# Initialize error recovery on module import
configure_error_recovery()