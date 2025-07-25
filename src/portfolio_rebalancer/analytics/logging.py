"""Analytics-specific logging configuration and utilities."""

import logging
import logging.config
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from functools import wraps
import uuid
import traceback

from ..common.logging import (
    get_logger, create_structured_logger, correlation_context,
    get_correlation_id, set_correlation_id
)
from .config import get_analytics_config
from .exceptions import AnalyticsError, ErrorSeverity, ErrorCategory


class AnalyticsLogFormatter(logging.Formatter):
    """Custom formatter for analytics operations with enhanced context."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with analytics-specific context."""
        # Base log entry
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        correlation_id = get_correlation_id()
        if correlation_id:
            log_entry['correlation_id'] = correlation_id
        
        # Add analytics-specific context
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'portfolio_id'):
            log_entry['portfolio_id'] = record.portfolio_id
        if hasattr(record, 'duration'):
            log_entry['duration_seconds'] = record.duration
        if hasattr(record, 'error_code'):
            log_entry['error_code'] = record.error_code
        if hasattr(record, 'error_category'):
            log_entry['error_category'] = record.error_category
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'operation', 'portfolio_id',
                          'duration', 'error_code', 'error_category']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class AnalyticsLogger:
    """Enhanced logger for analytics operations."""
    
    def __init__(self, name: str):
        """Initialize analytics logger."""
        self.logger = get_logger(f"analytics.{name}")
        self.config = get_analytics_config()
    
    def log_operation_start(self, operation: str, **context):
        """Log the start of an analytics operation."""
        # Filter out reserved log record attributes
        filtered_context = {k: v for k, v in context.items() 
                          if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                     'pathname', 'filename', 'module', 'lineno', 'funcName', 
                                     'created', 'msecs', 'relativeCreated', 'thread', 
                                     'threadName', 'processName', 'process', 'getMessage', 
                                     'exc_info', 'exc_text', 'stack_info', 'operation']}
        
        self.logger.info(
            f"Starting {operation}",
            extra={
                'operation': operation,
                'operation_phase': 'start',
                **filtered_context
            }
        )
    
    def log_operation_success(self, operation: str, duration: float, **context):
        """Log successful completion of an analytics operation."""
        # Filter out reserved log record attributes
        filtered_context = {k: v for k, v in context.items() 
                          if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                     'pathname', 'filename', 'module', 'lineno', 'funcName', 
                                     'created', 'msecs', 'relativeCreated', 'thread', 
                                     'threadName', 'processName', 'process', 'getMessage', 
                                     'exc_info', 'exc_text', 'stack_info', 'operation']}
        
        self.logger.info(
            f"Completed {operation} successfully",
            extra={
                'operation': operation,
                'operation_phase': 'success',
                'duration': duration,
                **filtered_context
            }
        )
    
    def log_operation_failure(self, operation: str, error: Exception, duration: float, **context):
        """Log failure of an analytics operation."""
        # Filter out reserved log record attributes
        filtered_context = {k: v for k, v in context.items() 
                          if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                                     'pathname', 'filename', 'module', 'lineno', 'funcName', 
                                     'created', 'msecs', 'relativeCreated', 'thread', 
                                     'threadName', 'processName', 'process', 'getMessage', 
                                     'exc_info', 'exc_text', 'stack_info', 'operation']}
        
        error_context = {
            'operation': operation,
            'operation_phase': 'failure',
            'duration': duration,
            **filtered_context
        }
        
        if isinstance(error, AnalyticsError):
            error_context.update({
                'error_code': error.error_code,
                'error_category': error.category.value,
                'error_severity': error.severity.value,
                'recoverable': error.recoverable,
                'retry_after': error.retry_after
            })
        
        self.logger.error(
            f"Failed {operation}: {str(error)}",
            extra=error_context,
            exc_info=True
        )
    
    def log_backtest_metrics(self, portfolio_id: str, metrics: Dict[str, Any]):
        """Log backtest performance metrics."""
        self.logger.info(
            "Backtest metrics calculated",
            extra={
                'operation': 'backtest',
                'portfolio_id': portfolio_id,
                'total_return': metrics.get('total_return'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'max_drawdown': metrics.get('max_drawdown'),
                'volatility': metrics.get('volatility')
            }
        )
    
    def log_monte_carlo_results(self, portfolio_id: str, results: Dict[str, Any]):
        """Log Monte Carlo simulation results."""
        self.logger.info(
            "Monte Carlo simulation completed",
            extra={
                'operation': 'monte_carlo',
                'portfolio_id': portfolio_id,
                'expected_value': results.get('expected_value'),
                'probability_of_loss': results.get('probability_of_loss'),
                'var_95': results.get('value_at_risk_95'),
                'num_simulations': results.get('num_simulations')
            }
        )
    
    def log_risk_analysis(self, portfolio_id: str, analysis: Dict[str, Any]):
        """Log risk analysis results."""
        self.logger.info(
            "Risk analysis completed",
            extra={
                'operation': 'risk_analysis',
                'portfolio_id': portfolio_id,
                'portfolio_beta': analysis.get('portfolio_beta'),
                'var_95': analysis.get('var_95'),
                'tracking_error': analysis.get('tracking_error'),
                'concentration_risk': analysis.get('concentration_risk')
            }
        )
    
    def log_performance_update(self, portfolio_id: str, metrics: Dict[str, Any]):
        """Log performance metrics update."""
        self.logger.info(
            "Performance metrics updated",
            extra={
                'operation': 'performance_tracking',
                'portfolio_id': portfolio_id,
                'total_return': metrics.get('total_return'),
                'annualized_return': metrics.get('annualized_return'),
                'sharpe_ratio': metrics.get('sharpe_ratio'),
                'alpha': metrics.get('alpha'),
                'beta': metrics.get('beta')
            }
        )
    
    def log_data_quality_issue(self, issue_type: str, details: Dict[str, Any]):
        """Log data quality issues."""
        self.logger.warning(
            f"Data quality issue detected: {issue_type}",
            extra={
                'operation': 'data_quality_check',
                'issue_type': issue_type,
                **details
            }
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: bool = None, **context):
        """Log cache operations."""
        log_level = logging.DEBUG
        message = f"Cache {operation}: {key}"
        
        extra_context = {
            'operation': 'cache',
            'cache_operation': operation,
            'cache_key': key,
            **context
        }
        
        if hit is not None:
            extra_context['cache_hit'] = hit
            if hit:
                message += " (HIT)"
            else:
                message += " (MISS)"
        
        self.logger.log(log_level, message, extra=extra_context)
    
    def log_database_operation(self, operation: str, table: str, duration: float, **context):
        """Log database operations."""
        self.logger.debug(
            f"Database {operation} on {table}",
            extra={
                'operation': 'database',
                'db_operation': operation,
                'table': table,
                'duration': duration,
                **context
            }
        )
    
    def log_configuration_loaded(self, config_source: str, config_sections: List[str]):
        """Log configuration loading."""
        self.logger.info(
            f"Analytics configuration loaded from {config_source}",
            extra={
                'operation': 'configuration',
                'config_source': config_source,
                'config_sections': config_sections
            }
        )
    
    def log_circuit_breaker_event(self, circuit_name: str, event: str, **context):
        """Log circuit breaker events."""
        log_level = logging.WARNING if event in ['opened', 'half_open'] else logging.INFO
        
        self.logger.log(
            log_level,
            f"Circuit breaker '{circuit_name}' {event}",
            extra={
                'operation': 'circuit_breaker',
                'circuit_name': circuit_name,
                'circuit_event': event,
                **context
            }
        )
    
    def log_fallback_activation(self, operation: str, fallback_type: str, **context):
        """Log fallback activation."""
        self.logger.warning(
            f"Fallback activated for {operation}: {fallback_type}",
            extra={
                'operation': 'fallback',
                'primary_operation': operation,
                'fallback_type': fallback_type,
                **context
            }
        )
    
    def log_degradation_change(self, old_level: str, new_level: str, reason: str):
        """Log service degradation level changes."""
        self.logger.warning(
            f"Service degradation changed: {old_level} -> {new_level}",
            extra={
                'operation': 'degradation',
                'old_level': old_level,
                'new_level': new_level,
                'reason': reason
            }
        )


def setup_analytics_logging():
    """Set up analytics-specific logging configuration."""
    config = get_analytics_config()
    
    if not config.monitoring.enable_structured_logging:
        return
    
    # Configure analytics logger
    analytics_logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "analytics": {
                "()": AnalyticsLogFormatter,
            }
        },
        "handlers": {
            "analytics_console": {
                "class": "logging.StreamHandler",
                "formatter": "analytics",
                "stream": sys.stdout,
            }
        },
        "loggers": {
            "portfolio_rebalancer.analytics": {
                "level": "INFO",
                "handlers": ["analytics_console"],
                "propagate": False,
            }
        }
    }
    
    # Add file handler if configured
    if hasattr(config, 'logging') and hasattr(config.logging, 'file_path') and config.logging.file_path:
        analytics_logging_config["handlers"]["analytics_file"] = {
            "class": "logging.FileHandler",
            "formatter": "analytics",
            "filename": f"{config.logging.file_path}.analytics",
        }
        analytics_logging_config["loggers"]["portfolio_rebalancer.analytics"]["handlers"].append("analytics_file")
    
    logging.config.dictConfig(analytics_logging_config)


@contextmanager
def analytics_operation_context(operation: str, portfolio_id: str = None, **context):
    """Context manager for analytics operations with automatic logging."""
    logger = AnalyticsLogger("operation")
    correlation_id = str(uuid.uuid4())
    
    operation_context = dict(context)  # Create a copy to avoid modifying the original
    if portfolio_id:
        operation_context['portfolio_id'] = portfolio_id
    
    with correlation_context(correlation_id):
        start_time = time.time()
        logger.log_operation_start(operation, **operation_context)
        
        try:
            yield logger
            duration = time.time() - start_time
            logger.log_operation_success(operation, duration, **operation_context)
        except Exception as e:
            duration = time.time() - start_time
            logger.log_operation_failure(operation, e, duration, **operation_context)
            raise


def log_analytics_operation(operation: str):
    """Decorator to automatically log analytics operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = AnalyticsLogger(func.__module__)
            
            # Extract portfolio_id from args/kwargs if available
            portfolio_id = None
            if 'portfolio_id' in kwargs:
                portfolio_id = kwargs['portfolio_id']
            elif len(args) > 0 and hasattr(args[0], 'portfolio_id'):
                portfolio_id = args[0].portfolio_id
            
            operation_context = {
                'func_name': func.__name__,  # Changed from 'function' to avoid conflicts
                'module_name': func.__module__  # Changed from 'module' to avoid conflicts
            }
            if portfolio_id:
                operation_context['portfolio_id'] = portfolio_id
            
            start_time = time.time()
            logger.log_operation_start(operation, **operation_context)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_operation_success(operation, duration, **operation_context)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_operation_failure(operation, e, duration, **operation_context)
                raise
        
        return wrapper
    return decorator


def get_analytics_logger(name: str) -> AnalyticsLogger:
    """Get an analytics logger instance."""
    return AnalyticsLogger(name)


# Initialize analytics logging on module import
setup_analytics_logging()