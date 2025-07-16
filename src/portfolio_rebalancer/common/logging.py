"""Structured logging configuration with JSON support."""

import logging
import logging.config
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
import uuid
from contextlib import contextmanager
from threading import local
import os

from .config import get_config


class JSONFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        correlation_id = getattr(_context, 'correlation_id', None)
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Standard text formatter with correlation ID support."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text with correlation ID."""
        correlation_id = getattr(_context, 'correlation_id', None)
        # Always set correlation_id attribute, even if None or empty string
        record.correlation_id = correlation_id or ""
        return super().format(record)


# Thread-local storage for correlation IDs
_context = local()


def setup_logging() -> None:
    """Set up logging configuration based on config settings."""
    config = get_config()
    
    # Determine formatter based on config
    if config.logging.format == "json":
        formatter_class = JSONFormatter
        format_string = None  # Not used for JSON formatter
    else:
        formatter_class = TextFormatter
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s"
    
    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": formatter_class,
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": sys.stdout,
            }
        },
        "root": {
            "level": config.logging.level,
            "handlers": ["console"],
        },
        "loggers": {
            "portfolio_rebalancer": {
                "level": config.logging.level,
                "handlers": ["console"],
                "propagate": False,
            }
        }
    }
    
    # Add file handler if file path is specified
    if config.logging.file_path:
        logging_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": config.logging.file_path,
        }
        logging_config["root"]["handlers"].append("file")
        logging_config["loggers"]["portfolio_rebalancer"]["handlers"].append("file")
    
    # Apply text formatter format string if using text format
    if config.logging.format == "text":
        logging_config["formatters"]["default"]["format"] = format_string
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(f"portfolio_rebalancer.{name}")


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for setting correlation ID in logs."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    old_correlation_id = getattr(_context, 'correlation_id', None)
    _context.correlation_id = correlation_id
    
    try:
        yield correlation_id
    finally:
        if old_correlation_id is not None:
            _context.correlation_id = old_correlation_id
        else:
            delattr(_context, 'correlation_id')


def log_execution_time(logger: logging.Logger, operation: str):
    """Decorator to log execution time of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"{operation} completed successfully", 
                           extra={"operation": operation, "duration_seconds": duration})
                return result
            except Exception as e:
                end_time = datetime.utcnow()
                duration = (end_time - start_time).total_seconds()
                logger.error(f"{operation} failed", 
                           extra={"operation": operation, "duration_seconds": duration, "error": str(e)})
                raise
        return wrapper
    return decorator


def log_error_with_context(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an error with comprehensive context information.
    
    Args:
        logger: Logger instance to use
        error: Exception that occurred
        context: Additional context information
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
    }
    
    if context:
        error_info.update(context)
    
    logger.error(f"Error occurred: {error_info['error_type']}", extra=error_info)


def create_structured_logger(component: str, **default_context) -> logging.Logger:
    """
    Create a logger with default structured context.
    
    Args:
        component: Component name for the logger
        **default_context: Default context fields to include in all log messages
        
    Returns:
        Configured logger instance
    """
    logger = get_logger(component)
    
    # Create a custom adapter that adds default context
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Merge default context with any extra context
            extra = kwargs.get('extra', {})
            extra.update(default_context)
            kwargs['extra'] = extra
            return msg, kwargs
    
    return ContextAdapter(logger, {})


def setup_container_logging() -> None:
    """
    Set up logging configuration optimized for containerized environments.
    This includes structured JSON logging and proper log aggregation support.
    """
    # Force JSON format for containers
    config = get_config()
    
    # Override format for container environments
    if os.getenv('CONTAINER_ENV', 'false').lower() == 'true':
        config.logging.format = "json"
    
    # Enhanced logging configuration for containers
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "text": {
                "()": TextFormatter,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if config.logging.format == "json" else "text",
                "stream": sys.stdout,
            }
        },
        "root": {
            "level": config.logging.level,
            "handlers": ["console"],
        },
        "loggers": {
            "portfolio_rebalancer": {
                "level": config.logging.level,
                "handlers": ["console"],
                "propagate": False,
            },
            # Suppress noisy third-party loggers in containers
            "urllib3": {
                "level": "WARNING",
            },
            "requests": {
                "level": "WARNING",
            },
            "yfinance": {
                "level": "WARNING",
            }
        }
    }
    
    # Add file handler if specified (useful for debugging in containers)
    if config.logging.file_path:
        logging_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "json" if config.logging.format == "json" else "text",
            "filename": config.logging.file_path,
        }
        logging_config["root"]["handlers"].append("file")
        logging_config["loggers"]["portfolio_rebalancer"]["handlers"].append("file")
    
    logging.config.dictConfig(logging_config)


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID from context.
    
    Returns:
        Current correlation ID or None if not set
    """
    return getattr(_context, 'correlation_id', None)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID in the current context.
    
    Args:
        correlation_id: Correlation ID to set
    """
    _context.correlation_id = correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    if hasattr(_context, 'correlation_id'):
        delattr(_context, 'correlation_id')


class LoggingMixin:
    """
    Mixin class to add structured logging capabilities to any class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(self.__class__.__name__.lower())
    
    def log_info(self, message: str, **context) -> None:
        """Log an info message with context."""
        self._logger.info(message, extra=context)
    
    def log_warning(self, message: str, **context) -> None:
        """Log a warning message with context."""
        self._logger.warning(message, extra=context)
    
    def log_error(self, message: str, error: Exception = None, **context) -> None:
        """Log an error message with context and optional exception."""
        if error:
            context.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
        self._logger.error(message, extra=context, exc_info=error is not None)
    
    def log_debug(self, message: str, **context) -> None:
        """Log a debug message with context."""
        self._logger.debug(message, extra=context)


# Initialize logging on module import
# Use container-optimized logging if in container environment
if os.getenv('CONTAINER_ENV', 'false').lower() == 'true':
    setup_container_logging()
else:
    setup_logging()