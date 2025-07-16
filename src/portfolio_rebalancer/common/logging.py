"""Structured logging configuration with JSON support."""

import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from contextlib import contextmanager
from threading import local

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
        if correlation_id:
            record.correlation_id = correlation_id
            return super().format(record)
        else:
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


# Initialize logging on module import
setup_logging()