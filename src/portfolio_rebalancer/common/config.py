"""Configuration management system with environment variable support."""

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime


class StorageType(Enum):
    """Supported storage types."""
    PARQUET = "parquet"
    SQLITE = "sqlite"


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"


class BrokerType(Enum):
    """Supported broker types."""
    ALPACA = "alpaca"
    IB = "ib"


class LogLevel(Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Supported log formats."""
    JSON = "json"
    TEXT = "text"


@dataclass
class DataConfig:
    """Configuration for data fetching and storage."""
    tickers: List[str] = field(default_factory=list)
    storage_type: str = StorageType.PARQUET.value
    storage_path: str = "data"
    backfill_days: int = 252  # ~1 year of trading days
    data_quality_checks: bool = True
    max_missing_days: int = 5
    price_change_threshold: float = 0.5  # Flag price changes > 50% as suspicious
    
    def __post_init__(self):
        """Validate data configuration after initialization."""
        if self.backfill_days < 1:
            raise ValueError("backfill_days must be positive")
        if self.max_missing_days < 0:
            raise ValueError("max_missing_days must be non-negative")
        if not 0 < self.price_change_threshold <= 1:
            raise ValueError("price_change_threshold must be between 0 and 1")
    
    
@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    user_age: int = 35
    risk_free_rate: float = 0.02
    lookback_days: int = 252
    min_weight: float = 0.0
    max_weight: float = 0.4
    safe_portfolio_bonds: float = 0.8
    optimization_method: str = "sharpe"  # "sharpe", "min_variance", "equal_weight"
    rebalance_frequency: str = "daily"  # "daily", "weekly", "monthly"
    covariance_regularization: float = 0.001
    
    def __post_init__(self):
        """Validate optimization configuration after initialization."""
        if not 18 <= self.user_age <= 100:
            raise ValueError("user_age must be between 18 and 100")
        if not 0 <= self.risk_free_rate <= 0.2:
            raise ValueError("risk_free_rate must be between 0 and 0.2")
        if self.lookback_days < 30:
            raise ValueError("lookback_days must be at least 30")
        if not 0 <= self.min_weight <= 1:
            raise ValueError("min_weight must be between 0 and 1")
        if not 0 <= self.max_weight <= 1:
            raise ValueError("max_weight must be between 0 and 1")
        if self.min_weight >= self.max_weight:
            raise ValueError("min_weight must be less than max_weight")
        if not 0 <= self.safe_portfolio_bonds <= 1:
            raise ValueError("safe_portfolio_bonds must be between 0 and 1")
        if self.optimization_method not in ["sharpe", "min_variance", "equal_weight"]:
            raise ValueError("optimization_method must be 'sharpe', 'min_variance', or 'equal_weight'")
        if self.rebalance_frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError("rebalance_frequency must be 'daily', 'weekly', or 'monthly'")
        if not 0 <= self.covariance_regularization <= 0.1:
            raise ValueError("covariance_regularization must be between 0 and 0.1")
    
    
@dataclass
class ExecutorConfig:
    """Configuration for trade execution."""
    rebalance_threshold: float = 0.05  # 5% drift threshold
    rebalance_absolute_threshold: float = 100.0  # Minimum dollar amount to trigger rebalancing
    order_type: str = OrderType.MARKET.value
    broker_type: str = BrokerType.ALPACA.value
    dry_run: bool = False  # If True, log trades but don't execute
    max_position_size: float = 0.5  # Maximum position size as fraction of portfolio
    trading_hours_only: bool = True
    order_timeout: int = 300  # Order timeout in seconds
    
    def __post_init__(self):
        """Validate executor configuration after initialization."""
        if not 0 <= self.rebalance_threshold <= 1:
            raise ValueError("rebalance_threshold must be between 0 and 1")
        if self.rebalance_absolute_threshold < 0:
            raise ValueError("rebalance_absolute_threshold must be non-negative")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        if self.order_timeout < 1:
            raise ValueError("order_timeout must be positive")
    
    
@dataclass
class BrokerConfig:
    """Configuration for broker API credentials."""
    # Alpaca configuration
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    alpaca_timeout: int = 30
    
    # Interactive Brokers configuration
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    ib_timeout: int = 30
    
    # General broker settings
    max_retries: int = 3
    retry_delay: int = 1
    
    def __post_init__(self):
        """Validate broker configuration after initialization."""
        if self.alpaca_timeout < 1:
            raise ValueError("alpaca_timeout must be positive")
        if not 1 <= self.ib_port <= 65535:
            raise ValueError("ib_port must be between 1 and 65535")
        if self.ib_client_id < 0:
            raise ValueError("ib_client_id must be non-negative")
        if self.ib_timeout < 1:
            raise ValueError("ib_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
    
    
@dataclass
class SchedulerConfig:
    """Configuration for scheduling and orchestration."""
    execution_time: str = "16:30"  # After market close
    timezone: str = "America/New_York"
    retry_attempts: int = 3
    retry_delay: int = 300  # 5 minutes
    health_check_interval: int = 60  # Health check interval in seconds
    pipeline_timeout: int = 3600  # Pipeline timeout in seconds
    enable_notifications: bool = False
    notification_webhook: Optional[str] = None
    
    def __post_init__(self):
        """Validate scheduler configuration after initialization."""
        # Validate execution time format (HH:MM)
        try:
            time_parts = self.execution_time.split(":")
            if len(time_parts) != 2:
                raise ValueError()
            hour, minute = int(time_parts[0]), int(time_parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError()
        except (ValueError, IndexError):
            raise ValueError("execution_time must be in HH:MM format")
        
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.health_check_interval < 1:
            raise ValueError("health_check_interval must be positive")
        if self.pipeline_timeout < 1:
            raise ValueError("pipeline_timeout must be positive")
    
    
@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = LogLevel.INFO.value
    format: str = LogFormat.JSON.value
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    correlation_id_header: str = "X-Correlation-ID"
    
    def __post_init__(self):
        """Validate logging configuration after initialization."""
        if self.max_file_size < 1024:  # Minimum 1KB
            raise ValueError("max_file_size must be at least 1024 bytes")
        if self.backup_count < 0:
            raise ValueError("backup_count must be non-negative")


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics."""
    enable_metrics: bool = True
    metrics_port: int = 8000
    metrics_path: str = "/metrics"
    enable_health_endpoint: bool = True
    health_path: str = "/health"
    
    def __post_init__(self):
        """Validate monitoring configuration after initialization."""
        if not 1024 <= self.metrics_port <= 65535:
            raise ValueError("metrics_port must be between 1024 and 65535")
    
    
@dataclass
class Config:
    """Main configuration class combining all subsystem configs."""
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    environment: str = "development"  # "development", "staging", "production"
    debug: bool = False
    
    def __post_init__(self):
        """Validate global configuration after initialization."""
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError("environment must be 'development', 'staging', or 'production'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, '__dataclass_fields__'):
                # Nested dataclass
                result[field_name] = {
                    nested_field: getattr(value, nested_field)
                    for nested_field in value.__dataclass_fields__
                }
            else:
                result[field_name] = value
        return result


class ConfigManager:
    """Manages configuration loading from environment variables and files."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config = None
    
    def load_config(self) -> Config:
        """Load configuration from environment variables and optional config file."""
        if self._config is not None:
            return self._config
            
        # Start with default config
        config = Config()
        
        # Load from config file if provided
        if self.config_file and Path(self.config_file).exists():
            config = self._load_from_file(config)
        
        # Override with environment variables
        config = self._load_from_env(config)
        
        # Validate configuration
        self._validate_config(config)
        
        self._config = config
        return config
    
    def _load_from_file(self, config: Config) -> Config:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update config with file values
            for section, values in file_config.items():
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error loading config file {self.config_file}: {e}")
        
        return config
    
    def _load_from_env(self, config: Config) -> Config:
        """Load configuration from environment variables."""
        # Data configuration
        if os.getenv("TICKERS"):
            config.data.tickers = os.getenv("TICKERS").split(",")
        config.data.storage_type = os.getenv("STORAGE_TYPE", config.data.storage_type)
        config.data.storage_path = os.getenv("STORAGE_PATH", config.data.storage_path)
        config.data.backfill_days = int(os.getenv("BACKFILL_DAYS", config.data.backfill_days))
        
        # Optimization configuration
        config.optimization.user_age = int(os.getenv("USER_AGE", config.optimization.user_age))
        config.optimization.risk_free_rate = float(os.getenv("RISK_FREE_RATE", config.optimization.risk_free_rate))
        config.optimization.lookback_days = int(os.getenv("LOOKBACK_DAYS", config.optimization.lookback_days))
        config.optimization.min_weight = float(os.getenv("MIN_WEIGHT", config.optimization.min_weight))
        config.optimization.max_weight = float(os.getenv("MAX_WEIGHT", config.optimization.max_weight))
        config.optimization.safe_portfolio_bonds = float(os.getenv("SAFE_PORTFOLIO_BONDS", config.optimization.safe_portfolio_bonds))
        
        # Executor configuration
        config.executor.rebalance_threshold = float(os.getenv("REBALANCE_THRESHOLD", config.executor.rebalance_threshold))
        config.executor.order_type = os.getenv("ORDER_TYPE", config.executor.order_type)
        config.executor.broker_type = os.getenv("BROKER_TYPE", config.executor.broker_type)
        
        # Broker configuration
        config.broker.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        config.broker.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
        config.broker.alpaca_base_url = os.getenv("ALPACA_BASE_URL", config.broker.alpaca_base_url)
        config.broker.ib_host = os.getenv("IB_HOST", config.broker.ib_host)
        config.broker.ib_port = int(os.getenv("IB_PORT", config.broker.ib_port))
        config.broker.ib_client_id = int(os.getenv("IB_CLIENT_ID", config.broker.ib_client_id))
        
        # Scheduler configuration
        config.scheduler.execution_time = os.getenv("EXECUTION_TIME", config.scheduler.execution_time)
        config.scheduler.timezone = os.getenv("TIMEZONE", config.scheduler.timezone)
        config.scheduler.retry_attempts = int(os.getenv("RETRY_ATTEMPTS", config.scheduler.retry_attempts))
        config.scheduler.retry_delay = int(os.getenv("RETRY_DELAY", config.scheduler.retry_delay))
        
        # Logging configuration
        config.logging.level = os.getenv("LOG_LEVEL", config.logging.level)
        config.logging.format = os.getenv("LOG_FORMAT", config.logging.format)
        config.logging.file_path = os.getenv("LOG_FILE_PATH")
        
        return config
    
    def _validate_config(self, config: Config) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Only validate required fields if they're needed for actual execution
        # During development/testing, these can be empty
        validate_required = os.getenv("VALIDATE_REQUIRED_CONFIG", "false").lower() == "true"
        
        if validate_required:
            if not config.data.tickers:
                errors.append("TICKERS must be specified")
            
            if config.executor.broker_type == "alpaca":
                if not config.broker.alpaca_api_key or not config.broker.alpaca_secret_key:
                    errors.append("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for Alpaca broker")
        
        # Validate ranges
        if not 0 <= config.optimization.min_weight <= 1:
            errors.append("MIN_WEIGHT must be between 0 and 1")
        
        if not 0 <= config.optimization.max_weight <= 1:
            errors.append("MAX_WEIGHT must be between 0 and 1")
        
        if config.optimization.min_weight >= config.optimization.max_weight:
            errors.append("MIN_WEIGHT must be less than MAX_WEIGHT")
        
        if not 0 <= config.executor.rebalance_threshold <= 1:
            errors.append("REBALANCE_THRESHOLD must be between 0 and 1")
        
        if config.executor.order_type not in ["market", "limit"]:
            errors.append("ORDER_TYPE must be 'market' or 'limit'")
        
        if config.executor.broker_type not in ["alpaca", "ib"]:
            errors.append("BROKER_TYPE must be 'alpaca' or 'ib'")
        
        if config.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        if config.logging.format not in ["json", "text"]:
            errors.append("LOG_FORMAT must be 'json' or 'text'")
        
        if errors:
            raise ValueError("Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors))


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config_manager.load_config()