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
    rebalance_relative_threshold: float = 0.05  # 5% relative drift threshold
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
        if not 0 <= self.rebalance_relative_threshold <= 1:
            raise ValueError("rebalance_relative_threshold must be between 0 and 1")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        if self.order_timeout < 1:
            raise ValueError("order_timeout must be positive")
    
    
@dataclass
class SecurityConfig:
    """Configuration for security features."""
    enable_encryption: bool = True
    credential_storage_path: str = ".credentials"
    key_storage_path: str = ".keys"
    credential_rotation_days: int = 90
    enable_secure_communication: bool = True
    ssl_verify: bool = True
    
    def __post_init__(self):
        """Validate security configuration after initialization."""
        if self.credential_rotation_days < 1:
            raise ValueError("credential_rotation_days must be positive")


@dataclass
class BrokerConfig:
    """Configuration for broker API credentials."""
    # Alpaca configuration
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    alpaca_timeout: int = 30
    
    # Interactive Brokers configuration
    ib_enabled: bool = True
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    ib_timeout: int = 30

    # Trading212 configurations
    t212_api_key: Optional[str] = None
    t212_demo: bool = True
    
    # General broker settings
    max_retries: int = 3
    retry_delay: int = 1
    
    # Security settings
    use_encrypted_credentials: bool = True
    
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
    schedule_interval_minutes: Optional[int] = None
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
            hour_str, minute_str = time_parts[0], time_parts[1]
            
            # Check format: must be exactly 2 digits for both hour and minute
            if len(hour_str) != 2 or len(minute_str) != 2:
                raise ValueError()
            
            hour, minute = int(hour_str), int(minute_str)
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
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
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


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class ConfigManager:
    """Manages configuration loading from environment variables and files."""
    
    # Environment variable mapping with type conversion
    ENV_MAPPING = {
        # Data configuration
        'data.tickers': ('TICKERS', lambda x: x.split(',') if x else []),
        'data.storage_type': ('STORAGE_TYPE', str),
        'data.storage_path': ('STORAGE_PATH', str),
        'data.backfill_days': ('BACKFILL_DAYS', int),
        'data.data_quality_checks': ('DATA_QUALITY_CHECKS', lambda x: x.lower() in ('true', '1', 'yes')),
        'data.max_missing_days': ('MAX_MISSING_DAYS', int),
        'data.price_change_threshold': ('PRICE_CHANGE_THRESHOLD', float),
        
        # Optimization configuration
        'optimization.user_age': ('USER_AGE', int),
        'optimization.risk_free_rate': ('RISK_FREE_RATE', float),
        'optimization.lookback_days': ('LOOKBACK_DAYS', int),
        'optimization.min_weight': ('MIN_WEIGHT', float),
        'optimization.max_weight': ('MAX_WEIGHT', float),
        'optimization.safe_portfolio_bonds': ('SAFE_PORTFOLIO_BONDS', float),
        'optimization.optimization_method': ('OPTIMIZATION_METHOD', str),
        'optimization.rebalance_frequency': ('REBALANCE_FREQUENCY', str),
        'optimization.covariance_regularization': ('COVARIANCE_REGULARIZATION', float),
        
        # Executor configuration
        'executor.rebalance_threshold': ('REBALANCE_THRESHOLD', float),
        'executor.rebalance_absolute_threshold': ('REBALANCE_ABSOLUTE_THRESHOLD', float),
        'executor.rebalance_relative_threshold': ('REBALANCE_RELATIVE_THRESHOLD', float),
        'executor.order_type': ('ORDER_TYPE', str),
        'executor.broker_type': ('BROKER_TYPE', str),
        'executor.dry_run': ('DRY_RUN', lambda x: x.lower() in ('true', '1', 'yes')),
        'executor.max_position_size': ('MAX_POSITION_SIZE', float),
        'executor.trading_hours_only': ('TRADING_HOURS_ONLY', lambda x: x.lower() in ('true', '1', 'yes')),
        'executor.order_timeout': ('ORDER_TIMEOUT', int),
        
        # Broker configuration
        'broker.alpaca_api_key': ('ALPACA_API_KEY', str),
        'broker.alpaca_secret_key': ('ALPACA_SECRET_KEY', str),
        'broker.alpaca_base_url': ('ALPACA_BASE_URL', str),
        'broker.alpaca_timeout': ('ALPACA_TIMEOUT', int),
        'broker.ib_enabled': ('IB_ENABLED', lambda x: x.lower() in ('true', '1', 'yes')),
        'broker.ib_host': ('IB_HOST', str),
        'broker.ib_port': ('IB_PORT', int),
        'broker.ib_client_id': ('IB_CLIENT_ID', int),
        'broker.ib_timeout': ('IB_TIMEOUT', int),
        'broker.t212_api_key': ('T212_API_KEY', str),
        'broker.t212_demo': ('T212_DEMO', str),
        'broker.max_retries': ('BROKER_MAX_RETRIES', int),
        'broker.retry_delay': ('BROKER_RETRY_DELAY', int),
        'broker.use_encrypted_credentials': ('USE_ENCRYPTED_CREDENTIALS', lambda x: x.lower() in ('true', '1', 'yes')),
        
        # Security configuration
        'security.enable_encryption': ('ENABLE_ENCRYPTION', lambda x: x.lower() in ('true', '1', 'yes')),
        'security.credential_storage_path': ('CREDENTIAL_STORAGE_PATH', str),
        'security.key_storage_path': ('KEY_STORAGE_PATH', str),
        'security.credential_rotation_days': ('CREDENTIAL_ROTATION_DAYS', int),
        'security.enable_secure_communication': ('ENABLE_SECURE_COMMUNICATION', lambda x: x.lower() in ('true', '1', 'yes')),
        'security.ssl_verify': ('SSL_VERIFY', lambda x: x.lower() in ('true', '1', 'yes')),
        
        # Scheduler configuration
        'scheduler.execution_time': ('EXECUTION_TIME', str),
        'scheduler.schedule_interval_minutes': ('SCHEDULE_INTERVAL_MINUTES', str),
        'scheduler.timezone': ('TIMEZONE', str),
        'scheduler.retry_attempts': ('RETRY_ATTEMPTS', int),
        'scheduler.retry_delay': ('RETRY_DELAY', int),
        'scheduler.health_check_interval': ('HEALTH_CHECK_INTERVAL', int),
        'scheduler.pipeline_timeout': ('PIPELINE_TIMEOUT', int),
        'scheduler.enable_notifications': ('ENABLE_NOTIFICATIONS', lambda x: x.lower() in ('true', '1', 'yes')),
        'scheduler.notification_webhook': ('NOTIFICATION_WEBHOOK', str),
        
        # Logging configuration
        'logging.level': ('LOG_LEVEL', str),
        'logging.format': ('LOG_FORMAT', str),
        'logging.file_path': ('LOG_FILE_PATH', str),
        'logging.max_file_size': ('LOG_MAX_FILE_SIZE', int),
        'logging.backup_count': ('LOG_BACKUP_COUNT', int),
        'logging.enable_console': ('LOG_ENABLE_CONSOLE', lambda x: x.lower() in ('true', '1', 'yes')),
        'logging.correlation_id_header': ('LOG_CORRELATION_ID_HEADER', str),
        
        # Monitoring configuration
        'monitoring.enable_metrics': ('ENABLE_METRICS', lambda x: x.lower() in ('true', '1', 'yes')),
        'monitoring.metrics_port': ('METRICS_PORT', int),
        'monitoring.metrics_path': ('METRICS_PATH', str),
        'monitoring.enable_health_endpoint': ('ENABLE_HEALTH_ENDPOINT', lambda x: x.lower() in ('true', '1', 'yes')),
        'monitoring.health_path': ('HEALTH_PATH', str),
        
        # Global configuration
        'environment': ('ENVIRONMENT', str),
        'debug': ('DEBUG', lambda x: x.lower() in ('true', '1', 'yes')),
    }
    
    def __init__(self, config_file: Optional[str] = None, auto_discover: bool = True):
        """
        Initialize ConfigManager.
        
        Args:
            config_file: Path to configuration file
            auto_discover: If True, automatically discover config files in common locations
        """
        self.config_file = config_file
        self.auto_discover = auto_discover
        self._config = None
        self._logger = logging.getLogger(__name__)
    
    def load_config(self) -> Config:
        """Load configuration from environment variables and optional config file."""
        if self._config is not None:
            return self._config
            
        try:
            # Start with default config
            config = Config()
            
            # Auto-discover config file if not provided
            if not self.config_file and self.auto_discover:
                self.config_file = self._discover_config_file()
            
            # Load from config file if available
            if self.config_file and Path(self.config_file).exists():
                config = self._load_from_file(config)
                self._logger.info(f"Loaded configuration from file: {self.config_file}")
            
            # Override with environment variables
            config = self._load_from_env(config)
            
            # Post-initialization validation will be called by dataclass __post_init__
            self._config = config
            self._logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e
    
    def _discover_config_file(self) -> Optional[str]:
        """Auto-discover configuration file in common locations."""
        search_paths = [
            "config.json",
            "config.yaml", 
            "config.yml",
            ".config/portfolio_rebalancer.json",
            ".config/portfolio_rebalancer.yaml",
            os.path.expanduser("~/.portfolio_rebalancer.json"),
            os.path.expanduser("~/.portfolio_rebalancer.yaml"),
        ]
        
        for path in search_paths:
            if Path(path).exists():
                self._logger.info(f"Auto-discovered config file: {path}")
                return path
        
        return None
    
    def _load_from_file(self, config: Config) -> Config:
        """Load configuration from JSON or YAML file."""
        file_path = Path(self.config_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        file_config = yaml.safe_load(f)
                    except ImportError:
                        raise ConfigurationError(
                            "PyYAML is required to load YAML configuration files. "
                            "Install with: pip install PyYAML"
                        )
                elif file_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    # Try JSON first, then YAML
                    content = f.read()
                    try:
                        file_config = json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            file_config = yaml.safe_load(content)
                        except ImportError:
                            raise ConfigurationError(
                                "Could not parse configuration file. "
                                "Ensure it's valid JSON or install PyYAML for YAML support."
                            )
            
            if not isinstance(file_config, dict):
                raise ConfigurationError("Configuration file must contain a JSON object or YAML mapping")
            
            # Update config with file values
            self._update_config_from_dict(config, file_config)
                            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Error parsing config file {self.config_file}: {e}")
        except FileNotFoundError:
            raise ConfigurationError(f"Config file not found: {self.config_file}")
        except Exception as e:
            raise ConfigurationError(f"Error loading config file {self.config_file}: {e}")
        
        return config
    
    def _update_config_from_dict(self, config: Config, config_dict: Dict[str, Any]) -> None:
        """Update configuration object from dictionary."""
        for section, values in config_dict.items():
            if hasattr(config, section) and isinstance(values, dict):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        # Type conversion based on current field type
                        current_value = getattr(section_config, key)
                        if current_value is not None:
                            target_type = type(current_value)
                            if target_type == bool and isinstance(value, str):
                                value = value.lower() in ('true', '1', 'yes', 'on')
                            elif target_type in (int, float) and isinstance(value, str):
                                value = target_type(value)
                            elif target_type == list and isinstance(value, str):
                                value = value.split(',')
                        setattr(section_config, key, value)
            elif hasattr(config, section):
                # Direct field assignment
                setattr(config, section, value)
    
    def _load_from_env(self, config: Config) -> Config:
        """Load configuration from environment variables with type conversion."""
        for config_path, (env_var, converter) in self.ENV_MAPPING.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert the value using the specified converter
                    if converter == str and env_value == '':
                        # Handle empty strings for optional string fields
                        converted_value = None
                    else:
                        converted_value = converter(env_value)
                    
                    # Set the value in the config object
                    self._set_nested_value(config, config_path, converted_value)
                    
                except (ValueError, TypeError) as e:
                    self._logger.warning(
                        f"Failed to convert environment variable {env_var}='{env_value}': {e}"
                    )
        
        return config
    
    def _set_nested_value(self, config: Config, path: str, value: Any) -> None:
        """Set a nested value in the configuration object."""
        parts = path.split('.')
        obj = config
        
        # Navigate to the parent object
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Set the final value
        setattr(obj, parts[-1], value)
    
    def reload_config(self) -> Config:
        """Reload configuration from sources."""
        self._config = None
        return self.load_config()
    
    def save_config(self, config: Config, file_path: str, format: str = 'json') -> None:
        """Save configuration to file."""
        config_dict = config.to_dict()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    try:
                        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                    except ImportError:
                        raise ConfigurationError(
                            "PyYAML is required to save YAML configuration files. "
                            "Install with: pip install PyYAML"
                        )
                else:
                    json.dump(config_dict, f, indent=2, default=str)
            
            self._logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {file_path}: {e}")
    
    def validate_runtime_config(self, config: Config) -> List[str]:
        """Validate configuration for runtime requirements."""
        errors = []
        
        # Check required fields for production
        if config.environment == "production":
            if not config.data.tickers:
                errors.append("TICKERS must be specified for production environment")
            
            if config.executor.broker_type == BrokerType.ALPACA.value:
                if not config.broker.alpaca_api_key or not config.broker.alpaca_secret_key:
                    errors.append("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for Alpaca broker")
        
        # Validate enum values
        try:
            StorageType(config.data.storage_type)
        except ValueError:
            errors.append(f"Invalid storage_type: {config.data.storage_type}")
        
        try:
            OrderType(config.executor.order_type)
        except ValueError:
            errors.append(f"Invalid order_type: {config.executor.order_type}")
        
        try:
            BrokerType(config.executor.broker_type)
        except ValueError:
            errors.append(f"Invalid broker_type: {config.executor.broker_type}")
        
        try:
            LogLevel(config.logging.level)
        except ValueError:
            errors.append(f"Invalid log_level: {config.logging.level}")
        
        try:
            LogFormat(config.logging.format)
        except ValueError:
            errors.append(f"Invalid log_format: {config.logging.format}")
        
        return errors


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config_manager.load_config()