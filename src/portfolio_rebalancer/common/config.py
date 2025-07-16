"""Configuration management system with environment variable support."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class DataConfig:
    """Configuration for data fetching and storage."""
    tickers: List[str] = field(default_factory=list)
    storage_type: str = "parquet"  # "parquet" or "sqlite"
    storage_path: str = "data"
    backfill_days: int = 252  # ~1 year of trading days
    
    
@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization."""
    user_age: int = 35
    risk_free_rate: float = 0.02
    lookback_days: int = 252
    min_weight: float = 0.0
    max_weight: float = 0.4
    safe_portfolio_bonds: float = 0.8
    
    
@dataclass
class ExecutorConfig:
    """Configuration for trade execution."""
    rebalance_threshold: float = 0.05  # 5% drift threshold
    order_type: str = "market"  # "market" or "limit"
    broker_type: str = "alpaca"  # "alpaca" or "ib"
    
    
@dataclass
class BrokerConfig:
    """Configuration for broker API credentials."""
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"  # Paper trading by default
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    
    
@dataclass
class SchedulerConfig:
    """Configuration for scheduling and orchestration."""
    execution_time: str = "16:30"  # After market close
    timezone: str = "America/New_York"
    retry_attempts: int = 3
    retry_delay: int = 300  # 5 minutes
    
    
@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "json"  # "json" or "text"
    file_path: Optional[str] = None
    
    
@dataclass
class Config:
    """Main configuration class combining all subsystem configs."""
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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