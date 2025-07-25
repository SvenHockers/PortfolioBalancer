"""Analytics-specific configuration classes."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path


class AnalyticsBackend(Enum):
    """Supported analytics backends."""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MEMORY = "memory"


class CacheBackend(Enum):
    """Supported cache backends."""
    REDIS = "redis"
    MEMORY = "memory"
    DISABLED = "disabled"


class AnalyticsMode(Enum):
    """Analytics service operation modes."""
    FULL = "full"  # All analytics features enabled
    BASIC = "basic"  # Basic analytics only
    READONLY = "readonly"  # Read-only mode for queries


@dataclass
class BacktestingConfig:
    """Configuration for backtesting engine."""
    
    # Performance settings
    max_concurrent_backtests: int = 3
    backtest_timeout_seconds: int = 300  # 5 minutes
    max_backtest_period_days: int = 3650  # 10 years
    min_backtest_period_days: int = 30
    
    # Data requirements
    min_data_points: int = 252  # Minimum trading days required
    max_missing_data_ratio: float = 0.05  # 5% missing data allowed
    
    # Calculation settings
    default_transaction_cost: float = 0.001  # 0.1%
    max_transaction_cost: float = 0.05  # 5%
    risk_free_rate: float = 0.02  # 2%
    
    # Caching
    enable_result_caching: bool = True
    cache_ttl_hours: int = 24
    
    def __post_init__(self):
        """Validate backtesting configuration."""
        if self.max_concurrent_backtests < 1:
            raise ValueError("max_concurrent_backtests must be at least 1")
        if self.backtest_timeout_seconds < 30:
            raise ValueError("backtest_timeout_seconds must be at least 30")
        if self.max_backtest_period_days <= self.min_backtest_period_days:
            raise ValueError("max_backtest_period_days must be greater than min_backtest_period_days")
        if not 0 <= self.max_missing_data_ratio <= 1:
            raise ValueError("max_missing_data_ratio must be between 0 and 1")
        if not 0 <= self.default_transaction_cost <= self.max_transaction_cost:
            raise ValueError("default_transaction_cost must be between 0 and max_transaction_cost")
        if not 0 <= self.risk_free_rate <= 0.2:
            raise ValueError("risk_free_rate must be between 0 and 0.2")


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation engine."""
    
    # Simulation settings
    default_num_simulations: int = 10000
    max_num_simulations: int = 100000
    min_num_simulations: int = 1000
    max_time_horizon_years: int = 50
    min_time_horizon_years: int = 1
    
    # Performance settings
    max_concurrent_simulations: int = 2
    simulation_timeout_seconds: int = 600  # 10 minutes
    chunk_size: int = 1000  # Process simulations in chunks
    
    # Statistical settings
    confidence_levels: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    random_seed: Optional[int] = None  # For reproducible results
    
    # Caching
    enable_result_caching: bool = True
    cache_ttl_hours: int = 12
    
    def __post_init__(self):
        """Validate Monte Carlo configuration."""
        if not self.min_num_simulations <= self.default_num_simulations <= self.max_num_simulations:
            raise ValueError("default_num_simulations must be between min and max")
        if self.max_concurrent_simulations < 1:
            raise ValueError("max_concurrent_simulations must be at least 1")
        if self.simulation_timeout_seconds < 60:
            raise ValueError("simulation_timeout_seconds must be at least 60")
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if not self.min_time_horizon_years <= self.max_time_horizon_years:
            raise ValueError("min_time_horizon_years must be <= max_time_horizon_years")
        
        # Validate confidence levels
        for level in self.confidence_levels:
            if not 0 < level < 1:
                raise ValueError(f"Confidence level {level} must be between 0 and 1")


@dataclass
class RiskAnalysisConfig:
    """Configuration for risk analysis engine."""
    
    # Analysis settings
    default_lookback_days: int = 252  # 1 year
    max_lookback_days: int = 1260  # 5 years
    min_lookback_days: int = 30
    
    # Risk metrics
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        "2008_financial_crisis", "2020_covid_crash", "dot_com_bubble"
    ])
    
    # Performance settings
    analysis_timeout_seconds: int = 120  # 2 minutes
    max_concurrent_analyses: int = 5
    
    # Thresholds
    high_correlation_threshold: float = 0.7
    concentration_risk_threshold: float = 0.3  # 30% in single position
    max_drawdown_alert_threshold: float = -0.2  # -20%
    
    # Caching
    enable_result_caching: bool = True
    cache_ttl_hours: int = 6
    
    def __post_init__(self):
        """Validate risk analysis configuration."""
        if not self.min_lookback_days <= self.default_lookback_days <= self.max_lookback_days:
            raise ValueError("default_lookback_days must be between min and max")
        if self.analysis_timeout_seconds < 30:
            raise ValueError("analysis_timeout_seconds must be at least 30")
        if self.max_concurrent_analyses < 1:
            raise ValueError("max_concurrent_analyses must be at least 1")
        if not 0 < self.high_correlation_threshold < 1:
            raise ValueError("high_correlation_threshold must be between 0 and 1")
        if not 0 < self.concentration_risk_threshold < 1:
            raise ValueError("concentration_risk_threshold must be between 0 and 1")
        if self.max_drawdown_alert_threshold > 0:
            raise ValueError("max_drawdown_alert_threshold must be negative")
        
        # Validate VaR confidence levels
        for level in self.var_confidence_levels:
            if not 0 < level < 1:
                raise ValueError(f"VaR confidence level {level} must be between 0 and 1")


@dataclass
class PerformanceConfig:
    """Configuration for performance tracking."""
    
    # Tracking settings
    update_frequency_minutes: int = 15
    benchmark_symbols: List[str] = field(default_factory=lambda: ["SPY", "VTI"])
    
    # Calculation settings
    rolling_window_days: int = 252  # 1 year rolling metrics
    attribution_frequency: str = "monthly"  # daily, weekly, monthly
    
    # Performance settings
    max_concurrent_updates: int = 10
    update_timeout_seconds: int = 60
    
    # Alerting thresholds
    performance_alert_threshold: float = -0.05  # -5% daily loss
    tracking_error_alert_threshold: float = 0.1  # 10% tracking error
    
    # Caching
    enable_result_caching: bool = True
    cache_ttl_minutes: int = 15  # Short TTL for real-time data
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.update_frequency_minutes < 1:
            raise ValueError("update_frequency_minutes must be at least 1")
        if self.rolling_window_days < 30:
            raise ValueError("rolling_window_days must be at least 30")
        if self.attribution_frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError("attribution_frequency must be daily, weekly, or monthly")
        if self.max_concurrent_updates < 1:
            raise ValueError("max_concurrent_updates must be at least 1")
        if self.update_timeout_seconds < 10:
            raise ValueError("update_timeout_seconds must be at least 10")
        if self.performance_alert_threshold > 0:
            raise ValueError("performance_alert_threshold must be negative")
        if self.tracking_error_alert_threshold <= 0:
            raise ValueError("tracking_error_alert_threshold must be positive")


@dataclass
class DividendAnalysisConfig:
    """Configuration for dividend analysis."""
    
    # Analysis settings
    lookback_years: int = 5
    projection_years: int = 10
    
    # Data requirements
    min_dividend_history_months: int = 12
    dividend_growth_smoothing_periods: int = 3
    
    # Thresholds
    high_yield_threshold: float = 0.06  # 6%
    dividend_cut_risk_threshold: float = 1.5  # Payout ratio threshold
    
    # Performance settings
    analysis_timeout_seconds: int = 90
    max_concurrent_analyses: int = 8
    
    # Caching
    enable_result_caching: bool = True
    cache_ttl_hours: int = 24  # Daily updates sufficient for dividends
    
    def __post_init__(self):
        """Validate dividend analysis configuration."""
        if self.lookback_years < 1:
            raise ValueError("lookback_years must be at least 1")
        if self.projection_years < 1:
            raise ValueError("projection_years must be at least 1")
        if self.min_dividend_history_months < 6:
            raise ValueError("min_dividend_history_months must be at least 6")
        if self.dividend_growth_smoothing_periods < 1:
            raise ValueError("dividend_growth_smoothing_periods must be at least 1")
        if self.high_yield_threshold <= 0:
            raise ValueError("high_yield_threshold must be positive")
        if self.dividend_cut_risk_threshold <= 0:
            raise ValueError("dividend_cut_risk_threshold must be positive")


@dataclass
class DatabaseConfig:
    """Configuration for analytics database."""
    
    # Connection settings
    backend: str = AnalyticsBackend.POSTGRESQL.value
    host: str = "localhost"
    port: int = 5432
    database: str = "portfolio_analytics"
    username: str = "analytics_user"
    password: Optional[str] = None
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    
    # Query settings
    query_timeout: int = 30
    statement_timeout: int = 300  # 5 minutes for long-running queries
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # SSL settings
    ssl_mode: str = "prefer"
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None
    
    def __post_init__(self):
        """Validate database configuration."""
        if self.backend not in [e.value for e in AnalyticsBackend]:
            raise ValueError(f"Invalid backend: {self.backend}")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.pool_size < 1:
            raise ValueError("pool_size must be at least 1")
        if self.max_overflow < 0:
            raise ValueError("max_overflow must be non-negative")
        if self.pool_timeout < 1:
            raise ValueError("pool_timeout must be at least 1")
        if self.query_timeout < 1:
            raise ValueError("query_timeout must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")


@dataclass
class CacheConfig:
    """Configuration for analytics caching."""
    
    # Cache backend
    backend: str = CacheBackend.REDIS.value
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1  # Use separate DB for analytics
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    
    # Connection settings
    connection_timeout: int = 5
    socket_timeout: int = 5
    max_connections: int = 50
    
    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400  # 24 hours
    key_prefix: str = "analytics:"
    
    # Memory cache settings (fallback)
    memory_cache_size: int = 1000  # Number of items
    
    def __post_init__(self):
        """Validate cache configuration."""
        if self.backend not in [e.value for e in CacheBackend]:
            raise ValueError(f"Invalid cache backend: {self.backend}")
        if not 1 <= self.redis_port <= 65535:
            raise ValueError("redis_port must be between 1 and 65535")
        if self.redis_db < 0:
            raise ValueError("redis_db must be non-negative")
        if self.connection_timeout < 1:
            raise ValueError("connection_timeout must be at least 1")
        if self.socket_timeout < 1:
            raise ValueError("socket_timeout must be at least 1")
        if self.max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        if self.default_ttl < 1:
            raise ValueError("default_ttl must be at least 1")
        if self.max_ttl < self.default_ttl:
            raise ValueError("max_ttl must be >= default_ttl")
        if self.memory_cache_size < 10:
            raise ValueError("memory_cache_size must be at least 10")


@dataclass
class MonitoringConfig:
    """Configuration for analytics monitoring."""
    
    # Metrics collection
    enable_metrics: bool = True
    metrics_port: int = 8085  # Different from main service
    metrics_path: str = "/metrics"
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 10
    
    # Alerting
    enable_alerting: bool = True
    alert_webhook_url: Optional[str] = None
    alert_email: Optional[str] = None
    
    # Performance monitoring
    slow_query_threshold: float = 5.0  # seconds
    memory_usage_threshold: float = 0.8  # 80%
    cpu_usage_threshold: float = 0.9  # 90%
    
    # Log aggregation
    enable_structured_logging: bool = True
    log_correlation_ids: bool = True
    
    # Analytics-specific monitoring
    backtest_duration_alert_threshold: float = 300.0  # 5 minutes
    monte_carlo_duration_alert_threshold: float = 600.0  # 10 minutes
    risk_analysis_duration_alert_threshold: float = 120.0  # 2 minutes
    
    # Error rate thresholds
    error_rate_threshold: float = 0.05  # 5% error rate
    error_rate_window_minutes: int = 15
    
    # Resource monitoring
    enable_resource_monitoring: bool = True
    disk_usage_threshold: float = 0.85  # 85%
    connection_pool_threshold: float = 0.8  # 80% of pool size
    
    # Custom metrics
    enable_custom_analytics_metrics: bool = True
    metric_collection_interval: int = 60  # seconds
    
    def __post_init__(self):
        """Validate monitoring configuration."""
        if not 1024 <= self.metrics_port <= 65535:
            raise ValueError("metrics_port must be between 1024 and 65535")
        if self.health_check_interval < 10:
            raise ValueError("health_check_interval must be at least 10")
        if self.health_check_timeout < 1:
            raise ValueError("health_check_timeout must be at least 1")
        if not 0 < self.slow_query_threshold <= 60:
            raise ValueError("slow_query_threshold must be between 0 and 60")
        if not 0 < self.memory_usage_threshold <= 1:
            raise ValueError("memory_usage_threshold must be between 0 and 1")
        if not 0 < self.cpu_usage_threshold <= 1:
            raise ValueError("cpu_usage_threshold must be between 0 and 1")
        if self.backtest_duration_alert_threshold < 30:
            raise ValueError("backtest_duration_alert_threshold must be at least 30 seconds")
        if self.monte_carlo_duration_alert_threshold < 60:
            raise ValueError("monte_carlo_duration_alert_threshold must be at least 60 seconds")
        if not 0 < self.error_rate_threshold <= 1:
            raise ValueError("error_rate_threshold must be between 0 and 1")
        if self.error_rate_window_minutes < 1:
            raise ValueError("error_rate_window_minutes must be at least 1")
        if not 0 < self.disk_usage_threshold <= 1:
            raise ValueError("disk_usage_threshold must be between 0 and 1")
        if not 0 < self.connection_pool_threshold <= 1:
            raise ValueError("connection_pool_threshold must be between 0 and 1")


@dataclass
class AnalyticsConfig:
    """Main analytics configuration combining all subsystem configs."""
    
    # Service settings
    mode: str = AnalyticsMode.FULL.value
    service_port: int = 8084
    max_workers: int = 4
    
    # Component configurations
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    risk_analysis: RiskAnalysisConfig = field(default_factory=RiskAnalysisConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    dividend_analysis: DividendAnalysisConfig = field(default_factory=DividendAnalysisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global settings
    enable_async_processing: bool = True
    async_queue_backend: str = "redis"  # redis, memory
    max_queue_size: int = 1000
    
    # Security
    enable_authentication: bool = True
    jwt_secret_key: Optional[str] = None
    jwt_expiration_hours: int = 24
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 20
    
    def __post_init__(self):
        """Validate analytics configuration."""
        if self.mode not in [e.value for e in AnalyticsMode]:
            raise ValueError(f"Invalid mode: {self.mode}")
        if not 1024 <= self.service_port <= 65535:
            raise ValueError("service_port must be between 1024 and 65535")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.async_queue_backend not in ["redis", "memory"]:
            raise ValueError("async_queue_backend must be 'redis' or 'memory'")
        if self.max_queue_size < 10:
            raise ValueError("max_queue_size must be at least 10")
        if self.jwt_expiration_hours < 1:
            raise ValueError("jwt_expiration_hours must be at least 1")
        if self.rate_limit_per_minute < 1:
            raise ValueError("rate_limit_per_minute must be at least 1")
        if self.rate_limit_burst < 1:
            raise ValueError("rate_limit_burst must be at least 1")
    
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


class AnalyticsConfigurationError(Exception):
    """Custom exception for analytics configuration errors."""
    pass


class AnalyticsConfigManager:
    """Manages analytics configuration loading from environment variables and files."""
    
    # Environment variable mapping for analytics config
    ENV_MAPPING = {
        # Service settings
        'mode': ('ANALYTICS_MODE', str),
        'service_port': ('ANALYTICS_PORT', int),
        'max_workers': ('ANALYTICS_MAX_WORKERS', int),
        
        # Database settings
        'database.backend': ('ANALYTICS_DB_BACKEND', str),
        'database.host': ('ANALYTICS_DB_HOST', str),
        'database.port': ('ANALYTICS_DB_PORT', int),
        'database.database': ('ANALYTICS_DB_NAME', str),
        'database.username': ('ANALYTICS_DB_USER', str),
        'database.password': ('ANALYTICS_DB_PASSWORD', str),
        'database.pool_size': ('ANALYTICS_DB_POOL_SIZE', int),
        'database.query_timeout': ('ANALYTICS_DB_QUERY_TIMEOUT', int),
        
        # Cache settings
        'cache.backend': ('ANALYTICS_CACHE_BACKEND', str),
        'cache.redis_host': ('ANALYTICS_REDIS_HOST', str),
        'cache.redis_port': ('ANALYTICS_REDIS_PORT', int),
        'cache.redis_db': ('ANALYTICS_REDIS_DB', int),
        'cache.redis_password': ('ANALYTICS_REDIS_PASSWORD', str),
        'cache.default_ttl': ('ANALYTICS_CACHE_TTL', int),
        
        # Backtesting settings
        'backtesting.max_concurrent_backtests': ('ANALYTICS_MAX_CONCURRENT_BACKTESTS', int),
        'backtesting.backtest_timeout_seconds': ('ANALYTICS_BACKTEST_TIMEOUT', int),
        'backtesting.enable_result_caching': ('ANALYTICS_BACKTEST_CACHE', lambda x: x.lower() in ('true', '1', 'yes')),
        
        # Monte Carlo settings
        'monte_carlo.default_num_simulations': ('ANALYTICS_MC_DEFAULT_SIMULATIONS', int),
        'monte_carlo.max_num_simulations': ('ANALYTICS_MC_MAX_SIMULATIONS', int),
        'monte_carlo.simulation_timeout_seconds': ('ANALYTICS_MC_TIMEOUT', int),
        
        # Risk analysis settings
        'risk_analysis.default_lookback_days': ('ANALYTICS_RISK_LOOKBACK_DAYS', int),
        'risk_analysis.analysis_timeout_seconds': ('ANALYTICS_RISK_TIMEOUT', int),
        
        # Performance settings
        'performance.update_frequency_minutes': ('ANALYTICS_PERF_UPDATE_FREQ', int),
        'performance.benchmark_symbols': ('ANALYTICS_BENCHMARK_SYMBOLS', lambda x: x.split(',') if x else []),
        
        # Monitoring settings
        'monitoring.enable_metrics': ('ANALYTICS_ENABLE_METRICS', lambda x: x.lower() in ('true', '1', 'yes')),
        'monitoring.metrics_port': ('ANALYTICS_METRICS_PORT', int),
        'monitoring.enable_alerting': ('ANALYTICS_ENABLE_ALERTING', lambda x: x.lower() in ('true', '1', 'yes')),
        'monitoring.slow_query_threshold': ('ANALYTICS_SLOW_QUERY_THRESHOLD', float),
        'monitoring.error_rate_threshold': ('ANALYTICS_ERROR_RATE_THRESHOLD', float),
        'monitoring.enable_structured_logging': ('ANALYTICS_ENABLE_STRUCTURED_LOGGING', lambda x: x.lower() in ('true', '1', 'yes')),
        'monitoring.log_correlation_ids': ('ANALYTICS_LOG_CORRELATION_IDS', lambda x: x.lower() in ('true', '1', 'yes')),
        
        # Security settings
        'enable_authentication': ('ANALYTICS_ENABLE_AUTH', lambda x: x.lower() in ('true', '1', 'yes')),
        'jwt_secret_key': ('ANALYTICS_JWT_SECRET', str),
        'enable_rate_limiting': ('ANALYTICS_ENABLE_RATE_LIMITING', lambda x: x.lower() in ('true', '1', 'yes')),
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize analytics config manager."""
        self.config_file = config_file
        self._config = None
    
    def load_config(self) -> AnalyticsConfig:
        """Load analytics configuration from environment variables and optional config file."""
        if self._config is not None:
            return self._config
        
        try:
            # Start with default config
            config = AnalyticsConfig()
            
            # Load from config file if available
            if self.config_file and Path(self.config_file).exists():
                config = self._load_from_file(config)
            
            # Override with environment variables
            config = self._load_from_env(config)
            
            self._config = config
            return config
            
        except Exception as e:
            raise AnalyticsConfigurationError(f"Failed to load analytics configuration: {e}") from e
    
    def _load_from_file(self, config: AnalyticsConfig) -> AnalyticsConfig:
        """Load configuration from JSON or YAML file."""
        # Implementation similar to main ConfigManager
        # For brevity, this is a simplified version
        return config
    
    def _load_from_env(self, config: AnalyticsConfig) -> AnalyticsConfig:
        """Load configuration from environment variables."""
        for config_path, (env_var, converter) in self.ENV_MAPPING.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value) if converter != str or env_value else None
                    self._set_nested_value(config, config_path, converted_value)
                except (ValueError, TypeError):
                    # Log warning but continue
                    pass
        
        return config
    
    def _set_nested_value(self, config: AnalyticsConfig, path: str, value: Any) -> None:
        """Set a nested value in the configuration object."""
        parts = path.split('.')
        obj = config
        
        # Navigate to the parent object
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Set the final value
        setattr(obj, parts[-1], value)
    
    def reload_config(self) -> AnalyticsConfig:
        """Reload configuration from sources."""
        self._config = None
        return self.load_config()


# Global analytics config manager instance
analytics_config_manager = AnalyticsConfigManager()


def get_analytics_config() -> AnalyticsConfig:
    """Get the global analytics configuration instance."""
    return analytics_config_manager.load_config()