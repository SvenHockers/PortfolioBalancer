"""Tests for core project setup and interfaces."""

import pytest
import os
from unittest.mock import patch
from src.portfolio_rebalancer.common.interfaces import (
    DataProvider, DataStorage, OptimizationStrategy, BrokerInterface
)
from src.portfolio_rebalancer.common.config import Config, ConfigManager
from src.portfolio_rebalancer.common.logging import get_logger, correlation_context


def test_interfaces_are_abstract():
    """Test that core interfaces are properly abstract."""
    # These should raise TypeError when instantiated directly
    with pytest.raises(TypeError):
        DataProvider()
    
    with pytest.raises(TypeError):
        DataStorage()
    
    with pytest.raises(TypeError):
        OptimizationStrategy()
    
    with pytest.raises(TypeError):
        BrokerInterface()


def test_config_defaults():
    """Test that configuration has sensible defaults."""
    config = Config()
    
    # Test data config defaults
    assert config.data.storage_type == "parquet"
    assert config.data.backfill_days == 252
    
    # Test optimization config defaults
    assert config.optimization.user_age == 35
    assert config.optimization.risk_free_rate == 0.02
    assert config.optimization.lookback_days == 252
    
    # Test executor config defaults
    assert config.executor.rebalance_threshold == 0.05
    assert config.executor.order_type == "market"
    assert config.executor.broker_type == "alpaca"


def test_config_manager_env_loading():
    """Test that ConfigManager loads from environment variables."""
    with patch.dict(os.environ, {
        'TICKERS': 'SPY,QQQ,VTI',
        'USER_AGE': '40',
        'REBALANCE_THRESHOLD': '0.03',
        'LOG_LEVEL': 'DEBUG'
    }):
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        assert config.data.tickers == ['SPY', 'QQQ', 'VTI']
        assert config.optimization.user_age == 40
        assert config.executor.rebalance_threshold == 0.03
        assert config.logging.level == 'DEBUG'


def test_config_validation():
    """Test configuration validation."""
    config_manager = ConfigManager()
    
    # Test validation with invalid values
    with patch.dict(os.environ, {
        'MIN_WEIGHT': '1.5',  # Invalid: > 1
        'ORDER_TYPE': 'invalid',  # Invalid order type
    }):
        with pytest.raises(ValueError, match="Configuration validation errors"):
            config_manager.load_config()


def test_logger_creation():
    """Test that loggers can be created."""
    logger = get_logger("test")
    assert logger.name == "portfolio_rebalancer.test"


def test_correlation_context():
    """Test correlation context manager."""
    logger = get_logger("test")
    
    with correlation_context("test-123") as correlation_id:
        assert correlation_id == "test-123"
        # In a real scenario, this would be captured in log output
        logger.info("Test message with correlation ID")


if __name__ == "__main__":
    pytest.main([__file__])