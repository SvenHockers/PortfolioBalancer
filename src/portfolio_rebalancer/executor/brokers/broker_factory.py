"""Factory for creating broker instances based on configuration."""

import logging
from typing import Optional
from portfolio_rebalancer.common.interfaces import BrokerInterface
from portfolio_rebalancer.common.config import get_config
from .alpaca_broker import AlpacaBroker
from .ib_broker import IBBroker
from .t212_broker import T212Broker

logger = logging.getLogger(__name__)


class BrokerFactory:
    """Factory for creating broker instances."""
    
    @staticmethod
    def create_broker() -> Optional[BrokerInterface]:
        """
        Create a broker instance based on configuration. Returns None on error.
        """
        try:
            config = get_config()
            broker_type = config.executor.broker_type.lower()
            logger.info(f"Creating broker instance for type: {broker_type}")
            
            if broker_type == "alpaca":
                try:
                    return AlpacaBroker()
                except Exception as e:
                    logger.error(f"Failed to create Alpaca broker: {e}")
                    # Don't return None, let the error propagate for now
                    # In the future, we could fall back to a different broker
                    raise
                    
            elif broker_type == "ib":
                try:
                    broker = IBBroker()
                    logger.info(f"IB Broker created successfully (mock_mode: {broker.mock_mode})")
                    return broker
                except Exception as e:
                    logger.error(f"Failed to create IB broker: {e}")
                    # IB broker should handle its own fallback to mock mode
                    # If it still fails, something is seriously wrong
                    raise
                    
            elif broker_type == "t212":
                try:
                    return T212Broker()
                except Exception as e:
                    logger.error(f"Failed to create T212 broker: {e}")
                    raise
                    
            else:
                logger.error(f"Unsupported broker type: {broker_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create broker: {e}")
            return None