"""Factory for creating broker instances based on configuration."""

import logging
from typing import Optional
from portfolio_rebalancer.common.interfaces import BrokerInterface
from portfolio_rebalancer.common.config import get_config
from .alpaca_broker import AlpacaBroker
from .ib_broker import IBBroker

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
                return AlpacaBroker()
            elif broker_type == "ib":
                return IBBroker()
            else:
                logger.error(f"Unsupported broker type: {broker_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create broker: {e}")
            return None