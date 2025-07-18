"""Base broker implementation with common functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
from portfolio_rebalancer.common.interfaces import BrokerInterface
from portfolio_rebalancer.common.models import OrderType, OrderSide
from portfolio_rebalancer.common.config import get_config

logger = logging.getLogger(__name__)


class BaseBroker(BrokerInterface, ABC):
    """Base class for broker implementations with common functionality."""
    
    def __init__(self, config: Optional[object] = None):
        """Initialize the broker with configuration."""
        self.config = config or get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def place_order(self, symbol: str, quantity: float, order_type: str) -> Optional[str]:
        """
        Place a trade order with error handling and logging.
        Returns None on error instead of raising, to ensure container stability.
        """
        try:
            # Determine order side based on quantity
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            
            # Ensure quantity is positive for the broker API
            abs_quantity = abs(quantity)
            
            # Validate order type
            if order_type not in [OrderType.MARKET, OrderType.LIMIT]:
                raise ValueError(f"Invalid order type: {order_type}")
            
            # Log order details
            self.logger.info(
                f"Placing {order_type} {side} order for {abs_quantity} shares of {symbol}",
                extra={
                    "symbol": symbol,
                    "quantity": quantity,
                    "order_type": order_type,
                    "side": side
                }
            )
            
            # Call broker-specific implementation
            order_id = self._place_order_impl(symbol, abs_quantity, order_type, side)
            
            self.logger.info(f"Order placed successfully with ID: {order_id}")
            return order_id
            
        except Exception as e:
            self.logger.error(
                f"Failed to place order for {symbol}: {str(e)}",
                extra={
                    "symbol": symbol,
                    "quantity": quantity,
                    "order_type": order_type,
                    "error": str(e)
                },
                exc_info=True
            )
            # Do not re-raise; return None to prevent container crash
            return None
    
    @abstractmethod
    def _place_order_impl(self, symbol: str, quantity: float, order_type: str, side: OrderSide) -> str:
        """
        Broker-specific implementation for placing orders.
        
        Args:
            symbol: Ticker symbol
            quantity: Absolute order quantity (always positive)
            order_type: Order type ('market' or 'limit')
            side: Order side ('buy' or 'sell')
            
        Returns:
            Order ID string
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """
        Get current portfolio positions.
        
        Returns:
            Dictionary mapping ticker symbols to position quantities
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """
        Get status of a placed order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status string
        """
        pass