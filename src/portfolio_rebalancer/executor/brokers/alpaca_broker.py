"""Alpaca broker implementation for trade execution."""

import logging
from datetime import datetime
import requests
from requests.exceptions import RequestException
from portfolio_rebalancer.common.models import OrderType, OrderSide, OrderStatus, TradeOrder
from .base_broker import BaseBroker

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation for trade execution."""
    
    def __init__(self, config: object = None):
        """Initialize the Alpaca broker with API credentials."""
        super().__init__(config)
        
        # Get API credentials from config
        self.api_key = self.config.broker.alpaca_api_key
        self.secret_key = self.config.broker.alpaca_secret_key
        self.base_url = self.config.broker.alpaca_base_url
        
        # Validate credentials
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key must be provided")
        
        # Set up API headers
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
        # Add SSL verification based on security config
        self.verify_ssl = getattr(self.config.security, 'ssl_verify', True)
        
        # Validate API connection
        self._validate_connection()
        
    def _validate_connection(self) -> None:
        """Validate API connection by checking account status. Returns None on error."""
        try:
            response = requests.get(
                f"{self.base_url}/v2/account", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.config.broker.alpaca_timeout
            )
            response.raise_for_status()
            self.logger.info("Successfully connected to Alpaca API")
        except RequestException as e:
            self.logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            # Do not raise; just log error
            return None
    
    def get_positions(self) -> dict[str, float]:
        """
        Get current portfolio positions from Alpaca. Returns empty dict on error.
        """
        try:
            response = requests.get(
                f"{self.base_url}/v2/positions", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.config.broker.alpaca_timeout
            )
            response.raise_for_status()
            positions = {}
            for position in response.json():
                symbol = position["symbol"]
                quantity = float(position["qty"])
                positions[symbol] = quantity
            self.logger.info(f"Retrieved {len(positions)} positions from Alpaca")
            return positions
        except RequestException as e:
            self.logger.error(f"Failed to get positions from Alpaca: {str(e)}")
            return {}
    
    def _place_order_impl(self, symbol: str, quantity: float, order_type: str, side: OrderSide) -> str:
        """
        Place an order with Alpaca API. Returns None on error.
        """
        try:
            payload = {
                "symbol": symbol,
                "qty": str(quantity),
                "side": side,
                "type": order_type,
                "time_in_force": "day"
            }
            response = requests.post(
                f"{self.base_url}/v2/orders", 
                json=payload, 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.config.broker.alpaca_timeout
            )
            response.raise_for_status()
            order_data = response.json()
            order_id = order_data["id"]
            return order_id
        except RequestException as e:
            self.logger.error(f"Failed to place order with Alpaca: {str(e)}")
            return None
    
    def get_order_status(self, order_id: str) -> str:
        """
        Get status of a placed order from Alpaca. Returns OrderStatus.PENDING on error.
        """
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.config.broker.alpaca_timeout
            )
            response.raise_for_status()
            order_data = response.json()
            alpaca_status = order_data["status"]
            status_mapping = {
                "new": OrderStatus.PENDING,
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
                "expired": OrderStatus.CANCELLED
            }
            return status_mapping.get(alpaca_status, OrderStatus.PENDING)
        except RequestException as e:
            self.logger.error(f"Failed to get order status from Alpaca: {str(e)}")
            return OrderStatus.PENDING
    
    def get_order_details(self, order_id: str) -> TradeOrder:
        """
        Get detailed information about an order. Returns None on error.
        """
        try:
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.config.broker.alpaca_timeout
            )
            response.raise_for_status()
            order_data = response.json()
            status_mapping = {
                "new": OrderStatus.PENDING,
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIALLY_FILLED,
                "canceled": OrderStatus.CANCELLED,
                "rejected": OrderStatus.REJECTED,
                "expired": OrderStatus.CANCELLED
            }
            quantity = float(order_data["qty"])
            if order_data["side"] == "sell":
                quantity = -quantity
            order = TradeOrder(
                order_id=order_data["id"],
                symbol=order_data["symbol"],
                quantity=quantity,
                order_type=OrderType(order_data["type"]),
                side=OrderSide(order_data["side"]),
                status=status_mapping.get(order_data["status"], OrderStatus.PENDING),
                timestamp=datetime.fromisoformat(order_data["created_at"].replace("Z", "+00:00")),
                fill_price=float(order_data["filled_avg_price"]) if order_data["filled_avg_price"] else None
            )
            return order
        except RequestException as e:
            self.logger.error(f"Failed to get order details from Alpaca: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order. Returns False on error.
        """
        try:
            response = requests.delete(
                f"{self.base_url}/v2/orders/{order_id}", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=self.config.broker.alpaca_timeout
            )
            if response.status_code == 404:
                self.logger.warning(f"Order {order_id} not found or already processed")
                return False
            response.raise_for_status()
            self.logger.info(f"Successfully cancelled order {order_id}")
            return True
        except RequestException as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False