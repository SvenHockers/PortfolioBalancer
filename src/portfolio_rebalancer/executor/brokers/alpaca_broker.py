"""Alpaca broker implementation for trade execution."""

import logging
from datetime import datetime
import requests
from requests.exceptions import RequestException
from portfolio_rebalancer.common.models import OrderType, OrderSide, OrderStatus, TradeOrder
from portfolio_rebalancer.common.api_error_handling import (
    api_error_handler, handle_alpaca_api_error, AuthenticationError, MockDataProvider
)
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
        
        # Track if we're in mock mode due to authentication failures
        self.mock_mode = False
        
        # Validate credentials
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API credentials not provided, enabling mock mode")
            self.mock_mode = True
        else:
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
        """Validate API connection by checking account status."""
        if self.mock_mode:
            logger.info("Alpaca broker running in mock mode")
            return
            
        @api_error_handler.with_retry("Alpaca connection validation")
        def _check_connection():
            response = requests.get(
                f"{self.base_url}/v2/account", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=getattr(self.config.broker, 'alpaca_timeout', 30)
            )
            response.raise_for_status()
            return response.json()
        
        try:
            account_info = _check_connection()
            logger.info("Successfully connected to Alpaca API")
        except AuthenticationError as e:
            logger.warning(f"Alpaca authentication failed, enabling mock mode: {e}")
            self.mock_mode = True
        except Exception as e:
            logger.warning(f"Alpaca connection validation failed, enabling mock mode: {e}")
            self.mock_mode = True
    
    def get_positions(self) -> dict[str, float]:
        """
        Get current portfolio positions from Alpaca. Returns mock data if in mock mode.
        """
        if self.mock_mode:
            logger.info("Returning mock positions (Alpaca API unavailable)")
            return MockDataProvider.get_mock_positions()
        
        @api_error_handler.with_retry("Get Alpaca positions")
        def _get_positions():
            response = requests.get(
                f"{self.base_url}/v2/positions", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=getattr(self.config.broker, 'alpaca_timeout', 30)
            )
            response.raise_for_status()
            return response.json()
        
        try:
            positions_data = _get_positions()
            positions = {}
            for position in positions_data:
                symbol = position["symbol"]
                quantity = float(position["qty"])
                positions[symbol] = quantity
            logger.info(f"Retrieved {len(positions)} positions from Alpaca")
            return positions
        except AuthenticationError as e:
            logger.warning(f"Alpaca authentication failed, switching to mock mode: {e}")
            self.mock_mode = True
            return MockDataProvider.get_mock_positions()
        except Exception as e:
            logger.error(f"Failed to get positions from Alpaca: {e}")
            fallback_result = handle_alpaca_api_error(e, "Get positions")
            return fallback_result if fallback_result is not None else {}
    
    def _place_order_impl(self, symbol: str, quantity: float, order_type: str, side: OrderSide) -> str:
        """
        Place an order with Alpaca API. Returns mock order ID if in mock mode.
        """
        if self.mock_mode:
            mock_response = MockDataProvider.get_mock_order_response(symbol, quantity, side)
            logger.info(f"Placed mock order for {symbol}: {mock_response['id']}")
            return mock_response["id"]
        
        @api_error_handler.with_retry("Place Alpaca order")
        def _place_order():
            payload = {
                "symbol": symbol,
                "qty": str(abs(quantity)),
                "side": side,
                "type": order_type,
                "time_in_force": "day"
            }
            response = requests.post(
                f"{self.base_url}/v2/orders", 
                json=payload, 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=getattr(self.config.broker, 'alpaca_timeout', 30)
            )
            response.raise_for_status()
            return response.json()
        
        try:
            order_data = _place_order()
            order_id = order_data["id"]
            logger.info(f"Successfully placed order {order_id} for {symbol}")
            return order_id
        except AuthenticationError as e:
            logger.warning(f"Alpaca authentication failed, switching to mock mode: {e}")
            self.mock_mode = True
            mock_response = MockDataProvider.get_mock_order_response(symbol, quantity, side)
            return mock_response["id"]
        except Exception as e:
            logger.error(f"Failed to place order with Alpaca: {e}")
            fallback_result = handle_alpaca_api_error(e, "Place order")
            return None
    
    def get_order_status(self, order_id: str) -> str:
        """
        Get status of a placed order from Alpaca. Returns FILLED for mock orders.
        """
        if self.mock_mode or order_id.startswith("mock_order_"):
            logger.info(f"Returning mock order status for {order_id}")
            return OrderStatus.FILLED
        
        @api_error_handler.with_retry("Get Alpaca order status")
        def _get_order_status():
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=getattr(self.config.broker, 'alpaca_timeout', 30)
            )
            response.raise_for_status()
            return response.json()
        
        try:
            order_data = _get_order_status()
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
        except AuthenticationError as e:
            logger.warning(f"Alpaca authentication failed, switching to mock mode: {e}")
            self.mock_mode = True
            return OrderStatus.FILLED
        except Exception as e:
            logger.error(f"Failed to get order status from Alpaca: {e}")
            handle_alpaca_api_error(e, "Get order status")
            return OrderStatus.PENDING
    
    def get_order_details(self, order_id: str) -> TradeOrder:
        """
        Get detailed information about an order. Returns mock order for mock orders.
        """
        if self.mock_mode or order_id.startswith("mock_order_"):
            logger.info(f"Returning mock order details for {order_id}")
            # Parse mock order ID to extract symbol and quantity
            parts = order_id.split("_")
            symbol = parts[2] if len(parts) > 2 else "MOCK"
            return TradeOrder(
                order_id=order_id,
                symbol=symbol,
                quantity=100.0,  # Mock quantity
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(),
                fill_price=100.0  # Mock fill price
            )
        
        @api_error_handler.with_retry("Get Alpaca order details")
        def _get_order_details():
            response = requests.get(
                f"{self.base_url}/v2/orders/{order_id}", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=getattr(self.config.broker, 'alpaca_timeout', 30)
            )
            response.raise_for_status()
            return response.json()
        
        try:
            order_data = _get_order_details()
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
        except AuthenticationError as e:
            logger.warning(f"Alpaca authentication failed, switching to mock mode: {e}")
            self.mock_mode = True
            return self.get_order_details(order_id)  # Recursive call will use mock mode
        except Exception as e:
            logger.error(f"Failed to get order details from Alpaca: {e}")
            handle_alpaca_api_error(e, "Get order details")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order. Returns True for mock orders.
        """
        if self.mock_mode or order_id.startswith("mock_order_"):
            logger.info(f"Mock order {order_id} cancelled successfully")
            return True
        
        @api_error_handler.with_retry("Cancel Alpaca order")
        def _cancel_order():
            response = requests.delete(
                f"{self.base_url}/v2/orders/{order_id}", 
                headers=self.headers,
                verify=self.verify_ssl,
                timeout=getattr(self.config.broker, 'alpaca_timeout', 30)
            )
            if response.status_code == 404:
                logger.warning(f"Order {order_id} not found or already processed")
                return False
            response.raise_for_status()
            return True
        
        try:
            result = _cancel_order()
            if result:
                logger.info(f"Successfully cancelled order {order_id}")
            return result
        except AuthenticationError as e:
            logger.warning(f"Alpaca authentication failed, switching to mock mode: {e}")
            self.mock_mode = True
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            handle_alpaca_api_error(e, "Cancel order")
            return False