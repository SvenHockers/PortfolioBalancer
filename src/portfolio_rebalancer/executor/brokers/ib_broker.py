"""Interactive Brokers implementation for trade execution."""

import logging
import time
from typing import Dict, Optional, List, Any
from datetime import datetime
import threading
from portfolio_rebalancer.common.models import OrderType, OrderSide, OrderStatus, TradeOrder
from .base_broker import BaseBroker

logger = logging.getLogger(__name__)

# Import IB API conditionally to allow for testing without the actual dependency
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order as IBOrder
    from ibapi.common import OrderId
    from ibapi.execution import Execution
    IB_API_AVAILABLE = True
except ImportError:
    logger.warning("Interactive Brokers API not available. Using mock implementation.")
    IB_API_AVAILABLE = False
    # Define dummy classes for testing
    class EClient: pass
    class EWrapper: pass
    class Contract: pass
    class IBOrder: pass
    OrderId = int
    class Execution: pass


class IBWrapper(EWrapper):
    """Wrapper class for IB API callbacks."""
    
    def __init__(self):
        super().__init__()
        self.positions = {}
        self.orders = {}
        self.order_status = {}
        self.executions = {}
        self.next_order_id = None
        self.next_order_id_ready = threading.Event()
        
    def nextValidId(self, orderId: int):
        """Called by IB when the next valid order ID is received."""
        self.next_order_id = orderId
        self.next_order_id_ready.set()
        
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Called by IB when position data is received."""
        symbol = contract.symbol
        self.positions[symbol] = position
        
    def orderStatus(self, orderId: OrderId, status: str, filled: float,
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int,
                   whyHeld: str, mktCapPrice: float):
        """Called by IB when order status changes."""
        self.order_status[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avgFillPrice": avgFillPrice
        }
        
    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        """Called by IB when execution details are received."""
        orderId = execution.orderId
        if orderId not in self.executions:
            self.executions[orderId] = []
        self.executions[orderId].append(execution)


class IBClient(EClient):
    """Client class for IB API requests."""
    
    def __init__(self, wrapper):
        super().__init__(wrapper)
        self.wrapper = wrapper


class IBBroker(BaseBroker):
    """Interactive Brokers implementation for trade execution."""
    
    def __init__(self):
        """Initialize the IB broker with API connection."""
        super().__init__()
        
        if not IB_API_AVAILABLE:
            raise ImportError("Interactive Brokers API is not available. Please install ibapi package.")
        
        # Get connection parameters from config
        self.host = self.config.broker.ib_host
        self.port = self.config.broker.ib_port
        self.client_id = self.config.broker.ib_client_id
        
        # Initialize API wrapper and client
        self.wrapper = IBWrapper()
        self.client = IBClient(self.wrapper)
        
        # Connect to IB TWS or Gateway
        self._connect()
        
        # Order tracking
        self.orders = {}
        
    def _connect(self) -> None:
        """Connect to Interactive Brokers TWS or Gateway."""
        try:
            self.client.connect(self.host, self.port, self.client_id)
            
            # Start client thread
            api_thread = threading.Thread(target=self.client.run)
            api_thread.daemon = True
            api_thread.start()
            
            # Wait for nextValidId
            if not self.wrapper.next_order_id_ready.wait(10):
                raise ConnectionError("Timed out waiting for IB API connection")
                
            self.logger.info(f"Successfully connected to IB API at {self.host}:{self.port}")
            
            # Request positions
            self.client.reqPositions()
            time.sleep(1)  # Give time for positions to be received
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IB API: {str(e)}")
            raise ConnectionError(f"Failed to connect to IB API: {str(e)}")
    
    def get_positions(self) -> Dict[str, float]:
        """
        Get current portfolio positions from IB.
        
        Returns:
            Dictionary mapping ticker symbols to position quantities
        """
        try:
            # Request positions update
            self.client.reqPositions()
            time.sleep(1)  # Give time for positions to be received
            
            return self.wrapper.positions.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to get positions from IB: {str(e)}")
            raise
    
    def _place_order_impl(self, symbol: str, quantity: float, order_type: str, side: OrderSide) -> str:
        """
        Place an order with IB API.
        
        Args:
            symbol: Ticker symbol
            quantity: Absolute order quantity (always positive)
            order_type: Order type ('market' or 'limit')
            side: Order side ('buy' or 'sell')
            
        Returns:
            Order ID string
        """
        try:
            # Create contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            
            # Create order
            ib_order = IBOrder()
            ib_order.action = "BUY" if side == OrderSide.BUY else "SELL"
            ib_order.totalQuantity = quantity
            ib_order.orderType = "MKT" if order_type == OrderType.MARKET else "LMT"
            
            # Get next valid order ID
            order_id = self.wrapper.next_order_id
            self.wrapper.next_order_id += 1
            
            # Place order
            self.client.placeOrder(order_id, contract, ib_order)
            
            # Store order details
            self.orders[order_id] = {
                "symbol": symbol,
                "quantity": quantity,
                "order_type": order_type,
                "side": side,
                "timestamp": datetime.now()
            }
            
            return str(order_id)
            
        except Exception as e:
            self.logger.error(f"Failed to place order with IB: {str(e)}")
            raise
    
    def get_order_status(self, order_id: str) -> str:
        """
        Get status of a placed order from IB.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status string
        """
        try:
            order_id_int = int(order_id)
            
            if order_id_int not in self.wrapper.order_status:
                return OrderStatus.PENDING
                
            ib_status = self.wrapper.order_status[order_id_int]["status"]
            
            # Map IB status to our OrderStatus enum
            status_mapping = {
                "Submitted": OrderStatus.PENDING,
                "PreSubmitted": OrderStatus.PENDING,
                "Filled": OrderStatus.FILLED,
                "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
                "Cancelled": OrderStatus.CANCELLED,
                "Rejected": OrderStatus.REJECTED,
                "Inactive": OrderStatus.CANCELLED
            }
            
            return status_mapping.get(ib_status, OrderStatus.PENDING)
            
        except Exception as e:
            self.logger.error(f"Failed to get order status from IB: {str(e)}")
            raise
    
    def get_order_details(self, order_id: str) -> TradeOrder:
        """
        Get detailed information about an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            TradeOrder object with order details
        """
        try:
            order_id_int = int(order_id)
            
            if order_id_int not in self.orders:
                raise ValueError(f"Order {order_id} not found")
                
            order_info = self.orders[order_id_int]
            
            # Get status
            status = self.get_order_status(order_id)
            
            # Get fill price if available
            fill_price = None
            if order_id_int in self.wrapper.order_status:
                status_info = self.wrapper.order_status[order_id_int]
                if status_info["filled"] > 0:
                    fill_price = status_info["avgFillPrice"]
            
            # Create TradeOrder object
            order = TradeOrder(
                order_id=order_id,
                symbol=order_info["symbol"],
                quantity=order_info["quantity"] if order_info["side"] == OrderSide.BUY else -order_info["quantity"],
                order_type=order_info["order_type"],
                side=order_info["side"],
                status=status,
                timestamp=order_info["timestamp"],
                fill_price=fill_price
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to get order details from IB: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        try:
            order_id_int = int(order_id)
            self.client.cancelOrder(order_id_int)
            self.logger.info(f"Cancellation request sent for order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from IB API."""
        if hasattr(self, 'client') and self.client.isConnected():
            self.client.disconnect()
            self.logger.info("Disconnected from IB API")
    
    def __del__(self):
        """Ensure API is disconnected when object is destroyed."""
        try:
            self.disconnect()
        except:
            pass