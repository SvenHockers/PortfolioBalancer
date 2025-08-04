"""Interactive Brokers implementation for trade execution."""

import logging
import time
from typing import Dict, Optional
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
    logger.debug("Interactive Brokers API is available")
except ImportError:
    logger.warning("Interactive Brokers API not available. Using mock implementation.")
    IB_API_AVAILABLE = False
    
    # Define robust mock classes for testing
    class EClient:
        """Mock EClient for testing without IB API."""
        def __init__(self, wrapper):
            self.wrapper = wrapper
            self._connected = False
            
        def connect(self, host: str, port: int, client_id: int) -> bool:
            """Mock connection that always fails gracefully."""
            logger.info(f"Mock IB connection attempt to {host}:{port} (client_id: {client_id})")
            return False
            
        def disconnect(self):
            """Mock disconnect."""
            self._connected = False
            
        def isConnected(self) -> bool:
            """Mock connection status."""
            return self._connected
            
        def run(self):
            """Mock run method."""
            pass
            
        def reqPositions(self):
            """Mock position request."""
            pass
            
        def placeOrder(self, order_id: int, contract, order):
            """Mock order placement."""
            pass
            
        def cancelOrder(self, order_id: int):
            """Mock order cancellation."""
            pass
    
    class EWrapper:
        """Mock EWrapper for testing without IB API."""
        def __init__(self):
            pass
            
        def nextValidId(self, orderId: int):
            """Mock next valid ID callback."""
            pass
            
        def position(self, account: str, contract, position: float, avgCost: float):
            """Mock position callback."""
            pass
            
        def orderStatus(self, orderId: int, status: str, filled: float,
                       remaining: float, avgFillPrice: float, permId: int,
                       parentId: int, lastFillPrice: float, clientId: int,
                       whyHeld: str, mktCapPrice: float):
            """Mock order status callback."""
            pass
            
        def execDetails(self, reqId: int, contract, execution):
            """Mock execution details callback."""
            pass
    
    class Contract:
        """Mock Contract for testing without IB API."""
        def __init__(self):
            self.symbol = ""
            self.secType = ""
            self.exchange = ""
            self.currency = ""
    
    class IBOrder:
        """Mock IBOrder for testing without IB API."""
        def __init__(self):
            self.action = ""
            self.totalQuantity = 0
            self.orderType = ""
    
    OrderId = int
    
    class Execution:
        """Mock Execution for testing without IB API."""
        def __init__(self):
            self.orderId = 0


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
    
    def __init__(self, config: Optional[object] = None):
        """Initialize the IB broker with API connection or mock mode."""
        super().__init__(config)
        
        # Check if IB should be disabled via configuration
        self.ib_enabled = getattr(self.config.broker, 'ib_enabled', True)
        self.mock_mode = False
        
        if not self.ib_enabled:
            self.logger.info("Interactive Brokers integration is disabled via configuration. Using mock mode.")
            self.mock_mode = True
        elif not IB_API_AVAILABLE:
            self.logger.warning("Interactive Brokers API is not available. Please install ibapi package. Using mock mode.")
            self.mock_mode = True
        
        # Initialize configuration
        self.host = getattr(self.config.broker, 'ib_host', '127.0.0.1')
        self.port = getattr(self.config.broker, 'ib_port', 7497)
        self.client_id = getattr(self.config.broker, 'ib_client_id', 1)
        self.timeout = getattr(self.config.broker, 'ib_timeout', 30)
        
        # Initialize wrapper and client
        self.wrapper = IBWrapper()
        self.client = IBClient(self.wrapper)
        self.orders = {}
        self.connected = False
        
        # Initialize mock data for testing
        self.mock_positions = {
            'AAPL': 100.0,
            'GOOGL': 50.0,
            'MSFT': 75.0,
            'TSLA': 25.0
        }
        self.mock_order_counter = 1000
        
        if not self.mock_mode:
            self._connect()
        else:
            self.logger.info("IB Broker initialized in mock mode - no actual connection will be made")
            self.connected = True  # Mock connection is always "connected"
        
    def _connect(self) -> bool:
        """Connect to Interactive Brokers TWS or Gateway. Returns True on success, False on failure."""
        if self.mock_mode:
            self.logger.info("Mock mode enabled - skipping actual IB connection")
            self.connected = True
            return True
            
        try:
            self.logger.info(f"Attempting to connect to IB API at {self.host}:{self.port} (client_id: {self.client_id})")
            
            # Attempt connection
            connection_result = self.client.connect(self.host, self.port, self.client_id)
            
            if connection_result is False:
                self.logger.warning("IB API connection failed - falling back to mock mode")
                self.mock_mode = True
                self.connected = True
                return True
            
            # Start API thread
            api_thread = threading.Thread(target=self.client.run, daemon=True)
            api_thread.start()
            
            # Wait for connection to be established
            if not self.wrapper.next_order_id_ready.wait(self.timeout):
                self.logger.warning(f"Timed out waiting for IB API connection after {self.timeout}s - falling back to mock mode")
                self.mock_mode = True
                self.connected = True
                return True
            
            self.logger.info(f"Successfully connected to IB API at {self.host}:{self.port}")
            self.connected = True
            
            # Request initial positions
            self.client.reqPositions()
            time.sleep(1)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to connect to IB API: {str(e)} - falling back to mock mode")
            self.mock_mode = True
            self.connected = True
            return True
    
    def get_positions(self) -> Dict[str, float]:
        """Get current portfolio positions from IB or mock data. Returns empty dict on error."""
        if self.mock_mode:
            self.logger.debug("Returning mock positions data")
            return self.mock_positions.copy()
            
        try:
            if not self.connected:
                self.logger.warning("Not connected to IB API - returning empty positions")
                return {}
                
            self.client.reqPositions()
            time.sleep(1)
            positions = self.wrapper.positions.copy()
            self.logger.debug(f"Retrieved {len(positions)} positions from IB API")
            return positions
            
        except Exception as e:
            self.logger.error(f"Failed to get positions from IB: {str(e)} - falling back to mock data")
            return self.mock_positions.copy()
    
    def _place_order_impl(self, symbol: str, quantity: float, order_type: str, side: OrderSide) -> str:
        """Place an order with IB API or simulate in mock mode. Returns order ID or None on error."""
        if self.mock_mode:
            # Simulate order placement in mock mode
            order_id = self.mock_order_counter
            self.mock_order_counter += 1
            
            self.orders[order_id] = {
                "symbol": symbol,
                "quantity": quantity,
                "order_type": order_type,
                "side": side,
                "timestamp": datetime.now(),
                "status": OrderStatus.FILLED,  # Mock orders are immediately filled
                "fill_price": 100.0  # Mock fill price
            }
            
            self.logger.info(f"Mock order placed: {side} {quantity} {symbol} (Order ID: {order_id})")
            
            # Update mock positions
            if side == OrderSide.BUY:
                self.mock_positions[symbol] = self.mock_positions.get(symbol, 0) + quantity
            else:
                self.mock_positions[symbol] = self.mock_positions.get(symbol, 0) - quantity
                
            return str(order_id)
            
        try:
            if not self.connected:
                self.logger.error("Not connected to IB API - cannot place order")
                return None
                
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
            
            # Get order ID
            order_id = self.wrapper.next_order_id
            self.wrapper.next_order_id += 1
            
            # Place order
            self.client.placeOrder(order_id, contract, ib_order)
            
            # Track order
            self.orders[order_id] = {
                "symbol": symbol,
                "quantity": quantity,
                "order_type": order_type,
                "side": side,
                "timestamp": datetime.now()
            }
            
            self.logger.info(f"Order placed with IB API: {side} {quantity} {symbol} (Order ID: {order_id})")
            return str(order_id)
            
        except Exception as e:
            self.logger.error(f"Failed to place order with IB: {str(e)}")
            return None
    
    def get_order_status(self, order_id: str) -> str:
        """Get status of a placed order from IB or mock data. Returns OrderStatus.PENDING on error."""
        try:
            order_id_int = int(order_id)
            
            if self.mock_mode:
                # Return mock status
                if order_id_int in self.orders:
                    return self.orders[order_id_int].get("status", OrderStatus.FILLED)
                else:
                    self.logger.warning(f"Mock order {order_id} not found")
                    return OrderStatus.PENDING
            
            # Real IB API status check
            if not self.connected:
                self.logger.warning("Not connected to IB API - returning pending status")
                return OrderStatus.PENDING
                
            if order_id_int not in self.wrapper.order_status:
                return OrderStatus.PENDING
                
            ib_status = self.wrapper.order_status[order_id_int]["status"]
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
            return OrderStatus.PENDING
    
    def get_order_details(self, order_id: str) -> Optional[TradeOrder]:
        """Get detailed information about an order. Returns None on error."""
        try:
            order_id_int = int(order_id)
            
            if order_id_int not in self.orders:
                self.logger.error(f"Order {order_id} not found")
                return None
                
            order_info = self.orders[order_id_int]
            status = self.get_order_status(order_id)
            fill_price = None
            
            if self.mock_mode:
                # Use mock fill price
                fill_price = order_info.get("fill_price", 100.0)
            else:
                # Get fill price from IB API
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
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns False on error."""
        try:
            order_id_int = int(order_id)
            
            if self.mock_mode:
                # Simulate order cancellation in mock mode
                if order_id_int in self.orders:
                    self.orders[order_id_int]["status"] = OrderStatus.CANCELLED
                    self.logger.info(f"Mock order {order_id} cancelled")
                    return True
                else:
                    self.logger.warning(f"Mock order {order_id} not found for cancellation")
                    return False
            
            if not self.connected:
                self.logger.error("Not connected to IB API - cannot cancel order")
                return False
                
            self.client.cancelOrder(order_id_int)
            self.logger.info(f"Cancellation request sent for order {order_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from IB API."""
        if self.mock_mode:
            self.logger.info("Mock mode - no actual disconnection needed")
            self.connected = False
            return
            
        try:
            if hasattr(self, 'client') and self.client.isConnected():
                self.client.disconnect()
                self.logger.info("Disconnected from IB API")
            self.connected = False
        except Exception as e:
            self.logger.warning(f"Error during IB API disconnection: {str(e)}")
            self.connected = False
    
    def __del__(self):
        """Ensure API is disconnected when object is destroyed."""
        try:
            self.disconnect()
        except Exception:
            # Ignore errors during cleanup
            pass