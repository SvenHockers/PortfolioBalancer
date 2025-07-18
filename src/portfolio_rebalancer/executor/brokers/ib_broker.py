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
    
    def __init__(self, config: object = None):
        """Initialize the IB broker with API connection."""
        super().__init__(config)
        if not IB_API_AVAILABLE:
            self.logger.error("Interactive Brokers API is not available. Please install ibapi package.")
            return
        self.host = self.config.broker.ib_host
        self.port = self.config.broker.ib_port
        self.client_id = self.config.broker.ib_client_id
        self.wrapper = IBWrapper()
        self.client = IBClient(self.wrapper)
        self._connect()
        self.orders = {}
        
    def _connect(self) -> None:
        """Connect to Interactive Brokers TWS or Gateway. Returns None on error."""
        try:
            self.client.connect(self.host, self.port, self.client_id)
            api_thread = threading.Thread(target=self.client.run)
            api_thread.daemon = True
            api_thread.start()
            if not self.wrapper.next_order_id_ready.wait(10):
                self.logger.error("Timed out waiting for IB API connection")
                return None
            self.logger.info(f"Successfully connected to IB API at {self.host}:{self.port}")
            self.client.reqPositions()
            time.sleep(1)
        except Exception as e:
            self.logger.error(f"Failed to connect to IB API: {str(e)}")
            return None
    
    def get_positions(self) -> Dict[str, float]:
        """Get current portfolio positions from IB. Returns empty dict on error."""
        try:
            self.client.reqPositions()
            time.sleep(1)
            return self.wrapper.positions.copy()
        except Exception as e:
            self.logger.error(f"Failed to get positions from IB: {str(e)}")
            return {}
    
    def _place_order_impl(self, symbol: str, quantity: float, order_type: str, side: OrderSide) -> str:
        """Place an order with IB API. Returns None on error."""
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            ib_order = IBOrder()
            ib_order.action = "BUY" if side == OrderSide.BUY else "SELL"
            ib_order.totalQuantity = quantity
            ib_order.orderType = "MKT" if order_type == OrderType.MARKET else "LMT"
            order_id = self.wrapper.next_order_id
            self.wrapper.next_order_id += 1
            self.client.placeOrder(order_id, contract, ib_order)
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
            return None
    
    def get_order_status(self, order_id: str) -> str:
        """Get status of a placed order from IB. Returns OrderStatus.PENDING on error."""
        try:
            order_id_int = int(order_id)
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
    
    def get_order_details(self, order_id: str) -> TradeOrder:
        """Get detailed information about an order. Returns None on error."""
        try:
            order_id_int = int(order_id)
            if order_id_int not in self.orders:
                self.logger.error(f"Order {order_id} not found")
                return None
            order_info = self.orders[order_id_int]
            status = self.get_order_status(order_id)
            fill_price = None
            if order_id_int in self.wrapper.order_status:
                status_info = self.wrapper.order_status[order_id_int]
                if status_info["filled"] > 0:
                    fill_price = status_info["avgFillPrice"]
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