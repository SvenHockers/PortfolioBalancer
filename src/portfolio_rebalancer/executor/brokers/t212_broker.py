"""Trading212 Broker implementation for trade execution"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional
from portfolio_rebalancer.common.models import OrderType, OrderSide, OrderStatus, TradeOrder
from .base_broker import BaseBroker
from .trading212_client import Trading212

logger = logging.getLogger(__name__)

class T212Broker(BaseBroker):
    """Trading212 implementation for trade executions using Trading212 client."""

    def __init__(self, config: object = None):
        super().__init__(config)
        self.config: Any
        self.api_key = getattr(self.config.broker, 't212_api_key', None)
        self.demo = getattr(self.config.broker, 't212_demo', True)
        if not self.api_key:
            raise ValueError("Trading212 API key must be provided in t212_api_key")
        self.client = Trading212(self.api_key, demo=self.demo)

    def get_positions(self) -> Dict[str, float]:
        try:
            positions = self.client.portfolio()
            return {pos["ticker"]: float(pos["quantity"]) for pos in positions}
        except Exception as e:
            self.logger.error(f"Failed to get positions from Trading212: {str(e)}")
            return {}

    def _place_order_impl(self, symbol: str, quantity: float, order_type: str, side: OrderSide, limit_price: Optional[float] = None) -> str:
        try:
            qty = abs(quantity)
            if side == OrderSide.SELL:
                qty = -qty
            if order_type == OrderType.MARKET:
                order = self.client.equity_order_place_market(symbol, int(abs(qty)))
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise ValueError("Limit orders require a limit_price argument.")
                order = self.client.equity_order_place_limit(symbol, int(abs(qty)), float(limit_price), "DAY")
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            return str(order["id"])
        except Exception as e:
            self.logger.error(f"Error in _place_order_impl: {str(e)}")
            raise

    def get_order_status(self, order_id: str) -> str:
        try:
            order_data = self.client.equity_order(int(order_id))
            t212_status = order_data["status"].upper()
            status_mapping = {
                "FILLED": OrderStatus.FILLED,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "LOCAL": OrderStatus.PENDING,
                "PENDING": OrderStatus.PENDING,
                "SUBMITTED": OrderStatus.PENDING,
                "INACTIVE": OrderStatus.CANCELLED
            }
            return status_mapping.get(t212_status, OrderStatus.PENDING)
        except Exception as e:
            self.logger.error(f"Failed to get order status from Trading212: {str(e)}")
            return OrderStatus.PENDING

    def get_order_details(self, order_id: str) -> Optional[TradeOrder]:
        try:
            order_data = self.client.equity_order(int(order_id))
            status_mapping = {
                "FILLED": OrderStatus.FILLED,
                "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED": OrderStatus.REJECTED,
                "LOCAL": OrderStatus.PENDING,
                "PENDING": OrderStatus.PENDING,
                "SUBMITTED": OrderStatus.PENDING,
                "INACTIVE": OrderStatus.CANCELLED
            }
            quantity = float(order_data["quantity"])
            side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
            order = TradeOrder(
                order_id=str(order_data["id"]),
                symbol=order_data["ticker"],
                quantity=quantity if side == OrderSide.BUY else -abs(quantity),
                order_type=OrderType(order_data["type"].lower()),
                side=side,
                status=status_mapping.get(order_data["status"].upper(), OrderStatus.PENDING),
                timestamp=datetime.fromisoformat(order_data["creationTime"].replace("Z", "+00:00")),
                fill_price=float(order_data["filledValue"]/order_data["filledQuantity"]) if order_data["filledQuantity"] else None
            )
            return order
        except Exception as e:
            self.logger.error(f"Failed to get order details from Trading212: {str(e)}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.equity_order_cancel(int(order_id))
            self.logger.info(f"Successfully cancelled order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False