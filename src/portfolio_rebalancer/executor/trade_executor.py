"""Trade executor service for coordinating rebalancing and broker operations."""

import logging
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from ..common.models import TargetAllocation, CurrentHolding, TradeOrder, OrderStatus, OrderType, OrderSide
from ..common.config import get_config
from ..common.logging import correlation_context, log_execution_time
from ..common.metrics import (
    timed, record_trade_execution, record_portfolio_drift, 
    record_trade_execution_latency, measure_latency
)
from .rebalance_calculator import RebalanceCalculator
from .brokers.broker_factory import BrokerFactory


class TradeExecutor:
    """
    Trade executor service that coordinates rebalancing and broker operations.
    
    This class is responsible for:
    1. Checking if rebalancing is needed based on current holdings and target allocation
    2. Calculating required trades to achieve target allocation
    3. Executing trades through the configured broker
    4. Tracking order status and logging execution details
    """
    
    def __init__(self):
        """Initialize the trade executor with configuration."""
        self.config = get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize rebalance calculator
        self.calculator = RebalanceCalculator()
        
        # Initialize broker interface
        self.broker = BrokerFactory.create_broker()
        
        # Track orders
        self.orders = {}
        
        self.logger.info("TradeExecutor initialized")
    
    @log_execution_time(logging.getLogger(__name__), "Portfolio rebalancing check")
    @timed("executor", "check_rebalancing")
    def check_rebalancing_needed(self, target_allocation: TargetAllocation) -> Tuple[bool, Dict[str, float]]:
        """
        Check if portfolio rebalancing is needed based on current holdings and target allocation.
        
        Args:
            target_allocation: Target portfolio allocation
            
        Returns:
            Tuple of (rebalancing_needed, drift_values)
        """
        with correlation_context():
            self.logger.info("Checking if rebalancing is needed")
            
            # Get current positions from broker
            try:
                positions = self.broker.get_positions()
                self.logger.info(f"Retrieved {len(positions)} positions from broker")
            except Exception as e:
                self.logger.error(f"Failed to get positions from broker: {str(e)}", exc_info=True)
                return False, {}
            
            # Convert positions to CurrentHolding objects
            current_holdings = self._positions_to_holdings(positions)
            
            # Check if rebalancing is needed
            rebalancing_needed, drift = self.calculator.is_rebalancing_needed(
                current_holdings, target_allocation
            )
            
            # Record portfolio drift metrics
            record_portfolio_drift(drift)
            
            # Log result
            if rebalancing_needed:
                self.logger.info("Rebalancing is needed")
                for symbol, drift_value in drift.items():
                    self.logger.info(
                        f"Drift for {symbol}: {drift_value:.2%}",
                        extra={"symbol": symbol, "drift": drift_value}
                    )
            else:
                self.logger.info("No rebalancing needed")
            
            return rebalancing_needed, drift
    
    @log_execution_time(logging.getLogger(__name__), "Portfolio rebalancing execution")
    @timed("executor", "execute_rebalancing")
    def execute_rebalancing(self, 
                           target_allocation: TargetAllocation,
                           cash_to_invest: float = 0.0) -> List[TradeOrder]:
        """
        Execute portfolio rebalancing based on target allocation.
        
        Args:
            target_allocation: Target portfolio allocation
            cash_to_invest: Additional cash to invest (positive) or withdraw (negative)
            
        Returns:
            List of executed trade orders
        """
        with correlation_context():
            self.logger.info(
                f"Executing rebalancing with {cash_to_invest:.2f} cash adjustment",
                extra={"cash_adjustment": cash_to_invest}
            )
            
            # Get current positions from broker
            try:
                positions = self.broker.get_positions()
                self.logger.info(f"Retrieved {len(positions)} positions from broker")
            except Exception as e:
                self.logger.error(f"Failed to get positions from broker: {str(e)}", exc_info=True)
                return []
            
            # Convert positions to CurrentHolding objects
            current_holdings = self._positions_to_holdings(positions)
            
            # Calculate required trades
            trades = self.calculator.calculate_trades(
                current_holdings, target_allocation, cash_to_invest
            )
            
            self.logger.info(f"Calculated {len(trades)} trades for rebalancing")
            
            # Execute trades
            executed_orders = []
            success_count = 0
            failure_count = 0
            broker_name = self.config.executor.broker_type
            
            for symbol, quantity in trades:
                try:
                    # Skip zero or near-zero quantity trades
                    if abs(quantity) < 0.0001:
                        self.logger.debug(f"Skipping negligible trade for {symbol}: {quantity}")
                        continue
                    
                    # Get order type from config
                    order_type = self.config.executor.order_type
                    
                    # Place order
                    self.logger.info(
                        f"Placing {order_type} order for {quantity} shares of {symbol}",
                        extra={"symbol": symbol, "quantity": quantity, "order_type": order_type}
                    )
                    
                    # Use context manager to measure trade execution latency
                    with measure_latency(broker_name, order_type):
                        order_id = self.broker.place_order(symbol, quantity, order_type)
                    
                    # Track order
                    self.orders[order_id] = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "order_type": order_type,
                        "timestamp": datetime.now()
                    }
                    
                    # Wait for order status
                    order = self._wait_for_order_status(order_id)
                    if order:
                        executed_orders.append(order)
                        
                        # Record trade execution metrics
                        status = "success" if order.status == OrderStatus.FILLED else "partial" if order.status == OrderStatus.PARTIALLY_FILLED else "pending"
                        record_trade_execution(broker_name, order_type, status)
                        success_count += 1
                    
                except Exception as e:
                    self.logger.error(
                        f"Failed to execute trade for {symbol}: {str(e)}",
                        extra={"symbol": symbol, "quantity": quantity, "error": str(e)},
                        exc_info=True
                    )
                    # Record trade execution failure
                    record_trade_execution(broker_name, order_type, "failure")
                    failure_count += 1
            
            self.logger.info(f"Completed rebalancing with {len(executed_orders)} executed orders")
            
            # Record overall success rate
            total_trades = success_count + failure_count
            if total_trades > 0:
                success_rate = success_count / total_trades
                self.logger.info(f"Trade execution success rate: {success_rate:.2%}")
            
            return executed_orders
    
    def get_order_status(self, order_id: str) -> Optional[TradeOrder]:
        """
        Get status of a placed order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            TradeOrder object with order details or None if not found
        """
        try:
            # Check if we have broker-specific get_order_details method
            if hasattr(self.broker, 'get_order_details') and callable(getattr(self.broker, 'get_order_details')):
                return self.broker.get_order_details(order_id)
            
            # Otherwise, construct TradeOrder from available information
            if order_id not in self.orders:
                self.logger.warning(f"Order {order_id} not found in tracking")
                return None
            
            order_info = self.orders[order_id]
            status = self.broker.get_order_status(order_id)
            
            # Determine side based on quantity
            side = OrderSide.BUY if order_info["quantity"] > 0 else OrderSide.SELL
            
            # Determine fill price based on status
            fill_price = None
            if status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                # Use a placeholder price for testing
                fill_price = 100.0
            
            # Create TradeOrder object
            order = TradeOrder(
                order_id=order_id,
                symbol=order_info["symbol"],
                quantity=order_info["quantity"],
                order_type=order_info["order_type"],
                side=side,
                status=status,
                timestamp=order_info["timestamp"],
                fill_price=fill_price
            )
            
            return order
            
        except Exception as e:
            self.logger.error(
                f"Failed to get order status for {order_id}: {str(e)}",
                extra={"order_id": order_id, "error": str(e)},
                exc_info=True
            )
            return None
    
    def _positions_to_holdings(self, positions: Dict[str, float]) -> Dict[str, CurrentHolding]:
        """
        Convert broker positions to CurrentHolding objects.
        
        Args:
            positions: Dictionary mapping ticker symbols to position quantities
            
        Returns:
            Dictionary mapping ticker symbols to CurrentHolding objects
        """
        # This is a simplified implementation that assumes we have price data
        # In a real implementation, we would need to fetch current prices
        
        holdings = {}
        total_value = 0.0
        
        # First pass: calculate market values (simplified)
        for symbol, quantity in positions.items():
            # In a real implementation, we would fetch current prices
            # For now, use a placeholder price of $100 for testing
            price = 100.0
            market_value = quantity * price
            
            holdings[symbol] = CurrentHolding(
                symbol=symbol,
                quantity=quantity,
                market_value=market_value,
                weight=0.0  # Will be calculated in second pass
            )
            
            total_value += market_value
        
        # Second pass: calculate weights
        if total_value > 0:
            for symbol, holding in holdings.items():
                weight = holding.market_value / total_value
                holdings[symbol] = CurrentHolding(
                    symbol=symbol,
                    quantity=holding.quantity,
                    market_value=holding.market_value,
                    weight=weight
                )
        
        return holdings
    
    def _wait_for_order_status(self, order_id: str, timeout: int = 30) -> Optional[TradeOrder]:
        """
        Wait for order status to be updated.
        
        Args:
            order_id: Order ID to check
            timeout: Maximum time to wait in seconds
            
        Returns:
            TradeOrder object with order details or None if timed out
        """
        start_time = time.time()
        poll_interval = 1.0  # 1 second
        
        while time.time() - start_time < timeout:
            try:
                order = self.get_order_status(order_id)
                
                if not order:
                    self.logger.warning(f"Order {order_id} not found")
                    return None
                
                # If order is no longer pending, return it
                if order.status != OrderStatus.PENDING:
                    self.logger.info(
                        f"Order {order_id} status: {order.status}",
                        extra={"order_id": order_id, "status": order.status}
                    )
                    return order
                
                # Wait before checking again
                time.sleep(poll_interval)
                
            except Exception as e:
                self.logger.error(
                    f"Error checking order status for {order_id}: {str(e)}",
                    extra={"order_id": order_id, "error": str(e)},
                    exc_info=True
                )
                return None
        
        self.logger.warning(
            f"Timed out waiting for order {order_id} status",
            extra={"order_id": order_id, "timeout": timeout}
        )
        return None