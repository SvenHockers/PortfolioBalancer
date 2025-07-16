"""Unit tests for data models."""

import pytest
from datetime import date, datetime
from decimal import Decimal
from pydantic import ValidationError

from src.portfolio_rebalancer.common.models import (
    PriceData, TargetAllocation, CurrentHolding, TradeOrder,
    OrderType, OrderSide, OrderStatus
)


class TestPriceData:
    """Test cases for PriceData model."""
    
    def test_valid_price_data(self):
        """Test creating valid price data."""
        price_data = PriceData(
            symbol="AAPL",
            date=date(2024, 1, 15),
            adjusted_close=150.25,
            volume=1000000
        )
        
        assert price_data.symbol == "AAPL"
        assert price_data.date == date(2024, 1, 15)
        assert price_data.adjusted_close == 150.25
        assert price_data.volume == 1000000
    
    def test_symbol_normalization(self):
        """Test symbol is normalized to uppercase."""
        price_data = PriceData(
            symbol="  aapl  ",
            date=date(2024, 1, 15),
            adjusted_close=150.25,
            volume=1000000
        )
        
        assert price_data.symbol == "AAPL"
    
    def test_empty_symbol_validation(self):
        """Test empty symbol raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PriceData(
                symbol="",
                date=date(2024, 1, 15),
                adjusted_close=150.25,
                volume=1000000
            )
        
        assert "Symbol cannot be empty" in str(exc_info.value)
    
    def test_negative_price_validation(self):
        """Test negative price raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PriceData(
                symbol="AAPL",
                date=date(2024, 1, 15),
                adjusted_close=-10.0,
                volume=1000000
            )
        
        assert "Price must be positive" in str(exc_info.value)
    
    def test_zero_price_validation(self):
        """Test zero price raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PriceData(
                symbol="AAPL",
                date=date(2024, 1, 15),
                adjusted_close=0.0,
                volume=1000000
            )
        
        assert "ensure this value is greater than 0" in str(exc_info.value)
    
    def test_extremely_high_price_validation(self):
        """Test extremely high price raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PriceData(
                symbol="AAPL",
                date=date(2024, 1, 15),
                adjusted_close=2000000.0,
                volume=1000000
            )
        
        assert "Price appears unreasonably high" in str(exc_info.value)
    
    def test_negative_volume_validation(self):
        """Test negative volume raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PriceData(
                symbol="AAPL",
                date=date(2024, 1, 15),
                adjusted_close=150.25,
                volume=-1000
            )
        
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)
    
    def test_price_rounding(self):
        """Test price is rounded to 4 decimal places."""
        price_data = PriceData(
            symbol="AAPL",
            date=date(2024, 1, 15),
            adjusted_close=150.123456789,
            volume=1000000
        )
        
        assert price_data.adjusted_close == 150.1235


class TestTargetAllocation:
    """Test cases for TargetAllocation model."""
    
    def test_valid_target_allocation(self):
        """Test creating valid target allocation."""
        allocation = TargetAllocation(
            timestamp=datetime(2024, 1, 15, 16, 0, 0),
            allocations={"AAPL": 0.6, "GOOGL": 0.4},
            expected_return=0.12,
            expected_volatility=0.15,
            sharpe_ratio=0.8
        )
        
        assert allocation.allocations == {"AAPL": 0.6, "GOOGL": 0.4}
        assert allocation.expected_return == 0.12
        assert allocation.expected_volatility == 0.15
        assert allocation.sharpe_ratio == 0.8
    
    def test_empty_allocations_validation(self):
        """Test empty allocations raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TargetAllocation(
                timestamp=datetime(2024, 1, 15, 16, 0, 0),
                allocations={},
                expected_return=0.12,
                expected_volatility=0.15,
                sharpe_ratio=0.8
            )
        
        assert "Allocations cannot be empty" in str(exc_info.value)
    
    def test_negative_weight_validation(self):
        """Test negative weight raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TargetAllocation(
                timestamp=datetime(2024, 1, 15, 16, 0, 0),
                allocations={"AAPL": -0.1, "GOOGL": 1.1},
                expected_return=0.12,
                expected_volatility=0.15,
                sharpe_ratio=0.8
            )
        
        assert "Weight for AAPL cannot be negative" in str(exc_info.value)
    
    def test_weight_exceeds_one_validation(self):
        """Test weight exceeding 1.0 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TargetAllocation(
                timestamp=datetime(2024, 1, 15, 16, 0, 0),
                allocations={"AAPL": 1.5, "GOOGL": 0.0},
                expected_return=0.12,
                expected_volatility=0.15,
                sharpe_ratio=0.8
            )
        
        assert "Weight for AAPL cannot exceed 100%" in str(exc_info.value)
    
    def test_weights_sum_validation(self):
        """Test weights must sum to approximately 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            TargetAllocation(
                timestamp=datetime(2024, 1, 15, 16, 0, 0),
                allocations={"AAPL": 0.3, "GOOGL": 0.3},  # Sum = 0.6
                expected_return=0.12,
                expected_volatility=0.15,
                sharpe_ratio=0.8
            )
        
        assert "Allocation weights must sum to 1.0" in str(exc_info.value)
    
    def test_weights_sum_tolerance(self):
        """Test weights sum tolerance allows small rounding errors."""
        # This should pass (sum = 1.005, within 0.01 tolerance)
        allocation = TargetAllocation(
            timestamp=datetime(2024, 1, 15, 16, 0, 0),
            allocations={"AAPL": 0.6025, "GOOGL": 0.4025},
            expected_return=0.12,
            expected_volatility=0.15,
            sharpe_ratio=0.8
        )
        
        assert allocation.allocations == {"AAPL": 0.6025, "GOOGL": 0.4025}
    
    def test_negative_volatility_validation(self):
        """Test negative volatility raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TargetAllocation(
                timestamp=datetime(2024, 1, 15, 16, 0, 0),
                allocations={"AAPL": 0.6, "GOOGL": 0.4},
                expected_return=0.12,
                expected_volatility=-0.15,
                sharpe_ratio=0.8
            )
        
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)
    
    def test_extremely_high_volatility_validation(self):
        """Test extremely high volatility raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TargetAllocation(
                timestamp=datetime(2024, 1, 15, 16, 0, 0),
                allocations={"AAPL": 0.6, "GOOGL": 0.4},
                expected_return=0.12,
                expected_volatility=3.0,
                sharpe_ratio=0.8
            )
        
        assert "Volatility appears unreasonably high" in str(exc_info.value)


class TestCurrentHolding:
    """Test cases for CurrentHolding model."""
    
    def test_valid_current_holding(self):
        """Test creating valid current holding."""
        holding = CurrentHolding(
            symbol="AAPL",
            quantity=100.0,
            market_value=15000.0,
            weight=0.6
        )
        
        assert holding.symbol == "AAPL"
        assert holding.quantity == 100.0
        assert holding.market_value == 15000.0
        assert holding.weight == 0.6
    
    def test_symbol_normalization(self):
        """Test symbol is normalized to uppercase."""
        holding = CurrentHolding(
            symbol="  aapl  ",
            quantity=100.0,
            market_value=15000.0,
            weight=0.6
        )
        
        assert holding.symbol == "AAPL"
    
    def test_zero_quantity_zero_value_consistency(self):
        """Test zero quantity with zero market value is valid."""
        holding = CurrentHolding(
            symbol="AAPL",
            quantity=0.0,
            market_value=0.0,
            weight=0.0
        )
        
        assert holding.quantity == 0.0
        assert holding.market_value == 0.0
    
    def test_zero_quantity_nonzero_value_validation(self):
        """Test zero quantity with non-zero market value raises error."""
        with pytest.raises(ValidationError) as exc_info:
            CurrentHolding(
                symbol="AAPL",
                quantity=0.0,
                market_value=1000.0,
                weight=0.1
            )
        
        assert "Market value should be zero when quantity is zero" in str(exc_info.value)
    
    def test_nonzero_quantity_zero_value_validation(self):
        """Test non-zero quantity with zero market value raises error."""
        with pytest.raises(ValidationError) as exc_info:
            CurrentHolding(
                symbol="AAPL",
                quantity=100.0,
                market_value=0.0,
                weight=0.1
            )
        
        assert "Market value must be positive when quantity is non-zero" in str(exc_info.value)
    
    def test_negative_market_value_validation(self):
        """Test negative market value raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CurrentHolding(
                symbol="AAPL",
                quantity=100.0,
                market_value=-1000.0,
                weight=0.1
            )
        
        assert "ensure this value is greater than or equal to 0" in str(exc_info.value)
    
    def test_weight_exceeds_one_validation(self):
        """Test weight exceeding 1.0 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CurrentHolding(
                symbol="AAPL",
                quantity=100.0,
                market_value=15000.0,
                weight=1.5
            )
        
        assert "ensure this value is less than or equal to 1" in str(exc_info.value)
    
    def test_extremely_large_quantity_validation(self):
        """Test extremely large quantity raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CurrentHolding(
                symbol="AAPL",
                quantity=2000000.0,
                market_value=15000.0,
                weight=0.6
            )
        
        assert "Quantity appears unreasonably large" in str(exc_info.value)


class TestTradeOrder:
    """Test cases for TradeOrder model."""
    
    def test_valid_buy_order(self):
        """Test creating valid buy order."""
        order = TradeOrder(
            order_id="ORD123",
            symbol="AAPL",
            quantity=100.0,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            status=OrderStatus.PENDING,
            timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        assert order.order_id == "ORD123"
        assert order.symbol == "AAPL"
        assert order.quantity == 100.0
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING
    
    def test_valid_sell_order(self):
        """Test creating valid sell order."""
        order = TradeOrder(
            order_id="ORD124",
            symbol="GOOGL",
            quantity=-50.0,
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            status=OrderStatus.FILLED,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            fill_price=2500.0
        )
        
        assert order.quantity == -50.0
        assert order.side == OrderSide.SELL
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 2500.0
    
    def test_empty_order_id_validation(self):
        """Test empty order ID raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                order_id="",
                symbol="AAPL",
                quantity=100.0,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                status=OrderStatus.PENDING,
                timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        
        assert "Order ID cannot be empty" in str(exc_info.value)
    
    def test_zero_quantity_validation(self):
        """Test zero quantity raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                order_id="ORD123",
                symbol="AAPL",
                quantity=0.0,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                status=OrderStatus.PENDING,
                timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        
        assert "Order quantity cannot be zero" in str(exc_info.value)
    
    def test_buy_negative_quantity_validation(self):
        """Test buy order with negative quantity raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                order_id="ORD123",
                symbol="AAPL",
                quantity=-100.0,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                status=OrderStatus.PENDING,
                timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        
        assert "Buy orders must have positive quantity" in str(exc_info.value)
    
    def test_sell_positive_quantity_validation(self):
        """Test sell order with positive quantity raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                order_id="ORD124",
                symbol="AAPL",
                quantity=100.0,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,
                status=OrderStatus.PENDING,
                timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        
        assert "Sell orders must have negative quantity" in str(exc_info.value)
    
    def test_filled_order_missing_fill_price_validation(self):
        """Test filled order without fill price raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                order_id="ORD123",
                symbol="AAPL",
                quantity=100.0,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                status=OrderStatus.FILLED,
                timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        
        assert "Fill price must be provided for filled orders" in str(exc_info.value)
    
    def test_pending_order_with_fill_price_validation(self):
        """Test pending order with fill price raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                order_id="ORD123",
                symbol="AAPL",
                quantity=100.0,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                status=OrderStatus.PENDING,
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                fill_price=150.0
            )
        
        assert "Fill price should not be provided for non-filled orders" in str(exc_info.value)
    
    def test_extremely_large_quantity_validation(self):
        """Test extremely large quantity raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TradeOrder(
                order_id="ORD123",
                symbol="AAPL",
                quantity=2000000.0,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                status=OrderStatus.PENDING,
                timestamp=datetime(2024, 1, 15, 10, 30, 0)
            )
        
        assert "Order quantity appears unreasonably large" in str(exc_info.value)