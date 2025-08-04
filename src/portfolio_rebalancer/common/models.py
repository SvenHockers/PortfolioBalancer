"""Pydantic data models for the portfolio rebalancer system."""

from datetime import date as Date, datetime as DateTime
from typing import Dict, Optional, List
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


class OrderType(str, Enum):
    """Enumeration for order types."""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(str, Enum):
    """Enumeration for order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Enumeration for order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PriceData(BaseModel):
    """Model for individual price data points."""
    
    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)
    
    symbol: str = Field(..., description="Ticker symbol")
    date: Date = Field(..., description="Date of the price data")
    adjusted_close: float = Field(..., gt=0, description="Adjusted closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate ticker symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator('adjusted_close')
    @classmethod
    def validate_price(cls, v):
        """Validate price is reasonable."""
        if v <= 0:
            raise ValueError("Price must be positive")
        if v > 1000000:  # Sanity check for extremely high prices
            raise ValueError("Price appears unreasonably high")
        return round(v, 4)  # Round to 4 decimal places


class TargetAllocation(BaseModel):
    """Model for target portfolio allocation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    timestamp: DateTime = Field(..., description="When the allocation was calculated")
    allocations: Dict[str, float] = Field(..., description="Symbol to weight mapping")
    expected_return: float = Field(..., description="Expected annual return")
    expected_volatility: float = Field(..., ge=0, description="Expected annual volatility")
    sharpe_ratio: float = Field(..., description="Expected Sharpe ratio")
    
    @field_validator('allocations')
    @classmethod
    def validate_allocations(cls, v):
        """Validate allocation weights."""
        if not v:
            raise ValueError("Allocations cannot be empty")
        
        # Check all weights are non-negative
        for symbol, weight in v.items():
            if weight < 0:
                raise ValueError(f"Weight for {symbol} cannot be negative")
            if weight > 1:
                raise ValueError(f"Weight for {symbol} cannot exceed 100%")
        
        # Check weights sum to approximately 1 (allowing for small rounding errors)
        total_weight = sum(v.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Allocation weights must sum to 1.0, got {total_weight}")
        
        return v
    
    @field_validator('expected_volatility')
    @classmethod
    def validate_volatility(cls, v):
        """Validate volatility is reasonable."""
        if v < 0:
            raise ValueError("Volatility cannot be negative")
        if v > 2.0:  # 200% volatility seems unreasonable
            raise ValueError("Volatility appears unreasonably high")
        return v


class CurrentHolding(BaseModel):
    """Model for current portfolio holdings."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    symbol: str = Field(..., description="Ticker symbol")
    quantity: float = Field(..., description="Number of shares held")
    market_value: float = Field(..., ge=0, description="Current market value")
    weight: float = Field(..., ge=0, le=1, description="Weight in portfolio")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate ticker symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        """Validate quantity is reasonable."""
        if abs(v) > 1000000:  # Sanity check for extremely large positions
            raise ValueError("Quantity appears unreasonably large")
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate consistency between quantity, market_value, and weight."""
        quantity = self.quantity
        market_value = self.market_value
        
        # If quantity is zero, market value should also be zero
        if quantity == 0 and market_value != 0:
            raise ValueError("Market value should be zero when quantity is zero")
        
        # If quantity is non-zero, market value should be positive
        if quantity != 0 and market_value <= 0:
            raise ValueError("Market value must be positive when quantity is non-zero")
        
        return self


class TradeOrder(BaseModel):
    """Model for trade orders."""
    
    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)
    
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Ticker symbol")
    quantity: float = Field(..., description="Order quantity (positive for buy, negative for sell)")
    order_type: OrderType = Field(..., description="Order type")
    side: OrderSide = Field(..., description="Order side")
    status: OrderStatus = Field(..., description="Order status")
    timestamp: DateTime = Field(..., description="Order creation timestamp")
    fill_price: Optional[float] = Field(None, gt=0, description="Fill price if executed")
    
    @field_validator('order_id')
    @classmethod
    def validate_order_id(cls, v):
        """Validate order ID format."""
        if not v or not v.strip():
            raise ValueError("Order ID cannot be empty")
        return v.strip()
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate ticker symbol format."""
        if not v or not v.strip():
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        """Validate order quantity."""
        if v == 0:
            raise ValueError("Order quantity cannot be zero")
        if abs(v) > 1000000:  # Sanity check
            raise ValueError("Order quantity appears unreasonably large")
        return v
    
    @model_validator(mode='after')
    def validate_side_quantity_consistency(self):
        """Validate consistency between side and quantity."""
        side = self.side
        quantity = self.quantity
        
        if side == OrderSide.BUY and quantity < 0:
            raise ValueError("Buy orders must have positive quantity")
        if side == OrderSide.SELL and quantity > 0:
            raise ValueError("Sell orders must have negative quantity")
        
        return self
    
    @model_validator(mode='after')
    def validate_fill_price_status(self):
        """Validate fill price is provided when order is filled."""
        status = self.status
        fill_price = self.fill_price
        
        if status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED] and fill_price is None:
            raise ValueError("Fill price must be provided for filled orders")
        
        if status in [OrderStatus.PENDING, OrderStatus.CANCELLED, OrderStatus.REJECTED] and fill_price is not None:
            raise ValueError("Fill price should not be provided for non-filled orders")
        
        return self


class Portfolio(BaseModel):
    """Model for portfolio definition."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    id: str = Field(..., description="Portfolio identifier")
    tickers: List[str] = Field(..., min_length=1, description="List of ticker symbols")
    weights: List[float] = Field(..., min_length=1, description="Portfolio weights")
    target_allocation: Dict[str, float] = Field(..., description="Target allocation mapping")
    
    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Validate ticker symbols."""
        if not v:
            raise ValueError("At least one ticker must be provided")
        
        validated_tickers = []
        for ticker in v:
            if not ticker or not ticker.strip():
                raise ValueError("Ticker symbols cannot be empty")
            validated_tickers.append(ticker.strip().upper())
        
        return validated_tickers
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Validate portfolio weights sum to 1."""
        if not v:
            raise ValueError("Portfolio weights cannot be empty")
        
        if any(w < 0 for w in v):
            raise ValueError("Portfolio weights cannot be negative")
        
        total_weight = sum(v)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight}")
        
        return v
    
    @model_validator(mode='after')
    def validate_portfolio_consistency(self):
        """Validate portfolio tickers and weights have same length."""
        if len(self.tickers) != len(self.weights):
            raise ValueError("Portfolio tickers and weights must have same length")
        
        # Validate target allocation consistency
        if set(self.tickers) != set(self.target_allocation.keys()):
            raise ValueError("Target allocation must include all tickers")
        
        return self