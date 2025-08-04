"""Analytics data models using Pydantic."""

from datetime import date as Date, datetime as DateTime
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
import pandas as pd


# Import exceptions from dedicated module
from .exceptions import (
    AnalyticsError,
    BacktestError,
    InsufficientDataError,
    SimulationError,
    RiskAnalysisError
)


class OptimizationStrategy(str, Enum):
    """Enumeration for optimization strategies."""
    SHARPE = "sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"


class RebalanceFrequency(str, Enum):
    """Enumeration for rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    
    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)
    
    tickers: List[str] = Field(..., min_length=1, description="List of ticker symbols")
    start_date: Date = Field(..., description="Start date for backtest")
    end_date: Date = Field(..., description="End date for backtest")
    strategy: OptimizationStrategy = Field(..., description="Optimization strategy")
    rebalance_frequency: RebalanceFrequency = Field(
        RebalanceFrequency.MONTHLY, 
        description="Rebalancing frequency"
    )
    transaction_cost: float = Field(
        0.001, 
        ge=0, 
        le=0.1, 
        description="Transaction cost as fraction"
    )
    initial_capital: float = Field(
        100000.0, 
        gt=0, 
        description="Initial capital amount"
    )
    
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
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate date range is logical."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        
        # Check for reasonable date range (at least 30 days)
        date_diff = (self.end_date - self.start_date).days
        if date_diff < 30:
            raise ValueError("Date range must be at least 30 days")
        
        return self


class BacktestResult(BaseModel):
    """Results from backtesting simulation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    config: BacktestConfig = Field(..., description="Backtest configuration")
    total_return: float = Field(..., description="Total return over period")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., ge=0, description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., le=0, description="Maximum drawdown")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    transaction_costs: float = Field(..., ge=0, description="Total transaction costs")
    num_rebalances: int = Field(..., ge=0, description="Number of rebalancing events")
    final_value: float = Field(..., gt=0, description="Final portfolio value")
    
    # Serialized data (stored as JSON in database)
    returns_data: Optional[Dict[str, Any]] = Field(None, description="Returns time series data")
    allocation_data: Optional[Dict[str, Any]] = Field(None, description="Allocation history data")
    
    @field_validator('max_drawdown')
    @classmethod
    def validate_max_drawdown(cls, v):
        """Validate max drawdown is negative or zero."""
        if v > 0:
            raise ValueError("Maximum drawdown must be negative or zero")
        return v


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    portfolio_tickers: List[str] = Field(..., min_length=1, description="Portfolio ticker symbols")
    portfolio_weights: List[float] = Field(..., min_length=1, description="Portfolio weights")
    time_horizon_years: int = Field(..., gt=0, le=50, description="Time horizon in years")
    num_simulations: int = Field(10000, gt=0, le=100000, description="Number of simulations")
    confidence_levels: List[float] = Field(
        [0.05, 0.25, 0.5, 0.75, 0.95], 
        description="Confidence levels for percentiles"
    )
    initial_value: float = Field(100000.0, gt=0, description="Initial portfolio value")
    
    @field_validator('portfolio_weights')
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
    
    @field_validator('confidence_levels')
    @classmethod
    def validate_confidence_levels(cls, v):
        """Validate confidence levels are between 0 and 1."""
        if not v:
            raise ValueError("At least one confidence level must be provided")
        
        for level in v:
            if not 0 < level < 1:
                raise ValueError(f"Confidence level {level} must be between 0 and 1")
        
        return sorted(v)
    
    @model_validator(mode='after')
    def validate_portfolio_consistency(self):
        """Validate portfolio tickers and weights have same length."""
        if len(self.portfolio_tickers) != len(self.portfolio_weights):
            raise ValueError("Portfolio tickers and weights must have same length")
        
        return self


class MonteCarloResult(BaseModel):
    """Results from Monte Carlo simulation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    config: MonteCarloConfig = Field(..., description="Monte Carlo configuration")
    expected_value: float = Field(..., gt=0, description="Expected final portfolio value")
    probability_of_loss: float = Field(..., ge=0, le=1, description="Probability of loss")
    value_at_risk_95: float = Field(..., description="Value at Risk at 95% confidence")
    conditional_var_95: float = Field(..., description="Conditional VaR at 95% confidence")
    
    # Percentile projections (stored as JSON)
    percentile_data: Optional[Dict[str, Any]] = Field(None, description="Percentile projection data")
    simulation_summary: Optional[Dict[str, Any]] = Field(None, description="Simulation summary statistics")


class RiskAnalysis(BaseModel):
    """Comprehensive risk analysis results."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    analysis_date: Date = Field(..., description="Date of analysis")
    portfolio_beta: float = Field(..., description="Portfolio beta")
    tracking_error: float = Field(..., ge=0, description="Tracking error")
    information_ratio: float = Field(..., description="Information ratio")
    var_95: float = Field(..., description="Value at Risk at 95%")
    cvar_95: float = Field(..., description="Conditional VaR at 95%")
    max_drawdown: float = Field(..., le=0, description="Maximum drawdown")
    concentration_risk: float = Field(..., ge=0, le=1, description="Concentration risk score")
    
    # Complex data stored as JSON
    correlation_data: Optional[Dict[str, Any]] = Field(None, description="Correlation matrix data")
    factor_exposures: Optional[Dict[str, float]] = Field(None, description="Factor exposure data")
    sector_exposures: Optional[Dict[str, float]] = Field(None, description="Sector exposure data")


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    calculation_date: Date = Field(..., description="Date of calculation")
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., ge=0, description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    alpha: float = Field(..., description="Alpha vs benchmark")
    beta: float = Field(..., description="Beta vs benchmark")
    r_squared: float = Field(..., ge=0, le=1, description="R-squared vs benchmark")
    tracking_error: float = Field(..., ge=0, description="Tracking error")
    information_ratio: float = Field(..., description="Information ratio")
    
    # Time series data
    performance_data: Optional[Dict[str, Any]] = Field(None, description="Performance time series data")


class DividendAnalysis(BaseModel):
    """Dividend analysis results."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    analysis_date: Date = Field(..., description="Date of analysis")
    current_yield: float = Field(..., ge=0, description="Current dividend yield")
    projected_annual_income: float = Field(..., ge=0, description="Projected annual income")
    dividend_growth_rate: float = Field(..., description="Historical dividend growth rate")
    payout_ratio: float = Field(..., ge=0, description="Average payout ratio")
    dividend_coverage: float = Field(..., ge=0, description="Dividend coverage ratio")
    income_sustainability_score: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Income sustainability score"
    )
    
    # Detailed data
    dividend_data: Optional[Dict[str, Any]] = Field(None, description="Detailed dividend data")
    top_contributors: Optional[List[Tuple[str, float]]] = Field(
        None, 
        description="Top dividend contributors"
    )


class StrategyComparison(BaseModel):
    """Results from strategy comparison."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    strategies: List[str] = Field(..., min_length=2, description="Compared strategies")
    comparison_period: Tuple[Date, Date] = Field(..., description="Comparison period")
    results: Dict[str, BacktestResult] = Field(..., description="Results by strategy")
    statistical_significance: Optional[Dict[str, Any]] = Field(
        None, 
        description="Statistical significance tests"
    )
    best_strategy: str = Field(..., description="Best performing strategy")
    ranking: List[str] = Field(..., description="Strategy ranking by performance")


class StressTestResult(BaseModel):
    """Results from stress testing."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    test_date: Date = Field(..., description="Date of stress test")
    scenarios: List[str] = Field(..., min_length=1, description="Tested scenarios")
    results: Dict[str, float] = Field(..., description="Results by scenario")
    worst_case_loss: float = Field(..., le=0, description="Worst case loss")
    recovery_time_estimate: Optional[int] = Field(None, description="Recovery time in days")


class VaRResult(BaseModel):
    """Value at Risk calculation result."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    calculation_date: Date = Field(..., description="Calculation date")
    confidence_level: float = Field(..., gt=0, lt=1, description="Confidence level")
    time_horizon_days: int = Field(..., gt=0, description="Time horizon in days")
    var_amount: float = Field(..., description="VaR amount")
    cvar_amount: float = Field(..., description="Conditional VaR amount")
    methodology: str = Field(..., description="VaR calculation methodology")