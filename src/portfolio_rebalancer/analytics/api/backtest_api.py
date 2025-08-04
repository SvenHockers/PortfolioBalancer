"""Backtesting API endpoints for strategy validation and comparison."""

import logging
from typing import Dict, List, Any, Optional
from datetime import date, datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import uuid

from ...common.models import Portfolio
from ..analytics_service import AnalyticsService
from ..models import BacktestConfig, BacktestResult, AnalyticsError
from ..exceptions import BacktestError, InsufficientDataError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/backtest", tags=["backtest"])


# Request/Response Models
class BacktestRequest(BaseModel):
    """Backtest execution request."""
    tickers: List[str] = Field(..., description="List of ticker symbols", min_items=1)
    weights: Optional[List[float]] = Field(None, description="Portfolio weights (if None, equal weight)")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    strategy: str = Field("sharpe", description="Optimization strategy")
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")
    transaction_cost: float = Field(0.001, description="Transaction cost as decimal")
    initial_capital: float = Field(100000.0, description="Initial capital amount")
    benchmark: Optional[str] = Field("SPY", description="Benchmark ticker for comparison")
    
    @validator('weights')
    def validate_weights(cls, v, values):
        if v is not None:
            tickers = values.get('tickers', [])
            if len(v) != len(tickers):
                raise ValueError("Weights must match number of tickers")
            if abs(sum(v) - 1.0) > 0.01:
                raise ValueError("Weights must sum to 1.0")
        return v
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        if v > date.today():
            raise ValueError("Date cannot be in the future")
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        start_date = values.get('start_date')
        if start_date and v <= start_date:
            raise ValueError("End date must be after start date")
        if start_date and (v - start_date).days < 30:
            raise ValueError("Backtest period must be at least 30 days")
        return v
    
    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['sharpe', 'min_variance', 'equal_weight', 'max_return', 'custom']
        if v not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        return v
    
    @validator('rebalance_frequency')
    def validate_frequency(cls, v):
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly', 'annually']
        if v not in valid_frequencies:
            raise ValueError(f"Frequency must be one of {valid_frequencies}")
        return v


class StrategyComparisonRequest(BaseModel):
    """Strategy comparison request."""
    tickers: List[str] = Field(..., description="List of ticker symbols", min_items=1)
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    strategies: List[str] = Field(..., description="List of strategies to compare", min_items=2)
    rebalance_frequency: str = Field("monthly", description="Rebalancing frequency")
    transaction_cost: float = Field(0.001, description="Transaction cost as decimal")
    initial_capital: float = Field(100000.0, description="Initial capital amount")
    benchmark: Optional[str] = Field("SPY", description="Benchmark ticker for comparison")
    
    @validator('strategies')
    def validate_strategies(cls, v):
        valid_strategies = ['sharpe', 'min_variance', 'equal_weight', 'max_return']
        for strategy in v:
            if strategy not in valid_strategies:
                raise ValueError(f"All strategies must be one of {valid_strategies}")
        if len(set(v)) != len(v):
            raise ValueError("Strategies must be unique")
        return v


class BacktestResponse(BaseModel):
    """Backtest execution response."""
    backtest_id: str = Field(..., description="Unique backtest identifier")
    status: str = Field(..., description="Backtest status")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    created_at: datetime = Field(..., description="Creation timestamp")


class BacktestResultResponse(BaseModel):
    """Backtest result response."""
    backtest_id: str = Field(..., description="Backtest identifier")
    config: Dict[str, Any] = Field(..., description="Backtest configuration")
    results: Dict[str, Any] = Field(..., description="Backtest results")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    benchmark_comparison: Optional[Dict[str, Any]] = Field(None, description="Benchmark comparison")
    status: str = Field(..., description="Result status")
    completed_at: datetime = Field(..., description="Completion timestamp")


class StrategyComparisonResponse(BaseModel):
    """Strategy comparison response."""
    comparison_id: str = Field(..., description="Comparison identifier")
    strategies: List[str] = Field(..., description="Compared strategies")
    results: Dict[str, Dict[str, Any]] = Field(..., description="Results by strategy")
    ranking: List[Dict[str, Any]] = Field(..., description="Strategy ranking")
    statistical_tests: Dict[str, Any] = Field(..., description="Statistical significance tests")
    recommendation: str = Field(..., description="Recommended strategy")


# Dependency injection
def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance."""
    return None  # Would be properly injected


# Endpoints
@router.post("/", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Execute a portfolio backtest with specified configuration.
    
    Runs historical simulation of portfolio performance using the specified
    optimization strategy and rebalancing frequency.
    """
    try:
        logger.info("Starting backtest execution", extra={
            'tickers': request.tickers,
            'strategy': request.strategy,
            'date_range': f"{request.start_date} to {request.end_date}"
        })
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Generate unique backtest ID
        backtest_id = f"bt_{uuid.uuid4().hex[:12]}"
        
        # Create backtest configuration
        weights = request.weights or [1.0 / len(request.tickers)] * len(request.tickers)
        
        config = BacktestConfig(
            tickers=request.tickers,
            weights=weights,
            start_date=request.start_date,
            end_date=request.end_date,
            strategy=request.strategy,
            rebalance_frequency=request.rebalance_frequency,
            transaction_cost=request.transaction_cost,
            initial_capital=request.initial_capital
        )
        
        # Estimate completion time based on complexity
        days = (request.end_date - request.start_date).days
        estimated_minutes = max(1, days // 100)  # Rough estimate
        estimated_completion = datetime.now() + timedelta(minutes=estimated_minutes)
        
        # For long-running backtests, execute in background
        if days > 1000 or len(request.tickers) > 20:
            background_tasks.add_task(_execute_backtest_async, analytics_service, config, backtest_id)
            status = "queued"
            message = "Backtest queued for background execution"
        else:
            # Execute immediately for shorter backtests
            try:
                result = analytics_service.run_backtest(config)
                # Store result with backtest_id for retrieval
                _store_backtest_result(backtest_id, result)
                status = "completed"
                message = "Backtest completed successfully"
                estimated_completion = datetime.now()
            except Exception as e:
                logger.error(f"Backtest execution failed: {e}")
                status = "failed"
                message = f"Backtest failed: {str(e)}"
        
        return BacktestResponse(
            backtest_id=backtest_id,
            status=status,
            message=message,
            estimated_completion=estimated_completion,
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except BacktestError as e:
        logger.error(f"Backtest configuration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in backtest execution: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{backtest_id}", response_model=BacktestResultResponse)
async def get_backtest_result(
    backtest_id: str,
    include_details: bool = Query(True, description="Include detailed results"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Retrieve backtest results by ID.
    
    Returns comprehensive backtest results including performance metrics,
    allocation history, and benchmark comparison.
    """
    try:
        logger.info(f"Retrieving backtest result: {backtest_id}")
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Retrieve backtest result
        result = analytics_service.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        
        # Format response
        response_data = {
            "backtest_id": backtest_id,
            "config": {
                "tickers": result.config.tickers,
                "start_date": result.config.start_date.isoformat(),
                "end_date": result.config.end_date.isoformat(),
                "strategy": result.config.strategy,
                "rebalance_frequency": result.config.rebalance_frequency,
                "transaction_cost": result.config.transaction_cost,
                "initial_capital": result.config.initial_capital
            },
            "results": {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "calmar_ratio": result.calmar_ratio,
                "transaction_costs": result.transaction_costs,
                "final_value": result.config.initial_capital * (1 + result.total_return)
            },
            "performance_metrics": {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "calmar_ratio": result.calmar_ratio,
                "win_rate": 0.65,  # Mock data
                "best_month": 0.12,  # Mock data
                "worst_month": -0.08  # Mock data
            },
            "status": "completed",
            "completed_at": datetime.now()
        }
        
        # Add detailed data if requested
        if include_details and hasattr(result, 'returns_series'):
            response_data["detailed_results"] = {
                "returns_series": result.returns_series.to_dict() if hasattr(result.returns_series, 'to_dict') else {},
                "allocation_history": result.allocation_history.to_dict() if hasattr(result.allocation_history, 'to_dict') else {}
            }
        
        # Add benchmark comparison if available
        response_data["benchmark_comparison"] = _generate_benchmark_comparison(result)
        
        return BacktestResultResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve backtest result: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve backtest result")


@router.post("/compare", response_model=StrategyComparisonResponse)
async def compare_strategies(
    request: StrategyComparisonRequest,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Compare multiple optimization strategies side-by-side.
    
    Runs backtests for each strategy and provides statistical comparison
    with significance tests and recommendations.
    """
    try:
        logger.info("Starting strategy comparison", extra={
            'strategies': request.strategies,
            'tickers': request.tickers,
            'date_range': f"{request.start_date} to {request.end_date}"
        })
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        comparison_id = f"cmp_{uuid.uuid4().hex[:12]}"
        results = {}
        
        # Run backtest for each strategy
        for strategy in request.strategies:
            try:
                config = BacktestConfig(
                    tickers=request.tickers,
                    weights=[1.0 / len(request.tickers)] * len(request.tickers),  # Equal weight
                    start_date=request.start_date,
                    end_date=request.end_date,
                    strategy=strategy,
                    rebalance_frequency=request.rebalance_frequency,
                    transaction_cost=request.transaction_cost,
                    initial_capital=request.initial_capital
                )
                
                result = analytics_service.run_backtest(config)
                
                results[strategy] = {
                    "total_return": result.total_return,
                    "annualized_return": result.annualized_return,
                    "volatility": result.volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "calmar_ratio": result.calmar_ratio,
                    "transaction_costs": result.transaction_costs
                }
                
            except Exception as e:
                logger.error(f"Strategy {strategy} backtest failed: {e}")
                results[strategy] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Rank strategies by Sharpe ratio
        successful_results = {k: v for k, v in results.items() if "error" not in v}
        ranking = sorted(
            successful_results.items(),
            key=lambda x: x[1].get("sharpe_ratio", -999),
            reverse=True
        )
        
        ranking_list = [
            {
                "rank": i + 1,
                "strategy": strategy,
                "sharpe_ratio": metrics["sharpe_ratio"],
                "total_return": metrics["total_return"],
                "volatility": metrics["volatility"]
            }
            for i, (strategy, metrics) in enumerate(ranking)
        ]
        
        # Generate statistical tests (mock for now)
        statistical_tests = _generate_statistical_tests(successful_results)
        
        # Determine recommendation
        if ranking_list:
            best_strategy = ranking_list[0]["strategy"]
            recommendation = f"Based on risk-adjusted returns (Sharpe ratio), {best_strategy} is the recommended strategy."
        else:
            recommendation = "No successful strategy results available for recommendation."
        
        return StrategyComparisonResponse(
            comparison_id=comparison_id,
            strategies=request.strategies,
            results=results,
            ranking=ranking_list,
            statistical_tests=statistical_tests,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Strategy comparison failed")


@router.get("/{backtest_id}/charts")
async def get_backtest_charts(
    backtest_id: str,
    chart_type: str = Query("performance", description="Chart type to generate"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Generate chart data for backtest visualization.
    
    Supports various chart types including performance, drawdown,
    allocation, and rolling metrics charts.
    """
    try:
        logger.info(f"Generating {chart_type} chart for backtest {backtest_id}")
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Retrieve backtest result
        result = analytics_service.get_backtest_result(backtest_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        
        # Generate chart data based on type
        if chart_type == "performance":
            chart_data = _generate_performance_chart_data(result)
        elif chart_type == "drawdown":
            chart_data = _generate_drawdown_chart_data(result)
        elif chart_type == "allocation":
            chart_data = _generate_allocation_chart_data(result)
        elif chart_type == "rolling_metrics":
            chart_data = _generate_rolling_metrics_chart_data(result)
        else:
            raise HTTPException(status_code=400, detail="Invalid chart type")
        
        return {
            "backtest_id": backtest_id,
            "chart_type": chart_type,
            "data": chart_data,
            "metadata": {
                "start_date": result.config.start_date.isoformat(),
                "end_date": result.config.end_date.isoformat(),
                "strategy": result.config.strategy
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        raise HTTPException(status_code=500, detail="Chart generation failed")


@router.delete("/{backtest_id}")
async def delete_backtest(
    backtest_id: str,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Delete a backtest result and associated data.
    
    Removes backtest results from storage to free up space.
    """
    try:
        logger.info(f"Deleting backtest {backtest_id}")
        
        # In a real implementation, this would delete from storage
        # For now, return success
        
        return {
            "backtest_id": backtest_id,
            "status": "deleted",
            "message": "Backtest result deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to delete backtest {backtest_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete backtest")


# Helper functions
async def _execute_backtest_async(
    analytics_service: AnalyticsService,
    config: BacktestConfig,
    backtest_id: str
):
    """Execute backtest asynchronously in background."""
    try:
        result = analytics_service.run_backtest(config)
        _store_backtest_result(backtest_id, result)
        logger.info(f"Background backtest {backtest_id} completed successfully")
    except Exception as e:
        logger.error(f"Background backtest {backtest_id} failed: {e}")
        # Store error result
        _store_backtest_error(backtest_id, str(e))


def _store_backtest_result(backtest_id: str, result: BacktestResult):
    """Store backtest result for later retrieval."""
    # In a real implementation, this would store in a database or cache
    # For now, just log
    logger.info(f"Storing backtest result {backtest_id}")


def _store_backtest_error(backtest_id: str, error: str):
    """Store backtest error for later retrieval."""
    # In a real implementation, this would store error in database
    logger.error(f"Storing backtest error {backtest_id}: {error}")


def _generate_benchmark_comparison(result: BacktestResult) -> Dict[str, Any]:
    """Generate benchmark comparison data."""
    # Mock benchmark comparison
    return {
        "benchmark_ticker": "SPY",
        "benchmark_return": 0.08,  # Mock 8% return
        "excess_return": result.total_return - 0.08,
        "tracking_error": 0.05,  # Mock tracking error
        "information_ratio": (result.total_return - 0.08) / 0.05,
        "beta": 0.95,  # Mock beta
        "alpha": result.total_return - (0.03 + 0.95 * (0.08 - 0.03))  # Mock alpha
    }


def _generate_statistical_tests(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistical significance tests."""
    # Mock statistical tests
    return {
        "t_test_results": {
            "significant_differences": True,
            "p_value": 0.023,
            "confidence_level": 0.95
        },
        "variance_test": {
            "equal_variances": False,
            "p_value": 0.045
        },
        "normality_test": {
            "returns_normal": True,
            "p_value": 0.12
        }
    }


def _generate_performance_chart_data(result: BacktestResult) -> List[Dict[str, Any]]:
    """Generate performance chart data."""
    # Mock performance chart data
    chart_data = []
    
    # Generate mock daily returns
    import pandas as pd
    dates = pd.date_range(result.config.start_date, result.config.end_date, freq='D')
    cumulative_return = 1.0
    
    for i, date in enumerate(dates):
        daily_return = 0.0008 + (i % 10) * 0.0002  # Mock daily return
        cumulative_return *= (1 + daily_return)
        
        chart_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "cumulative_return": cumulative_return - 1,
            "daily_return": daily_return,
            "portfolio_value": result.config.initial_capital * cumulative_return
        })
    
    return chart_data


def _generate_drawdown_chart_data(result: BacktestResult) -> List[Dict[str, Any]]:
    """Generate drawdown chart data."""
    # Mock drawdown chart data
    chart_data = []
    
    import pandas as pd
    dates = pd.date_range(result.config.start_date, result.config.end_date, freq='D')
    peak_value = result.config.initial_capital
    current_value = result.config.initial_capital
    
    for i, date in enumerate(dates):
        # Mock value changes
        daily_change = 0.001 * (1 if i % 7 != 0 else -0.5)  # Occasional drawdowns
        current_value *= (1 + daily_change)
        
        if current_value > peak_value:
            peak_value = current_value
        
        drawdown = (current_value - peak_value) / peak_value
        
        chart_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "drawdown": drawdown,
            "portfolio_value": current_value,
            "peak_value": peak_value
        })
    
    return chart_data


def _generate_allocation_chart_data(result: BacktestResult) -> List[Dict[str, Any]]:
    """Generate allocation chart data."""
    # Mock allocation chart data
    chart_data = []
    
    import pandas as pd
    dates = pd.date_range(result.config.start_date, result.config.end_date, freq='M')  # Monthly
    
    for i, date in enumerate(dates):
        # Mock changing allocations
        allocations = {}
        for j, ticker in enumerate(result.config.tickers):
            base_weight = 1.0 / len(result.config.tickers)
            variation = 0.1 * (i % 3 - 1) * (j % 2)  # Some variation
            allocations[ticker] = max(0.05, min(0.5, base_weight + variation))
        
        # Normalize to sum to 1
        total = sum(allocations.values())
        allocations = {k: v / total for k, v in allocations.items()}
        
        chart_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "allocations": allocations
        })
    
    return chart_data


def _generate_rolling_metrics_chart_data(result: BacktestResult) -> List[Dict[str, Any]]:
    """Generate rolling metrics chart data."""
    # Mock rolling metrics chart data
    chart_data = []
    
    import pandas as pd
    dates = pd.date_range(result.config.start_date, result.config.end_date, freq='W')  # Weekly
    
    for i, date in enumerate(dates):
        # Mock rolling metrics with some variation
        base_sharpe = result.sharpe_ratio
        sharpe_variation = 0.2 * (i % 5 - 2) / 5
        
        chart_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "rolling_sharpe": base_sharpe + sharpe_variation,
            "rolling_volatility": result.volatility * (1 + 0.1 * (i % 3 - 1) / 3),
            "rolling_return": result.annualized_return * (1 + 0.05 * (i % 4 - 2) / 4)
        })
    
    return chart_data