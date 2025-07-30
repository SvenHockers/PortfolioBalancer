"""Dividend analytics API endpoints for income tracking and reporting."""

import logging
from typing import Dict, List, Any, Optional
from datetime import date, datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...common.models import Portfolio
from ..analytics_service import AnalyticsService
from ..models import DividendAnalysis, AnalyticsError
from ..engines.dividend_analysis import DividendAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dividends", tags=["dividends"])


class DividendRequest(BaseModel):
    """Request model for dividend analysis."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    tickers: List[str] = Field(..., description="List of ticker symbols")
    weights: List[float] = Field(..., description="Portfolio weights")
    portfolio_value: float = Field(100000.0, description="Total portfolio value")


class IncomeProjectionRequest(BaseModel):
    """Request model for income projection."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    tickers: List[str] = Field(..., description="List of ticker symbols")
    weights: List[float] = Field(..., description="Portfolio weights")
    portfolio_value: float = Field(100000.0, description="Total portfolio value")
    years: int = Field(5, ge=1, le=20, description="Number of years to project")
    scenarios: Optional[List[str]] = Field(None, description="Scenarios to model")


class SustainabilityRequest(BaseModel):
    """Request model for sustainability analysis."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    tickers: List[str] = Field(..., description="List of ticker symbols")
    weights: List[float] = Field(..., description="Portfolio weights")
    portfolio_value: float = Field(100000.0, description="Total portfolio value")


class OptimizationRequest(BaseModel):
    """Request model for income optimization."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    tickers: List[str] = Field(..., description="List of ticker symbols")
    weights: List[float] = Field(..., description="Portfolio weights")
    portfolio_value: float = Field(100000.0, description="Total portfolio value")
    target_yield: float = Field(0.04, description="Target dividend yield")
    risk_tolerance: str = Field("moderate", description="Risk tolerance level")


class IncomeGoalRequest(BaseModel):
    """Request model for income goal tracking."""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    annual_income_goal: float = Field(..., description="Annual income goal")
    target_date: date = Field(..., description="Target date to achieve goal")


# Dependency to get analytics service
def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance."""
    return None  # Would be properly injected


def get_dividend_analyzer() -> DividendAnalyzer:
    """Get dividend analyzer instance."""
    from ...common.interfaces import DataStorage
    from unittest.mock import Mock
    mock_storage = Mock(spec=DataStorage)
    return DividendAnalyzer(mock_storage)

@router.post("/analyze")
async def analyze_dividend_income(
    request: DividendRequest,
    analyzer: DividendAnalyzer = Depends(get_dividend_analyzer)
):
    """
    Analyze current and projected dividend income for a portfolio.
    
    Args:
        request: Dividend analysis request
        analyzer: Dividend analyzer instance
        
    Returns:
        Comprehensive dividend analysis results
    """
    try:
        logger.info(f"Analyzing dividend income for portfolio {request.portfolio_id}")
        
        analysis = analyzer.analyze_dividend_income(
            portfolio_id=request.portfolio_id,
            tickers=request.tickers,
            weights=request.weights,
            portfolio_value=request.portfolio_value
        )
        
        return {
            "analysis": analysis.dict(),
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except AnalyticsError as e:
        logger.error(f"Dividend analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in dividend analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/project")
async def project_income(
    request: IncomeProjectionRequest,
    analyzer: DividendAnalyzer = Depends(get_dividend_analyzer)
):
    """
    Project future dividend income with multiple scenarios.
    
    Args:
        request: Income projection request
        analyzer: Dividend analyzer instance
        
    Returns:
        Multi-scenario income projections
    """
    try:
        logger.info(f"Projecting income for portfolio {request.portfolio_id}")
        
        projection = analyzer.project_income(
            portfolio_id=request.portfolio_id,
            tickers=request.tickers,
            weights=request.weights,
            years=request.years,
            portfolio_value=request.portfolio_value,
            scenarios=request.scenarios
        )
        
        return {
            "projection": projection,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except AnalyticsError as e:
        logger.error(f"Income projection failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in income projection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sustainability")
async def analyze_sustainability(
    request: SustainabilityRequest,
    analyzer: DividendAnalyzer = Depends(get_dividend_analyzer)
):
    """
    Analyze dividend sustainability and coverage ratios.
    
    Args:
        request: Sustainability analysis request
        analyzer: Dividend analyzer instance
        
    Returns:
        Comprehensive sustainability analysis
    """
    try:
        logger.info(f"Analyzing sustainability for portfolio {request.portfolio_id}")
        
        sustainability = analyzer.analyze_sustainability(
            portfolio_id=request.portfolio_id,
            tickers=request.tickers,
            weights=request.weights,
            portfolio_value=request.portfolio_value
        )
        
        return {
            "sustainability": sustainability,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except AnalyticsError as e:
        logger.error(f"Sustainability analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in sustainability analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboard/{portfolio_id}")
async def get_income_dashboard_data(
    portfolio_id: str,
    tickers: str = Query(..., description="Comma-separated ticker symbols"),
    weights: str = Query(..., description="Comma-separated weights"),
    portfolio_value: float = Query(100000.0, description="Portfolio value"),
    analyzer: DividendAnalyzer = Depends(get_dividend_analyzer)
):
    """
    Get comprehensive income dashboard data for visualization.
    
    Args:
        portfolio_id: Portfolio identifier
        tickers: Comma-separated ticker symbols
        weights: Comma-separated weights
        portfolio_value: Total portfolio value
        analyzer: Dividend analyzer instance
        
    Returns:
        Comprehensive dashboard data for income tracking
    """
    try:
        logger.info(f"Getting income dashboard data for portfolio {portfolio_id}")
        
        # Parse inputs
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        weight_list = [float(w.strip()) for w in weights.split(',')]
        
        if len(ticker_list) != len(weight_list):
            raise HTTPException(status_code=400, detail="Tickers and weights must have same length")
        
        # Get comprehensive analysis
        analysis = analyzer.analyze_dividend_income(
            portfolio_id=portfolio_id,
            tickers=ticker_list,
            weights=weight_list,
            portfolio_value=portfolio_value
        )
        
        # Get yield metrics
        yield_metrics = analyzer.calculate_yield_metrics(
            tickers=ticker_list,
            weights=weight_list,
            portfolio_value=portfolio_value
        )
        
        # Get top contributors
        contributors = analyzer.get_top_dividend_contributors(
            tickers=ticker_list,
            weights=weight_list,
            portfolio_value=portfolio_value,
            top_n=5
        )
        
        # Get sustainability analysis
        sustainability = analyzer.analyze_sustainability(
            portfolio_id=portfolio_id,
            tickers=ticker_list,
            weights=weight_list,
            portfolio_value=portfolio_value
        )
        
        dashboard_data = {
            "portfolio_id": portfolio_id,
            "summary": {
                "current_yield": analysis.current_yield,
                "annual_income": analysis.projected_annual_income,
                "growth_rate": analysis.dividend_growth_rate,
                "sustainability_score": analysis.income_sustainability_score,
                "portfolio_value": portfolio_value
            },
            "yield_metrics": yield_metrics,
            "top_contributors": [
                {"ticker": ticker, "annual_income": income}
                for ticker, income in contributors
            ],
            "sustainability": {
                "overall_score": sustainability["overall_sustainability_score"],
                "risk_factors": sustainability["risk_factors"],
                "recommendations": sustainability["recommendations"][:3]  # Top 3 recommendations
            },
            "charts": {
                "yield_by_holding": {
                    ticker: yield_metrics["yield_by_holding"].get(ticker, 0)
                    for ticker in ticker_list
                },
                "income_by_holding": {
                    ticker: income for ticker, income in contributors
                }
            }
        }
        
        return {
            "dashboard": dashboard_data,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Invalid input parameters: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
    except AnalyticsError as e:
        logger.error(f"Dashboard data generation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")