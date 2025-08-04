"""Performance dashboard API endpoints for Grafana integration."""

import logging
from typing import Dict, List, Any, Optional
from datetime import date, datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...common.models import Portfolio
from ..analytics_service import AnalyticsService
from ..models import PerformanceMetrics, AnalyticsError
from ..exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])


class GrafanaTimeSeriesPoint(BaseModel):
    """Grafana time series data point."""
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    value: float = Field(..., description="Metric value")


class GrafanaTimeSeriesResponse(BaseModel):
    """Grafana time series response format."""
    target: str = Field(..., description="Series name")
    datapoints: List[List[Any]] = Field(..., description="[value, timestamp] pairs")


class PerformanceDashboardData(BaseModel):
    """Performance dashboard data for Grafana."""
    portfolio_id: str
    current_metrics: Dict[str, float]
    time_series: Dict[str, List[GrafanaTimeSeriesPoint]]
    benchmark_comparison: Dict[str, Any]
    attribution_data: Optional[Dict[str, Any]] = None


class PerformanceAlert(BaseModel):
    """Performance alert configuration."""
    portfolio_id: str
    metric: str
    threshold: float
    condition: str  # "above", "below"
    enabled: bool = True


# Dependency to get analytics service (would be injected in real implementation)
def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance."""
    # This would be properly injected in a real implementation
    # For now, return None and handle in the endpoints
    return None


@router.get("/dashboard/{portfolio_id}", response_model=PerformanceDashboardData)
async def get_performance_dashboard_data(
    portfolio_id: str,
    days: int = Query(252, description="Number of days to look back"),
    benchmark: str = Query("SPY", description="Benchmark ticker"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get comprehensive performance dashboard data for Grafana.
    
    Args:
        portfolio_id: Portfolio identifier
        days: Number of days to look back
        benchmark: Benchmark ticker symbol
        analytics_service: Analytics service instance
        
    Returns:
        Performance dashboard data
    """
    try:
        logger.info(f"Getting dashboard data for portfolio {portfolio_id}")
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Get current performance metrics
        current_date = date.today()
        current_metrics = analytics_service.get_performance_metrics(portfolio_id, current_date)
        
        if not current_metrics:
            raise HTTPException(status_code=404, detail="No performance data found for portfolio")
        
        # Get historical performance data
        start_date = current_date - timedelta(days=days)
        performance_history = analytics_service.get_performance_history(
            portfolio_id, start_date, current_date
        )
        
        # Convert to Grafana time series format
        time_series = _convert_to_grafana_time_series(performance_history)
        
        # Get benchmark comparison (mock data for now)
        benchmark_comparison = {
            "benchmark_ticker": benchmark,
            "portfolio_vs_benchmark": {
                "portfolio_return": current_metrics.annualized_return,
                "benchmark_return": 0.10,  # Mock benchmark return
                "excess_return": current_metrics.annualized_return - 0.10,
                "tracking_error": current_metrics.tracking_error,
                "information_ratio": current_metrics.information_ratio
            }
        }
        
        dashboard_data = PerformanceDashboardData(
            portfolio_id=portfolio_id,
            current_metrics={
                "total_return": current_metrics.total_return,
                "annualized_return": current_metrics.annualized_return,
                "volatility": current_metrics.volatility,
                "sharpe_ratio": current_metrics.sharpe_ratio,
                "alpha": current_metrics.alpha,
                "beta": current_metrics.beta,
                "max_drawdown": -0.15,  # Mock data
                "current_value": 125000.0  # Mock data
            },
            time_series=time_series,
            benchmark_comparison=benchmark_comparison
        )
        
        return dashboard_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/grafana/query/{portfolio_id}")
async def grafana_query_endpoint(
    portfolio_id: str,
    metric: str = Query(..., description="Metric to query"),
    from_timestamp: int = Query(..., description="Start timestamp (ms)"),
    to_timestamp: int = Query(..., description="End timestamp (ms)"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Grafana-compatible query endpoint for time series data.
    
    Args:
        portfolio_id: Portfolio identifier
        metric: Metric name to query
        from_timestamp: Start timestamp in milliseconds
        to_timestamp: End timestamp in milliseconds
        analytics_service: Analytics service instance
        
    Returns:
        Grafana-compatible time series data
    """
    try:
        logger.info(f"Grafana query for portfolio {portfolio_id}, metric {metric}")
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Convert timestamps to dates
        start_date = datetime.fromtimestamp(from_timestamp / 1000).date()
        end_date = datetime.fromtimestamp(to_timestamp / 1000).date()
        
        # Get performance history
        performance_history = analytics_service.get_performance_history(
            portfolio_id, start_date, end_date
        )
        
        if not performance_history:
            return []
        
        # Extract the requested metric
        datapoints = []
        for perf_metric in performance_history:
            timestamp_ms = int(perf_metric.calculation_date.strftime('%s')) * 1000
            
            # Get the metric value
            metric_value = _get_metric_value(perf_metric, metric)
            if metric_value is not None:
                datapoints.append([metric_value, timestamp_ms])
        
        response = GrafanaTimeSeriesResponse(
            target=f"{portfolio_id}_{metric}",
            datapoints=datapoints
        )
        
        return [response.dict()]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grafana query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/grafana/search")
async def grafana_search_endpoint(
    query: str = Query("", description="Search query"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Grafana search endpoint for available metrics.
    
    Args:
        query: Search query string
        analytics_service: Analytics service instance
        
    Returns:
        List of available metrics
    """
    try:
        # Available performance metrics
        available_metrics = [
            "total_return",
            "annualized_return", 
            "volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "alpha",
            "beta",
            "r_squared",
            "tracking_error",
            "information_ratio",
            "portfolio_value",
            "drawdown"
        ]
        
        # Filter by query if provided
        if query:
            available_metrics = [m for m in available_metrics if query.lower() in m.lower()]
        
        return available_metrics
        
    except Exception as e:
        logger.error(f"Grafana search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/alerts", response_model=Dict[str, str])
async def create_performance_alert(
    alert: PerformanceAlert,
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Create a performance alert for significant changes.
    
    Args:
        alert: Alert configuration
        analytics_service: Analytics service instance
        
    Returns:
        Alert creation confirmation
    """
    try:
        logger.info(f"Creating performance alert for portfolio {alert.portfolio_id}")
        
        # In a real implementation, this would store the alert configuration
        # and set up monitoring to trigger when conditions are met
        
        alert_id = f"alert_{alert.portfolio_id}_{alert.metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Mock alert storage
        logger.info(f"Alert created: {alert_id}")
        
        return {
            "alert_id": alert_id,
            "status": "created",
            "message": f"Alert created for {alert.metric} on portfolio {alert.portfolio_id}"
        }
        
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.get("/reports/{portfolio_id}/generate")
async def generate_performance_report(
    portfolio_id: str,
    report_type: str = Query("standard", description="Report type"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    format: str = Query("json", description="Report format (json, pdf)"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Generate automated performance report with customizable templates.
    
    Args:
        portfolio_id: Portfolio identifier
        report_type: Type of report to generate
        start_date: Report start date
        end_date: Report end date
        format: Output format
        analytics_service: Analytics service instance
        
    Returns:
        Generated performance report
    """
    try:
        logger.info(f"Generating {report_type} report for portfolio {portfolio_id}")
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Parse dates
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start_dt = date.today() - timedelta(days=90)  # Default 3 months
            
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_dt = date.today()
        
        # Get performance data
        performance_history = analytics_service.get_performance_history(
            portfolio_id, start_dt, end_dt
        )
        
        if not performance_history:
            raise HTTPException(status_code=404, detail="No performance data found for the specified period")
        
        # Generate report based on type
        if report_type == "standard":
            report = _generate_standard_report(portfolio_id, performance_history, start_dt, end_dt)
        elif report_type == "detailed":
            report = _generate_detailed_report(portfolio_id, performance_history, start_dt, end_dt)
        elif report_type == "summary":
            report = _generate_summary_report(portfolio_id, performance_history, start_dt, end_dt)
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")
        
        if format == "pdf":
            # In a real implementation, this would generate a PDF
            return {"message": "PDF generation not implemented yet", "report_data": report}
        else:
            return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/charts/{portfolio_id}/data")
async def get_chart_data(
    portfolio_id: str,
    chart_type: str = Query(..., description="Chart type"),
    period: str = Query("1Y", description="Time period"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Get interactive chart data for performance visualization.
    
    Args:
        portfolio_id: Portfolio identifier
        chart_type: Type of chart (returns, drawdown, rolling_metrics)
        period: Time period (1M, 3M, 6M, 1Y, 2Y)
        analytics_service: Analytics service instance
        
    Returns:
        Chart data formatted for visualization
    """
    try:
        logger.info(f"Getting {chart_type} chart data for portfolio {portfolio_id}")
        
        if not analytics_service:
            raise HTTPException(status_code=503, detail="Analytics service not available")
        
        # Parse period
        days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
        days = days_map.get(period, 365)
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get performance data
        performance_history = analytics_service.get_performance_history(
            portfolio_id, start_date, end_date
        )
        
        if not performance_history:
            raise HTTPException(status_code=404, detail="No performance data found")
        
        # Generate chart data based on type
        if chart_type == "returns":
            chart_data = _generate_returns_chart_data(performance_history)
        elif chart_type == "drawdown":
            chart_data = _generate_drawdown_chart_data(performance_history)
        elif chart_type == "rolling_metrics":
            chart_data = _generate_rolling_metrics_chart_data(performance_history)
        elif chart_type == "risk_return":
            chart_data = _generate_risk_return_chart_data(performance_history)
        else:
            raise HTTPException(status_code=400, detail="Invalid chart type")
        
        return {
            "portfolio_id": portfolio_id,
            "chart_type": chart_type,
            "period": period,
            "data": chart_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chart data generation failed: {str(e)}")


# Helper functions

def _convert_to_grafana_time_series(performance_history: List[PerformanceMetrics]) -> Dict[str, List[GrafanaTimeSeriesPoint]]:
    """Convert performance history to Grafana time series format."""
    time_series = {
        "returns": [],
        "volatility": [],
        "sharpe_ratio": [],
        "alpha": [],
        "beta": []
    }
    
    for metric in performance_history:
        timestamp = int(metric.calculation_date.strftime('%s')) * 1000
        
        time_series["returns"].append(GrafanaTimeSeriesPoint(
            timestamp=timestamp,
            value=metric.annualized_return
        ))
        time_series["volatility"].append(GrafanaTimeSeriesPoint(
            timestamp=timestamp,
            value=metric.volatility
        ))
        time_series["sharpe_ratio"].append(GrafanaTimeSeriesPoint(
            timestamp=timestamp,
            value=metric.sharpe_ratio
        ))
        time_series["alpha"].append(GrafanaTimeSeriesPoint(
            timestamp=timestamp,
            value=metric.alpha
        ))
        time_series["beta"].append(GrafanaTimeSeriesPoint(
            timestamp=timestamp,
            value=metric.beta
        ))
    
    return time_series


def _get_metric_value(performance_metric: PerformanceMetrics, metric_name: str) -> Optional[float]:
    """Extract metric value from performance metrics object."""
    metric_mapping = {
        "total_return": performance_metric.total_return,
        "annualized_return": performance_metric.annualized_return,
        "volatility": performance_metric.volatility,
        "sharpe_ratio": performance_metric.sharpe_ratio,
        "sortino_ratio": performance_metric.sortino_ratio,
        "alpha": performance_metric.alpha,
        "beta": performance_metric.beta,
        "r_squared": performance_metric.r_squared,
        "tracking_error": performance_metric.tracking_error,
        "information_ratio": performance_metric.information_ratio
    }
    
    return metric_mapping.get(metric_name)


def _generate_standard_report(portfolio_id: str, performance_history: List[PerformanceMetrics], 
                            start_date: date, end_date: date) -> Dict[str, Any]:
    """Generate standard performance report."""
    latest_metrics = performance_history[-1] if performance_history else None
    
    if not latest_metrics:
        return {"error": "No performance data available"}
    
    return {
        "report_type": "standard",
        "portfolio_id": portfolio_id,
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": (end_date - start_date).days
        },
        "summary": {
            "total_return": latest_metrics.total_return,
            "annualized_return": latest_metrics.annualized_return,
            "volatility": latest_metrics.volatility,
            "sharpe_ratio": latest_metrics.sharpe_ratio,
            "max_drawdown": -0.12,  # Mock data
            "best_month": 0.08,     # Mock data
            "worst_month": -0.15    # Mock data
        },
        "risk_metrics": {
            "beta": latest_metrics.beta,
            "alpha": latest_metrics.alpha,
            "r_squared": latest_metrics.r_squared,
            "tracking_error": latest_metrics.tracking_error,
            "information_ratio": latest_metrics.information_ratio
        },
        "generated_at": datetime.now().isoformat()
    }


def _generate_detailed_report(portfolio_id: str, performance_history: List[PerformanceMetrics], 
                            start_date: date, end_date: date) -> Dict[str, Any]:
    """Generate detailed performance report."""
    standard_report = _generate_standard_report(portfolio_id, performance_history, start_date, end_date)
    
    # Add detailed analysis
    standard_report["detailed_analysis"] = {
        "monthly_returns": [],  # Would calculate monthly returns
        "rolling_metrics": {},  # Would calculate rolling metrics
        "drawdown_analysis": {},  # Would calculate drawdown periods
        "attribution_analysis": {}  # Would include attribution data
    }
    
    return standard_report


def _generate_summary_report(portfolio_id: str, performance_history: List[PerformanceMetrics], 
                           start_date: date, end_date: date) -> Dict[str, Any]:
    """Generate summary performance report."""
    latest_metrics = performance_history[-1] if performance_history else None
    
    if not latest_metrics:
        return {"error": "No performance data available"}
    
    return {
        "report_type": "summary",
        "portfolio_id": portfolio_id,
        "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
        "key_metrics": {
            "return": f"{latest_metrics.annualized_return:.2%}",
            "volatility": f"{latest_metrics.volatility:.2%}",
            "sharpe_ratio": f"{latest_metrics.sharpe_ratio:.2f}",
            "alpha": f"{latest_metrics.alpha:.2%}"
        },
        "generated_at": datetime.now().isoformat()
    }


def _generate_returns_chart_data(performance_history: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
    """Generate returns chart data."""
    chart_data = []
    
    for metric in performance_history:
        chart_data.append({
            "date": metric.calculation_date.isoformat(),
            "return": metric.total_return,
            "annualized_return": metric.annualized_return
        })
    
    return chart_data


def _generate_drawdown_chart_data(performance_history: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
    """Generate drawdown chart data."""
    # Mock drawdown calculation
    chart_data = []
    
    for i, metric in enumerate(performance_history):
        # Simple mock drawdown calculation
        drawdown = -0.05 * (i % 10) / 10  # Mock varying drawdown
        
        chart_data.append({
            "date": metric.calculation_date.isoformat(),
            "drawdown": drawdown,
            "underwater_curve": drawdown
        })
    
    return chart_data


def _generate_rolling_metrics_chart_data(performance_history: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
    """Generate rolling metrics chart data."""
    chart_data = []
    
    for metric in performance_history:
        chart_data.append({
            "date": metric.calculation_date.isoformat(),
            "rolling_sharpe": metric.sharpe_ratio,
            "rolling_volatility": metric.volatility,
            "rolling_alpha": metric.alpha,
            "rolling_beta": metric.beta
        })
    
    return chart_data


def _generate_risk_return_chart_data(performance_history: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
    """Generate risk-return scatter plot data."""
    chart_data = []
    
    for metric in performance_history:
        chart_data.append({
            "return": metric.annualized_return,
            "risk": metric.volatility,
            "sharpe": metric.sharpe_ratio,
            "date": metric.calculation_date.isoformat()
        })
    
    return chart_data