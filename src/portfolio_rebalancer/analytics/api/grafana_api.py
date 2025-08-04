"""Grafana-compatible data source API endpoints for analytics metrics."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, Depends, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from ..models import AnalyticsError
from ..analytics_service import AnalyticsService
from .auth import get_current_active_user, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/grafana", tags=["grafana"])


def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance."""
    # This would be properly injected in a real implementation
    # For now, return None and handle in the endpoints
    return None


@router.get("/")
async def grafana_health_check():
    """Grafana data source health check endpoint."""
    return {"status": "ok", "message": "Portfolio Analytics Grafana API"}


@router.post("/search")
async def grafana_search(
    target: str = Query("", description="Search target"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: TokenData = Depends(get_current_active_user)
):
    """
    Grafana search endpoint for available metrics and portfolios.
    
    Args:
        target: Search query string
        
    Returns:
        List of available metrics/targets
    """
    try:
        # Available metric categories
        metrics = [
            # Performance metrics
            "portfolio_total_return",
            "portfolio_annualized_return", 
            "portfolio_volatility",
            "portfolio_sharpe_ratio",
            "portfolio_sortino_ratio",
            "portfolio_alpha",
            "portfolio_beta",
            "portfolio_tracking_error",
            "portfolio_information_ratio",
            "portfolio_cumulative_return",
            "portfolio_drawdown",
            "portfolio_rolling_sharpe",
            "portfolio_rolling_volatility",
            
            # Risk metrics
            "portfolio_var_95",
            "portfolio_cvar_95",
            "portfolio_max_drawdown",
            "portfolio_correlation_risk",
            "portfolio_concentration_risk",
            
            # Attribution metrics
            "portfolio_attribution_allocation",
            "portfolio_attribution_selection", 
            "portfolio_attribution_interaction",
            "portfolio_excess_return",
            
            # Dividend metrics
            "portfolio_dividend_yield",
            "portfolio_dividend_growth",
            "portfolio_income_projection",
            "portfolio_payout_ratio",
            
            # Benchmark metrics
            "benchmark_total_return",
            "benchmark_cumulative_return",
            "benchmark_volatility",
            
            # Portfolio composition
            "portfolio_allocation_drift",
            "portfolio_rebalance_events",
            
            # Alerts and monitoring
            "portfolio_alerts",
            "portfolio_risk_alerts",
            "portfolio_performance_alerts"
        ]
        
        # Filter metrics based on search target
        if target:
            filtered_metrics = [m for m in metrics if target.lower() in m.lower()]
        else:
            filtered_metrics = metrics
            
        return filtered_metrics
        
    except Exception as e:
        logger.error(f"Grafana search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.post("/query")
async def grafana_query(
    request_data: Dict[str, Any],
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: TokenData = Depends(get_current_active_user)
):
    """
    Grafana query endpoint for time series data.
    
    Args:
        request_data: Grafana query request data
        
    Returns:
        Time series data in Grafana format
    """
    try:
        targets = request_data.get("targets", [])
        range_data = request_data.get("range", {})
        interval = request_data.get("interval", "1m")
        
        # Parse time range
        start_time = datetime.fromisoformat(range_data.get("from", "").replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(range_data.get("to", "").replace("Z", "+00:00"))
        
        results = []
        
        for target in targets:
            target_expr = target.get("expr", "")
            ref_id = target.get("refId", "A")
            
            try:
                # Parse metric and labels from expression
                metric_data = _parse_prometheus_expression(target_expr)
                metric_name = metric_data["metric"]
                labels = metric_data["labels"]
                
                # Get time series data for the metric
                time_series = await _get_metric_time_series(
                    metric_name, labels, start_time, end_time, interval, analytics_service
                )
                
                if time_series:
                    results.append({
                        "target": target_expr,
                        "refId": ref_id,
                        "datapoints": time_series
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to process target {target_expr}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Grafana query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.get("/annotations")
async def grafana_annotations(
    query: str = Query("", description="Annotation query"),
    from_time: str = Query(..., alias="from", description="Start time"),
    to_time: str = Query(..., alias="to", description="End time"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: TokenData = Depends(get_current_active_user)
):
    """
    Grafana annotations endpoint for portfolio events.
    
    Args:
        query: Annotation query
        from_time: Start time
        to_time: End time
        
    Returns:
        List of annotations
    """
    try:
        start_time = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(to_time.replace("Z", "+00:00"))
        
        annotations = []
        
        # Parse query to extract portfolio_id
        portfolio_id = _extract_portfolio_id_from_query(query)
        
        if portfolio_id and analytics_service:
            # Get rebalancing events
            rebalance_events = await _get_rebalance_events(
                portfolio_id, start_time, end_time, analytics_service
            )
            
            for event in rebalance_events:
                annotations.append({
                    "time": int(event["timestamp"].timestamp() * 1000),
                    "title": "Portfolio Rebalanced",
                    "text": event.get("description", "Portfolio rebalancing executed"),
                    "tags": ["rebalance", portfolio_id]
                })
            
            # Get alert events
            alert_events = await _get_alert_events(
                portfolio_id, start_time, end_time, analytics_service
            )
            
            for event in alert_events:
                annotations.append({
                    "time": int(event["timestamp"].timestamp() * 1000),
                    "title": f"Alert: {event['severity'].upper()}",
                    "text": event.get("message", "Performance alert triggered"),
                    "tags": ["alert", event["severity"], portfolio_id]
                })
        
        return annotations
        
    except Exception as e:
        logger.error(f"Grafana annotations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Annotations failed: {e}")


@router.get("/tag-keys")
async def grafana_tag_keys(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: TokenData = Depends(get_current_active_user)
):
    """
    Grafana tag keys endpoint for available label keys.
    
    Returns:
        List of available tag keys
    """
    try:
        tag_keys = [
            {"type": "string", "text": "portfolio_id"},
            {"type": "string", "text": "benchmark"},
            {"type": "string", "text": "strategy"},
            {"type": "string", "text": "asset_class"},
            {"type": "string", "text": "sector"},
            {"type": "string", "text": "region"}
        ]
        
        return tag_keys
        
    except Exception as e:
        logger.error(f"Grafana tag keys failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tag keys failed: {e}")


@router.get("/tag-values")
async def grafana_tag_values(
    key: str = Query(..., description="Tag key"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    current_user: TokenData = Depends(get_current_active_user)
):
    """
    Grafana tag values endpoint for available label values.
    
    Args:
        key: Tag key to get values for
        
    Returns:
        List of available tag values
    """
    try:
        values = []
        
        if key == "portfolio_id":
            # Get available portfolio IDs
            if analytics_service:
                portfolios = await _get_available_portfolios(analytics_service)
                values = [{"text": p} for p in portfolios]
            else:
                values = [{"text": "demo-portfolio"}]
                
        elif key == "benchmark":
            values = [
                {"text": "SPY"},
                {"text": "VTI"}, 
                {"text": "QQQ"},
                {"text": "BND"},
                {"text": "VEA"},
                {"text": "VWO"}
            ]
            
        elif key == "strategy":
            values = [
                {"text": "sharpe_optimization"},
                {"text": "minimum_variance"},
                {"text": "equal_weight"},
                {"text": "risk_parity"},
                {"text": "maximum_diversification"}
            ]
            
        elif key == "asset_class":
            values = [
                {"text": "equity"},
                {"text": "fixed_income"},
                {"text": "commodity"},
                {"text": "real_estate"},
                {"text": "alternative"}
            ]
            
        return values
        
    except Exception as e:
        logger.error(f"Grafana tag values failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tag values failed: {e}")


async def _get_metric_time_series(
    metric_name: str,
    labels: Dict[str, str],
    start_time: datetime,
    end_time: datetime,
    interval: str,
    analytics_service: Optional[AnalyticsService]
) -> List[List[float]]:
    """
    Get time series data for a specific metric.
    
    Args:
        metric_name: Name of the metric
        labels: Metric labels/filters
        start_time: Start time
        end_time: End time
        interval: Data interval
        analytics_service: Analytics service instance
        
    Returns:
        List of [value, timestamp] pairs
    """
    try:
        portfolio_id = labels.get("portfolio_id", "demo-portfolio")
        
        # Generate mock time series data for now
        # In a real implementation, this would query the analytics service
        time_points = pd.date_range(start=start_time, end=end_time, freq=interval)
        
        if metric_name == "portfolio_total_return":
            # Mock cumulative return data
            base_return = 0.08  # 8% annual return
            daily_return = base_return / 252
            values = [(i * daily_return + 0.01 * (i % 10 - 5) / 10, 
                      int(ts.timestamp() * 1000)) for i, ts in enumerate(time_points)]
                      
        elif metric_name == "portfolio_volatility":
            # Mock volatility data
            base_vol = 0.15  # 15% volatility
            values = [(base_vol + 0.02 * (i % 20 - 10) / 10, 
                      int(ts.timestamp() * 1000)) for i, ts in enumerate(time_points)]
                      
        elif metric_name == "portfolio_sharpe_ratio":
            # Mock Sharpe ratio data
            base_sharpe = 1.2
            values = [(base_sharpe + 0.3 * (i % 15 - 7) / 7, 
                      int(ts.timestamp() * 1000)) for i, ts in enumerate(time_points)]
                      
        elif metric_name == "portfolio_drawdown":
            # Mock drawdown data (negative values)
            values = [(-0.05 * abs((i % 30 - 15) / 15), 
                      int(ts.timestamp() * 1000)) for i, ts in enumerate(time_points)]
                      
        elif metric_name == "portfolio_beta":
            # Mock beta data
            base_beta = 1.0
            values = [(base_beta + 0.2 * (i % 25 - 12) / 12, 
                      int(ts.timestamp() * 1000)) for i, ts in enumerate(time_points)]
                      
        else:
            # Default mock data
            values = [(0.5 + 0.1 * (i % 10 - 5) / 5, 
                      int(ts.timestamp() * 1000)) for i, ts in enumerate(time_points)]
        
        return values
        
    except Exception as e:
        logger.error(f"Failed to get time series for {metric_name}: {e}")
        return []


def _parse_prometheus_expression(expr: str) -> Dict[str, Any]:
    """
    Parse a Prometheus-style expression to extract metric name and labels.
    
    Args:
        expr: Prometheus expression (e.g., 'metric_name{label1="value1"}')
        
    Returns:
        Dictionary with metric name and labels
    """
    try:
        # Simple parsing for basic expressions
        if "{" in expr:
            metric_name = expr.split("{")[0]
            labels_str = expr.split("{")[1].split("}")[0]
            
            labels = {}
            if labels_str:
                # Parse labels like: label1="value1",label2="value2"
                for label_pair in labels_str.split(","):
                    if "=" in label_pair:
                        key, value = label_pair.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"')
                        labels[key] = value
        else:
            metric_name = expr
            labels = {}
            
        return {"metric": metric_name, "labels": labels}
        
    except Exception as e:
        logger.warning(f"Failed to parse expression {expr}: {e}")
        return {"metric": expr, "labels": {}}


def _extract_portfolio_id_from_query(query: str) -> Optional[str]:
    """Extract portfolio_id from annotation query."""
    try:
        if "portfolio_id=" in query:
            # Extract portfolio_id from query like: portfolio_rebalance_events{portfolio_id="demo"}
            start = query.find('portfolio_id="') + len('portfolio_id="')
            end = query.find('"', start)
            return query[start:end]
        return None
    except Exception:
        return None


async def _get_rebalance_events(
    portfolio_id: str,
    start_time: datetime,
    end_time: datetime,
    analytics_service: AnalyticsService
) -> List[Dict[str, Any]]:
    """Get portfolio rebalancing events for annotations."""
    try:
        # Mock rebalancing events
        events = []
        current_time = start_time
        
        while current_time < end_time:
            # Add monthly rebalancing events
            events.append({
                "timestamp": current_time,
                "description": f"Monthly rebalancing executed for {portfolio_id}",
                "portfolio_id": portfolio_id
            })
            current_time += timedelta(days=30)
            
        return events
        
    except Exception as e:
        logger.error(f"Failed to get rebalance events: {e}")
        return []


async def _get_alert_events(
    portfolio_id: str,
    start_time: datetime,
    end_time: datetime,
    analytics_service: AnalyticsService
) -> List[Dict[str, Any]]:
    """Get portfolio alert events for annotations."""
    try:
        # Mock alert events - in real implementation would query alert system
        events = []
        current_time = start_time
        
        while current_time < end_time:
            # Add some mock alert events
            if current_time.day % 7 == 0:  # Weekly alerts
                events.append({
                    "timestamp": current_time,
                    "severity": "warning",
                    "message": f"High volatility detected for {portfolio_id}",
                    "portfolio_id": portfolio_id
                })
            current_time += timedelta(days=1)
            
        return events
        
    except Exception as e:
        logger.error(f"Failed to get alert events: {e}")
        return []


async def _get_available_portfolios(analytics_service: AnalyticsService) -> List[str]:
    """Get list of available portfolio IDs."""
    try:
        # In a real implementation, this would query the analytics service
        return ["demo-portfolio", "conservative-portfolio", "aggressive-portfolio"]
    except Exception as e:
        logger.error(f"Failed to get available portfolios: {e}")
        return ["demo-portfolio"]