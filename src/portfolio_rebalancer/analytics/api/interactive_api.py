"""API endpoints for interactive reporting and alerting features."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Query, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models import AnalyticsError
from ..analytics_service import AnalyticsService
from ..interactive_reporting import (
    InteractiveReportingSystem, 
    InteractiveChartConfig,
    AlertConfiguration,
    ReportScheduleConfig,
    ReportSchedule,
    AlertCondition
)
from .auth import get_current_active_user, TokenData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/interactive", tags=["interactive"])


# Pydantic models for request/response
class ChartConfigRequest(BaseModel):
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_time_selection: bool = True
    enable_crossfilter: bool = True
    enable_brush_selection: bool = True
    default_time_range: str = "1y"
    refresh_interval: str = "1m"
    auto_refresh: bool = True
    mobile_responsive: bool = True


class AlertConfigRequest(BaseModel):
    metric: str
    warning_threshold: float
    critical_threshold: float
    condition: str  # "above", "below", "equals"
    notification_emails: List[str] = []
    webhook_url: Optional[str] = None
    mobile_push: bool = False


class ReportScheduleRequest(BaseModel):
    schedule: str  # "daily", "weekly", "monthly", "quarterly"
    recipients: List[str]
    report_types: List[str] = ["performance"]
    include_charts: bool = True
    format: str = "html"
    custom_template: Optional[str] = None


def get_interactive_reporting_system() -> InteractiveReportingSystem:
    """Get interactive reporting system instance."""
    # This would be properly injected in a real implementation
    # For now, return None and handle in the endpoints
    return None


@router.get("/dashboard/{portfolio_id}")
async def get_interactive_dashboard(
    portfolio_id: str,
    mobile: bool = Query(False, description="Generate mobile-optimized dashboard"),
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Get interactive dashboard configuration for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        mobile: Whether to generate mobile-optimized version
        
    Returns:
        Interactive dashboard configuration
    """
    try:
        if not reporting_system:
            # Mock response for demonstration
            return {
                "dashboard": {
                    "title": f"Interactive Portfolio Analytics - {portfolio_id}",
                    "mobile_optimized": mobile,
                    "features": {
                        "zoom": True,
                        "pan": True,
                        "time_selection": True,
                        "crossfilter": True,
                        "alerts": True,
                        "export": True
                    }
                }
            }
        
        if mobile:
            config = reporting_system.create_mobile_responsive_config(portfolio_id)
        else:
            config = reporting_system.create_interactive_dashboard_config(portfolio_id)
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get interactive dashboard for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {e}")


@router.post("/dashboard/{portfolio_id}/configure")
async def configure_interactive_dashboard(
    portfolio_id: str,
    config: ChartConfigRequest,
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Configure interactive dashboard settings for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        config: Chart configuration settings
        
    Returns:
        Updated dashboard configuration
    """
    try:
        chart_config = InteractiveChartConfig(
            enable_zoom=config.enable_zoom,
            enable_pan=config.enable_pan,
            enable_time_selection=config.enable_time_selection,
            enable_crossfilter=config.enable_crossfilter,
            enable_brush_selection=config.enable_brush_selection,
            default_time_range=config.default_time_range,
            refresh_interval=config.refresh_interval,
            auto_refresh=config.auto_refresh,
            mobile_responsive=config.mobile_responsive
        )
        
        if reporting_system:
            dashboard_config = reporting_system.create_interactive_dashboard_config(
                portfolio_id, chart_config
            )
            return dashboard_config
        else:
            # Mock response
            return {
                "status": "configured",
                "portfolio_id": portfolio_id,
                "config": config.dict()
            }
        
    except Exception as e:
        logger.error(f"Failed to configure dashboard for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {e}")


@router.get("/alerts/{portfolio_id}")
async def get_alert_status(
    portfolio_id: str,
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Get current alert status for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        
    Returns:
        Alert status information
    """
    try:
        if reporting_system:
            return reporting_system.get_alert_status(portfolio_id)
        else:
            # Mock response
            return {
                "portfolio_id": portfolio_id,
                "total_alerts": 3,
                "active_alerts": 1,
                "triggered_alerts": [
                    {
                        "metric": "volatility",
                        "current_value": 0.18,
                        "threshold": 0.15,
                        "severity": "warning",
                        "triggered_at": datetime.now().isoformat(),
                        "message": "Portfolio volatility above warning threshold"
                    }
                ],
                "last_check": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Failed to get alert status for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Alert status retrieval failed: {e}")


@router.post("/alerts/{portfolio_id}/configure")
async def configure_alerts(
    portfolio_id: str,
    alerts: List[AlertConfigRequest],
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Configure performance and risk alerts for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        alerts: List of alert configurations
        
    Returns:
        Configuration status
    """
    try:
        alert_configs = []
        
        for alert_req in alerts:
            # Convert string condition to enum
            try:
                condition = AlertCondition(alert_req.condition.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid condition: {alert_req.condition}")
            
            alert_config = AlertConfiguration(
                portfolio_id=portfolio_id,
                metric=alert_req.metric,
                warning_threshold=alert_req.warning_threshold,
                critical_threshold=alert_req.critical_threshold,
                condition=condition,
                notification_emails=alert_req.notification_emails,
                webhook_url=alert_req.webhook_url,
                mobile_push=alert_req.mobile_push
            )
            alert_configs.append(alert_config)
        
        if reporting_system:
            reporting_system.configure_alerts(portfolio_id, alert_configs)
        
        return {
            "status": "configured",
            "portfolio_id": portfolio_id,
            "alerts_configured": len(alert_configs),
            "configuration_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to configure alerts for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Alert configuration failed: {e}")


@router.post("/reports/{portfolio_id}/schedule")
async def schedule_automated_reports(
    portfolio_id: str,
    schedules: List[ReportScheduleRequest],
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Schedule automated report generation and distribution.
    
    Args:
        portfolio_id: Portfolio identifier
        schedules: List of report schedule configurations
        
    Returns:
        Scheduling status
    """
    try:
        schedule_configs = []
        
        for schedule_req in schedules:
            # Convert string schedule to enum
            try:
                schedule = ReportSchedule(schedule_req.schedule.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid schedule: {schedule_req.schedule}")
            
            schedule_config = ReportScheduleConfig(
                portfolio_id=portfolio_id,
                schedule=schedule,
                recipients=schedule_req.recipients,
                report_types=schedule_req.report_types,
                include_charts=schedule_req.include_charts,
                format=schedule_req.format,
                custom_template=schedule_req.custom_template
            )
            schedule_configs.append(schedule_config)
        
        if reporting_system:
            reporting_system.schedule_automated_reports(portfolio_id, schedule_configs)
        
        return {
            "status": "scheduled",
            "portfolio_id": portfolio_id,
            "schedules_configured": len(schedule_configs),
            "next_reports": [
                {
                    "schedule": config.schedule.value,
                    "recipients": len(config.recipients),
                    "report_types": config.report_types
                }
                for config in schedule_configs
            ],
            "configuration_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to schedule reports for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Report scheduling failed: {e}")


@router.get("/reports/{portfolio_id}/schedules")
async def get_report_schedules(
    portfolio_id: str,
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Get current report schedules for a portfolio.
    
    Args:
        portfolio_id: Portfolio identifier
        
    Returns:
        List of configured report schedules
    """
    try:
        if reporting_system and portfolio_id in reporting_system.schedule_configs:
            configs = reporting_system.schedule_configs[portfolio_id]
            return {
                "portfolio_id": portfolio_id,
                "schedules": [
                    {
                        "schedule": config.schedule.value,
                        "recipients": config.recipients,
                        "report_types": config.report_types,
                        "enabled": config.enabled,
                        "format": config.format
                    }
                    for config in configs
                ]
            }
        else:
            # Mock response
            return {
                "portfolio_id": portfolio_id,
                "schedules": [
                    {
                        "schedule": "daily",
                        "recipients": ["user@example.com"],
                        "report_types": ["performance"],
                        "enabled": True,
                        "format": "html"
                    }
                ]
            }
        
    except Exception as e:
        logger.error(f"Failed to get report schedules for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Schedule retrieval failed: {e}")


@router.post("/reports/{portfolio_id}/generate")
async def generate_interactive_report(
    portfolio_id: str,
    report_types: List[str] = Body(["performance"], description="Types of reports to generate"),
    include_charts: bool = Body(True, description="Include charts in report"),
    format: str = Body("html", description="Report format (html, pdf)"),
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Generate an interactive report on-demand.
    
    Args:
        portfolio_id: Portfolio identifier
        report_types: Types of reports to generate
        include_charts: Whether to include charts
        format: Report format
        
    Returns:
        Generated report data
    """
    try:
        if not reporting_system:
            # Mock response
            return {
                "portfolio_id": portfolio_id,
                "report_types": report_types,
                "format": format,
                "generated_at": datetime.now().isoformat(),
                "status": "generated",
                "download_url": f"/api/v1/reports/{portfolio_id}/download"
            }
        
        # Generate report using the reporting system
        # This would typically generate and return the actual report
        report_data = {
            "portfolio_id": portfolio_id,
            "report_types": report_types,
            "include_charts": include_charts,
            "format": format,
            "generated_at": datetime.now().isoformat(),
            "status": "generated"
        }
        
        return report_data
        
    except Exception as e:
        logger.error(f"Failed to generate report for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@router.get("/dashboard/{portfolio_id}/export")
async def export_dashboard_config(
    portfolio_id: str,
    mobile: bool = Query(False, description="Export mobile-optimized version"),
    current_user: TokenData = Depends(get_current_active_user),
    reporting_system: InteractiveReportingSystem = Depends(get_interactive_reporting_system)
):
    """
    Export interactive dashboard configuration.
    
    Args:
        portfolio_id: Portfolio identifier
        mobile: Whether to export mobile-optimized version
        
    Returns:
        Dashboard configuration for export
    """
    try:
        if reporting_system:
            if mobile:
                config = reporting_system.create_mobile_responsive_config(portfolio_id)
            else:
                config = reporting_system.create_interactive_dashboard_config(portfolio_id)
            
            return {
                "portfolio_id": portfolio_id,
                "mobile_optimized": mobile,
                "config": config,
                "export_time": datetime.now().isoformat()
            }
        else:
            # Mock response
            return {
                "portfolio_id": portfolio_id,
                "mobile_optimized": mobile,
                "config": {
                    "dashboard": {
                        "title": f"Interactive Portfolio Analytics - {portfolio_id}",
                        "mobile_optimized": mobile
                    }
                },
                "export_time": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Failed to export dashboard config for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard export failed: {e}")


@router.get("/health")
async def interactive_health_check():
    """Health check endpoint for interactive reporting system."""
    return {
        "status": "healthy",
        "service": "interactive_reporting",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "interactive_dashboards": True,
            "automated_alerts": True,
            "scheduled_reports": True,
            "mobile_responsive": True,
            "real_time_updates": True
        }
    }