"""Interactive reporting and alerting system with enhanced dashboard controls."""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import asyncio
from pathlib import Path

from .models import AnalyticsError
from .analytics_service import AnalyticsService
from .reporting import PerformanceReportGenerator, PerformanceAlertSystem, AlertSeverity, AlertCondition

logger = logging.getLogger(__name__)


class ReportSchedule(str, Enum):
    """Report scheduling options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class InteractiveChartConfig:
    """Configuration for interactive chart controls."""
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_time_selection: bool = True
    enable_crossfilter: bool = True
    enable_brush_selection: bool = True
    default_time_range: str = "1y"
    refresh_interval: str = "1m"
    auto_refresh: bool = True
    mobile_responsive: bool = True


@dataclass
class AlertConfiguration:
    """Enhanced alert configuration with thresholds."""
    portfolio_id: str
    metric: str
    warning_threshold: float
    critical_threshold: float
    condition: AlertCondition
    enabled: bool = True
    notification_emails: List[str] = None
    webhook_url: Optional[str] = None
    mobile_push: bool = False
    
    def __post_init__(self):
        if self.notification_emails is None:
            self.notification_emails = []


@dataclass
class ReportScheduleConfig:
    """Configuration for automated report scheduling."""
    portfolio_id: str
    schedule: ReportSchedule
    recipients: List[str]
    report_types: List[str] = None  # ['performance', 'risk', 'attribution']
    enabled: bool = True
    include_charts: bool = True
    format: str = "html"  # html, pdf
    custom_template: Optional[str] = None
    
    def __post_init__(self):
        if self.report_types is None:
            self.report_types = ['performance']


class InteractiveReportingSystem:
    """Enhanced reporting system with interactive controls and automation."""
    
    def __init__(self, analytics_service: AnalyticsService,
                 smtp_config: Dict[str, str] = None):
        """
        Initialize interactive reporting system.
        
        Args:
            analytics_service: Analytics service instance
            smtp_config: SMTP configuration for email reports
        """
        self.analytics_service = analytics_service
        self.smtp_config = smtp_config or {}
        
        # Initialize subsystems
        self.report_generator = PerformanceReportGenerator(analytics_service)
        self.alert_system = PerformanceAlertSystem(
            analytics_service,
            smtp_server=smtp_config.get('server'),
            smtp_port=smtp_config.get('port', 587),
            smtp_username=smtp_config.get('username'),
            smtp_password=smtp_config.get('password')
        )
        
        # Storage for configurations
        self.chart_configs: Dict[str, InteractiveChartConfig] = {}
        self.alert_configs: Dict[str, List[AlertConfiguration]] = {}
        self.schedule_configs: Dict[str, List[ReportScheduleConfig]] = {}
        
        # Background tasks
        self._scheduled_tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info("Interactive reporting system initialized")
    
    def create_interactive_dashboard_config(self, portfolio_id: str,
                                          chart_config: InteractiveChartConfig = None) -> Dict[str, Any]:
        """
        Create enhanced dashboard configuration with interactive controls.
        
        Args:
            portfolio_id: Portfolio identifier
            chart_config: Interactive chart configuration
            
        Returns:
            Enhanced dashboard configuration
        """
        if chart_config is None:
            chart_config = InteractiveChartConfig()
        
        self.chart_configs[portfolio_id] = chart_config
        
        # Enhanced dashboard with interactive features
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": f"Interactive Portfolio Analytics - {portfolio_id}",
                "tags": ["portfolio", "interactive", "analytics"],
                "timezone": "browser",
                "refresh": chart_config.refresh_interval,
                "time": {
                    "from": f"now-{chart_config.default_time_range}",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d", "90d", "1y"],
                    "nowDelay": "1m"
                },
                "templating": {
                    "list": [
                        {
                            "name": "portfolio",
                            "type": "custom",
                            "label": "Portfolio",
                            "options": [{"text": portfolio_id, "value": portfolio_id, "selected": True}],
                            "current": {"text": portfolio_id, "value": portfolio_id}
                        },
                        {
                            "name": "time_range",
                            "type": "interval",
                            "label": "Time Range",
                            "auto": True,
                            "auto_count": 30,
                            "auto_min": "1m"
                        },
                        {
                            "name": "benchmark",
                            "type": "query",
                            "label": "Benchmark",
                            "query": "label_values(benchmark_total_return, benchmark)",
                            "refresh": 1,
                            "multi": True,
                            "includeAll": True
                        }
                    ]
                },
                "panels": self._create_interactive_panels(portfolio_id, chart_config),
                "links": [
                    {
                        "title": "Export Report",
                        "url": f"/api/v1/reports/export?portfolio_id={portfolio_id}",
                        "type": "link",
                        "icon": "external link"
                    },
                    {
                        "title": "Configure Alerts",
                        "url": f"/api/v1/alerts/configure?portfolio_id={portfolio_id}",
                        "type": "link",
                        "icon": "bell"
                    }
                ],
                "annotations": {
                    "list": [
                        {
                            "name": "Rebalancing Events",
                            "datasource": "Portfolio Analytics API",
                            "enable": True,
                            "iconColor": "blue",
                            "query": f"portfolio_rebalance_events{{portfolio_id=\"{portfolio_id}\"}}"
                        },
                        {
                            "name": "Alert Events",
                            "datasource": "Portfolio Analytics API", 
                            "enable": True,
                            "iconColor": "red",
                            "query": f"portfolio_alert_events{{portfolio_id=\"{portfolio_id}\"}}"
                        }
                    ]
                }
            }
        }
        
        # Add mobile responsiveness
        if chart_config.mobile_responsive:
            dashboard_config["dashboard"]["style"] = "dark"
            dashboard_config["dashboard"]["graphTooltip"] = 1  # Shared crosshair
            
        return dashboard_config
    
    def _create_interactive_panels(self, portfolio_id: str, 
                                 chart_config: InteractiveChartConfig) -> List[Dict[str, Any]]:
        """Create interactive dashboard panels with enhanced controls."""
        panels = []
        
        # Performance overview with drill-down capability
        panels.append({
            "id": 1,
            "title": "Performance Overview - Interactive",
            "type": "stat",
            "targets": [
                {
                    "expr": f"portfolio_total_return{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Total Return",
                    "refId": "A"
                },
                {
                    "expr": f"portfolio_sharpe_ratio{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Sharpe Ratio", 
                    "refId": "B"
                },
                {
                    "expr": f"portfolio_max_drawdown{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Max Drawdown",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "decimals": 2,
                    "links": [
                        {
                            "title": "Drill Down to Details",
                            "url": f"/d/portfolio-details?var-portfolio={portfolio_id}&${{__url_time_range}}",
                            "targetBlank": True
                        }
                    ],
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": -10},
                            {"color": "yellow", "value": 0},
                            {"color": "green", "value": 5}
                        ]
                    }
                }
            },
            "options": {
                "reduceOptions": {"calcs": ["lastNotNull"]},
                "orientation": "horizontal",
                "textMode": "value_and_name",
                "colorMode": "background"
            },
            "gridPos": {"h": 4, "w": 24, "x": 0, "y": 0}
        })
        
        # Interactive time series with zoom and pan
        panels.append({
            "id": 2,
            "title": "Interactive Returns Chart",
            "type": "timeseries",
            "targets": [
                {
                    "expr": f"portfolio_cumulative_return{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Portfolio",
                    "refId": "A"
                },
                {
                    "expr": f"benchmark_cumulative_return{{benchmark=\"$benchmark\"}}",
                    "legendFormat": "{{benchmark}}",
                    "refId": "B"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "gradientMode": "opacity",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "none"},
                        "axisPlacement": "auto",
                        "scaleDistribution": {"type": "linear"},
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False},
                        "thresholdsStyle": {"mode": "off"}
                    }
                }
            },
            "options": {
                "tooltip": {"mode": "multi", "sort": "desc"},
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "values": ["min", "max", "mean", "last"]
                },
                "zoom": {"enabled": chart_config.enable_zoom},
                "pan": {"enabled": chart_config.enable_pan}
            },
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 4}
        })
        
        # Risk metrics with interactive gauges
        panels.append({
            "id": 3,
            "title": "Risk Metrics - Interactive",
            "type": "gauge",
            "targets": [
                {
                    "expr": f"portfolio_volatility{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Volatility",
                    "refId": "A"
                },
                {
                    "expr": f"portfolio_beta{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Beta",
                    "refId": "B"
                },
                {
                    "expr": f"portfolio_var_95{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "VaR 95%",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 2,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 0.15},
                            {"color": "red", "value": 0.25}
                        ]
                    }
                }
            },
            "options": {
                "reduceOptions": {"calcs": ["lastNotNull"]},
                "orientation": "auto",
                "showThresholdLabels": True,
                "showThresholdMarkers": True
            },
            "gridPos": {"h": 6, "w": 12, "x": 0, "y": 12}
        })
        
        # Alert status panel
        panels.append({
            "id": 4,
            "title": "Active Alerts",
            "type": "table",
            "targets": [
                {
                    "expr": f"portfolio_active_alerts{{portfolio_id=\"{portfolio_id}\"}}",
                    "format": "table",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "align": "left",
                        "displayMode": "color-background"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Severity"},
                        "properties": [
                            {
                                "id": "custom.displayMode",
                                "value": "color-background"
                            },
                            {
                                "id": "thresholds",
                                "value": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 1},
                                        {"color": "red", "value": 2}
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
            "options": {
                "showHeader": True,
                "sortBy": [{"desc": True, "displayName": "Triggered At"}]
            },
            "gridPos": {"h": 6, "w": 12, "x": 12, "y": 12}
        })
        
        return panels
    
    def configure_alerts(self, portfolio_id: str, 
                        alert_configs: List[AlertConfiguration]) -> None:
        """
        Configure performance and risk alerts for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            alert_configs: List of alert configurations
        """
        self.alert_configs[portfolio_id] = alert_configs
        
        # Create alerts in the alert system
        for config in alert_configs:
            # Warning alert
            warning_alert_id = self.alert_system.create_alert(
                portfolio_id=config.portfolio_id,
                metric=config.metric,
                condition=config.condition,
                threshold=config.warning_threshold,
                severity=AlertSeverity.WARNING,
                notification_emails=config.notification_emails
            )
            
            # Critical alert
            critical_alert_id = self.alert_system.create_alert(
                portfolio_id=config.portfolio_id,
                metric=config.metric,
                condition=config.condition,
                threshold=config.critical_threshold,
                severity=AlertSeverity.CRITICAL,
                notification_emails=config.notification_emails
            )
            
            logger.info(f"Configured alerts for {portfolio_id} metric {config.metric}")
    
    def schedule_automated_reports(self, portfolio_id: str,
                                 schedule_configs: List[ReportScheduleConfig]) -> None:
        """
        Schedule automated report generation and distribution.
        
        Args:
            portfolio_id: Portfolio identifier
            schedule_configs: List of report schedule configurations
        """
        self.schedule_configs[portfolio_id] = schedule_configs
        
        for config in schedule_configs:
            if config.enabled:
                self._create_scheduled_task(config)
        
        logger.info(f"Scheduled {len(schedule_configs)} automated reports for {portfolio_id}")
    
    def _create_scheduled_task(self, config: ReportScheduleConfig) -> None:
        """Create a scheduled task for automated reporting."""
        async def scheduled_report_task():
            while self._running:
                try:
                    # Calculate next run time
                    next_run = self._calculate_next_run_time(config.schedule)
                    wait_seconds = (next_run - datetime.now()).total_seconds()
                    
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                    
                    # Generate and send report
                    await self._generate_and_send_report(config)
                    
                except Exception as e:
                    logger.error(f"Scheduled report task failed: {e}")
                    await asyncio.sleep(3600)  # Wait 1 hour before retry
        
        # Only create task if event loop is running
        try:
            task = asyncio.create_task(scheduled_report_task())
            self._scheduled_tasks.append(task)
        except RuntimeError:
            # No event loop running, store task for later
            logger.warning(f"No event loop running, scheduled task for {config.portfolio_id} will be created when background tasks start")
    
    def _calculate_next_run_time(self, schedule: ReportSchedule) -> datetime:
        """Calculate the next run time for a scheduled report."""
        now = datetime.now()
        
        if schedule == ReportSchedule.DAILY:
            # Run at 8 AM daily
            next_run = now.replace(hour=8, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif schedule == ReportSchedule.WEEKLY:
            # Run on Monday at 8 AM
            days_ahead = 0 - now.weekday()  # Monday is 0
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now.replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        elif schedule == ReportSchedule.MONTHLY:
            # Run on the 1st of each month at 8 AM
            if now.day == 1 and now.hour < 8:
                next_run = now.replace(hour=8, minute=0, second=0, microsecond=0)
            else:
                if now.month == 12:
                    next_run = now.replace(year=now.year + 1, month=1, day=1, hour=8, minute=0, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month + 1, day=1, hour=8, minute=0, second=0, microsecond=0)
        elif schedule == ReportSchedule.QUARTERLY:
            # Run on the 1st of each quarter at 8 AM
            current_quarter = (now.month - 1) // 3 + 1
            next_quarter_month = current_quarter * 3 + 1
            if next_quarter_month > 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=8, minute=0, second=0, microsecond=0)
            else:
                next_run = now.replace(month=next_quarter_month, day=1, hour=8, minute=0, second=0, microsecond=0)
        else:
            next_run = now + timedelta(hours=1)  # Default fallback
        
        return next_run
    
    async def _generate_and_send_report(self, config: ReportScheduleConfig) -> None:
        """Generate and send a scheduled report."""
        try:
            logger.info(f"Generating scheduled {config.schedule.value} report for {config.portfolio_id}")
            
            # Generate reports based on configuration
            reports = {}
            
            if 'performance' in config.report_types:
                if config.schedule == ReportSchedule.DAILY:
                    reports['performance'] = self.report_generator.generate_daily_report(config.portfolio_id)
                elif config.schedule == ReportSchedule.WEEKLY:
                    reports['performance'] = self.report_generator.generate_weekly_report(config.portfolio_id)
                elif config.schedule in [ReportSchedule.MONTHLY, ReportSchedule.QUARTERLY]:
                    reports['performance'] = self.report_generator.generate_monthly_report(config.portfolio_id)
            
            # Create email content
            email_content = self._create_report_email(reports, config)
            
            # Send email to recipients
            await self._send_report_email(email_content, config.recipients, config)
            
            logger.info(f"Successfully sent scheduled report for {config.portfolio_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate and send scheduled report: {e}")
            raise
    
    def _create_report_email(self, reports: Dict[str, Any], 
                           config: ReportScheduleConfig) -> Dict[str, str]:
        """Create email content for scheduled reports."""
        subject = f"Portfolio {config.schedule.value.title()} Report - {config.portfolio_id}"
        
        # Create HTML email body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Portfolio {config.schedule.value.title()} Report</h2>
                <p><strong>Portfolio:</strong> {config.portfolio_id}</p>
                <p><strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Add performance report content
        if 'performance' in reports:
            perf_report = reports['performance']
            html_body += f"""
            <h3>Performance Summary</h3>
            <div class="metric">
                <strong>Total Return:</strong> 
                <span class="{'positive' if perf_report.get('current_metrics', {}).get('total_return', 0) > 0 else 'negative'}">
                    {perf_report.get('current_metrics', {}).get('total_return', 0):.2%}
                </span>
            </div>
            <div class="metric">
                <strong>Sharpe Ratio:</strong> {perf_report.get('current_metrics', {}).get('sharpe_ratio', 0):.2f}
            </div>
            <div class="metric">
                <strong>Volatility:</strong> {perf_report.get('current_metrics', {}).get('volatility', 0):.2%}
            </div>
            """
            
            if 'summary' in perf_report:
                html_body += f"<p><strong>Summary:</strong> {perf_report['summary']}</p>"
        
        html_body += """
        <hr>
        <p><em>This is an automated report from the Portfolio Analytics System.</em></p>
        </body>
        </html>
        """
        
        return {
            'subject': subject,
            'html_body': html_body,
            'text_body': f"Portfolio {config.schedule.value.title()} Report for {config.portfolio_id}"
        }
    
    async def _send_report_email(self, email_content: Dict[str, str], 
                               recipients: List[str], config: ReportScheduleConfig) -> None:
        """Send report email to recipients."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            if not self.smtp_config.get('server'):
                logger.warning("SMTP server not configured, skipping email send")
                return
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = email_content['subject']
            msg['From'] = self.smtp_config.get('username', 'noreply@portfolio-analytics.com')
            
            # Add text and HTML parts
            text_part = MIMEText(email_content['text_body'], 'plain')
            html_part = MIMEText(email_content['html_body'], 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config.get('port', 587)) as server:
                server.starttls()
                if self.smtp_config.get('username') and self.smtp_config.get('password'):
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                
                for recipient in recipients:
                    msg['To'] = recipient
                    server.send_message(msg)
                    del msg['To']
            
            logger.info(f"Report email sent to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send report email: {e}")
            raise
    
    def create_mobile_responsive_config(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Create mobile-responsive dashboard configuration.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Mobile-optimized dashboard configuration
        """
        mobile_config = InteractiveChartConfig(
            mobile_responsive=True,
            default_time_range="30d",  # Shorter default for mobile
            refresh_interval="5m"  # Less frequent refresh for mobile
        )
        
        dashboard_config = self.create_interactive_dashboard_config(portfolio_id, mobile_config)
        
        # Mobile-specific optimizations
        dashboard_config["dashboard"]["style"] = "dark"  # Better for mobile
        dashboard_config["dashboard"]["graphTooltip"] = 1  # Shared crosshair
        
        # Adjust panel sizes for mobile
        for panel in dashboard_config["dashboard"]["panels"]:
            # Make panels full width on mobile
            if panel["gridPos"]["w"] < 24:
                panel["gridPos"]["w"] = 24
            
            # Reduce panel heights for mobile scrolling
            if panel["gridPos"]["h"] > 6:
                panel["gridPos"]["h"] = 6
        
        return dashboard_config
    
    async def start_background_tasks(self) -> None:
        """Start background tasks for scheduled reporting and alerting."""
        self._running = True
        logger.info("Started interactive reporting background tasks")
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        self._running = False
        
        # Cancel all scheduled tasks
        for task in self._scheduled_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)
        
        self._scheduled_tasks.clear()
        logger.info("Stopped interactive reporting background tasks")
    
    def get_alert_status(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get current alert status for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Alert status information
        """
        try:
            # Check alerts
            triggered_alerts = self.alert_system.check_alerts(portfolio_id)
            
            # Get alert configurations
            configs = self.alert_configs.get(portfolio_id, [])
            
            return {
                "portfolio_id": portfolio_id,
                "total_alerts": len(configs),
                "active_alerts": len(triggered_alerts),
                "triggered_alerts": [
                    {
                        "metric": alert.metric,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "severity": alert.severity.value,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "message": alert.message
                    }
                    for alert in triggered_alerts
                ],
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert status for {portfolio_id}: {e}")
            return {"error": str(e)}
    
    def export_dashboard_config(self, portfolio_id: str, output_path: str,
                              mobile_optimized: bool = False) -> None:
        """
        Export interactive dashboard configuration to file.
        
        Args:
            portfolio_id: Portfolio identifier
            output_path: Output file path
            mobile_optimized: Whether to create mobile-optimized version
        """
        try:
            if mobile_optimized:
                config = self.create_mobile_responsive_config(portfolio_id)
            else:
                config = self.create_interactive_dashboard_config(portfolio_id)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Interactive dashboard config exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export dashboard config: {e}")
            raise