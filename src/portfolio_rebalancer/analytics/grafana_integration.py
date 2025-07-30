"""Grafana dashboard integration and provisioning."""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class GrafanaDashboardProvisioner:
    """Handles Grafana dashboard provisioning and configuration."""
    
    def __init__(self, grafana_url: str = "http://localhost:3000", 
                 api_key: Optional[str] = None):
        """
        Initialize Grafana dashboard provisioner.
        
        Args:
            grafana_url: Grafana server URL
            api_key: Grafana API key for authentication
        """
        self.grafana_url = grafana_url
        self.api_key = api_key
        logger.info(f"Grafana provisioner initialized for {grafana_url}")
    
    def create_portfolio_performance_dashboard(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Create a comprehensive portfolio performance dashboard.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Dashboard configuration
        """
        dashboard = {
            "dashboard": {
                "id": None,
                "title": f"Portfolio Performance - {portfolio_id}",
                "tags": ["portfolio", "performance", "analytics"],
                "timezone": "browser",
                "panels": [
                    self._create_performance_overview_panel(portfolio_id),
                    self._create_returns_chart_panel(portfolio_id),
                    self._create_risk_metrics_panel(portfolio_id),
                    self._create_benchmark_comparison_panel(portfolio_id),
                    self._create_drawdown_panel(portfolio_id),
                    self._create_rolling_metrics_panel(portfolio_id),
                    self._create_attribution_panel(portfolio_id),
                    self._create_alerts_panel(portfolio_id)
                ],
                "time": {
                    "from": "now-1y",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"]
                },
                "templating": {
                    "list": [
                        {
                            "name": "portfolio",
                            "type": "custom",
                            "options": [
                                {"text": portfolio_id, "value": portfolio_id, "selected": True}
                            ],
                            "current": {"text": portfolio_id, "value": portfolio_id}
                        },
                        {
                            "name": "benchmark",
                            "type": "custom",
                            "options": [
                                {"text": "S&P 500 (SPY)", "value": "SPY", "selected": True},
                                {"text": "Total Market (VTI)", "value": "VTI"},
                                {"text": "NASDAQ 100 (QQQ)", "value": "QQQ"}
                            ],
                            "current": {"text": "S&P 500 (SPY)", "value": "SPY"}
                        }
                    ]
                },
                "refresh": "1m",
                "schemaVersion": 30,
                "version": 1
            },
            "overwrite": True
        }
        
        return dashboard
    
    def _create_performance_overview_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create performance overview stat panel."""
        return {
            "id": 1,
            "title": "Performance Overview",
            "type": "stat",
            "targets": [
                {
                    "expr": f"portfolio_total_return{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Total Return",
                    "refId": "A"
                },
                {
                    "expr": f"portfolio_annualized_return{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Annualized Return",
                    "refId": "B"
                },
                {
                    "expr": f"portfolio_sharpe_ratio{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Sharpe Ratio",
                    "refId": "C"
                },
                {
                    "expr": f"portfolio_volatility{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Volatility",
                    "refId": "D"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "decimals": 2,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 0.05},
                            {"color": "green", "value": 0.10}
                        ]
                    }
                }
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "horizontal",
                "textMode": "value_and_name",
                "colorMode": "background"
            },
            "gridPos": {"h": 4, "w": 24, "x": 0, "y": 0}
        }
    
    def _create_returns_chart_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create returns time series chart panel."""
        return {
            "id": 2,
            "title": "Portfolio Returns vs Benchmark",
            "type": "timeseries",
            "targets": [
                {
                    "expr": f"portfolio_cumulative_return{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Portfolio",
                    "refId": "A"
                },
                {
                    "expr": f"benchmark_cumulative_return{{benchmark=\"$benchmark\"}}",
                    "legendFormat": "Benchmark ($benchmark)",
                    "refId": "B"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {"mode": "none", "group": "A"},
                        "axisPlacement": "auto",
                        "axisLabel": "",
                        "scaleDistribution": {"type": "linear"},
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False},
                        "thresholdsStyle": {"mode": "off"}
                    }
                }
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "visible", "placement": "bottom"}
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4}
        }
    
    def _create_risk_metrics_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create risk metrics gauge panel."""
        return {
            "id": 3,
            "title": "Risk Metrics",
            "type": "gauge",
            "targets": [
                {
                    "expr": f"portfolio_beta{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Beta",
                    "refId": "A"
                },
                {
                    "expr": f"portfolio_alpha{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Alpha",
                    "refId": "B"
                },
                {
                    "expr": f"portfolio_tracking_error{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Tracking Error",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": -2,
                    "max": 2,
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": -2},
                            {"color": "yellow", "value": -0.5},
                            {"color": "red", "value": 0.5}
                        ]
                    }
                }
            },
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            },
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4}
        }
    
    def _create_benchmark_comparison_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create benchmark comparison table panel."""
        return {
            "id": 4,
            "title": "Benchmark Comparison",
            "type": "table",
            "targets": [
                {
                    "expr": f"portfolio_vs_benchmark_metrics{{portfolio_id=\"{portfolio_id}\"}}",
                    "format": "table",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "align": "auto",
                        "displayMode": "auto"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Excess Return"},
                        "properties": [
                            {
                                "id": "custom.displayMode",
                                "value": "color-background"
                            },
                            {
                                "id": "thresholds",
                                "value": {
                                    "steps": [
                                        {"color": "red", "value": -0.05},
                                        {"color": "yellow", "value": 0},
                                        {"color": "green", "value": 0.02}
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
            "options": {
                "showHeader": True
            },
            "gridPos": {"h": 6, "w": 24, "x": 0, "y": 12}
        }
    
    def _create_drawdown_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create drawdown chart panel."""
        return {
            "id": 5,
            "title": "Drawdown Analysis",
            "type": "timeseries",
            "targets": [
                {
                    "expr": f"portfolio_drawdown{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Drawdown",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 2,
                        "fillOpacity": 30,
                        "gradientMode": "opacity"
                    },
                    "color": {"mode": "palette-classic"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": -0.05},
                            {"color": "red", "value": -0.10}
                        ]
                    }
                }
            },
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "visible", "placement": "bottom"}
            },
            "gridPos": {"h": 6, "w": 12, "x": 0, "y": 18}
        }
    
    def _create_rolling_metrics_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create rolling metrics panel."""
        return {
            "id": 6,
            "title": "Rolling Metrics (30-day)",
            "type": "timeseries",
            "targets": [
                {
                    "expr": f"portfolio_rolling_sharpe{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Rolling Sharpe",
                    "refId": "A"
                },
                {
                    "expr": f"portfolio_rolling_volatility{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Rolling Volatility",
                    "refId": "B"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineWidth": 1,
                        "fillOpacity": 0
                    }
                }
            },
            "gridPos": {"h": 6, "w": 12, "x": 12, "y": 18}
        }
    
    def _create_attribution_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create attribution analysis panel."""
        return {
            "id": 7,
            "title": "Performance Attribution",
            "type": "barchart",
            "targets": [
                {
                    "expr": f"portfolio_attribution_allocation{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Asset Allocation",
                    "refId": "A"
                },
                {
                    "expr": f"portfolio_attribution_selection{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Security Selection",
                    "refId": "B"
                },
                {
                    "expr": f"portfolio_attribution_interaction{{portfolio_id=\"{portfolio_id}\"}}",
                    "legendFormat": "Interaction",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "custom": {
                        "orientation": "horizontal",
                        "barWidth": 0.6,
                        "groupWidth": 0.7
                    }
                }
            },
            "gridPos": {"h": 6, "w": 12, "x": 0, "y": 24}
        }
    
    def _create_alerts_panel(self, portfolio_id: str) -> Dict[str, Any]:
        """Create alerts status panel."""
        return {
            "id": 8,
            "title": "Active Alerts",
            "type": "logs",
            "targets": [
                {
                    "expr": f"portfolio_alerts{{portfolio_id=\"{portfolio_id}\"}}",
                    "refId": "A"
                }
            ],
            "options": {
                "showTime": True,
                "showLabels": False,
                "showCommonLabels": False,
                "wrapLogMessage": False,
                "prettifyLogMessage": False,
                "enableLogDetails": True,
                "dedupStrategy": "none",
                "sortOrder": "Descending"
            },
            "gridPos": {"h": 6, "w": 12, "x": 12, "y": 24}
        }
    
    def create_data_source_config(self, analytics_api_url: str) -> Dict[str, Any]:
        """
        Create Grafana data source configuration for analytics API.
        
        Args:
            analytics_api_url: Analytics API base URL
            
        Returns:
            Data source configuration
        """
        return {
            "name": "Portfolio Analytics API",
            "type": "prometheus",  # Using Prometheus format for time series
            "url": analytics_api_url,
            "access": "proxy",
            "isDefault": False,
            "jsonData": {
                "httpMethod": "GET",
                "keepCookies": [],
                "timeInterval": "30s"
            },
            "secureJsonData": {
                "httpHeaderValue1": self.api_key if self.api_key else ""
            },
            "version": 1,
            "readOnly": False
        }
    
    def create_dashboard_provisioning_config(self, dashboards_path: str) -> Dict[str, Any]:
        """
        Create dashboard provisioning configuration.
        
        Args:
            dashboards_path: Path to dashboard JSON files
            
        Returns:
            Provisioning configuration
        """
        return {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "portfolio-analytics",
                    "orgId": 1,
                    "folder": "Portfolio Analytics",
                    "type": "file",
                    "disableDeletion": False,
                    "updateIntervalSeconds": 10,
                    "allowUiUpdates": True,
                    "options": {
                        "path": dashboards_path
                    }
                }
            ]
        }
    
    def export_dashboard_config(self, portfolio_id: str, output_path: str) -> None:
        """
        Export dashboard configuration to JSON file.
        
        Args:
            portfolio_id: Portfolio identifier
            output_path: Output file path
        """
        try:
            dashboard_config = self.create_portfolio_performance_dashboard(portfolio_id)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            logger.info(f"Dashboard configuration exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export dashboard config: {e}")
            raise
    
    def create_alert_rules(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """
        Create Grafana alert rules for portfolio monitoring.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            List of alert rule configurations
        """
        alert_rules = [
            {
                "uid": f"portfolio-drawdown-{portfolio_id}",
                "title": f"High Drawdown Alert - {portfolio_id}",
                "condition": "A",
                "data": [
                    {
                        "refId": "A",
                        "queryType": "",
                        "relativeTimeRange": {
                            "from": 600,
                            "to": 0
                        },
                        "model": {
                            "expr": f"portfolio_drawdown{{portfolio_id=\"{portfolio_id}\"}} < -0.10",
                            "interval": "",
                            "refId": "A"
                        }
                    }
                ],
                "noDataState": "NoData",
                "execErrState": "Alerting",
                "for": "5m",
                "annotations": {
                    "description": f"Portfolio {portfolio_id} has experienced a drawdown greater than 10%",
                    "summary": "High portfolio drawdown detected"
                },
                "labels": {
                    "portfolio": portfolio_id,
                    "severity": "warning"
                }
            },
            {
                "uid": f"portfolio-volatility-{portfolio_id}",
                "title": f"High Volatility Alert - {portfolio_id}",
                "condition": "A",
                "data": [
                    {
                        "refId": "A",
                        "queryType": "",
                        "relativeTimeRange": {
                            "from": 600,
                            "to": 0
                        },
                        "model": {
                            "expr": f"portfolio_volatility{{portfolio_id=\"{portfolio_id}\"}} > 0.25",
                            "interval": "",
                            "refId": "A"
                        }
                    }
                ],
                "noDataState": "NoData",
                "execErrState": "Alerting",
                "for": "10m",
                "annotations": {
                    "description": f"Portfolio {portfolio_id} volatility has exceeded 25%",
                    "summary": "High portfolio volatility detected"
                },
                "labels": {
                    "portfolio": portfolio_id,
                    "severity": "critical"
                }
            },
            {
                "uid": f"portfolio-underperformance-{portfolio_id}",
                "title": f"Underperformance Alert - {portfolio_id}",
                "condition": "A",
                "data": [
                    {
                        "refId": "A",
                        "queryType": "",
                        "relativeTimeRange": {
                            "from": 2592000,  # 30 days
                            "to": 0
                        },
                        "model": {
                            "expr": f"portfolio_excess_return{{portfolio_id=\"{portfolio_id}\"}} < -0.05",
                            "interval": "",
                            "refId": "A"
                        }
                    }
                ],
                "noDataState": "NoData",
                "execErrState": "Alerting",
                "for": "1h",
                "annotations": {
                    "description": f"Portfolio {portfolio_id} has underperformed benchmark by more than 5% over 30 days",
                    "summary": "Portfolio underperformance detected"
                },
                "labels": {
                    "portfolio": portfolio_id,
                    "severity": "warning"
                }
            }
        ]
        
        return alert_rules


def setup_grafana_integration(portfolio_ids: List[str], 
                            analytics_api_url: str,
                            grafana_config_path: str = "./monitoring/grafana") -> None:
    """
    Set up complete Grafana integration for portfolio analytics.
    
    Args:
        portfolio_ids: List of portfolio identifiers
        analytics_api_url: Analytics API base URL
        grafana_config_path: Path to Grafana configuration directory
    """
    try:
        provisioner = GrafanaDashboardProvisioner()
        
        # Create configuration directories
        config_path = Path(grafana_config_path)
        dashboards_path = config_path / "dashboards"
        provisioning_path = config_path / "provisioning"
        datasources_path = provisioning_path / "datasources"
        dashboard_provisioning_path = provisioning_path / "dashboards"
        
        for path in [dashboards_path, datasources_path, dashboard_provisioning_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Create data source configuration
        datasource_config = provisioner.create_data_source_config(analytics_api_url)
        with open(datasources_path / "analytics-datasource.yml", 'w') as f:
            json.dump({"datasources": [datasource_config]}, f, indent=2)
        
        # Create dashboard provisioning configuration
        dashboard_provisioning_config = provisioner.create_dashboard_provisioning_config(
            str(dashboards_path)
        )
        with open(dashboard_provisioning_path / "dashboards.yml", 'w') as f:
            json.dump(dashboard_provisioning_config, f, indent=2)
        
        # Create dashboards for each portfolio
        for portfolio_id in portfolio_ids:
            dashboard_file = dashboards_path / f"portfolio-{portfolio_id}-dashboard.json"
            provisioner.export_dashboard_config(portfolio_id, str(dashboard_file))
        
        logger.info(f"Grafana integration setup completed for {len(portfolio_ids)} portfolios")
        
    except Exception as e:
        logger.error(f"Failed to setup Grafana integration: {e}")
        raise