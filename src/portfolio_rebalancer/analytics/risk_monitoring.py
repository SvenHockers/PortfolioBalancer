"""Risk monitoring and alerting system."""

import logging
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from threading import Thread, Event
import time
from concurrent.futures import ThreadPoolExecutor

from .engines.risk_analysis import RiskAnalyzer, RiskLimit, RiskLimitBreach, RiskLimitType
from .models import RiskAnalysis
from ..common.interfaces import DataStorage

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Alert configuration."""
    email_enabled: bool = False
    email_recipients: List[str] = None
    webhook_enabled: bool = False
    webhook_url: str = ""
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    alert_cooldown_minutes: int = 60  # Minimum time between alerts for same breach


@dataclass
class MonitoringConfig:
    """Risk monitoring configuration."""
    portfolio_id: str
    tickers: List[str]
    weights: List[float]
    risk_limits: List[RiskLimit]
    monitoring_interval_minutes: int = 15
    alert_config: AlertConfig = None
    enabled: bool = True


@dataclass
class RiskAlert:
    """Risk alert information."""
    portfolio_id: str
    breach: RiskLimitBreach
    alert_id: str
    created_at: datetime
    sent_at: Optional[datetime] = None
    acknowledged: bool = False


class RiskMonitor:
    """Real-time risk monitoring system."""
    
    def __init__(self, data_storage: DataStorage, benchmark_ticker: str = "SPY"):
        """
        Initialize risk monitor.
        
        Args:
            data_storage: Data storage interface
            benchmark_ticker: Benchmark ticker for risk analysis
        """
        self.data_storage = data_storage
        self.risk_analyzer = RiskAnalyzer(data_storage, benchmark_ticker)
        self.monitoring_configs: Dict[str, MonitoringConfig] = {}
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.monitoring_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Risk monitor initialized")
    
    def add_portfolio_monitoring(self, config: MonitoringConfig) -> None:
        """
        Add portfolio to monitoring.
        
        Args:
            config: Monitoring configuration
        """
        try:
            self.monitoring_configs[config.portfolio_id] = config
            logger.info(f"Added portfolio {config.portfolio_id} to monitoring")
            
            # Start monitoring thread if not already running
            if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
                self.start_monitoring()
                
        except Exception as e:
            logger.error(f"Failed to add portfolio monitoring: {e}")
            raise
    
    def remove_portfolio_monitoring(self, portfolio_id: str) -> None:
        """
        Remove portfolio from monitoring.
        
        Args:
            portfolio_id: Portfolio identifier
        """
        try:
            if portfolio_id in self.monitoring_configs:
                del self.monitoring_configs[portfolio_id]
                logger.info(f"Removed portfolio {portfolio_id} from monitoring")
            
            # Clear related alerts
            self.active_alerts = {
                alert_id: alert for alert_id, alert in self.active_alerts.items()
                if alert.portfolio_id != portfolio_id
            }
            
        except Exception as e:
            logger.error(f"Failed to remove portfolio monitoring: {e}")
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        try:
            if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
                logger.warning("Monitoring thread already running")
                return
            
            self.stop_event.clear()
            self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Risk monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        try:
            self.stop_event.set()
            if self.monitoring_thread is not None:
                self.monitoring_thread.join(timeout=30)
            logger.info("Risk monitoring stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    def get_portfolio_risk_status(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Get current risk status for portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Risk status information
        """
        try:
            if portfolio_id not in self.monitoring_configs:
                raise ValueError(f"Portfolio {portfolio_id} not being monitored")
            
            config = self.monitoring_configs[portfolio_id]
            
            # Get current risk analysis
            risk_analysis = self.risk_analyzer.analyze_portfolio_risk(
                portfolio_id, config.tickers, config.weights
            )
            
            # Check for breaches
            breaches = self.risk_analyzer.monitor_risk_limits(
                portfolio_id, config.tickers, config.weights, config.risk_limits
            )
            
            # Get active alerts for this portfolio
            portfolio_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.portfolio_id == portfolio_id
            ]
            
            return {
                'portfolio_id': portfolio_id,
                'last_updated': datetime.now().isoformat(),
                'risk_analysis': risk_analysis.model_dump() if hasattr(risk_analysis, 'model_dump') else risk_analysis.__dict__,
                'breaches': [asdict(breach) for breach in breaches],
                'active_alerts': len(portfolio_alerts),
                'monitoring_enabled': config.enabled
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk status for {portfolio_id}: {e}")
            raise
    
    def get_grafana_metrics(self) -> Dict[str, Any]:
        """
        Get metrics formatted for Grafana dashboard.
        
        Returns:
            Metrics data for Grafana
        """
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'portfolios': {},
                'system': {
                    'monitored_portfolios': len(self.monitoring_configs),
                    'active_alerts': len(self.active_alerts),
                    'monitoring_status': 'running' if self._is_monitoring_active() else 'stopped'
                }
            }
            
            # Get metrics for each monitored portfolio
            for portfolio_id, config in self.monitoring_configs.items():
                try:
                    if not config.enabled:
                        continue
                    
                    risk_analysis = self.risk_analyzer.analyze_portfolio_risk(
                        portfolio_id, config.tickers, config.weights
                    )
                    
                    breaches = self.risk_analyzer.monitor_risk_limits(
                        portfolio_id, config.tickers, config.weights, config.risk_limits
                    )
                    
                    metrics['portfolios'][portfolio_id] = {
                        'var_95': abs(risk_analysis.var_95),
                        'max_drawdown': abs(risk_analysis.max_drawdown),
                        'tracking_error': risk_analysis.tracking_error,
                        'portfolio_beta': risk_analysis.portfolio_beta,
                        'concentration_risk': risk_analysis.concentration_risk,
                        'information_ratio': risk_analysis.information_ratio,
                        'breach_count': len(breaches),
                        'warning_count': len([b for b in breaches if b.severity == 'warning']),
                        'critical_count': len([b for b in breaches if b.severity == 'breach'])
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to get metrics for portfolio {portfolio_id}: {e}")
                    metrics['portfolios'][portfolio_id] = {'error': str(e)}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get Grafana metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an active alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was acknowledged
        """
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    def get_alert_history(self, 
                         portfolio_id: Optional[str] = None,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            portfolio_id: Optional portfolio filter
            hours: Number of hours to look back
            
        Returns:
            List of historical alerts
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_alerts = [
                alert for alert in self.alert_history
                if alert.created_at >= cutoff_time and
                (portfolio_id is None or alert.portfolio_id == portfolio_id)
            ]
            
            return [asdict(alert) for alert in filtered_alerts]
            
        except Exception as e:
            logger.error(f"Failed to get alert history: {e}")
            return []
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while not self.stop_event.is_set():
            try:
                # Check each monitored portfolio
                for portfolio_id, config in self.monitoring_configs.items():
                    if not config.enabled:
                        continue
                    
                    try:
                        self._check_portfolio_risks(portfolio_id, config)
                    except Exception as e:
                        logger.error(f"Error checking risks for portfolio {portfolio_id}: {e}")
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Wait for next check
                self.stop_event.wait(timeout=60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stop_event.wait(timeout=60)
        
        logger.info("Monitoring loop stopped")
    
    def _check_portfolio_risks(self, portfolio_id: str, config: MonitoringConfig) -> None:
        """Check risks for a specific portfolio."""
        try:
            # Get current breaches
            breaches = self.risk_analyzer.monitor_risk_limits(
                portfolio_id, config.tickers, config.weights, config.risk_limits
            )
            
            # Process each breach
            for breach in breaches:
                alert_key = f"{portfolio_id}_{breach.limit_type.value}"
                
                # Check if we already have an active alert for this breach
                existing_alert = None
                for alert_id, alert in self.active_alerts.items():
                    if (alert.portfolio_id == portfolio_id and 
                        alert.breach.limit_type == breach.limit_type):
                        existing_alert = alert
                        break
                
                # Check cooldown period
                if existing_alert and not self._is_alert_cooled_down(existing_alert, config):
                    continue
                
                # Create new alert
                alert = RiskAlert(
                    portfolio_id=portfolio_id,
                    breach=breach,
                    alert_id=f"{alert_key}_{int(datetime.now().timestamp())}",
                    created_at=datetime.now()
                )
                
                # Store alert
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                
                # Send alert
                if config.alert_config:
                    self.executor.submit(self._send_alert, alert, config.alert_config)
                
                logger.warning(f"Risk alert created: {alert.alert_id} - {breach.description}")
            
        except Exception as e:
            logger.error(f"Failed to check portfolio risks for {portfolio_id}: {e}")
    
    def _send_alert(self, alert: RiskAlert, alert_config: AlertConfig) -> None:
        """Send alert through configured channels."""
        try:
            # Send email alert
            if alert_config.email_enabled and alert_config.email_recipients:
                self._send_email_alert(alert, alert_config)
            
            # Send webhook alert
            if alert_config.webhook_enabled and alert_config.webhook_url:
                self._send_webhook_alert(alert, alert_config)
            
            # Send Slack alert
            if alert_config.slack_enabled and alert_config.slack_webhook_url:
                self._send_slack_alert(alert, alert_config)
            
            alert.sent_at = datetime.now()
            logger.info(f"Alert {alert.alert_id} sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send alert {alert.alert_id}: {e}")
    
    def _send_email_alert(self, alert: RiskAlert, alert_config: AlertConfig) -> None:
        """Send email alert."""
        try:
            # This is a simplified implementation
            # In production, you would configure SMTP settings and use proper email libraries
            logger.info(f"Email alert would be sent to {alert_config.email_recipients}")
            logger.info(f"Alert content: {alert.breach.description}")
            
            # Simulate email sending
            email_body = f"""
            Risk Alert: {alert.portfolio_id}
            
            Breach Type: {alert.breach.limit_type.value}
            Severity: {alert.breach.severity}
            Description: {alert.breach.description}
            Current Value: {alert.breach.current_value:.4f}
            Threshold: {alert.breach.threshold:.4f}
            Timestamp: {alert.created_at.isoformat()}
            """
            
            logger.info(f"Email content prepared: {len(email_body)} characters")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: RiskAlert, alert_config: AlertConfig) -> None:
        """Send webhook alert."""
        try:
            import requests
            
            payload = {
                'alert_id': alert.alert_id,
                'portfolio_id': alert.portfolio_id,
                'breach_type': alert.breach.limit_type.value,
                'severity': alert.breach.severity,
                'description': alert.breach.description,
                'current_value': alert.breach.current_value,
                'threshold': alert.breach.threshold,
                'timestamp': alert.created_at.isoformat()
            }
            
            response = requests.post(alert_config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent successfully to {alert_config.webhook_url}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_slack_alert(self, alert: RiskAlert, alert_config: AlertConfig) -> None:
        """Send Slack alert."""
        try:
            import requests
            
            color = "danger" if alert.breach.severity == "breach" else "warning"
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Risk Alert - {alert.portfolio_id}",
                        "text": alert.breach.description,
                        "fields": [
                            {
                                "title": "Breach Type",
                                "value": alert.breach.limit_type.value,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.breach.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": f"{alert.breach.current_value:.4f}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.breach.threshold:.4f}",
                                "short": True
                            }
                        ],
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            response = requests.post(alert_config.slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _is_alert_cooled_down(self, alert: RiskAlert, config: MonitoringConfig) -> bool:
        """Check if alert is past cooldown period."""
        if not config.alert_config:
            return True
        
        cooldown_minutes = config.alert_config.alert_cooldown_minutes
        time_since_alert = datetime.now() - alert.created_at
        
        return time_since_alert.total_seconds() >= (cooldown_minutes * 60)
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts."""
        try:
            # Remove acknowledged alerts older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            alerts_to_remove = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.acknowledged and alert.created_at < cutoff_time
            ]
            
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
            
            if alerts_to_remove:
                logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")
    
    def _is_monitoring_active(self) -> bool:
        """Check if monitoring is active."""
        return (self.monitoring_thread is not None and 
                self.monitoring_thread.is_alive() and 
                not self.stop_event.is_set())
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop_monitoring()
            self.executor.shutdown(wait=False)
        except:
            pass