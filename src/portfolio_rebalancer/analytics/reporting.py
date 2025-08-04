"""Performance reporting and alert system."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
import base64

from .models import PerformanceMetrics, AnalyticsError
from .analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCondition(str, Enum):
    """Alert condition types."""
    ABOVE = "above"
    BELOW = "below"
    EQUALS = "equals"
    CHANGE_ABOVE = "change_above"
    CHANGE_BELOW = "change_below"


@dataclass
class PerformanceAlert:
    """Performance alert configuration."""
    id: str
    portfolio_id: str
    metric: str
    condition: AlertCondition
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    notification_emails: List[str] = None
    created_at: datetime = None
    last_triggered: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.notification_emails is None:
            self.notification_emails = []


@dataclass
class AlertTrigger:
    """Alert trigger event."""
    alert_id: str
    portfolio_id: str
    metric: str
    current_value: float
    threshold: float
    condition: AlertCondition
    severity: AlertSeverity
    triggered_at: datetime
    message: str


class PerformanceReportGenerator:
    """Generates automated performance reports."""
    
    def __init__(self, analytics_service: AnalyticsService):
        """
        Initialize report generator.
        
        Args:
            analytics_service: Analytics service instance
        """
        self.analytics_service = analytics_service
        logger.info("Performance report generator initialized")
    
    def generate_daily_report(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Generate daily performance report.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Daily performance report
        """
        try:
            logger.info(f"Generating daily report for portfolio {portfolio_id}")
            
            today = date.today()
            yesterday = today - timedelta(days=1)
            
            # Get current performance metrics
            current_metrics = self.analytics_service.get_performance_metrics(portfolio_id, today)
            previous_metrics = self.analytics_service.get_performance_metrics(portfolio_id, yesterday)
            
            if not current_metrics:
                return {"error": "No current performance data available"}
            
            # Calculate daily changes
            daily_changes = {}
            if previous_metrics:
                daily_changes = {
                    "return_change": current_metrics.total_return - previous_metrics.total_return,
                    "volatility_change": current_metrics.volatility - previous_metrics.volatility,
                    "sharpe_change": current_metrics.sharpe_ratio - previous_metrics.sharpe_ratio,
                    "alpha_change": current_metrics.alpha - previous_metrics.alpha,
                    "beta_change": current_metrics.beta - previous_metrics.beta
                }
            
            report = {
                "report_type": "daily",
                "portfolio_id": portfolio_id,
                "report_date": today.isoformat(),
                "current_metrics": {
                    "total_return": current_metrics.total_return,
                    "annualized_return": current_metrics.annualized_return,
                    "volatility": current_metrics.volatility,
                    "sharpe_ratio": current_metrics.sharpe_ratio,
                    "alpha": current_metrics.alpha,
                    "beta": current_metrics.beta,
                    "tracking_error": current_metrics.tracking_error,
                    "information_ratio": current_metrics.information_ratio
                },
                "daily_changes": daily_changes,
                "summary": self._generate_daily_summary(current_metrics, daily_changes),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            raise AnalyticsError(f"Failed to generate daily report: {e}")
    
    def generate_weekly_report(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Generate weekly performance report.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            Weekly performance report
        """
        try:
            logger.info(f"Generating weekly report for portfolio {portfolio_id}")
            
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
            
            # Get performance history
            performance_history = self.analytics_service.get_performance_history(
                portfolio_id, start_date, end_date
            )
            
            if not performance_history:
                return {"error": "No performance data available for the week"}
            
            # Calculate weekly statistics
            weekly_stats = self._calculate_weekly_statistics(performance_history)
            
            report = {
                "report_type": "weekly",
                "portfolio_id": portfolio_id,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "weekly_statistics": weekly_stats,
                "performance_trend": self._analyze_performance_trend(performance_history),
                "risk_analysis": self._analyze_weekly_risk(performance_history),
                "recommendations": self._generate_weekly_recommendations(weekly_stats),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")
            raise AnalyticsError(f"Failed to generate weekly report: {e}")
    
    def generate_monthly_report(self, portfolio_id: str, month: Optional[int] = None, 
                              year: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate monthly performance report.
        
        Args:
            portfolio_id: Portfolio identifier
            month: Month (1-12), defaults to current month
            year: Year, defaults to current year
            
        Returns:
            Monthly performance report
        """
        try:
            logger.info(f"Generating monthly report for portfolio {portfolio_id}")
            
            today = date.today()
            if month is None:
                month = today.month
            if year is None:
                year = today.year
            
            # Calculate month date range
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, month + 1, 1) - timedelta(days=1)
            
            # Get performance history
            performance_history = self.analytics_service.get_performance_history(
                portfolio_id, start_date, end_date
            )
            
            if not performance_history:
                return {"error": "No performance data available for the month"}
            
            # Calculate monthly statistics
            monthly_stats = self._calculate_monthly_statistics(performance_history)
            
            # Get attribution analysis if available
            attribution_analysis = None
            try:
                # This would require benchmark data - simplified for now
                attribution_analysis = {"message": "Attribution analysis not available"}
            except Exception:
                pass
            
            report = {
                "report_type": "monthly",
                "portfolio_id": portfolio_id,
                "period": {
                    "month": month,
                    "year": year,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "monthly_statistics": monthly_stats,
                "performance_analysis": self._analyze_monthly_performance(performance_history),
                "risk_metrics": self._calculate_monthly_risk_metrics(performance_history),
                "attribution_analysis": attribution_analysis,
                "benchmark_comparison": self._generate_benchmark_comparison(performance_history),
                "key_insights": self._generate_monthly_insights(monthly_stats),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate monthly report: {e}")
            raise AnalyticsError(f"Failed to generate monthly report: {e}")
    
    def _generate_daily_summary(self, current_metrics: PerformanceMetrics, 
                              daily_changes: Dict[str, float]) -> str:
        """Generate daily performance summary."""
        summary_parts = []
        
        # Performance summary
        if current_metrics.total_return > 0:
            summary_parts.append(f"Portfolio is up {current_metrics.total_return:.2%} overall")
        else:
            summary_parts.append(f"Portfolio is down {abs(current_metrics.total_return):.2%} overall")
        
        # Daily changes
        if daily_changes.get("return_change"):
            change = daily_changes["return_change"]
            if change > 0:
                summary_parts.append(f"gained {change:.2%} today")
            else:
                summary_parts.append(f"lost {abs(change):.2%} today")
        
        # Risk assessment
        if current_metrics.sharpe_ratio > 1.0:
            summary_parts.append("showing strong risk-adjusted returns")
        elif current_metrics.sharpe_ratio < 0:
            summary_parts.append("showing poor risk-adjusted returns")
        
        return ". ".join(summary_parts) + "."
    
    def _calculate_weekly_statistics(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate weekly performance statistics."""
        if not performance_history:
            return {}
        
        returns = [metric.total_return for metric in performance_history]
        sharpe_ratios = [metric.sharpe_ratio for metric in performance_history]
        volatilities = [metric.volatility for metric in performance_history]
        
        return {
            "avg_return": sum(returns) / len(returns),
            "return_volatility": self._calculate_std(returns),
            "avg_sharpe": sum(sharpe_ratios) / len(sharpe_ratios),
            "avg_volatility": sum(volatilities) / len(volatilities),
            "best_day_return": max(returns),
            "worst_day_return": min(returns),
            "positive_days": sum(1 for r in returns if r > 0),
            "total_days": len(returns)
        }
    
    def _calculate_monthly_statistics(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate monthly performance statistics."""
        if not performance_history:
            return {}
        
        latest = performance_history[-1]
        earliest = performance_history[0]
        
        monthly_return = latest.total_return - earliest.total_return
        
        returns = [metric.total_return for metric in performance_history]
        
        return {
            "monthly_return": monthly_return,
            "avg_daily_return": sum(returns) / len(returns),
            "return_volatility": self._calculate_std(returns),
            "max_return": max(returns),
            "min_return": min(returns),
            "final_sharpe": latest.sharpe_ratio,
            "final_volatility": latest.volatility,
            "final_alpha": latest.alpha,
            "final_beta": latest.beta,
            "trading_days": len(performance_history)
        }
    
    def _analyze_performance_trend(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance trend over the period."""
        if len(performance_history) < 2:
            return {"trend": "insufficient_data"}
        
        returns = [metric.total_return for metric in performance_history]
        
        # Simple trend analysis
        first_half = returns[:len(returns)//2]
        second_half = returns[len(returns)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.01:
            trend = "improving"
        elif second_avg < first_avg * 0.99:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "first_half_avg": first_avg,
            "second_half_avg": second_avg,
            "trend_strength": abs(second_avg - first_avg) / first_avg if first_avg != 0 else 0
        }
    
    def _analyze_weekly_risk(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze risk metrics for the week."""
        if not performance_history:
            return {}
        
        latest = performance_history[-1]
        volatilities = [metric.volatility for metric in performance_history]
        
        return {
            "current_volatility": latest.volatility,
            "avg_volatility": sum(volatilities) / len(volatilities),
            "volatility_trend": "increasing" if latest.volatility > sum(volatilities) / len(volatilities) else "decreasing",
            "risk_level": "high" if latest.volatility > 0.20 else "medium" if latest.volatility > 0.15 else "low"
        }
    
    def _generate_weekly_recommendations(self, weekly_stats: Dict[str, Any]) -> List[str]:
        """Generate weekly recommendations based on performance."""
        recommendations = []
        
        if weekly_stats.get("avg_return", 0) < 0:
            recommendations.append("Consider reviewing portfolio allocation due to negative weekly performance")
        
        if weekly_stats.get("return_volatility", 0) > 0.05:
            recommendations.append("High return volatility detected - consider risk management measures")
        
        positive_ratio = weekly_stats.get("positive_days", 0) / max(weekly_stats.get("total_days", 1), 1)
        if positive_ratio < 0.4:
            recommendations.append("Low percentage of positive days - review investment strategy")
        
        if not recommendations:
            recommendations.append("Portfolio performance is within normal parameters")
        
        return recommendations
    
    def _analyze_monthly_performance(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze monthly performance patterns."""
        if not performance_history:
            return {}
        
        returns = [metric.total_return for metric in performance_history]
        
        # Calculate performance quartiles
        sorted_returns = sorted(returns)
        n = len(sorted_returns)
        
        q1 = sorted_returns[n//4] if n >= 4 else sorted_returns[0]
        q2 = sorted_returns[n//2] if n >= 2 else sorted_returns[0]
        q3 = sorted_returns[3*n//4] if n >= 4 else sorted_returns[-1]
        
        return {
            "quartiles": {"q1": q1, "q2": q2, "q3": q3},
            "consistency": self._calculate_consistency_score(returns),
            "momentum": self._calculate_momentum_score(returns),
            "drawdown_periods": self._identify_drawdown_periods(returns)
        }
    
    def _calculate_monthly_risk_metrics(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate comprehensive monthly risk metrics."""
        if not performance_history:
            return {}
        
        latest = performance_history[-1]
        returns = [metric.total_return for metric in performance_history]
        
        # Calculate Value at Risk (simplified)
        sorted_returns = sorted(returns)
        var_95 = sorted_returns[int(0.05 * len(sorted_returns))] if len(sorted_returns) > 20 else min(returns)
        
        return {
            "value_at_risk_95": var_95,
            "max_drawdown": min(returns) - max(returns),
            "volatility": latest.volatility,
            "beta": latest.beta,
            "tracking_error": latest.tracking_error,
            "risk_adjusted_return": latest.sharpe_ratio
        }
    
    def _generate_benchmark_comparison(self, performance_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate benchmark comparison analysis."""
        if not performance_history:
            return {}
        
        latest = performance_history[-1]
        
        # Mock benchmark data - in real implementation, would fetch actual benchmark data
        benchmark_return = 0.08  # 8% benchmark return
        
        return {
            "portfolio_return": latest.total_return,
            "benchmark_return": benchmark_return,
            "excess_return": latest.total_return - benchmark_return,
            "alpha": latest.alpha,
            "beta": latest.beta,
            "r_squared": latest.r_squared,
            "tracking_error": latest.tracking_error,
            "information_ratio": latest.information_ratio,
            "outperformance": latest.total_return > benchmark_return
        }
    
    def _generate_monthly_insights(self, monthly_stats: Dict[str, Any]) -> List[str]:
        """Generate key insights for monthly report."""
        insights = []
        
        monthly_return = monthly_stats.get("monthly_return", 0)
        if monthly_return > 0.02:
            insights.append("Strong monthly performance with returns above 2%")
        elif monthly_return < -0.02:
            insights.append("Challenging month with returns below -2%")
        
        sharpe = monthly_stats.get("final_sharpe", 0)
        if sharpe > 1.0:
            insights.append("Excellent risk-adjusted returns with Sharpe ratio above 1.0")
        elif sharpe < 0:
            insights.append("Poor risk-adjusted returns - consider strategy review")
        
        volatility = monthly_stats.get("final_volatility", 0)
        if volatility > 0.25:
            insights.append("High volatility detected - portfolio may benefit from diversification")
        
        return insights
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _calculate_consistency_score(self, returns: List[float]) -> float:
        """Calculate consistency score (0-1, higher is more consistent)."""
        if len(returns) < 2:
            return 1.0
        
        volatility = self._calculate_std(returns)
        mean_return = sum(returns) / len(returns)
        
        # Consistency is inverse of coefficient of variation
        if mean_return == 0:
            return 0.0
        
        cv = abs(volatility / mean_return)
        return max(0, 1 - cv)
    
    def _calculate_momentum_score(self, returns: List[float]) -> float:
        """Calculate momentum score (-1 to 1)."""
        if len(returns) < 3:
            return 0.0
        
        # Simple momentum: compare recent performance to earlier performance
        recent = returns[-len(returns)//3:]
        earlier = returns[:len(returns)//3]
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        if earlier_avg == 0:
            return 0.0
        
        momentum = (recent_avg - earlier_avg) / abs(earlier_avg)
        return max(-1, min(1, momentum))
    
    def _identify_drawdown_periods(self, returns: List[float]) -> List[Dict[str, Any]]:
        """Identify significant drawdown periods."""
        drawdowns = []
        
        if len(returns) < 2:
            return drawdowns
        
        peak = returns[0]
        peak_idx = 0
        in_drawdown = False
        drawdown_start = 0
        
        for i, return_val in enumerate(returns):
            if return_val > peak:
                # New peak
                if in_drawdown:
                    # End of drawdown period
                    drawdowns.append({
                        "start_day": drawdown_start,
                        "end_day": i - 1,
                        "peak_value": peak,
                        "trough_value": min(returns[drawdown_start:i]),
                        "drawdown_magnitude": min(returns[drawdown_start:i]) - peak,
                        "duration_days": i - drawdown_start
                    })
                    in_drawdown = False
                
                peak = return_val
                peak_idx = i
            elif return_val < peak * 0.95:  # 5% drawdown threshold
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = peak_idx
        
        # Handle ongoing drawdown
        if in_drawdown:
            drawdowns.append({
                "start_day": drawdown_start,
                "end_day": len(returns) - 1,
                "peak_value": peak,
                "trough_value": min(returns[drawdown_start:]),
                "drawdown_magnitude": min(returns[drawdown_start:]) - peak,
                "duration_days": len(returns) - drawdown_start,
                "ongoing": True
            })
        
        return drawdowns


class PerformanceAlertSystem:
    """Performance alert monitoring and notification system."""
    
    def __init__(self, analytics_service: AnalyticsService, 
                 smtp_server: str = None, smtp_port: int = 587,
                 smtp_username: str = None, smtp_password: str = None):
        """
        Initialize alert system.
        
        Args:
            analytics_service: Analytics service instance
            smtp_server: SMTP server for email notifications
            smtp_port: SMTP port
            smtp_username: SMTP username
            smtp_password: SMTP password
        """
        self.analytics_service = analytics_service
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        
        # In-memory alert storage (in production, would use database)
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[AlertTrigger] = []
        
        logger.info("Performance alert system initialized")
    
    def create_alert(self, portfolio_id: str, metric: str, condition: AlertCondition,
                    threshold: float, severity: AlertSeverity = AlertSeverity.WARNING,
                    notification_emails: List[str] = None) -> str:
        """
        Create a new performance alert.
        
        Args:
            portfolio_id: Portfolio identifier
            metric: Metric to monitor
            condition: Alert condition
            threshold: Threshold value
            severity: Alert severity
            notification_emails: Email addresses for notifications
            
        Returns:
            Alert ID
        """
        alert_id = f"alert_{portfolio_id}_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = PerformanceAlert(
            id=alert_id,
            portfolio_id=portfolio_id,
            metric=metric,
            condition=condition,
            threshold=threshold,
            severity=severity,
            notification_emails=notification_emails or []
        )
        
        self.alerts[alert_id] = alert
        logger.info(f"Created alert {alert_id} for portfolio {portfolio_id}")
        
        return alert_id
    
    def check_alerts(self, portfolio_id: str) -> List[AlertTrigger]:
        """
        Check all alerts for a portfolio and trigger if conditions are met.
        
        Args:
            portfolio_id: Portfolio identifier
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        try:
            # Get current performance metrics
            current_metrics = self.analytics_service.get_performance_metrics(
                portfolio_id, date.today()
            )
            
            if not current_metrics:
                logger.warning(f"No performance data available for portfolio {portfolio_id}")
                return triggered_alerts
            
            # Check each alert for this portfolio
            portfolio_alerts = [alert for alert in self.alerts.values() 
                              if alert.portfolio_id == portfolio_id and alert.enabled]
            
            for alert in portfolio_alerts:
                current_value = self._get_metric_value(current_metrics, alert.metric)
                
                if current_value is None:
                    continue
                
                if self._check_alert_condition(current_value, alert.condition, alert.threshold):
                    trigger = AlertTrigger(
                        alert_id=alert.id,
                        portfolio_id=portfolio_id,
                        metric=alert.metric,
                        current_value=current_value,
                        threshold=alert.threshold,
                        condition=alert.condition,
                        severity=alert.severity,
                        triggered_at=datetime.now(),
                        message=self._generate_alert_message(alert, current_value)
                    )
                    
                    triggered_alerts.append(trigger)
                    self.alert_history.append(trigger)
                    alert.last_triggered = trigger.triggered_at
                    
                    # Send notification
                    self._send_alert_notification(trigger, alert)
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Failed to check alerts for portfolio {portfolio_id}: {e}")
            return triggered_alerts
    
    def _get_metric_value(self, metrics: PerformanceMetrics, metric_name: str) -> Optional[float]:
        """Extract metric value from performance metrics."""
        metric_mapping = {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "volatility": metrics.volatility,
            "sharpe_ratio": metrics.sharpe_ratio,
            "alpha": metrics.alpha,
            "beta": metrics.beta,
            "tracking_error": metrics.tracking_error,
            "information_ratio": metrics.information_ratio
        }
        
        return metric_mapping.get(metric_name)
    
    def _check_alert_condition(self, current_value: float, condition: AlertCondition, 
                             threshold: float) -> bool:
        """Check if alert condition is met."""
        if condition == AlertCondition.ABOVE:
            return current_value > threshold
        elif condition == AlertCondition.BELOW:
            return current_value < threshold
        elif condition == AlertCondition.EQUALS:
            return abs(current_value - threshold) < 0.001
        else:
            return False
    
    def _generate_alert_message(self, alert: PerformanceAlert, current_value: float) -> str:
        """Generate alert message."""
        return (f"Portfolio {alert.portfolio_id} {alert.metric} is {current_value:.4f}, "
                f"which is {alert.condition.value} the threshold of {alert.threshold:.4f}")
    
    def _send_alert_notification(self, trigger: AlertTrigger, alert: PerformanceAlert) -> None:
        """Send alert notification via email."""
        if not alert.notification_emails or not self.smtp_server:
            return
        
        try:
            subject = f"Portfolio Alert: {trigger.severity.value.upper()} - {trigger.portfolio_id}"
            body = self._create_alert_email_body(trigger, alert)
            
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                for email in alert.notification_emails:
                    msg['To'] = email
                    server.send_message(msg)
                    del msg['To']
            
            logger.info(f"Alert notification sent for {trigger.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    def _create_alert_email_body(self, trigger: AlertTrigger, alert: PerformanceAlert) -> str:
        """Create HTML email body for alert notification."""
        severity_colors = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107", 
            AlertSeverity.CRITICAL: "#dc3545"
        }
        
        color = severity_colors.get(trigger.severity, "#6c757d")
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h2>Portfolio Performance Alert</h2>
                    <p style="margin: 0; font-size: 18px;">{trigger.severity.value.upper()}</p>
                </div>
                
                <div style="padding: 20px; background-color: #f8f9fa;">
                    <h3>Alert Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Portfolio:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{trigger.portfolio_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Metric:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{trigger.metric}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Current Value:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{trigger.current_value:.4f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Threshold:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{trigger.threshold:.4f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Condition:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{trigger.condition.value}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Triggered At:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{trigger.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
                    </table>
                    
                    <div style="margin-top: 20px; padding: 15px; background-color: white; border-left: 4px solid {color};">
                        <p><strong>Message:</strong> {trigger.message}</p>
                    </div>
                </div>
                
                <div style="padding: 20px; text-align: center; color: #6c757d; font-size: 12px;">
                    <p>This is an automated alert from the Portfolio Analytics System.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def get_alert_history(self, portfolio_id: str = None, 
                         limit: int = 100) -> List[AlertTrigger]:
        """
        Get alert history.
        
        Args:
            portfolio_id: Filter by portfolio ID (optional)
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert triggers
        """
        history = self.alert_history
        
        if portfolio_id:
            history = [alert for alert in history if alert.portfolio_id == portfolio_id]
        
        # Sort by triggered_at descending and limit
        history.sort(key=lambda x: x.triggered_at, reverse=True)
        return history[:limit]
    
    def disable_alert(self, alert_id: str) -> bool:
        """
        Disable an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was disabled, False if not found
        """
        if alert_id in self.alerts:
            self.alerts[alert_id].enabled = False
            logger.info(f"Disabled alert {alert_id}")
            return True
        return False
    
    def delete_alert(self, alert_id: str) -> bool:
        """
        Delete an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was deleted, False if not found
        """
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Deleted alert {alert_id}")
            return True
        return False