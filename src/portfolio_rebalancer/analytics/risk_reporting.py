"""Risk reporting and PDF generation module."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np
from io import BytesIO
import base64

from .engines.risk_analysis import RiskAnalyzer, FactorExposure, TailRiskMetrics, GeographicExposure
from .models import RiskAnalysis
from ..common.interfaces import DataStorage

logger = logging.getLogger(__name__)


class RiskReportGenerator:
    """Generate comprehensive risk reports with PDF export."""
    
    def __init__(self, data_storage: DataStorage, benchmark_ticker: str = "SPY"):
        """
        Initialize risk report generator.
        
        Args:
            data_storage: Data storage interface
            benchmark_ticker: Benchmark ticker for analysis
        """
        self.data_storage = data_storage
        self.risk_analyzer = RiskAnalyzer(data_storage, benchmark_ticker)
        logger.info("Risk report generator initialized")
    
    def generate_comprehensive_report(self, 
                                    portfolio_id: str,
                                    tickers: List[str], 
                                    weights: List[float],
                                    lookback_days: int = 252,
                                    include_charts: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            lookback_days: Number of days for analysis
            include_charts: Whether to include charts in report
            
        Returns:
            Comprehensive risk report data
        """
        try:
            logger.info(f"Generating comprehensive risk report for portfolio {portfolio_id}")
            
            # Basic risk analysis
            risk_analysis = self.risk_analyzer.analyze_portfolio_risk(
                portfolio_id, tickers, weights, lookback_days
            )
            
            # Enhanced factor exposure
            factor_exposure = self.risk_analyzer.calculate_enhanced_factor_exposure(
                tickers, weights, lookback_days
            )
            
            # Tail risk metrics
            tail_risk = self.risk_analyzer.calculate_tail_risk_metrics(
                tickers, weights, lookback_days
            )
            
            # Geographic exposure
            geographic_exposure = self.risk_analyzer.calculate_geographic_exposure(
                tickers, weights
            )
            
            # Rolling risk metrics
            rolling_metrics = self.risk_analyzer.calculate_rolling_risk_metrics(
                tickers, weights, window_days=63, lookback_days=lookback_days
            )
            
            # Beta analysis
            beta_analysis = self.risk_analyzer.calculate_beta_analysis(
                tickers, weights, lookback_days
            )
            
            # Tracking error analysis
            tracking_error_analysis = self.risk_analyzer.calculate_tracking_error_analysis(
                tickers, weights, lookback_days
            )
            
            # Compile report
            report = {
                'metadata': {
                    'portfolio_id': portfolio_id,
                    'report_date': datetime.now().isoformat(),
                    'analysis_period_days': lookback_days,
                    'tickers': tickers,
                    'weights': weights
                },
                'executive_summary': self._generate_executive_summary(
                    risk_analysis, factor_exposure, tail_risk
                ),
                'risk_analysis': self._format_risk_analysis(risk_analysis),
                'factor_exposure': self._format_factor_exposure(factor_exposure),
                'tail_risk_metrics': self._format_tail_risk_metrics(tail_risk),
                'geographic_exposure': self._format_geographic_exposure(geographic_exposure),
                'beta_analysis': beta_analysis,
                'tracking_error_analysis': tracking_error_analysis,
                'rolling_metrics_summary': self._summarize_rolling_metrics(rolling_metrics),
                'risk_recommendations': self._generate_risk_recommendations(
                    risk_analysis, factor_exposure, tail_risk, geographic_exposure
                )
            }
            
            # Add charts if requested
            if include_charts:
                report['charts'] = self._generate_chart_data(
                    rolling_metrics, risk_analysis, factor_exposure
                )
            
            logger.info(f"Risk report generated successfully for portfolio {portfolio_id}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate risk report: {e}")
            raise
    
    def export_to_pdf(self, report_data: Dict[str, Any]) -> bytes:
        """
        Export risk report to PDF.
        
        Args:
            report_data: Report data from generate_comprehensive_report
            
        Returns:
            PDF bytes
        """
        try:
            logger.info("Exporting risk report to PDF")
            
            # This is a simplified implementation
            # In production, you would use a proper PDF library like reportlab
            html_content = self._generate_html_report(report_data)
            
            # For now, return HTML as bytes (in production, convert to PDF)
            pdf_bytes = html_content.encode('utf-8')
            
            logger.info("PDF export completed")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")
            raise
    
    def _generate_executive_summary(self, 
                                  risk_analysis: RiskAnalysis,
                                  factor_exposure: FactorExposure,
                                  tail_risk: TailRiskMetrics) -> Dict[str, Any]:
        """Generate executive summary."""
        try:
            # Risk level assessment
            risk_level = "Low"
            if abs(risk_analysis.var_95) > 0.05 or abs(risk_analysis.max_drawdown) > 0.20:
                risk_level = "High"
            elif abs(risk_analysis.var_95) > 0.03 or abs(risk_analysis.max_drawdown) > 0.15:
                risk_level = "Medium"
            
            # Key risk factors
            key_risks = []
            if abs(risk_analysis.var_95) > 0.05:
                key_risks.append("High Value at Risk")
            if abs(risk_analysis.max_drawdown) > 0.20:
                key_risks.append("Significant Maximum Drawdown")
            if risk_analysis.concentration_risk > 0.5:
                key_risks.append("High Concentration Risk")
            if risk_analysis.tracking_error > 0.10:
                key_risks.append("High Tracking Error")
            
            # Factor exposures summary
            dominant_factors = []
            factor_dict = {
                'Size': factor_exposure.size_factor,
                'Value': factor_exposure.value_factor,
                'Growth': factor_exposure.growth_factor,
                'Momentum': factor_exposure.momentum_factor,
                'Quality': factor_exposure.quality_factor,
                'Low Volatility': factor_exposure.low_volatility_factor
            }
            
            for factor, exposure in factor_dict.items():
                if abs(exposure) > 0.3:
                    dominant_factors.append(f"{factor} ({exposure:.2f})")
            
            return {
                'overall_risk_level': risk_level,
                'key_risk_factors': key_risks,
                'dominant_factor_exposures': dominant_factors,
                'var_95_percent': abs(risk_analysis.var_95) * 100,
                'max_drawdown_percent': abs(risk_analysis.max_drawdown) * 100,
                'sharpe_equivalent': risk_analysis.information_ratio,
                'tail_risk_score': min(max(tail_risk.sortino_ratio, 0), 3),
                'diversification_score': risk_analysis.correlation_data.get('diversification_ratio', 1.0) if risk_analysis.correlation_data else 1.0
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate executive summary: {e}")
            return {'error': str(e)}
    
    def _format_risk_analysis(self, risk_analysis: RiskAnalysis) -> Dict[str, Any]:
        """Format risk analysis for report."""
        return {
            'portfolio_beta': risk_analysis.portfolio_beta,
            'tracking_error': risk_analysis.tracking_error,
            'information_ratio': risk_analysis.information_ratio,
            'var_95_percent': abs(risk_analysis.var_95) * 100,
            'cvar_95_percent': abs(risk_analysis.cvar_95) * 100,
            'max_drawdown_percent': abs(risk_analysis.max_drawdown) * 100,
            'concentration_risk': risk_analysis.concentration_risk,
            'correlation_summary': risk_analysis.correlation_data or {},
            'sector_exposures': risk_analysis.sector_exposures or {}
        }
    
    def _format_factor_exposure(self, factor_exposure: FactorExposure) -> Dict[str, Any]:
        """Format factor exposure for report."""
        return {
            'size_factor': factor_exposure.size_factor,
            'value_factor': factor_exposure.value_factor,
            'growth_factor': factor_exposure.growth_factor,
            'momentum_factor': factor_exposure.momentum_factor,
            'quality_factor': factor_exposure.quality_factor,
            'low_volatility_factor': factor_exposure.low_volatility_factor,
            'profitability_factor': factor_exposure.profitability_factor,
            'investment_factor': factor_exposure.investment_factor,
            'factor_summary': self._summarize_factor_exposure(factor_exposure)
        }
    
    def _format_tail_risk_metrics(self, tail_risk: TailRiskMetrics) -> Dict[str, Any]:
        """Format tail risk metrics for report."""
        return {
            'max_drawdown_percent': abs(tail_risk.max_drawdown) * 100,
            'max_drawdown_duration_days': tail_risk.max_drawdown_duration,
            'recovery_time_days': tail_risk.recovery_time,
            'tail_ratio': tail_risk.tail_ratio,
            'downside_deviation': tail_risk.downside_deviation,
            'sortino_ratio': tail_risk.sortino_ratio,
            'worst_month_percent': tail_risk.worst_month * 100,
            'worst_quarter_percent': tail_risk.worst_quarter * 100,
            'worst_year_percent': tail_risk.worst_year * 100,
            'skewness': tail_risk.skewness,
            'kurtosis': tail_risk.kurtosis,
            'tail_risk_assessment': self._assess_tail_risk(tail_risk)
        }
    
    def _format_geographic_exposure(self, geo_exposure: GeographicExposure) -> Dict[str, Any]:
        """Format geographic exposure for report."""
        return {
            'domestic_exposure_percent': geo_exposure.domestic_exposure * 100,
            'developed_markets_percent': geo_exposure.developed_markets_exposure * 100,
            'emerging_markets_percent': geo_exposure.emerging_markets_exposure * 100,
            'regional_breakdown': {
                region: weight * 100 for region, weight in geo_exposure.regional_breakdown.items()
            },
            'currency_exposure': {
                currency: weight * 100 for currency, weight in geo_exposure.currency_exposure.items()
            },
            'geographic_diversification': len(geo_exposure.regional_breakdown)
        }
    
    def _summarize_rolling_metrics(self, rolling_metrics: pd.DataFrame) -> Dict[str, Any]:
        """Summarize rolling metrics."""
        try:
            if rolling_metrics.empty:
                return {'error': 'No rolling metrics data available'}
            
            return {
                'volatility': {
                    'current': rolling_metrics['volatility'].iloc[-1],
                    'average': rolling_metrics['volatility'].mean(),
                    'min': rolling_metrics['volatility'].min(),
                    'max': rolling_metrics['volatility'].max(),
                    'trend': 'increasing' if rolling_metrics['volatility'].iloc[-1] > rolling_metrics['volatility'].mean() else 'decreasing'
                },
                'beta': {
                    'current': rolling_metrics['beta'].iloc[-1],
                    'average': rolling_metrics['beta'].mean(),
                    'stability': rolling_metrics['beta'].std()
                },
                'tracking_error': {
                    'current': rolling_metrics['tracking_error'].iloc[-1],
                    'average': rolling_metrics['tracking_error'].mean(),
                    'trend': 'increasing' if rolling_metrics['tracking_error'].iloc[-1] > rolling_metrics['tracking_error'].mean() else 'decreasing'
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to summarize rolling metrics: {e}")
            return {'error': str(e)}
    
    def _generate_risk_recommendations(self, 
                                     risk_analysis: RiskAnalysis,
                                     factor_exposure: FactorExposure,
                                     tail_risk: TailRiskMetrics,
                                     geo_exposure: GeographicExposure) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        try:
            # VaR recommendations
            if abs(risk_analysis.var_95) > 0.05:
                recommendations.append(
                    "Consider reducing position sizes or adding hedging instruments to lower Value at Risk."
                )
            
            # Concentration risk
            if risk_analysis.concentration_risk > 0.5:
                recommendations.append(
                    "Portfolio shows high concentration risk. Consider diversifying across more positions."
                )
            
            # Drawdown recommendations
            if abs(risk_analysis.max_drawdown) > 0.20:
                recommendations.append(
                    "Historical maximum drawdown is significant. Consider implementing stop-loss strategies."
                )
            
            # Factor exposure recommendations
            if abs(factor_exposure.size_factor) > 0.5:
                recommendations.append(
                    f"High {'small' if factor_exposure.size_factor > 0 else 'large'} cap exposure. "
                    "Consider balancing with opposite size exposure."
                )
            
            if abs(factor_exposure.momentum_factor) > 0.5:
                recommendations.append(
                    "High momentum exposure detected. Monitor for potential reversals."
                )
            
            # Tail risk recommendations
            if tail_risk.sortino_ratio < 1.0:
                recommendations.append(
                    "Low Sortino ratio indicates poor downside risk-adjusted returns. "
                    "Consider strategies to reduce downside volatility."
                )
            
            # Geographic diversification
            if geo_exposure.domestic_exposure > 0.8:
                recommendations.append(
                    "High domestic concentration. Consider international diversification."
                )
            
            # Tracking error
            if risk_analysis.tracking_error > 0.10:
                recommendations.append(
                    "High tracking error suggests significant deviation from benchmark. "
                    "Review if this aligns with investment objectives."
                )
            
            # Default recommendation if no issues found
            if not recommendations:
                recommendations.append(
                    "Portfolio risk metrics appear to be within acceptable ranges. "
                    "Continue regular monitoring and rebalancing as needed."
                )
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error."]
    
    def _summarize_factor_exposure(self, factor_exposure: FactorExposure) -> str:
        """Summarize factor exposure in text."""
        try:
            factors = {
                'Size': factor_exposure.size_factor,
                'Value': factor_exposure.value_factor,
                'Growth': factor_exposure.growth_factor,
                'Momentum': factor_exposure.momentum_factor,
                'Quality': factor_exposure.quality_factor,
                'Low Volatility': factor_exposure.low_volatility_factor
            }
            
            # Find dominant factor
            dominant_factor = max(factors.items(), key=lambda x: abs(x[1]))
            
            if abs(dominant_factor[1]) > 0.3:
                direction = "positive" if dominant_factor[1] > 0 else "negative"
                return f"Portfolio shows {direction} {dominant_factor[0].lower()} factor exposure ({dominant_factor[1]:.2f})"
            else:
                return "Portfolio shows balanced factor exposures with no dominant factor"
                
        except Exception as e:
            return f"Unable to summarize factor exposure: {e}"
    
    def _assess_tail_risk(self, tail_risk: TailRiskMetrics) -> str:
        """Assess tail risk level."""
        try:
            risk_score = 0
            
            # Assess various tail risk components
            if abs(tail_risk.max_drawdown) > 0.25:
                risk_score += 2
            elif abs(tail_risk.max_drawdown) > 0.15:
                risk_score += 1
            
            if tail_risk.sortino_ratio < 0.5:
                risk_score += 2
            elif tail_risk.sortino_ratio < 1.0:
                risk_score += 1
            
            if tail_risk.skewness < -1.0:
                risk_score += 1
            
            if tail_risk.kurtosis > 3.0:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 4:
                return "High tail risk - significant potential for extreme losses"
            elif risk_score >= 2:
                return "Moderate tail risk - some potential for large losses"
            else:
                return "Low tail risk - limited potential for extreme losses"
                
        except Exception as e:
            return f"Unable to assess tail risk: {e}"
    
    def _generate_chart_data(self, 
                           rolling_metrics: pd.DataFrame,
                           risk_analysis: RiskAnalysis,
                           factor_exposure: FactorExposure) -> Dict[str, Any]:
        """Generate chart data for visualization."""
        try:
            charts = {}
            
            # Rolling metrics chart data
            if not rolling_metrics.empty:
                charts['rolling_volatility'] = {
                    'dates': rolling_metrics.index.strftime('%Y-%m-%d').tolist(),
                    'values': rolling_metrics['volatility'].tolist(),
                    'title': 'Rolling Volatility (63-day window)'
                }
                
                charts['rolling_beta'] = {
                    'dates': rolling_metrics.index.strftime('%Y-%m-%d').tolist(),
                    'values': rolling_metrics['beta'].tolist(),
                    'title': 'Rolling Beta (63-day window)'
                }
            
            # Factor exposure radar chart data
            charts['factor_exposure'] = {
                'factors': ['Size', 'Value', 'Growth', 'Momentum', 'Quality', 'Low Vol'],
                'values': [
                    factor_exposure.size_factor,
                    factor_exposure.value_factor,
                    factor_exposure.growth_factor,
                    factor_exposure.momentum_factor,
                    factor_exposure.quality_factor,
                    factor_exposure.low_volatility_factor
                ],
                'title': 'Factor Exposure Profile'
            }
            
            # Sector exposure pie chart data
            if risk_analysis.sector_exposures:
                charts['sector_exposure'] = {
                    'labels': list(risk_analysis.sector_exposures.keys()),
                    'values': list(risk_analysis.sector_exposures.values()),
                    'title': 'Sector Exposure'
                }
            
            return charts
            
        except Exception as e:
            logger.warning(f"Failed to generate chart data: {e}")
            return {}
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML version of the report."""
        try:
            metadata = report_data.get('metadata', {})
            executive_summary = report_data.get('executive_summary', {})
            risk_analysis = report_data.get('risk_analysis', {})
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Risk Analysis Report - {metadata.get('portfolio_id', 'Unknown')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 30px; }}
                    .metric {{ margin: 10px 0; }}
                    .risk-high {{ color: #d32f2f; }}
                    .risk-medium {{ color: #f57c00; }}
                    .risk-low {{ color: #388e3c; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Portfolio Risk Analysis Report</h1>
                    <h2>{metadata.get('portfolio_id', 'Unknown Portfolio')}</h2>
                    <p>Report Date: {metadata.get('report_date', 'Unknown')}</p>
                </div>
                
                <div class="section">
                    <h3>Executive Summary</h3>
                    <div class="metric">
                        <strong>Overall Risk Level:</strong> 
                        <span class="risk-{executive_summary.get('overall_risk_level', 'unknown').lower()}">
                            {executive_summary.get('overall_risk_level', 'Unknown')}
                        </span>
                    </div>
                    <div class="metric">
                        <strong>Value at Risk (95%):</strong> {executive_summary.get('var_95_percent', 0):.2f}%
                    </div>
                    <div class="metric">
                        <strong>Maximum Drawdown:</strong> {executive_summary.get('max_drawdown_percent', 0):.2f}%
                    </div>
                </div>
                
                <div class="section">
                    <h3>Risk Metrics</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Portfolio Beta</td><td>{risk_analysis.get('portfolio_beta', 0):.4f}</td></tr>
                        <tr><td>Tracking Error</td><td>{risk_analysis.get('tracking_error', 0):.4f}</td></tr>
                        <tr><td>Information Ratio</td><td>{risk_analysis.get('information_ratio', 0):.4f}</td></tr>
                        <tr><td>Concentration Risk</td><td>{risk_analysis.get('concentration_risk', 0):.4f}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h3>Recommendations</h3>
                    <ul>
            """
            
            for recommendation in report_data.get('risk_recommendations', []):
                html += f"<li>{recommendation}</li>"
            
            html += """
                    </ul>
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return f"<html><body><h1>Error generating report: {e}</h1></body></html>"