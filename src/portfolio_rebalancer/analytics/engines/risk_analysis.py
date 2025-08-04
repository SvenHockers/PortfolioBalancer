"""Risk analysis engine for portfolio risk assessment."""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum

from ...common.interfaces import DataStorage
from ..models import RiskAnalysis, RiskAnalysisError

logger = logging.getLogger(__name__)


class RiskLimitType(Enum):
    """Types of risk limits."""
    VAR_95 = "var_95"
    MAX_DRAWDOWN = "max_drawdown"
    CONCENTRATION = "concentration"
    SECTOR_CONCENTRATION = "sector_concentration"
    TRACKING_ERROR = "tracking_error"
    BETA = "beta"


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    limit_type: RiskLimitType
    threshold: float
    warning_threshold: Optional[float] = None
    description: str = ""


@dataclass
class RiskLimitBreach:
    """Risk limit breach information."""
    limit_type: RiskLimitType
    current_value: float
    threshold: float
    severity: str  # "warning" or "breach"
    description: str
    timestamp: datetime


@dataclass
class FactorExposure:
    """Detailed factor exposure analysis."""
    size_factor: float
    value_factor: float
    growth_factor: float
    momentum_factor: float
    quality_factor: float
    low_volatility_factor: float
    profitability_factor: float
    investment_factor: float


@dataclass
class TailRiskMetrics:
    """Comprehensive tail risk analysis."""
    max_drawdown: float
    max_drawdown_duration: int
    recovery_time: int
    tail_ratio: float
    downside_deviation: float
    sortino_ratio: float
    worst_month: float
    worst_quarter: float
    worst_year: float
    skewness: float
    kurtosis: float


@dataclass
class GeographicExposure:
    """Geographic exposure analysis."""
    domestic_exposure: float
    developed_markets_exposure: float
    emerging_markets_exposure: float
    regional_breakdown: Dict[str, float]
    currency_exposure: Dict[str, float]


class RiskAnalyzer:
    """Comprehensive portfolio risk analysis."""
    
    def __init__(self, data_storage: DataStorage, benchmark_ticker: str = "SPY"):
        """
        Initialize risk analyzer.
        
        Args:
            data_storage: Data storage interface for historical data
            benchmark_ticker: Benchmark ticker for beta and tracking error calculations
        """
        self.data_storage = data_storage
        self.benchmark_ticker = benchmark_ticker
        logger.info(f"Risk analyzer initialized with benchmark: {benchmark_ticker}")
    
    def analyze_portfolio_risk(self, 
                             portfolio_id: str,
                             tickers: List[str], 
                             weights: List[float],
                             lookback_days: int = 252) -> RiskAnalysis:
        """
        Comprehensive risk analysis of portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            lookback_days: Number of days for historical analysis
            
        Returns:
            Risk analysis results
            
        Raises:
            RiskAnalysisError: If risk analysis fails
        """
        try:
            logger.info(f"Starting comprehensive risk analysis for portfolio {portfolio_id}")
            
            # Validate inputs
            if len(tickers) != len(weights):
                raise RiskAnalysisError("Tickers and weights must have same length")
            
            if abs(sum(weights) - 1.0) > 0.01:
                raise RiskAnalysisError("Portfolio weights must sum to 1.0")
            
            # Get historical price data
            all_tickers = tickers + [self.benchmark_ticker]
            price_data = self.data_storage.get_prices(all_tickers, lookback_days)
            
            if price_data.empty:
                raise RiskAnalysisError("No historical price data available")
            
            # Calculate returns
            returns_data = self._calculate_returns(price_data)
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(returns_data, tickers, weights)
            benchmark_returns = returns_data[self.benchmark_ticker] if self.benchmark_ticker in returns_data.columns else None
            
            # Calculate risk metrics
            portfolio_beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
            information_ratio = self._calculate_information_ratio(portfolio_returns, benchmark_returns)
            var_95, cvar_95 = self._calculate_var_cvar(portfolio_returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            concentration_risk = self._calculate_concentration_risk(weights)
            
            # Calculate correlation analysis
            correlation_data = self._analyze_correlations(returns_data, tickers, weights)
            
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(returns_data, tickers, weights)
            
            # Calculate sector exposures (simplified implementation)
            sector_exposures = self._calculate_sector_exposures(tickers, weights)
            
            analysis = RiskAnalysis(
                portfolio_id=portfolio_id,
                analysis_date=date.today(),
                portfolio_beta=portfolio_beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                concentration_risk=concentration_risk,
                correlation_data=correlation_data,
                factor_exposures=factor_exposures,
                sector_exposures=sector_exposures
            )
            
            logger.info(f"Risk analysis completed for portfolio {portfolio_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            raise RiskAnalysisError(f"Risk analysis failed: {e}")
    
    def calculate_beta_analysis(self, 
                              tickers: List[str], 
                              weights: List[float],
                              lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate beta analysis for portfolio components.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            lookback_days: Number of days for analysis
            
        Returns:
            Dictionary with beta analysis results
        """
        try:
            logger.info("Calculating beta analysis")
            
            # Get price data
            all_tickers = tickers + [self.benchmark_ticker]
            price_data = self.data_storage.get_prices(all_tickers, lookback_days)
            
            if price_data.empty:
                raise RiskAnalysisError("No price data available for beta analysis")
            
            returns_data = self._calculate_returns(price_data)
            benchmark_returns = returns_data[self.benchmark_ticker] if self.benchmark_ticker in returns_data.columns else None
            
            if benchmark_returns is None:
                raise RiskAnalysisError(f"Benchmark data not available for {self.benchmark_ticker}")
            
            # Calculate individual asset betas
            individual_betas = {}
            for ticker in tickers:
                if ticker in returns_data.columns:
                    asset_returns = returns_data[ticker]
                    beta = self._calculate_beta(asset_returns, benchmark_returns)
                    individual_betas[ticker] = beta
            
            # Calculate portfolio beta
            portfolio_returns = self._calculate_portfolio_returns(returns_data, tickers, weights)
            portfolio_beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            
            return {
                'portfolio_beta': portfolio_beta,
                'individual_betas': individual_betas,
                'weighted_average_beta': sum(individual_betas.get(ticker, 0) * weight 
                                           for ticker, weight in zip(tickers, weights))
            }
            
        except Exception as e:
            logger.error(f"Beta analysis failed: {e}")
            raise RiskAnalysisError(f"Beta analysis failed: {e}")
    
    def calculate_tracking_error_analysis(self, 
                                        tickers: List[str], 
                                        weights: List[float],
                                        lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate tracking error and information ratio analysis.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            lookback_days: Number of days for analysis
            
        Returns:
            Dictionary with tracking error analysis results
        """
        try:
            logger.info("Calculating tracking error analysis")
            
            # Get price data
            all_tickers = tickers + [self.benchmark_ticker]
            price_data = self.data_storage.get_prices(all_tickers, lookback_days)
            
            if price_data.empty:
                raise RiskAnalysisError("No price data available for tracking error analysis")
            
            returns_data = self._calculate_returns(price_data)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, tickers, weights)
            benchmark_returns = returns_data[self.benchmark_ticker] if self.benchmark_ticker in returns_data.columns else None
            
            if benchmark_returns is None:
                raise RiskAnalysisError(f"Benchmark data not available for {self.benchmark_ticker}")
            
            # Calculate tracking error metrics
            tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
            information_ratio = self._calculate_information_ratio(portfolio_returns, benchmark_returns)
            
            # Calculate active return statistics
            active_returns = portfolio_returns - benchmark_returns
            active_return_mean = active_returns.mean() * 252  # Annualized
            active_return_std = active_returns.std() * np.sqrt(252)  # Annualized
            
            return {
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'active_return_mean': active_return_mean,
                'active_return_volatility': active_return_std,
                'active_return_sharpe': active_return_mean / active_return_std if active_return_std > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Tracking error analysis failed: {e}")
            raise RiskAnalysisError(f"Tracking error analysis failed: {e}")
    
    def calculate_rolling_risk_metrics(self, 
                                     tickers: List[str], 
                                     weights: List[float],
                                     window_days: int = 63,
                                     lookback_days: int = 252) -> pd.DataFrame:
        """
        Calculate rolling risk metrics with configurable time windows.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            window_days: Rolling window size in days
            lookback_days: Total lookback period in days
            
        Returns:
            DataFrame with rolling risk metrics
        """
        try:
            logger.info(f"Calculating rolling risk metrics with {window_days}-day window")
            
            # Get price data
            all_tickers = tickers + [self.benchmark_ticker]
            price_data = self.data_storage.get_prices(all_tickers, lookback_days)
            
            if price_data.empty:
                raise RiskAnalysisError("No price data available for rolling metrics")
            
            returns_data = self._calculate_returns(price_data)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, tickers, weights)
            benchmark_returns = returns_data[self.benchmark_ticker] if self.benchmark_ticker in returns_data.columns else None
            
            # Calculate rolling metrics
            rolling_metrics = []
            
            for i in range(window_days, len(portfolio_returns)):
                window_portfolio = portfolio_returns.iloc[i-window_days:i]
                window_benchmark = benchmark_returns.iloc[i-window_days:i] if benchmark_returns is not None else None
                
                # Calculate metrics for this window
                volatility = window_portfolio.std() * np.sqrt(252)
                
                if window_benchmark is not None:
                    beta = self._calculate_beta(window_portfolio, window_benchmark)
                    tracking_error = self._calculate_tracking_error(window_portfolio, window_benchmark)
                    information_ratio = self._calculate_information_ratio(window_portfolio, window_benchmark)
                else:
                    beta = np.nan
                    tracking_error = np.nan
                    information_ratio = np.nan
                
                var_95, cvar_95 = self._calculate_var_cvar(window_portfolio)
                
                rolling_metrics.append({
                    'date': portfolio_returns.index[i],
                    'volatility': volatility,
                    'beta': beta,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'var_95': var_95,
                    'cvar_95': cvar_95
                })
            
            return pd.DataFrame(rolling_metrics).set_index('date')
            
        except Exception as e:
            logger.error(f"Rolling risk metrics calculation failed: {e}")
            raise RiskAnalysisError(f"Rolling risk metrics calculation failed: {e}")
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data."""
        # Pivot to get tickers as columns
        prices_pivot = price_data.reset_index().pivot(index='date', columns='symbol', values='adjusted_close')
        
        # Calculate daily returns
        returns = prices_pivot.pct_change().dropna()
        
        return returns
    
    def _calculate_portfolio_returns(self, returns_data: pd.DataFrame, tickers: List[str], weights: List[float]) -> pd.Series:
        """Calculate portfolio returns from individual asset returns."""
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        
        for ticker, weight in zip(tickers, weights):
            if ticker in returns_data.columns:
                portfolio_returns += returns_data[ticker] * weight
        
        return portfolio_returns
    
    def _calculate_beta(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        if benchmark_returns is None or len(asset_returns) == 0 or len(benchmark_returns) == 0:
            return 1.0
        
        # Align the series
        aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:  # Need sufficient data points
            return 1.0
        
        asset_aligned = aligned_data.iloc[:, 0]
        benchmark_aligned = aligned_data.iloc[:, 1]
        
        covariance = np.cov(asset_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    def _calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error (annualized)."""
        if benchmark_returns is None:
            return 0.0
        
        # Align the series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:
            return 0.0
        
        active_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        return active_returns.std() * np.sqrt(252)
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        if benchmark_returns is None:
            return 0.0
        
        # Align the series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:
            return 0.0
        
        active_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        active_return_mean = active_returns.mean() * 252  # Annualized
        tracking_error = active_returns.std() * np.sqrt(252)  # Annualized
        
        if tracking_error == 0:
            return 0.0
        
        return active_return_mean / tracking_error
    
    def _calculate_var_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR."""
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Sort returns in ascending order
        sorted_returns = returns.sort_values()
        
        # Calculate VaR (the percentile value, should be negative for losses)
        var_index = int(confidence_level * len(sorted_returns))
        var = sorted_returns.iloc[var_index] if var_index < len(sorted_returns) else 0.0
        
        # Calculate CVaR (average of returns below VaR)
        tail_returns = sorted_returns.iloc[:var_index]
        cvar = tail_returns.mean() if len(tail_returns) > 0 else var
        
        return var, cvar
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_concentration_risk(self, weights: List[float]) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index."""
        return sum(w**2 for w in weights)
    
    def _analyze_correlations(self, returns_data: pd.DataFrame, tickers: List[str], weights: List[float]) -> Dict[str, Any]:
        """Analyze asset correlations and concentration risk."""
        try:
            # Filter returns data to only include portfolio tickers
            portfolio_returns = returns_data[tickers].dropna()
            
            if portfolio_returns.empty:
                return {
                    'correlation_matrix': {},
                    'avg_correlation': 0.0,
                    'max_correlation': 0.0,
                    'min_correlation': 0.0,
                    'diversification_ratio': 1.0
                }
            
            # Calculate correlation matrix
            correlation_matrix = portfolio_returns.corr()
            
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            correlations = correlation_matrix.where(mask).stack().values
            
            # Calculate statistics
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(correlations)
            min_correlation = np.min(correlations)
            
            # Calculate diversification ratio
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(portfolio_returns.cov() * 252, weights)))
            weighted_avg_volatility = sum(w * portfolio_returns[ticker].std() * np.sqrt(252) 
                                        for ticker, w in zip(tickers, weights))
            diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1.0
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'avg_correlation': float(avg_correlation),
                'max_correlation': float(max_correlation),
                'min_correlation': float(min_correlation),
                'diversification_ratio': float(diversification_ratio)
            }
            
        except Exception as e:
            logger.warning(f"Correlation analysis failed: {e}")
            return {
                'correlation_matrix': {},
                'avg_correlation': 0.0,
                'max_correlation': 0.0,
                'min_correlation': 0.0,
                'diversification_ratio': 1.0
            }
    
    def _calculate_factor_exposures(self, returns_data: pd.DataFrame, tickers: List[str], weights: List[float]) -> Dict[str, float]:
        """Calculate factor exposures (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In practice, you would use factor models like Fama-French
            
            portfolio_returns = self._calculate_portfolio_returns(returns_data, tickers, weights)
            
            if len(portfolio_returns) == 0:
                return {}
            
            # Calculate basic style factors based on return characteristics
            volatility = portfolio_returns.std() * np.sqrt(252)
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            # Simplified factor exposures
            return {
                'size_factor': min(max(-1.0, (volatility - 0.15) / 0.1), 1.0),  # Normalized size exposure
                'value_factor': min(max(-1.0, skewness / 2.0), 1.0),  # Value based on skewness
                'momentum_factor': min(max(-1.0, portfolio_returns.tail(21).mean() * 252 / 0.1), 1.0),  # Recent momentum
                'quality_factor': min(max(-1.0, -kurtosis / 5.0), 1.0),  # Quality based on kurtosis
                'low_volatility_factor': min(max(-1.0, (0.15 - volatility) / 0.1), 1.0)  # Low vol factor
            }
            
        except Exception as e:
            logger.warning(f"Factor exposure calculation failed: {e}")
            return {}
    
    def calculate_enhanced_factor_exposure(self, 
                                         tickers: List[str], 
                                         weights: List[float],
                                         lookback_days: int = 252) -> FactorExposure:
        """
        Calculate enhanced factor exposure analysis.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            lookback_days: Number of days for analysis
            
        Returns:
            Detailed factor exposure analysis
        """
        try:
            logger.info("Calculating enhanced factor exposure analysis")
            
            # Get price data
            all_tickers = tickers + [self.benchmark_ticker]
            price_data = self.data_storage.get_prices(all_tickers, lookback_days)
            
            if price_data.empty:
                logger.warning("No price data available for factor analysis")
                return FactorExposure(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            returns_data = self._calculate_returns(price_data)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, tickers, weights)
            
            if len(portfolio_returns) == 0:
                return FactorExposure(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Calculate enhanced factor exposures
            size_factor = self._calculate_size_factor(portfolio_returns, returns_data)
            value_factor = self._calculate_value_factor(portfolio_returns, returns_data)
            growth_factor = self._calculate_growth_factor(portfolio_returns, returns_data)
            momentum_factor = self._calculate_momentum_factor(portfolio_returns, returns_data)
            quality_factor = self._calculate_quality_factor(portfolio_returns, returns_data)
            low_vol_factor = self._calculate_low_volatility_factor(portfolio_returns, returns_data)
            profitability_factor = self._calculate_profitability_factor(portfolio_returns, returns_data)
            investment_factor = self._calculate_investment_factor(portfolio_returns, returns_data)
            
            return FactorExposure(
                size_factor=size_factor,
                value_factor=value_factor,
                growth_factor=growth_factor,
                momentum_factor=momentum_factor,
                quality_factor=quality_factor,
                low_volatility_factor=low_vol_factor,
                profitability_factor=profitability_factor,
                investment_factor=investment_factor
            )
            
        except Exception as e:
            logger.error(f"Enhanced factor exposure calculation failed: {e}")
            return FactorExposure(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def calculate_tail_risk_metrics(self, 
                                  tickers: List[str], 
                                  weights: List[float],
                                  lookback_days: int = 252) -> TailRiskMetrics:
        """
        Calculate comprehensive tail risk metrics.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            lookback_days: Number of days for analysis
            
        Returns:
            Comprehensive tail risk analysis
        """
        try:
            logger.info("Calculating tail risk metrics")
            
            # Get price data
            price_data = self.data_storage.get_prices(tickers, lookback_days)
            
            if price_data.empty:
                logger.warning("No price data available for tail risk analysis")
                return TailRiskMetrics(0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            returns_data = self._calculate_returns(price_data)
            portfolio_returns = self._calculate_portfolio_returns(returns_data, tickers, weights)
            
            if len(portfolio_returns) == 0:
                return TailRiskMetrics(0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Calculate tail risk metrics
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            max_dd_duration, recovery_time = self._calculate_drawdown_duration(portfolio_returns)
            tail_ratio = self._calculate_tail_ratio(portfolio_returns)
            downside_deviation = self._calculate_downside_deviation(portfolio_returns)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            
            # Calculate worst period returns
            worst_month = self._calculate_worst_period_return(portfolio_returns, 21)  # ~1 month
            worst_quarter = self._calculate_worst_period_return(portfolio_returns, 63)  # ~3 months
            worst_year = self._calculate_worst_period_return(portfolio_returns, 252)  # ~1 year
            
            # Calculate distribution metrics
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            return TailRiskMetrics(
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                recovery_time=recovery_time,
                tail_ratio=tail_ratio,
                downside_deviation=downside_deviation,
                sortino_ratio=sortino_ratio,
                worst_month=worst_month,
                worst_quarter=worst_quarter,
                worst_year=worst_year,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
        except Exception as e:
            logger.error(f"Tail risk metrics calculation failed: {e}")
            return TailRiskMetrics(0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def calculate_geographic_exposure(self, 
                                    tickers: List[str], 
                                    weights: List[float]) -> GeographicExposure:
        """
        Calculate geographic and currency exposure analysis.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            
        Returns:
            Geographic exposure analysis
        """
        try:
            logger.info("Calculating geographic exposure analysis")
            
            # This is a simplified implementation
            # In practice, you would use actual geographic classification data
            domestic_exposure = 0.0
            developed_markets_exposure = 0.0
            emerging_markets_exposure = 0.0
            regional_breakdown = {}
            currency_exposure = {}
            
            for ticker, weight in zip(tickers, weights):
                # Simplified geographic mapping based on ticker patterns
                if self._is_domestic_ticker(ticker):
                    domestic_exposure += weight
                    region = "North America"
                    currency = "USD"
                elif self._is_developed_market_ticker(ticker):
                    developed_markets_exposure += weight
                    region = self._get_ticker_region(ticker)
                    currency = self._get_ticker_currency(ticker)
                else:
                    emerging_markets_exposure += weight
                    region = "Emerging Markets"
                    currency = self._get_ticker_currency(ticker)
                
                # Update regional breakdown
                regional_breakdown[region] = regional_breakdown.get(region, 0.0) + weight
                
                # Update currency exposure
                currency_exposure[currency] = currency_exposure.get(currency, 0.0) + weight
            
            return GeographicExposure(
                domestic_exposure=domestic_exposure,
                developed_markets_exposure=developed_markets_exposure,
                emerging_markets_exposure=emerging_markets_exposure,
                regional_breakdown=regional_breakdown,
                currency_exposure=currency_exposure
            )
            
        except Exception as e:
            logger.error(f"Geographic exposure calculation failed: {e}")
            return GeographicExposure(0.0, 0.0, 0.0, {}, {})
    
    def monitor_risk_limits(self, 
                          portfolio_id: str,
                          tickers: List[str], 
                          weights: List[float],
                          risk_limits: List[RiskLimit],
                          lookback_days: int = 252) -> List[RiskLimitBreach]:
        """
        Monitor portfolio against configured risk limits.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            risk_limits: List of risk limits to monitor
            lookback_days: Number of days for analysis
            
        Returns:
            List of risk limit breaches
        """
        try:
            logger.info(f"Monitoring risk limits for portfolio {portfolio_id}")
            
            breaches = []
            
            # Get current risk analysis
            risk_analysis = self.analyze_portfolio_risk(portfolio_id, tickers, weights, lookback_days)
            
            # Check each risk limit
            for limit in risk_limits:
                current_value = self._get_risk_metric_value(risk_analysis, limit.limit_type)
                
                # Check for breach
                if self._is_limit_breached(current_value, limit.threshold, limit.limit_type):
                    breaches.append(RiskLimitBreach(
                        limit_type=limit.limit_type,
                        current_value=current_value,
                        threshold=limit.threshold,
                        severity="breach",
                        description=f"{limit.limit_type.value} breach: {current_value:.4f} exceeds limit {limit.threshold:.4f}",
                        timestamp=datetime.now()
                    ))
                
                # Check for warning
                elif (limit.warning_threshold is not None and 
                      self._is_limit_breached(current_value, limit.warning_threshold, limit.limit_type)):
                    breaches.append(RiskLimitBreach(
                        limit_type=limit.limit_type,
                        current_value=current_value,
                        threshold=limit.warning_threshold,
                        severity="warning",
                        description=f"{limit.limit_type.value} warning: {current_value:.4f} exceeds warning {limit.warning_threshold:.4f}",
                        timestamp=datetime.now()
                    ))
            
            return breaches
            
        except Exception as e:
            logger.error(f"Risk limit monitoring failed: {e}")
            return []
    
    def _calculate_sector_exposures(self, tickers: List[str], weights: List[float]) -> Dict[str, float]:
        """Calculate sector exposures (simplified implementation)."""
        try:
            # This is a very simplified implementation
            # In practice, you would use actual sector classification data
            
            # Mock sector mapping based on ticker patterns
            sector_mapping = {}
            for ticker in tickers:
                if ticker.startswith(('AAPL', 'MSFT', 'GOOGL', 'AMZN')):
                    sector_mapping[ticker] = 'Technology'
                elif ticker.startswith(('JPM', 'BAC', 'WFC', 'GS')):
                    sector_mapping[ticker] = 'Financials'
                elif ticker.startswith(('JNJ', 'PFE', 'UNH', 'ABBV')):
                    sector_mapping[ticker] = 'Healthcare'
                elif ticker.startswith(('XOM', 'CVX', 'COP', 'SLB')):
                    sector_mapping[ticker] = 'Energy'
                else:
                    sector_mapping[ticker] = 'Other'
            
            # Calculate sector weights
            sector_exposures = {}
            for ticker, weight in zip(tickers, weights):
                sector = sector_mapping.get(ticker, 'Other')
                sector_exposures[sector] = sector_exposures.get(sector, 0.0) + weight
            
            return sector_exposures
            
        except Exception as e:
            logger.warning(f"Sector exposure calculation failed: {e}")
            return {}
    
    # Enhanced factor calculation methods
    def _calculate_size_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate size factor exposure."""
        try:
            volatility = portfolio_returns.std() * np.sqrt(252)
            # Size factor: higher volatility suggests smaller cap exposure
            return min(max(-1.0, (volatility - 0.15) / 0.1), 1.0)
        except:
            return 0.0
    
    def _calculate_value_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate value factor exposure."""
        try:
            # Value factor based on return skewness (value stocks tend to have positive skew)
            skewness = portfolio_returns.skew()
            return min(max(-1.0, skewness / 2.0), 1.0)
        except:
            return 0.0
    
    def _calculate_growth_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate growth factor exposure."""
        try:
            # Growth factor opposite of value factor
            value_factor = self._calculate_value_factor(portfolio_returns, returns_data)
            return -value_factor
        except:
            return 0.0
    
    def _calculate_momentum_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate momentum factor exposure."""
        try:
            # Momentum based on recent performance
            recent_returns = portfolio_returns.tail(21).mean() * 252  # Last month annualized
            return min(max(-1.0, recent_returns / 0.1), 1.0)
        except:
            return 0.0
    
    def _calculate_quality_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate quality factor exposure."""
        try:
            # Quality based on return consistency (negative kurtosis indicates more consistent returns)
            kurtosis = portfolio_returns.kurtosis()
            return min(max(-1.0, -kurtosis / 5.0), 1.0)
        except:
            return 0.0
    
    def _calculate_low_volatility_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate low volatility factor exposure."""
        try:
            volatility = portfolio_returns.std() * np.sqrt(252)
            # Low volatility factor: lower volatility = higher exposure
            return min(max(-1.0, (0.15 - volatility) / 0.1), 1.0)
        except:
            return 0.0
    
    def _calculate_profitability_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate profitability factor exposure."""
        try:
            # Profitability based on return-to-volatility ratio
            mean_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            if volatility > 0:
                profitability_ratio = mean_return / volatility
                return min(max(-1.0, profitability_ratio / 2.0), 1.0)
            return 0.0
        except:
            return 0.0
    
    def _calculate_investment_factor(self, portfolio_returns: pd.Series, returns_data: pd.DataFrame) -> float:
        """Calculate investment factor exposure."""
        try:
            # Investment factor based on return stability
            rolling_std = portfolio_returns.rolling(21).std().mean() * np.sqrt(252)
            overall_std = portfolio_returns.std() * np.sqrt(252)
            if overall_std > 0:
                stability_ratio = rolling_std / overall_std
                return min(max(-1.0, (1.0 - stability_ratio) * 2.0 - 1.0), 1.0)
            return 0.0
        except:
            return 0.0
    
    # Tail risk calculation methods
    def _calculate_drawdown_duration(self, returns: pd.Series) -> Tuple[int, int]:
        """Calculate maximum drawdown duration and recovery time."""
        try:
            if len(returns) == 0:
                return 0, 0
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown period
            max_dd_start = None
            max_dd_end = None
            max_dd_duration = 0
            current_dd_start = None
            
            for i, dd in enumerate(drawdown):
                if dd < 0 and current_dd_start is None:
                    current_dd_start = i
                elif dd >= 0 and current_dd_start is not None:
                    duration = i - current_dd_start
                    if duration > max_dd_duration:
                        max_dd_duration = duration
                        max_dd_start = current_dd_start
                        max_dd_end = i
                    current_dd_start = None
            
            # Handle case where drawdown continues to end
            if current_dd_start is not None:
                duration = len(drawdown) - current_dd_start
                if duration > max_dd_duration:
                    max_dd_duration = duration
                    max_dd_start = current_dd_start
                    max_dd_end = len(drawdown)
            
            # Calculate recovery time (simplified)
            recovery_time = max_dd_duration // 2 if max_dd_duration > 0 else 0
            
            return max_dd_duration, recovery_time
            
        except Exception as e:
            logger.warning(f"Drawdown duration calculation failed: {e}")
            return 0, 0
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        try:
            if len(returns) == 0:
                return 0.0
            
            p95 = returns.quantile(0.95)
            p5 = returns.quantile(0.05)
            
            if p5 != 0:
                return abs(p95 / p5)
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate downside deviation."""
        try:
            if len(returns) == 0:
                return 0.0
            
            downside_returns = returns[returns < target_return]
            if len(downside_returns) == 0:
                return 0.0
            
            downside_variance = ((downside_returns - target_return) ** 2).mean()
            return np.sqrt(downside_variance) * np.sqrt(252)  # Annualized
            
        except:
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_return = returns.mean() * 252 - target_return  # Annualized
            downside_deviation = self._calculate_downside_deviation(returns, target_return / 252)
            
            if downside_deviation > 0:
                return excess_return / downside_deviation
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_worst_period_return(self, returns: pd.Series, period_days: int) -> float:
        """Calculate worst return over a rolling period."""
        try:
            if len(returns) < period_days:
                return returns.sum() if len(returns) > 0 else 0.0
            
            rolling_returns = returns.rolling(period_days).sum()
            return rolling_returns.min()
            
        except:
            return 0.0
    
    # Geographic exposure helper methods
    def _is_domestic_ticker(self, ticker: str) -> bool:
        """Check if ticker is domestic (US)."""
        # Simplified implementation - in practice, use actual classification data
        return not any(suffix in ticker for suffix in ['.TO', '.L', '.HK', '.T'])
    
    def _is_developed_market_ticker(self, ticker: str) -> bool:
        """Check if ticker is from developed markets."""
        # Simplified implementation
        developed_suffixes = ['.TO', '.L', '.PA', '.DE', '.SW']
        return any(suffix in ticker for suffix in developed_suffixes)
    
    def _get_ticker_region(self, ticker: str) -> str:
        """Get region for ticker."""
        # Simplified implementation
        if '.TO' in ticker:
            return "North America"
        elif '.L' in ticker:
            return "Europe"
        elif '.HK' in ticker or '.T' in ticker:
            return "Asia Pacific"
        else:
            return "North America"
    
    def _get_ticker_currency(self, ticker: str) -> str:
        """Get currency for ticker."""
        # Simplified implementation
        if '.TO' in ticker:
            return "CAD"
        elif '.L' in ticker:
            return "GBP"
        elif '.PA' in ticker:
            return "EUR"
        elif '.HK' in ticker:
            return "HKD"
        elif '.T' in ticker:
            return "JPY"
        else:
            return "USD"
    
    # Risk limit monitoring helper methods
    def _get_risk_metric_value(self, risk_analysis: RiskAnalysis, limit_type: RiskLimitType) -> float:
        """Get risk metric value from analysis."""
        if limit_type == RiskLimitType.VAR_95:
            return abs(risk_analysis.var_95)  # Use absolute value for comparison
        elif limit_type == RiskLimitType.MAX_DRAWDOWN:
            return abs(risk_analysis.max_drawdown)
        elif limit_type == RiskLimitType.CONCENTRATION:
            return risk_analysis.concentration_risk
        elif limit_type == RiskLimitType.TRACKING_ERROR:
            return risk_analysis.tracking_error
        elif limit_type == RiskLimitType.BETA:
            return abs(risk_analysis.portfolio_beta)
        elif limit_type == RiskLimitType.SECTOR_CONCENTRATION:
            # Calculate max sector concentration
            if risk_analysis.sector_exposures:
                return max(risk_analysis.sector_exposures.values())
            return 0.0
        else:
            return 0.0
    
    def _is_limit_breached(self, current_value: float, threshold: float, limit_type: RiskLimitType) -> bool:
        """Check if risk limit is breached."""
        # For most risk metrics, breach occurs when current value exceeds threshold
        return current_value > threshold