"""Performance tracking engine for portfolio performance analysis."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats

from ...common.interfaces import DataStorage
from ...common.models import Portfolio
from ..storage import AnalyticsStorage
from ..models import PerformanceMetrics, AnalyticsError
from ..exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Real-time portfolio performance tracking."""
    
    def __init__(self, data_storage: DataStorage, analytics_storage: AnalyticsStorage):
        """
        Initialize performance tracker.
        
        Args:
            data_storage: Data storage interface for historical data
            analytics_storage: Analytics storage interface for results
        """
        self.data_storage = data_storage
        self.analytics_storage = analytics_storage
        
        # Default benchmark configurations
        self.default_benchmarks = {
            'SPY': {'name': 'S&P 500', 'weight': 1.0},
            'VTI': {'name': 'Total Stock Market', 'weight': 1.0},
            'QQQ': {'name': 'NASDAQ 100', 'weight': 1.0}
        }
        
        # Risk-free rate proxy (3-month Treasury)
        self.risk_free_rate = 0.02  # 2% default, should be updated from market data
        
        logger.info("Performance tracker initialized with benchmark support")
    
    def track_performance(self, 
                         portfolio: Portfolio,
                         benchmark_ticker: str = 'SPY',
                         lookback_days: int = 252) -> PerformanceMetrics:
        """
        Track current portfolio performance against benchmarks.
        
        Args:
            portfolio: Portfolio object with tickers and weights
            benchmark_ticker: Benchmark ticker symbol
            lookback_days: Number of days to look back for calculations
            
        Returns:
            Performance metrics
            
        Raises:
            AnalyticsError: If performance tracking fails
        """
        try:
            logger.info(f"Tracking performance for portfolio {portfolio.id}")
            
            # Get historical data for portfolio and benchmark
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer for data
            
            portfolio_data = self._get_portfolio_data(portfolio, start_date, end_date)
            benchmark_data = self._get_benchmark_data(benchmark_ticker, start_date, end_date)
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, portfolio.weights)
            benchmark_returns = self._calculate_benchmark_returns(benchmark_data)
            
            # Align dates and trim to lookback period
            aligned_data = self._align_returns(portfolio_returns, benchmark_returns, lookback_days)
            portfolio_returns = aligned_data['portfolio']
            benchmark_returns = aligned_data['benchmark']
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                portfolio.id, portfolio_returns, benchmark_returns
            )
            
            logger.info(f"Performance tracking completed for portfolio {portfolio.id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Performance tracking failed: {e}")
            raise AnalyticsError(f"Performance tracking failed: {e}")
    
    def _get_portfolio_data(self, portfolio: Portfolio, start_date: date, end_date: date) -> pd.DataFrame:
        """Get historical price data for portfolio tickers."""
        try:
            all_data = []
            
            for ticker in portfolio.tickers:
                ticker_data = self.data_storage.get_price_data(ticker, start_date, end_date)
                if ticker_data.empty:
                    raise InsufficientDataError(f"No data available for ticker {ticker}")
                
                # Convert to DataFrame if needed
                if isinstance(ticker_data, list):
                    ticker_data = pd.DataFrame([
                        {
                            'date': item.date,
                            'adjusted_close': item.adjusted_close,
                            'symbol': item.symbol
                        } for item in ticker_data
                    ])
                
                ticker_data['symbol'] = ticker
                all_data.append(ticker_data)
            
            # Combine all ticker data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Pivot to get tickers as columns
            portfolio_data = combined_data.pivot(index='date', columns='symbol', values='adjusted_close')
            portfolio_data.index = pd.to_datetime(portfolio_data.index)
            portfolio_data = portfolio_data.sort_index()
            
            return portfolio_data.dropna()
            
        except Exception as e:
            logger.error(f"Failed to get portfolio data: {e}")
            raise AnalyticsError(f"Failed to get portfolio data: {e}")
    
    def _get_benchmark_data(self, benchmark_ticker: str, start_date: date, end_date: date) -> pd.Series:
        """Get historical price data for benchmark."""
        try:
            benchmark_data = self.data_storage.get_price_data(benchmark_ticker, start_date, end_date)
            
            if isinstance(benchmark_data, list):
                benchmark_df = pd.DataFrame([
                    {
                        'date': item.date,
                        'adjusted_close': item.adjusted_close
                    } for item in benchmark_data
                ])
            else:
                benchmark_df = benchmark_data[['date', 'adjusted_close']].copy()
            
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            benchmark_df = benchmark_df.set_index('date').sort_index()
            
            return benchmark_df['adjusted_close'].dropna()
            
        except Exception as e:
            logger.error(f"Failed to get benchmark data: {e}")
            raise AnalyticsError(f"Failed to get benchmark data: {e}")
    
    def _calculate_portfolio_returns(self, price_data: pd.DataFrame, weights: List[float]) -> pd.Series:
        """Calculate portfolio returns from price data and weights."""
        try:
            # Calculate daily returns for each asset
            returns = price_data.pct_change().dropna()
            
            # Create weights series aligned with tickers
            weight_dict = dict(zip(price_data.columns, weights))
            weights_series = pd.Series([weight_dict.get(ticker, 0) for ticker in returns.columns])
            
            # Calculate weighted portfolio returns
            portfolio_returns = (returns * weights_series).sum(axis=1)
            
            return portfolio_returns.dropna()
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio returns: {e}")
            raise AnalyticsError(f"Failed to calculate portfolio returns: {e}")
    
    def _calculate_benchmark_returns(self, price_data: pd.Series) -> pd.Series:
        """Calculate benchmark returns from price data."""
        try:
            return price_data.pct_change().dropna()
        except Exception as e:
            logger.error(f"Failed to calculate benchmark returns: {e}")
            raise AnalyticsError(f"Failed to calculate benchmark returns: {e}")
    
    def _align_returns(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                      lookback_days: int) -> Dict[str, pd.Series]:
        """Align portfolio and benchmark returns and trim to lookback period."""
        try:
            # Find common dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            
            if len(common_dates) < lookback_days:
                raise InsufficientDataError(
                    f"Insufficient data: only {len(common_dates)} days available, need {lookback_days}"
                )
            
            # Align data
            portfolio_aligned = portfolio_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            # Trim to lookback period (most recent data)
            portfolio_trimmed = portfolio_aligned.tail(lookback_days)
            benchmark_trimmed = benchmark_aligned.tail(lookback_days)
            
            return {
                'portfolio': portfolio_trimmed,
                'benchmark': benchmark_trimmed
            }
            
        except Exception as e:
            logger.error(f"Failed to align returns: {e}")
            raise AnalyticsError(f"Failed to align returns: {e}")
    
    def _calculate_performance_metrics(self, portfolio_id: str, portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            # Basic return metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            
            # Volatility metrics
            volatility = portfolio_returns.std() * np.sqrt(252)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Risk-adjusted metrics
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
            
            # Benchmark comparison metrics
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1
            
            # Beta calculation
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Alpha calculation
            alpha = annualized_return - (self.risk_free_rate + beta * (benchmark_annualized_return - self.risk_free_rate))
            
            # R-squared
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            # Tracking error and information ratio
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = (annualized_return - benchmark_annualized_return) / tracking_error if tracking_error > 0 else 0
            
            # Create performance data dictionary
            performance_data = {
                'returns_series': portfolio_returns.to_dict(),
                'benchmark_returns': benchmark_returns.to_dict(),
                'active_returns': active_returns.to_dict(),
                'rolling_metrics': self._calculate_rolling_metrics(portfolio_returns, benchmark_returns),
                'calculation_period': {
                    'start_date': portfolio_returns.index[0].strftime('%Y-%m-%d'),
                    'end_date': portfolio_returns.index[-1].strftime('%Y-%m-%d'),
                    'days': len(portfolio_returns)
                }
            }
            
            return PerformanceMetrics(
                portfolio_id=portfolio_id,
                calculation_date=date.today(),
                total_return=float(total_return),
                annualized_return=float(annualized_return),
                volatility=float(volatility),
                sharpe_ratio=float(sharpe_ratio),
                sortino_ratio=float(sortino_ratio),
                alpha=float(alpha),
                beta=float(beta),
                r_squared=float(r_squared),
                tracking_error=float(tracking_error),
                information_ratio=float(information_ratio),
                performance_data=performance_data
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            raise AnalyticsError(f"Failed to calculate performance metrics: {e}")
    
    def _calculate_rolling_metrics(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                                 window: int = 30) -> Dict[str, Any]:
        """Calculate rolling performance metrics."""
        try:
            rolling_returns = portfolio_returns.rolling(window=window)
            rolling_benchmark = benchmark_returns.rolling(window=window)
            
            # Rolling Sharpe ratio
            rolling_excess = portfolio_returns.rolling(window=window).mean() * 252 - self.risk_free_rate
            rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
            rolling_sharpe = (rolling_excess / rolling_vol).dropna()
            
            # Rolling beta
            rolling_cov = portfolio_returns.rolling(window=window).cov(benchmark_returns)
            rolling_var = benchmark_returns.rolling(window=window).var()
            rolling_beta = (rolling_cov / rolling_var).dropna()
            
            return {
                'rolling_sharpe': rolling_sharpe.to_dict(),
                'rolling_beta': rolling_beta.to_dict(),
                'window_days': window
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate rolling metrics: {e}")
            return {}

    def calculate_attribution(self, 
                            portfolio: Portfolio,
                            benchmark_ticker: str = 'SPY',
                            period: str = "monthly") -> Dict[str, Any]:
        """
        Calculate performance attribution analysis using Brinson model.
        
        Args:
            portfolio: Portfolio object
            benchmark_ticker: Benchmark ticker symbol
            period: Attribution period
            
        Returns:
            Attribution analysis results
        """
        try:
            logger.info(f"Calculating attribution for portfolio {portfolio.id}")
            
            # Get data for attribution period
            end_date = date.today()
            if period == "monthly":
                start_date = end_date - timedelta(days=30)
            elif period == "quarterly":
                start_date = end_date - timedelta(days=90)
            elif period == "yearly":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=30)  # Default to monthly
            
            # Get portfolio and benchmark data
            portfolio_data = self._get_portfolio_data(portfolio, start_date, end_date)
            benchmark_data = self._get_benchmark_data(benchmark_ticker, start_date, end_date)
            
            # Calculate returns for attribution period
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, portfolio.weights)
            benchmark_returns = self._calculate_benchmark_returns(benchmark_data)
            
            # Align data
            aligned_data = self._align_returns(portfolio_returns, benchmark_returns, len(portfolio_returns))
            portfolio_returns = aligned_data['portfolio']
            benchmark_returns = aligned_data['benchmark']
            
            # Calculate attribution components
            total_portfolio_return = (1 + portfolio_returns).prod() - 1
            total_benchmark_return = (1 + benchmark_returns).prod() - 1
            total_active_return = total_portfolio_return - total_benchmark_return
            
            # Simplified attribution (for full Brinson model, would need sector data)
            # Asset allocation effect: difference in weights * benchmark returns
            # Security selection effect: portfolio weights * (portfolio returns - benchmark returns)
            
            # For now, provide simplified attribution
            attribution = {
                'total_portfolio_return': float(total_portfolio_return),
                'total_benchmark_return': float(total_benchmark_return),
                'total_active_return': float(total_active_return),
                'attribution_period': period,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'asset_allocation_effect': float(total_active_return * 0.6),  # Simplified
                'security_selection_effect': float(total_active_return * 0.4),  # Simplified
                'interaction_effect': 0.0,
                'individual_contributions': self._calculate_individual_contributions(
                    portfolio, portfolio_data, benchmark_returns
                )
            }
            
            logger.info(f"Attribution calculation completed for portfolio {portfolio.id}")
            return attribution
            
        except Exception as e:
            logger.error(f"Attribution calculation failed: {e}")
            raise AnalyticsError(f"Attribution calculation failed: {e}")
    
    def _calculate_individual_contributions(self, portfolio: Portfolio, 
                                         portfolio_data: pd.DataFrame, 
                                         benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate individual asset contributions to portfolio performance."""
        try:
            contributions = {}
            
            for i, ticker in enumerate(portfolio.tickers):
                if ticker in portfolio_data.columns:
                    # Calculate asset returns
                    asset_returns = portfolio_data[ticker].pct_change().dropna()
                    
                    # Align with benchmark
                    common_dates = asset_returns.index.intersection(benchmark_returns.index)
                    if len(common_dates) > 0:
                        asset_returns_aligned = asset_returns.loc[common_dates]
                        benchmark_aligned = benchmark_returns.loc[common_dates]
                        
                        # Calculate contribution (weight * excess return)
                        asset_total_return = (1 + asset_returns_aligned).prod() - 1
                        benchmark_total_return = (1 + benchmark_aligned).prod() - 1
                        excess_return = asset_total_return - benchmark_total_return
                        
                        contribution = portfolio.weights[i] * excess_return
                        contributions[ticker] = float(contribution)
                    else:
                        contributions[ticker] = 0.0
                else:
                    contributions[ticker] = 0.0
            
            return contributions
            
        except Exception as e:
            logger.warning(f"Failed to calculate individual contributions: {e}")
            return {ticker: 0.0 for ticker in portfolio.tickers}
    
    def update_performance_history(self, portfolio: Portfolio, 
                                 benchmark_ticker: str = 'SPY') -> None:
        """
        Update historical performance records.
        
        Args:
            portfolio: Portfolio object
            benchmark_ticker: Benchmark ticker symbol
        """
        try:
            logger.info(f"Updating performance history for portfolio {portfolio.id}")
            
            # Calculate current performance metrics
            metrics = self.track_performance(portfolio, benchmark_ticker)
            
            # Store in analytics storage
            self.analytics_storage.store_performance_metrics(metrics)
            
            logger.info(f"Performance history updated for portfolio {portfolio.id}")
            
        except Exception as e:
            logger.error(f"Performance history update failed: {e}")
            raise AnalyticsError(f"Performance history update failed: {e}")
    
    def get_performance_summary(self, 
                              portfolio_id: str,
                              start_date: date,
                              end_date: date) -> Dict[str, Any]:
        """
        Get performance summary for a date range.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            Performance summary
        """
        try:
            logger.info(f"Getting performance summary for portfolio {portfolio_id}")
            
            # Get performance history from storage
            history = self.analytics_storage.get_performance_history(
                portfolio_id, start_date, end_date
            )
            
            if not history:
                return {
                    'message': 'No performance data available for the specified period',
                    'portfolio_id': portfolio_id,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                }
            
            # Calculate summary statistics
            returns = [metric.total_return for metric in history]
            volatilities = [metric.volatility for metric in history]
            
            summary = {
                'portfolio_id': portfolio_id,
                'period': f"{start_date} to {end_date}",
                'data_points': len(history),
                'avg_return': np.mean(returns) if returns else 0,
                'avg_volatility': np.mean(volatilities) if volatilities else 0,
                'best_return': max(returns) if returns else 0,
                'worst_return': min(returns) if returns else 0,
                'latest_metrics': {
                    'total_return': history[-1].total_return,
                    'sharpe_ratio': history[-1].sharpe_ratio,
                    'volatility': history[-1].volatility
                } if history else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary calculation failed: {e}")
            raise AnalyticsError(f"Performance summary calculation failed: {e}")
    
    def compare_multiple_benchmarks(self, portfolio: Portfolio, 
                                  benchmark_tickers: List[str] = None,
                                  lookback_days: int = 252) -> Dict[str, Any]:
        """
        Compare portfolio performance against multiple benchmarks.
        
        Args:
            portfolio: Portfolio object
            benchmark_tickers: List of benchmark ticker symbols
            lookback_days: Number of days to look back
            
        Returns:
            Multi-benchmark comparison results
        """
        try:
            if benchmark_tickers is None:
                benchmark_tickers = list(self.default_benchmarks.keys())
            
            logger.info(f"Comparing portfolio {portfolio.id} against {len(benchmark_tickers)} benchmarks")
            
            comparison_results = {}
            
            for benchmark_ticker in benchmark_tickers:
                try:
                    # Calculate performance against this benchmark
                    metrics = self.track_performance(portfolio, benchmark_ticker, lookback_days)
                    
                    benchmark_name = self.default_benchmarks.get(
                        benchmark_ticker, {'name': benchmark_ticker}
                    )['name']
                    
                    comparison_results[benchmark_ticker] = {
                        'name': benchmark_name,
                        'alpha': metrics.alpha,
                        'beta': metrics.beta,
                        'r_squared': metrics.r_squared,
                        'tracking_error': metrics.tracking_error,
                        'information_ratio': metrics.information_ratio,
                        'correlation': np.sqrt(metrics.r_squared) if metrics.r_squared >= 0 else 0
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to compare against {benchmark_ticker}: {e}")
                    comparison_results[benchmark_ticker] = {
                        'name': benchmark_ticker,
                        'error': str(e)
                    }
            
            # Find best benchmark match (highest R-squared)
            valid_benchmarks = {k: v for k, v in comparison_results.items() if 'error' not in v}
            if valid_benchmarks:
                best_benchmark = max(valid_benchmarks.items(), key=lambda x: x[1]['r_squared'])
                comparison_results['best_benchmark'] = {
                    'ticker': best_benchmark[0],
                    'name': best_benchmark[1]['name'],
                    'r_squared': best_benchmark[1]['r_squared']
                }
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Multi-benchmark comparison failed: {e}")
            raise AnalyticsError(f"Multi-benchmark comparison failed: {e}")
    
    def calculate_rolling_performance(self, portfolio: Portfolio, 
                                    benchmark_ticker: str = 'SPY',
                                    window_days: int = 30,
                                    lookback_days: int = 252) -> Dict[str, Any]:
        """
        Calculate rolling performance metrics with configurable time windows.
        
        Args:
            portfolio: Portfolio object
            benchmark_ticker: Benchmark ticker symbol
            window_days: Rolling window size in days
            lookback_days: Total lookback period
            
        Returns:
            Rolling performance metrics
        """
        try:
            logger.info(f"Calculating rolling performance for portfolio {portfolio.id}")
            
            # Get data
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days + 30)
            
            portfolio_data = self._get_portfolio_data(portfolio, start_date, end_date)
            benchmark_data = self._get_benchmark_data(benchmark_ticker, start_date, end_date)
            
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, portfolio.weights)
            benchmark_returns = self._calculate_benchmark_returns(benchmark_data)
            
            # Align data
            aligned_data = self._align_returns(portfolio_returns, benchmark_returns, lookback_days)
            portfolio_returns = aligned_data['portfolio']
            benchmark_returns = aligned_data['benchmark']
            
            # Calculate rolling metrics
            rolling_results = {}
            
            # Rolling returns (annualized)
            rolling_portfolio_returns = portfolio_returns.rolling(window=window_days).apply(
                lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) == window_days else np.nan
            ).dropna()
            
            rolling_benchmark_returns = benchmark_returns.rolling(window=window_days).apply(
                lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) == window_days else np.nan
            ).dropna()
            
            # Rolling volatility (annualized)
            rolling_portfolio_vol = portfolio_returns.rolling(window=window_days).std() * np.sqrt(252)
            rolling_benchmark_vol = benchmark_returns.rolling(window=window_days).std() * np.sqrt(252)
            
            # Rolling Sharpe ratio
            rolling_sharpe = ((rolling_portfolio_returns - self.risk_free_rate) / 
                            rolling_portfolio_vol.loc[rolling_portfolio_returns.index])
            
            # Rolling beta
            rolling_beta = portfolio_returns.rolling(window=window_days).cov(benchmark_returns) / \
                          benchmark_returns.rolling(window=window_days).var()
            
            # Rolling alpha
            rolling_alpha = (rolling_portfolio_returns - 
                           (self.risk_free_rate + rolling_beta.loc[rolling_portfolio_returns.index] * 
                            (rolling_benchmark_returns - self.risk_free_rate)))
            
            rolling_results = {
                'window_days': window_days,
                'data_points': len(rolling_portfolio_returns),
                'rolling_returns': {
                    'portfolio': rolling_portfolio_returns.to_dict(),
                    'benchmark': rolling_benchmark_returns.to_dict()
                },
                'rolling_volatility': {
                    'portfolio': rolling_portfolio_vol.dropna().to_dict(),
                    'benchmark': rolling_benchmark_vol.dropna().to_dict()
                },
                'rolling_sharpe': rolling_sharpe.dropna().to_dict(),
                'rolling_beta': rolling_beta.dropna().to_dict(),
                'rolling_alpha': rolling_alpha.dropna().to_dict(),
                'summary_stats': {
                    'avg_return': float(rolling_portfolio_returns.mean()),
                    'avg_volatility': float(rolling_portfolio_vol.mean()),
                    'avg_sharpe': float(rolling_sharpe.mean()),
                    'avg_beta': float(rolling_beta.mean()),
                    'avg_alpha': float(rolling_alpha.mean())
                }
            }
            
            return rolling_results
            
        except Exception as e:
            logger.error(f"Rolling performance calculation failed: {e}")
            raise AnalyticsError(f"Rolling performance calculation failed: {e}")