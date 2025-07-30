"""Performance attribution analysis engine."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

from ...common.interfaces import DataStorage
from ...common.models import Portfolio
from ..storage import AnalyticsStorage
from ..models import AnalyticsError
from ..exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


class AttributionAnalyzer:
    """Performance attribution analysis using Brinson model."""
    
    def __init__(self, data_storage: DataStorage, analytics_storage: AnalyticsStorage):
        """
        Initialize attribution analyzer.
        
        Args:
            data_storage: Data storage interface for historical data
            analytics_storage: Analytics storage interface for results
        """
        self.data_storage = data_storage
        self.analytics_storage = analytics_storage
        
        # Sector mapping for attribution analysis
        self.sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
            'GOOG': 'Technology', 'NVDA': 'Technology', 'META': 'Technology',
            'TSLA': 'Technology', 'NFLX': 'Technology', 'CRM': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare',
            
            # Financials
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
            'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
            
            # Consumer Discretionary
            'AMZN': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
            'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary',
            
            # Consumer Staples
            'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
            'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            
            # Industrials
            'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
            
            # ETFs (treated as broad market)
            'SPY': 'Broad Market', 'VTI': 'Broad Market', 'QQQ': 'Technology',
            'BND': 'Fixed Income', 'VXUS': 'International'
        }
        
        logger.info("Attribution analyzer initialized")
    
    def calculate_brinson_attribution(self, portfolio: Portfolio, 
                                    benchmark_portfolio: Portfolio,
                                    start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Calculate Brinson attribution model for return decomposition.
        
        Args:
            portfolio: Portfolio object
            benchmark_portfolio: Benchmark portfolio object
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Brinson attribution results
        """
        try:
            logger.info(f"Calculating Brinson attribution for portfolio {portfolio.id}")
            
            # Get portfolio and benchmark data
            portfolio_data = self._get_portfolio_data(portfolio, start_date, end_date)
            benchmark_data = self._get_portfolio_data(benchmark_portfolio, start_date, end_date)
            
            # Calculate returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, portfolio.weights)
            benchmark_returns = self._calculate_portfolio_returns(benchmark_data, benchmark_portfolio.weights)
            
            # Calculate sector-level attribution
            sector_attribution = self._calculate_sector_attribution(
                portfolio, benchmark_portfolio, portfolio_data, benchmark_data
            )
            
            # Calculate total attribution effects
            total_portfolio_return = (1 + portfolio_returns).prod() - 1
            total_benchmark_return = (1 + benchmark_returns).prod() - 1
            total_active_return = total_portfolio_return - total_benchmark_return
            
            # Brinson attribution components
            asset_allocation_effect = sum(attr['allocation_effect'] for attr in sector_attribution.values())
            security_selection_effect = sum(attr['selection_effect'] for attr in sector_attribution.values())
            interaction_effect = sum(attr['interaction_effect'] for attr in sector_attribution.values())
            
            attribution_result = {
                'portfolio_id': portfolio.id,
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': (end_date - start_date).days
                },
                'total_returns': {
                    'portfolio': float(total_portfolio_return),
                    'benchmark': float(total_benchmark_return),
                    'active': float(total_active_return)
                },
                'attribution_effects': {
                    'asset_allocation': float(asset_allocation_effect),
                    'security_selection': float(security_selection_effect),
                    'interaction': float(interaction_effect),
                    'total_explained': float(asset_allocation_effect + security_selection_effect + interaction_effect)
                },
                'sector_attribution': sector_attribution,
                'individual_contributions': self._calculate_individual_contributions(
                    portfolio, benchmark_portfolio, portfolio_data, benchmark_data
                )
            }
            
            logger.info(f"Brinson attribution completed for portfolio {portfolio.id}")
            return attribution_result
            
        except Exception as e:
            logger.error(f"Brinson attribution calculation failed: {e}")
            raise AnalyticsError(f"Brinson attribution calculation failed: {e}")
    
    def _get_portfolio_data(self, portfolio: Portfolio, start_date: date, end_date: date) -> pd.DataFrame:
        """Get historical price data for portfolio tickers."""
        try:
            all_data = []
            
            for ticker in portfolio.tickers:
                ticker_data = self.data_storage.get_price_data(ticker, start_date, end_date)
                if ticker_data.empty if hasattr(ticker_data, 'empty') else len(ticker_data) == 0:
                    logger.warning(f"No data available for ticker {ticker}")
                    continue
                
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
            
            if not all_data:
                raise InsufficientDataError("No data available for any portfolio tickers")
            
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
    
    def _calculate_sector_attribution(self, portfolio: Portfolio, benchmark_portfolio: Portfolio,
                                    portfolio_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate sector-level attribution analysis."""
        try:
            # Group assets by sector
            portfolio_sectors = self._group_by_sector(portfolio)
            benchmark_sectors = self._group_by_sector(benchmark_portfolio)
            
            sector_attribution = {}
            
            # Get all sectors
            all_sectors = set(portfolio_sectors.keys()) | set(benchmark_sectors.keys())
            
            for sector in all_sectors:
                # Portfolio sector data
                portfolio_sector_tickers = portfolio_sectors.get(sector, [])
                portfolio_sector_weights = [
                    portfolio.weights[portfolio.tickers.index(ticker)] 
                    for ticker in portfolio_sector_tickers if ticker in portfolio.tickers
                ]
                portfolio_sector_weight = sum(portfolio_sector_weights)
                
                # Benchmark sector data
                benchmark_sector_tickers = benchmark_sectors.get(sector, [])
                benchmark_sector_weights = [
                    benchmark_portfolio.weights[benchmark_portfolio.tickers.index(ticker)]
                    for ticker in benchmark_sector_tickers if ticker in benchmark_portfolio.tickers
                ]
                benchmark_sector_weight = sum(benchmark_sector_weights)
                
                # Calculate sector returns
                portfolio_sector_return = self._calculate_sector_return(
                    portfolio_sector_tickers, portfolio_sector_weights, portfolio_data
                )
                benchmark_sector_return = self._calculate_sector_return(
                    benchmark_sector_tickers, benchmark_sector_weights, benchmark_data
                )
                
                # Brinson attribution effects
                # Asset Allocation Effect = (Wp - Wb) * Rb
                allocation_effect = (portfolio_sector_weight - benchmark_sector_weight) * benchmark_sector_return
                
                # Security Selection Effect = Wb * (Rp - Rb)
                selection_effect = benchmark_sector_weight * (portfolio_sector_return - benchmark_sector_return)
                
                # Interaction Effect = (Wp - Wb) * (Rp - Rb)
                interaction_effect = (portfolio_sector_weight - benchmark_sector_weight) * \
                                   (portfolio_sector_return - benchmark_sector_return)
                
                sector_attribution[sector] = {
                    'portfolio_weight': float(portfolio_sector_weight),
                    'benchmark_weight': float(benchmark_sector_weight),
                    'portfolio_return': float(portfolio_sector_return),
                    'benchmark_return': float(benchmark_sector_return),
                    'allocation_effect': float(allocation_effect),
                    'selection_effect': float(selection_effect),
                    'interaction_effect': float(interaction_effect),
                    'total_contribution': float(allocation_effect + selection_effect + interaction_effect)
                }
            
            return sector_attribution
            
        except Exception as e:
            logger.error(f"Sector attribution calculation failed: {e}")
            raise AnalyticsError(f"Sector attribution calculation failed: {e}")
    
    def _group_by_sector(self, portfolio: Portfolio) -> Dict[str, List[str]]:
        """Group portfolio tickers by sector."""
        sectors = {}
        
        for ticker in portfolio.tickers:
            sector = self.sector_mapping.get(ticker, 'Other')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        
        return sectors
    
    def _calculate_sector_return(self, tickers: List[str], weights: List[float], 
                               price_data: pd.DataFrame) -> float:
        """Calculate sector return given tickers and weights."""
        try:
            if not tickers or not weights:
                return 0.0
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            
            normalized_weights = [w / total_weight for w in weights]
            
            # Calculate sector return
            sector_return = 0.0
            for ticker, weight in zip(tickers, normalized_weights):
                if ticker in price_data.columns:
                    ticker_returns = price_data[ticker].pct_change().dropna()
                    ticker_total_return = (1 + ticker_returns).prod() - 1
                    sector_return += weight * ticker_total_return
            
            return sector_return
            
        except Exception as e:
            logger.warning(f"Failed to calculate sector return: {e}")
            return 0.0
    
    def _calculate_individual_contributions(self, portfolio: Portfolio, benchmark_portfolio: Portfolio,
                                          portfolio_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate individual asset contributions to active return."""
        try:
            contributions = {}
            
            for i, ticker in enumerate(portfolio.tickers):
                # Portfolio weight
                portfolio_weight = portfolio.weights[i]
                
                # Benchmark weight (0 if not in benchmark)
                benchmark_weight = 0.0
                if ticker in benchmark_portfolio.tickers:
                    benchmark_idx = benchmark_portfolio.tickers.index(ticker)
                    benchmark_weight = benchmark_portfolio.weights[benchmark_idx]
                
                # Calculate returns
                if ticker in portfolio_data.columns:
                    ticker_returns = portfolio_data[ticker].pct_change().dropna()
                    ticker_total_return = (1 + ticker_returns).prod() - 1
                else:
                    ticker_total_return = 0.0
                
                # Benchmark return for this asset
                benchmark_return = 0.0
                if ticker in benchmark_data.columns:
                    benchmark_ticker_returns = benchmark_data[ticker].pct_change().dropna()
                    benchmark_return = (1 + benchmark_ticker_returns).prod() - 1
                
                # Contribution to active return
                # = Portfolio_weight * Asset_return - Benchmark_weight * Benchmark_return
                contribution = portfolio_weight * ticker_total_return - benchmark_weight * benchmark_return
                
                contributions[ticker] = {
                    'portfolio_weight': float(portfolio_weight),
                    'benchmark_weight': float(benchmark_weight),
                    'asset_return': float(ticker_total_return),
                    'benchmark_return': float(benchmark_return),
                    'active_contribution': float(contribution),
                    'sector': self.sector_mapping.get(ticker, 'Other')
                }
            
            return contributions
            
        except Exception as e:
            logger.error(f"Individual contributions calculation failed: {e}")
            raise AnalyticsError(f"Individual contributions calculation failed: {e}")
    
    def calculate_multi_period_attribution(self, portfolio: Portfolio, 
                                         benchmark_portfolio: Portfolio,
                                         periods: List[Tuple[date, date]]) -> Dict[str, Any]:
        """
        Calculate attribution analysis for multiple time periods.
        
        Args:
            portfolio: Portfolio object
            benchmark_portfolio: Benchmark portfolio object
            periods: List of (start_date, end_date) tuples
            
        Returns:
            Multi-period attribution results
        """
        try:
            logger.info(f"Calculating multi-period attribution for {len(periods)} periods")
            
            period_results = {}
            
            for i, (start_date, end_date) in enumerate(periods):
                period_name = f"Period_{i+1}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                
                try:
                    attribution = self.calculate_brinson_attribution(
                        portfolio, benchmark_portfolio, start_date, end_date
                    )
                    period_results[period_name] = attribution
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate attribution for period {period_name}: {e}")
                    period_results[period_name] = {'error': str(e)}
            
            # Calculate aggregate statistics
            valid_periods = {k: v for k, v in period_results.items() if 'error' not in v}
            
            if valid_periods:
                aggregate_stats = self._calculate_aggregate_attribution_stats(valid_periods)
                period_results['aggregate_statistics'] = aggregate_stats
            
            return {
                'portfolio_id': portfolio.id,
                'total_periods': len(periods),
                'successful_periods': len(valid_periods),
                'period_results': period_results
            }
            
        except Exception as e:
            logger.error(f"Multi-period attribution calculation failed: {e}")
            raise AnalyticsError(f"Multi-period attribution calculation failed: {e}")
    
    def _calculate_aggregate_attribution_stats(self, period_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate statistics across multiple periods."""
        try:
            # Extract attribution effects from all periods
            allocation_effects = []
            selection_effects = []
            interaction_effects = []
            active_returns = []
            
            for period_data in period_results.values():
                if 'attribution_effects' in period_data:
                    allocation_effects.append(period_data['attribution_effects']['asset_allocation'])
                    selection_effects.append(period_data['attribution_effects']['security_selection'])
                    interaction_effects.append(period_data['attribution_effects']['interaction'])
                
                if 'total_returns' in period_data:
                    active_returns.append(period_data['total_returns']['active'])
            
            aggregate_stats = {
                'average_effects': {
                    'asset_allocation': float(np.mean(allocation_effects)) if allocation_effects else 0.0,
                    'security_selection': float(np.mean(selection_effects)) if selection_effects else 0.0,
                    'interaction': float(np.mean(interaction_effects)) if interaction_effects else 0.0,
                    'active_return': float(np.mean(active_returns)) if active_returns else 0.0
                },
                'volatility_effects': {
                    'asset_allocation': float(np.std(allocation_effects)) if allocation_effects else 0.0,
                    'security_selection': float(np.std(selection_effects)) if selection_effects else 0.0,
                    'interaction': float(np.std(interaction_effects)) if interaction_effects else 0.0,
                    'active_return': float(np.std(active_returns)) if active_returns else 0.0
                },
                'consistency_metrics': {
                    'positive_allocation_periods': sum(1 for x in allocation_effects if x > 0),
                    'positive_selection_periods': sum(1 for x in selection_effects if x > 0),
                    'positive_active_return_periods': sum(1 for x in active_returns if x > 0),
                    'total_periods': len(period_results)
                }
            }
            
            return aggregate_stats
            
        except Exception as e:
            logger.warning(f"Failed to calculate aggregate attribution stats: {e}")
            return {}