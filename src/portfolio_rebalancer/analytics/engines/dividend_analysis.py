"""Dividend analysis engine for income-focused portfolio analysis."""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from collections import defaultdict

from ...common.interfaces import DataStorage
from ..models import DividendAnalysis, AnalyticsError

logger = logging.getLogger(__name__)


class DividendAnalyzer:
    """Dividend and income analysis for income-focused portfolio analysis."""
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize dividend analyzer.
        
        Args:
            data_storage: Data storage interface for historical data
        """
        self.data_storage = data_storage
        self._dividend_cache = {}  # Cache for dividend data
        self._yield_cache = {}     # Cache for yield data
        logger.info("Dividend analyzer initialized")
    
    def analyze_dividend_income(self, 
                              portfolio_id: str,
                              tickers: List[str], 
                              weights: List[float],
                              portfolio_value: float = 100000.0) -> DividendAnalysis:
        """
        Analyze current and projected dividend income.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            portfolio_value: Total portfolio value for income calculations
            
        Returns:
            Dividend analysis results
            
        Raises:
            AnalyticsError: If dividend analysis fails
        """
        try:
            logger.info(f"Starting dividend analysis for portfolio {portfolio_id} with {len(tickers)} holdings")
            
            if len(tickers) != len(weights):
                raise AnalyticsError("Tickers and weights must have the same length")
            
            if abs(sum(weights) - 1.0) > 0.01:
                raise AnalyticsError("Portfolio weights must sum to 1.0")
            
            # Fetch dividend data for all holdings
            dividend_data = self._fetch_dividend_data(tickers)
            
            # Calculate individual holding metrics
            holding_metrics = []
            total_annual_income = 0.0
            weighted_yield = 0.0
            weighted_growth_rate = 0.0
            weighted_payout_ratio = 0.0
            weighted_coverage = 0.0
            
            for ticker, weight in zip(tickers, weights):
                holding_value = portfolio_value * weight
                
                # Get dividend metrics for this holding
                holding_data = dividend_data.get(ticker, {})
                annual_dividend = holding_data.get('annual_dividend', 0.0)
                current_price = holding_data.get('current_price', 1.0)
                dividend_yield = holding_data.get('dividend_yield', 0.0)
                growth_rate = holding_data.get('growth_rate', 0.0)
                payout_ratio = holding_data.get('payout_ratio', 0.0)
                coverage_ratio = holding_data.get('coverage_ratio', 1.0)
                
                # Calculate income from this holding
                shares = holding_value / current_price if current_price > 0 else 0
                holding_annual_income = shares * annual_dividend
                total_annual_income += holding_annual_income
                
                # Weight the metrics by portfolio allocation
                weighted_yield += dividend_yield * weight
                weighted_growth_rate += growth_rate * weight
                weighted_payout_ratio += payout_ratio * weight
                weighted_coverage += coverage_ratio * weight
                
                holding_metrics.append({
                    'ticker': ticker,
                    'weight': weight,
                    'holding_value': holding_value,
                    'shares': shares,
                    'annual_dividend_per_share': annual_dividend,
                    'annual_income': holding_annual_income,
                    'dividend_yield': dividend_yield,
                    'growth_rate': growth_rate,
                    'payout_ratio': payout_ratio,
                    'coverage_ratio': coverage_ratio
                })
            
            # Calculate portfolio-level yield
            portfolio_yield = total_annual_income / portfolio_value if portfolio_value > 0 else 0.0
            
            # Calculate sustainability score
            sustainability_score = self._calculate_sustainability_score(holding_metrics)
            
            # Get top dividend contributors
            top_contributors = self._get_top_contributors(holding_metrics)
            
            # Create detailed dividend data
            detailed_data = {
                'portfolio_value': portfolio_value,
                'total_annual_income': total_annual_income,
                'holding_metrics': holding_metrics,
                'weighted_metrics': {
                    'yield': weighted_yield,
                    'growth_rate': weighted_growth_rate,
                    'payout_ratio': weighted_payout_ratio,
                    'coverage_ratio': weighted_coverage
                },
                'analysis_date': date.today().isoformat(),
                'num_dividend_paying_holdings': sum(1 for h in holding_metrics if h['annual_dividend_per_share'] > 0)
            }
            
            analysis = DividendAnalysis(
                portfolio_id=portfolio_id,
                analysis_date=date.today(),
                current_yield=portfolio_yield,
                projected_annual_income=total_annual_income,
                dividend_growth_rate=weighted_growth_rate,
                payout_ratio=weighted_payout_ratio,
                dividend_coverage=weighted_coverage,
                income_sustainability_score=sustainability_score,
                dividend_data=detailed_data,
                top_contributors=top_contributors
            )
            
            logger.info(f"Dividend analysis completed for portfolio {portfolio_id}: "
                       f"Yield={portfolio_yield:.3f}, Income=${total_annual_income:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Dividend analysis failed: {e}")
            raise AnalyticsError(f"Dividend analysis failed: {e}")
    
    def _fetch_dividend_data(self, tickers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetch dividend data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to dividend metrics
        """
        dividend_data = {}
        
        for ticker in tickers:
            try:
                # Check cache first
                if ticker in self._dividend_cache:
                    cache_time, cached_data = self._dividend_cache[ticker]
                    if (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                        dividend_data[ticker] = cached_data
                        continue
                
                logger.debug(f"Fetching dividend data for {ticker}")
                
                # Fetch data using yfinance
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                
                # Get dividend and financial metrics
                annual_dividend = info.get('dividendRate', 0.0) or 0.0
                dividend_yield = info.get('dividendYield', 0.0) or 0.0
                current_price = info.get('currentPrice', 0.0) or info.get('regularMarketPrice', 1.0)
                payout_ratio = info.get('payoutRatio', 0.0) or 0.0
                
                # Calculate dividend coverage ratio (inverse of payout ratio)
                coverage_ratio = 1.0 / payout_ratio if payout_ratio > 0 else 1.0
                
                # Get historical dividends for growth rate calculation
                growth_rate = self._calculate_dividend_growth_rate(yf_ticker)
                
                ticker_data = {
                    'annual_dividend': annual_dividend,
                    'dividend_yield': dividend_yield,
                    'current_price': current_price,
                    'payout_ratio': payout_ratio,
                    'coverage_ratio': coverage_ratio,
                    'growth_rate': growth_rate
                }
                
                dividend_data[ticker] = ticker_data
                
                # Cache the data
                self._dividend_cache[ticker] = (datetime.now(), ticker_data)
                
            except Exception as e:
                logger.warning(f"Failed to fetch dividend data for {ticker}: {e}")
                # Use default values for failed fetches
                dividend_data[ticker] = {
                    'annual_dividend': 0.0,
                    'dividend_yield': 0.0,
                    'current_price': 1.0,
                    'payout_ratio': 0.0,
                    'coverage_ratio': 1.0,
                    'growth_rate': 0.0
                }
        
        return dividend_data
    
    def _calculate_dividend_growth_rate(self, yf_ticker) -> float:
        """
        Calculate historical dividend growth rate.
        
        Args:
            yf_ticker: yfinance Ticker object
            
        Returns:
            Annualized dividend growth rate
        """
        try:
            # Get dividend history for the last 5 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)
            
            dividends = yf_ticker.dividends
            if dividends.empty:
                return 0.0
            
            # Filter to last 5 years
            recent_dividends = dividends[dividends.index >= start_date]
            if len(recent_dividends) < 2:
                return 0.0
            
            # Group by year and sum annual dividends
            annual_dividends = recent_dividends.groupby(recent_dividends.index.year).sum()
            
            if len(annual_dividends) < 2:
                return 0.0
            
            # Calculate compound annual growth rate (CAGR)
            first_year_dividend = annual_dividends.iloc[0]
            last_year_dividend = annual_dividends.iloc[-1]
            years = len(annual_dividends) - 1
            
            if first_year_dividend <= 0:
                return 0.0
            
            growth_rate = (last_year_dividend / first_year_dividend) ** (1/years) - 1
            
            # Cap growth rate at reasonable bounds (-50% to +50%)
            growth_rate = max(-0.5, min(0.5, growth_rate))
            
            return growth_rate
            
        except Exception as e:
            logger.debug(f"Could not calculate dividend growth rate: {e}")
            return 0.0
    
    def _calculate_sustainability_score(self, holding_metrics: List[Dict[str, Any]]) -> float:
        """
        Calculate income sustainability score based on various factors.
        
        Args:
            holding_metrics: List of holding metrics
            
        Returns:
            Sustainability score between 0 and 1
        """
        if not holding_metrics:
            return 0.0
        
        total_weight = sum(h['weight'] for h in holding_metrics)
        if total_weight == 0:
            return 0.0
        
        weighted_score = 0.0
        
        for holding in holding_metrics:
            weight = holding['weight']
            payout_ratio = holding['payout_ratio']
            coverage_ratio = holding['coverage_ratio']
            growth_rate = holding['growth_rate']
            
            # Score components (each 0-1)
            payout_score = max(0, 1 - max(0, payout_ratio - 0.6) / 0.4)  # Penalize payout > 60%
            coverage_score = min(1, coverage_ratio / 2.0)  # Good coverage is 2x or higher
            growth_score = min(1, max(0, (growth_rate + 0.1) / 0.2))  # Reward positive growth
            
            # Weighted average of components
            holding_score = (payout_score * 0.4 + coverage_score * 0.4 + growth_score * 0.2)
            weighted_score += holding_score * weight
        
        return weighted_score / total_weight
    
    def _get_top_contributors(self, holding_metrics: List[Dict[str, Any]], top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top dividend contributing holdings.
        
        Args:
            holding_metrics: List of holding metrics
            top_n: Number of top contributors to return
            
        Returns:
            List of tuples (ticker, annual_income)
        """
        # Sort by annual income contribution
        sorted_holdings = sorted(
            holding_metrics, 
            key=lambda x: x['annual_income'], 
            reverse=True
        )
        
        return [(h['ticker'], h['annual_income']) for h in sorted_holdings[:top_n]]

    def project_income(self, 
                      portfolio_id: str,
                      tickers: List[str],
                      weights: List[float],
                      years: int = 5,
                      portfolio_value: float = 100000.0,
                      scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Project future dividend income with multiple scenarios and detailed analysis.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            years: Number of years to project
            portfolio_value: Current portfolio value
            scenarios: List of scenarios to model ('conservative', 'base', 'optimistic')
            
        Returns:
            Comprehensive income projection results
        """
        try:
            logger.info(f"Projecting income for {years} years for portfolio {portfolio_id}")
            
            if scenarios is None:
                scenarios = ['conservative', 'base', 'optimistic']
            
            # Get current dividend analysis
            current_analysis = self.analyze_dividend_income(portfolio_id, tickers, weights, portfolio_value)
            current_income = current_analysis.projected_annual_income
            base_growth_rate = current_analysis.dividend_growth_rate
            
            # Define scenario parameters
            scenario_params = {
                'conservative': {
                    'growth_multiplier': 0.5,  # 50% of base growth rate
                    'volatility': 0.15,        # 15% volatility
                    'recession_probability': 0.3  # 30% chance of recession impact
                },
                'base': {
                    'growth_multiplier': 1.0,  # Base growth rate
                    'volatility': 0.10,        # 10% volatility
                    'recession_probability': 0.2  # 20% chance of recession impact
                },
                'optimistic': {
                    'growth_multiplier': 1.5,  # 150% of base growth rate
                    'volatility': 0.20,        # 20% volatility
                    'recession_probability': 0.1  # 10% chance of recession impact
                }
            }
            
            scenario_projections = {}
            
            for scenario in scenarios:
                params = scenario_params.get(scenario, scenario_params['base'])
                scenario_growth_rate = base_growth_rate * params['growth_multiplier']
                
                projections = {}
                cumulative_income = 0.0
                
                for year in range(1, years + 1):
                    # Base projection with growth
                    base_projected_income = current_income * ((1 + scenario_growth_rate) ** year)
                    
                    # Add volatility and recession risk
                    volatility_factor = 1.0 + (np.random.normal(0, params['volatility']) if year > 1 else 0)
                    recession_factor = 0.7 if (year > 2 and np.random.random() < params['recession_probability']) else 1.0
                    
                    projected_income = base_projected_income * volatility_factor * recession_factor
                    projected_income = max(0, projected_income)  # Ensure non-negative
                    
                    cumulative_income += projected_income
                    
                    projections[f"year_{year}"] = {
                        'projected_income': projected_income,
                        'growth_from_current': (projected_income / current_income) - 1 if current_income > 0 else 0,
                        'cumulative_income': cumulative_income,
                        'volatility_factor': volatility_factor,
                        'recession_factor': recession_factor
                    }
                
                scenario_projections[scenario] = {
                    'growth_rate': scenario_growth_rate,
                    'projections': projections,
                    'total_projected_income': cumulative_income,
                    'final_year_income': projections[f"year_{years}"]['projected_income'] if years > 0 else current_income,
                    'average_annual_income': cumulative_income / years if years > 0 else current_income
                }
            
            # Calculate holding-level projections
            holding_projections = self._project_holding_level_income(
                tickers, weights, portfolio_value, years, base_growth_rate
            )
            
            # Analyze seasonal patterns
            seasonal_analysis = self._analyze_seasonal_patterns(tickers, weights)
            
            projection_result = {
                'portfolio_id': portfolio_id,
                'projection_years': years,
                'current_annual_income': current_income,
                'base_growth_rate': base_growth_rate,
                'scenario_projections': scenario_projections,
                'holding_projections': holding_projections,
                'seasonal_analysis': seasonal_analysis,
                'projection_date': date.today().isoformat(),
                'confidence_intervals': self._calculate_projection_confidence_intervals(scenario_projections)
            }
            
            return projection_result
            
        except Exception as e:
            logger.error(f"Income projection failed: {e}")
            raise AnalyticsError(f"Income projection failed: {e}")
    
    def _project_holding_level_income(self, 
                                    tickers: List[str], 
                                    weights: List[float], 
                                    portfolio_value: float,
                                    years: int,
                                    base_growth_rate: float) -> Dict[str, Any]:
        """
        Project income at the individual holding level.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            portfolio_value: Total portfolio value
            years: Number of years to project
            base_growth_rate: Base growth rate assumption
            
        Returns:
            Holding-level income projections
        """
        dividend_data = self._fetch_dividend_data(tickers)
        holding_projections = {}
        
        for ticker, weight in zip(tickers, weights):
            holding_value = portfolio_value * weight
            holding_data = dividend_data.get(ticker, {})
            
            annual_dividend = holding_data.get('annual_dividend', 0.0)
            current_price = holding_data.get('current_price', 1.0)
            holding_growth_rate = holding_data.get('growth_rate', base_growth_rate)
            
            if current_price > 0:
                shares = holding_value / current_price
                current_annual_income = shares * annual_dividend
                
                yearly_projections = {}
                for year in range(1, years + 1):
                    projected_dividend = annual_dividend * ((1 + holding_growth_rate) ** year)
                    projected_income = shares * projected_dividend
                    
                    yearly_projections[f"year_{year}"] = {
                        'projected_dividend_per_share': projected_dividend,
                        'projected_annual_income': projected_income
                    }
                
                holding_projections[ticker] = {
                    'current_shares': shares,
                    'current_annual_income': current_annual_income,
                    'growth_rate': holding_growth_rate,
                    'yearly_projections': yearly_projections
                }
        
        return holding_projections
    
    def _analyze_seasonal_patterns(self, tickers: List[str], weights: List[float]) -> Dict[str, Any]:
        """
        Analyze seasonal dividend payment patterns.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            
        Returns:
            Seasonal pattern analysis
        """
        try:
            # Simplified seasonal analysis
            # In a full implementation, this would analyze historical dividend payment dates
            
            quarterly_weight = 0.0
            monthly_weight = 0.0
            semi_annual_weight = 0.0
            annual_weight = 0.0
            
            for ticker, weight in zip(tickers, weights):
                # Most US stocks pay quarterly, bonds may pay monthly
                if 'BND' in ticker or 'BOND' in ticker:
                    monthly_weight += weight
                else:
                    quarterly_weight += weight
            
            # Estimate monthly income distribution
            monthly_distribution = {}
            for month in range(1, 13):
                if month in [3, 6, 9, 12]:  # Typical dividend months
                    monthly_distribution[f"month_{month}"] = quarterly_weight * 0.25
                else:
                    monthly_distribution[f"month_{month}"] = monthly_weight / 12
            
            return {
                'payment_frequency_weights': {
                    'quarterly': quarterly_weight,
                    'monthly': monthly_weight,
                    'semi_annual': semi_annual_weight,
                    'annual': annual_weight
                },
                'monthly_income_distribution': monthly_distribution,
                'peak_income_months': [3, 6, 9, 12],  # Typical dividend payment months
                'estimated_quarterly_income': quarterly_weight * 0.25
            }
            
        except Exception as e:
            logger.warning(f"Seasonal pattern analysis failed: {e}")
            return {}
    
    def _calculate_projection_confidence_intervals(self, scenario_projections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate confidence intervals for income projections.
        
        Args:
            scenario_projections: Projections for different scenarios
            
        Returns:
            Confidence interval data
        """
        try:
            if not scenario_projections:
                return {}
            
            # Extract final year incomes for each scenario
            final_incomes = []
            total_incomes = []
            
            for scenario, data in scenario_projections.items():
                final_incomes.append(data['final_year_income'])
                total_incomes.append(data['total_projected_income'])
            
            if not final_incomes:
                return {}
            
            # Calculate percentiles
            final_income_p10 = np.percentile(final_incomes, 10)
            final_income_p50 = np.percentile(final_incomes, 50)
            final_income_p90 = np.percentile(final_incomes, 90)
            
            total_income_p10 = np.percentile(total_incomes, 10)
            total_income_p50 = np.percentile(total_incomes, 50)
            total_income_p90 = np.percentile(total_incomes, 90)
            
            return {
                'final_year_income': {
                    'p10': final_income_p10,
                    'p50': final_income_p50,
                    'p90': final_income_p90,
                    'range': final_income_p90 - final_income_p10
                },
                'total_projected_income': {
                    'p10': total_income_p10,
                    'p50': total_income_p50,
                    'p90': total_income_p90,
                    'range': total_income_p90 - total_income_p10
                }
            }
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return {}
    
    def analyze_sustainability(self, 
                             portfolio_id: str,
                             tickers: List[str],
                             weights: List[float],
                             portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """
        Analyze dividend sustainability and coverage ratios.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            portfolio_value: Total portfolio value
            
        Returns:
            Sustainability analysis results
        """
        try:
            logger.info(f"Analyzing dividend sustainability for portfolio {portfolio_id}")
            
            # Get current dividend analysis
            analysis = self.analyze_dividend_income(portfolio_id, tickers, weights, portfolio_value)
            holding_metrics = analysis.dividend_data.get('holding_metrics', [])
            
            # Calculate risk factors
            high_payout_holdings = []
            declining_earnings_holdings = []
            low_coverage_holdings = []
            
            sustainability_by_holding = {}
            
            for holding in holding_metrics:
                ticker = holding['ticker']
                payout_ratio = holding['payout_ratio']
                coverage_ratio = holding['coverage_ratio']
                growth_rate = holding['growth_rate']
                
                # Identify risk factors
                if payout_ratio > 0.8:  # Payout ratio > 80%
                    high_payout_holdings.append(ticker)
                
                if growth_rate < -0.05:  # Declining dividend growth > 5%
                    declining_earnings_holdings.append(ticker)
                
                if coverage_ratio < 1.2:  # Coverage ratio < 1.2x
                    low_coverage_holdings.append(ticker)
                
                # Calculate individual sustainability score
                payout_score = max(0, 1 - max(0, payout_ratio - 0.6) / 0.4)
                coverage_score = min(1, coverage_ratio / 2.0)
                growth_score = min(1, max(0, (growth_rate + 0.1) / 0.2))
                
                holding_sustainability = (payout_score * 0.4 + coverage_score * 0.4 + growth_score * 0.2)
                
                sustainability_by_holding[ticker] = {
                    'sustainability_score': holding_sustainability,
                    'payout_ratio': payout_ratio,
                    'coverage_ratio': coverage_ratio,
                    'growth_rate': growth_rate,
                    'risk_factors': {
                        'high_payout': payout_ratio > 0.8,
                        'declining_growth': growth_rate < -0.05,
                        'low_coverage': coverage_ratio < 1.2
                    }
                }
            
            # Generate recommendations
            recommendations = []
            
            if high_payout_holdings:
                recommendations.append(f"Monitor {len(high_payout_holdings)} holdings with high payout ratios (>80%): {', '.join(high_payout_holdings[:3])}")
            
            if declining_earnings_holdings:
                recommendations.append(f"Review {len(declining_earnings_holdings)} holdings with declining dividend growth: {', '.join(declining_earnings_holdings[:3])}")
            
            if low_coverage_holdings:
                recommendations.append(f"Consider reducing exposure to {len(low_coverage_holdings)} holdings with low coverage ratios: {', '.join(low_coverage_holdings[:3])}")
            
            if analysis.income_sustainability_score < 0.6:
                recommendations.append("Overall portfolio sustainability is below 60% - consider rebalancing toward more sustainable dividend payers")
            
            if not recommendations:
                recommendations.append("Portfolio dividend sustainability appears healthy based on current metrics")
            
            sustainability_analysis = {
                'portfolio_id': portfolio_id,
                'analysis_date': date.today().isoformat(),
                'overall_sustainability_score': analysis.income_sustainability_score,
                'coverage_metrics': {
                    'avg_payout_ratio': analysis.payout_ratio,
                    'avg_coverage_ratio': analysis.dividend_coverage,
                    'weighted_growth_rate': analysis.dividend_growth_rate
                },
                'risk_factors': {
                    'high_payout_ratio_holdings': len(high_payout_holdings),
                    'declining_earnings_holdings': len(declining_earnings_holdings),
                    'low_coverage_holdings': len(low_coverage_holdings)
                },
                'risk_holdings': {
                    'high_payout': high_payout_holdings,
                    'declining_growth': declining_earnings_holdings,
                    'low_coverage': low_coverage_holdings
                },
                'sustainability_by_holding': sustainability_by_holding,
                'recommendations': recommendations
            }
            
            return sustainability_analysis
            
        except Exception as e:
            logger.error(f"Sustainability analysis failed: {e}")
            raise AnalyticsError(f"Sustainability analysis failed: {e}")
    
    def analyze_dividend_coverage_ratios(self, 
                                       tickers: List[str],
                                       weights: List[float]) -> Dict[str, Any]:
        """
        Perform detailed dividend coverage ratio analysis.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            
        Returns:
            Detailed coverage ratio analysis
        """
        try:
            logger.info("Analyzing dividend coverage ratios")
            
            dividend_data = self._fetch_dividend_data(tickers)
            coverage_analysis = {}
            
            weighted_coverage = 0.0
            weighted_payout = 0.0
            coverage_distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
            
            for ticker, weight in zip(tickers, weights):
                holding_data = dividend_data.get(ticker, {})
                coverage_ratio = holding_data.get('coverage_ratio', 1.0)
                payout_ratio = holding_data.get('payout_ratio', 0.0)
                
                weighted_coverage += coverage_ratio * weight
                weighted_payout += payout_ratio * weight
                
                # Classify coverage quality
                if coverage_ratio >= 2.0:
                    coverage_quality = 'excellent'
                elif coverage_ratio >= 1.5:
                    coverage_quality = 'good'
                elif coverage_ratio >= 1.2:
                    coverage_quality = 'fair'
                else:
                    coverage_quality = 'poor'
                
                coverage_distribution[coverage_quality] += weight
                
                coverage_analysis[ticker] = {
                    'coverage_ratio': coverage_ratio,
                    'payout_ratio': payout_ratio,
                    'coverage_quality': coverage_quality,
                    'weight': weight,
                    'sustainability_risk': 'high' if coverage_ratio < 1.2 else 'medium' if coverage_ratio < 1.5 else 'low'
                }
            
            return {
                'weighted_coverage_ratio': weighted_coverage,
                'weighted_payout_ratio': weighted_payout,
                'coverage_distribution': coverage_distribution,
                'holding_analysis': coverage_analysis,
                'portfolio_coverage_quality': self._assess_portfolio_coverage_quality(weighted_coverage),
                'recommendations': self._generate_coverage_recommendations(coverage_analysis)
            }
            
        except Exception as e:
            logger.error(f"Coverage ratio analysis failed: {e}")
            raise AnalyticsError(f"Coverage ratio analysis failed: {e}")
    
    def _assess_portfolio_coverage_quality(self, weighted_coverage: float) -> str:
        """Assess overall portfolio coverage quality."""
        if weighted_coverage >= 2.0:
            return 'excellent'
        elif weighted_coverage >= 1.5:
            return 'good'
        elif weighted_coverage >= 1.2:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_coverage_recommendations(self, coverage_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on coverage analysis."""
        recommendations = []
        
        poor_coverage_holdings = [
            ticker for ticker, data in coverage_analysis.items()
            if data['coverage_ratio'] < 1.2
        ]
        
        high_payout_holdings = [
            ticker for ticker, data in coverage_analysis.items()
            if data['payout_ratio'] > 0.8
        ]
        
        if poor_coverage_holdings:
            recommendations.append(
                f"Consider reducing exposure to holdings with poor coverage ratios: {', '.join(poor_coverage_holdings[:3])}"
            )
        
        if high_payout_holdings:
            recommendations.append(
                f"Monitor holdings with high payout ratios (>80%): {', '.join(high_payout_holdings[:3])}"
            )
        
        if not recommendations:
            recommendations.append("Portfolio dividend coverage appears healthy")
        
        return recommendations
    
    def analyze_income_optimization_opportunities(self, 
                                                portfolio_id: str,
                                                tickers: List[str],
                                                weights: List[float],
                                                target_yield: float = 0.04,
                                                risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Analyze opportunities to optimize income while maintaining risk targets.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            target_yield: Target dividend yield
            risk_tolerance: Risk tolerance level ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Income optimization analysis and suggestions
        """
        try:
            logger.info(f"Analyzing income optimization opportunities for {portfolio_id}")
            
            current_analysis = self.analyze_dividend_income(portfolio_id, tickers, weights)
            current_yield = current_analysis.current_yield
            
            # Define risk parameters
            risk_params = {
                'conservative': {'max_payout_ratio': 0.6, 'min_coverage_ratio': 2.0, 'max_concentration': 0.1},
                'moderate': {'max_payout_ratio': 0.75, 'min_coverage_ratio': 1.5, 'max_concentration': 0.15},
                'aggressive': {'max_payout_ratio': 0.9, 'min_coverage_ratio': 1.2, 'max_concentration': 0.25}
            }
            
            params = risk_params.get(risk_tolerance, risk_params['moderate'])
            
            # Analyze current holdings
            dividend_data = self._fetch_dividend_data(tickers)
            holding_analysis = []
            
            for ticker, weight in zip(tickers, weights):
                holding_data = dividend_data.get(ticker, {})
                yield_contribution = holding_data.get('dividend_yield', 0.0) * weight
                
                holding_analysis.append({
                    'ticker': ticker,
                    'weight': weight,
                    'dividend_yield': holding_data.get('dividend_yield', 0.0),
                    'yield_contribution': yield_contribution,
                    'payout_ratio': holding_data.get('payout_ratio', 0.0),
                    'coverage_ratio': holding_data.get('coverage_ratio', 1.0),
                    'meets_risk_criteria': (
                        holding_data.get('payout_ratio', 0.0) <= params['max_payout_ratio'] and
                        holding_data.get('coverage_ratio', 1.0) >= params['min_coverage_ratio']
                    )
                })
            
            # Identify optimization opportunities
            yield_gap = target_yield - current_yield
            
            optimization_suggestions = []
            
            if yield_gap > 0:
                # Need to increase yield
                high_yield_candidates = [
                    h for h in holding_analysis 
                    if h['dividend_yield'] > target_yield and h['meets_risk_criteria']
                ]
                
                if high_yield_candidates:
                    best_candidate = max(high_yield_candidates, key=lambda x: x['dividend_yield'])
                    optimization_suggestions.append({
                        'action': 'increase_allocation',
                        'ticker': best_candidate['ticker'],
                        'current_weight': best_candidate['weight'],
                        'suggested_weight': min(best_candidate['weight'] + 0.05, params['max_concentration']),
                        'expected_yield_improvement': best_candidate['dividend_yield'] * 0.05,
                        'rationale': f"High yield ({best_candidate['dividend_yield']:.2%}) with acceptable risk metrics"
                    })
                
                # Identify low-yield holdings to reduce
                low_yield_holdings = [
                    h for h in holding_analysis 
                    if h['dividend_yield'] < current_yield * 0.5 and h['weight'] > 0.05
                ]
                
                for holding in low_yield_holdings[:2]:  # Top 2 candidates
                    optimization_suggestions.append({
                        'action': 'reduce_allocation',
                        'ticker': holding['ticker'],
                        'current_weight': holding['weight'],
                        'suggested_weight': max(holding['weight'] - 0.03, 0.02),
                        'yield_drag': holding['yield_contribution'],
                        'rationale': f"Low yield ({holding['dividend_yield']:.2%}) dragging down portfolio income"
                    })
            
            elif yield_gap < -0.01:  # Current yield significantly above target
                optimization_suggestions.append({
                    'action': 'maintain_current',
                    'rationale': f"Current yield ({current_yield:.2%}) exceeds target ({target_yield:.2%})"
                })
            
            return {
                'portfolio_id': portfolio_id,
                'current_yield': current_yield,
                'target_yield': target_yield,
                'yield_gap': yield_gap,
                'risk_tolerance': risk_tolerance,
                'risk_parameters': params,
                'holding_analysis': holding_analysis,
                'optimization_suggestions': optimization_suggestions,
                'expected_yield_after_optimization': self._estimate_optimized_yield(
                    holding_analysis, optimization_suggestions
                )
            }
            
        except Exception as e:
            logger.error(f"Income optimization analysis failed: {e}")
            raise AnalyticsError(f"Income optimization analysis failed: {e}")
    
    def _estimate_optimized_yield(self, 
                                holding_analysis: List[Dict[str, Any]], 
                                suggestions: List[Dict[str, Any]]) -> float:
        """
        Estimate portfolio yield after applying optimization suggestions.
        
        Args:
            holding_analysis: Current holding analysis
            suggestions: Optimization suggestions
            
        Returns:
            Estimated optimized yield
        """
        try:
            # Create a copy of current weights
            optimized_weights = {h['ticker']: h['weight'] for h in holding_analysis}
            
            # Apply suggestions
            for suggestion in suggestions:
                if suggestion['action'] in ['increase_allocation', 'reduce_allocation']:
                    ticker = suggestion['ticker']
                    optimized_weights[ticker] = suggestion['suggested_weight']
            
            # Normalize weights to sum to 1
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v/total_weight for k, v in optimized_weights.items()}
            
            # Calculate optimized yield
            optimized_yield = 0.0
            for holding in holding_analysis:
                ticker = holding['ticker']
                new_weight = optimized_weights.get(ticker, holding['weight'])
                optimized_yield += holding['dividend_yield'] * new_weight
            
            return optimized_yield
            
        except Exception as e:
            logger.warning(f"Could not estimate optimized yield: {e}")
            return 0.0
    
    def get_top_dividend_contributors(self, 
                                    tickers: List[str],
                                    weights: List[float],
                                    portfolio_value: float = 100000.0,
                                    top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top dividend contributing holdings.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            portfolio_value: Total portfolio value
            top_n: Number of top contributors to return
            
        Returns:
            List of tuples (ticker, annual_income)
        """
        try:
            logger.info(f"Getting top {top_n} dividend contributors")
            
            # Fetch dividend data
            dividend_data = self._fetch_dividend_data(tickers)
            
            contributors = []
            for ticker, weight in zip(tickers, weights):
                holding_value = portfolio_value * weight
                holding_data = dividend_data.get(ticker, {})
                annual_dividend = holding_data.get('annual_dividend', 0.0)
                current_price = holding_data.get('current_price', 1.0)
                
                if current_price > 0:
                    shares = holding_value / current_price
                    annual_income = shares * annual_dividend
                    contributors.append((ticker, annual_income))
            
            # Sort by annual income and return top N
            contributors.sort(key=lambda x: x[1], reverse=True)
            return contributors[:top_n]
            
        except Exception as e:
            logger.error(f"Top contributors calculation failed: {e}")
            raise AnalyticsError(f"Top contributors calculation failed: {e}")
    
    def calculate_yield_metrics(self, 
                              tickers: List[str],
                              weights: List[float],
                              portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """
        Calculate various yield metrics for the portfolio.
        
        Args:
            tickers: List of ticker symbols
            weights: Portfolio weights
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary of yield metrics
        """
        try:
            logger.info("Calculating yield metrics")
            
            # Fetch dividend data
            dividend_data = self._fetch_dividend_data(tickers)
            
            # Calculate weighted metrics
            weighted_yield = 0.0
            weighted_forward_yield = 0.0
            total_annual_income = 0.0
            
            # Distribution frequency analysis
            frequency_weights = defaultdict(float)
            
            for ticker, weight in zip(tickers, weights):
                holding_data = dividend_data.get(ticker, {})
                dividend_yield = holding_data.get('dividend_yield', 0.0)
                annual_dividend = holding_data.get('annual_dividend', 0.0)
                current_price = holding_data.get('current_price', 1.0)
                
                # Weight the yields
                weighted_yield += dividend_yield * weight
                
                # Calculate income contribution
                holding_value = portfolio_value * weight
                if current_price > 0:
                    shares = holding_value / current_price
                    annual_income = shares * annual_dividend
                    total_annual_income += annual_income
                
                # Estimate distribution frequency (simplified)
                if annual_dividend > 0:
                    # Most US stocks pay quarterly
                    frequency_weights['quarterly'] += weight
                else:
                    frequency_weights['none'] += weight
            
            # Calculate portfolio yield
            portfolio_yield = total_annual_income / portfolio_value if portfolio_value > 0 else 0.0
            
            # Estimate trailing 12-month yield (assume similar to current)
            trailing_12m_yield = portfolio_yield * 0.95  # Slightly lower estimate
            
            # Estimate forward yield (based on growth expectations)
            forward_yield = portfolio_yield * 1.03  # Slightly higher estimate
            
            # Normalize frequency weights
            total_weight = sum(frequency_weights.values())
            if total_weight > 0:
                for freq in frequency_weights:
                    frequency_weights[freq] /= total_weight
            
            yield_metrics = {
                'current_yield': portfolio_yield,
                'trailing_12m_yield': trailing_12m_yield,
                'forward_yield': forward_yield,
                'weighted_yield': weighted_yield,
                'total_annual_income': total_annual_income,
                'distribution_frequency': dict(frequency_weights),
                'yield_by_holding': {
                    ticker: dividend_data.get(ticker, {}).get('dividend_yield', 0.0)
                    for ticker in tickers
                }
            }
            
            return yield_metrics
            
        except Exception as e:
            logger.error(f"Yield metrics calculation failed: {e}")
            raise AnalyticsError(f"Yield metrics calculation failed: {e}")
    
    def record_dividend_payment(self, 
                              portfolio_id: str,
                              ticker: str,
                              payment_date: date,
                              amount_per_share: float,
                              shares: float) -> None:
        """
        Record a dividend payment for tracking income history.
        
        Args:
            portfolio_id: Portfolio identifier
            ticker: Ticker symbol
            payment_date: Date of dividend payment
            amount_per_share: Dividend amount per share
            shares: Number of shares owned
        """
        try:
            logger.info(f"Recording dividend payment: {ticker} ${amount_per_share}/share on {payment_date}")
            
            # This would typically store to a dividend payments table
            # For now, we'll log the information
            total_payment = amount_per_share * shares
            
            logger.info(f"Dividend recorded: {portfolio_id} received ${total_payment:.2f} "
                       f"from {ticker} ({shares} shares @ ${amount_per_share}/share)")
            
            # In a full implementation, this would:
            # 1. Store to dividend_payments table
            # 2. Update portfolio income tracking
            # 3. Trigger income analytics updates
            
        except Exception as e:
            logger.error(f"Failed to record dividend payment: {e}")
            raise AnalyticsError(f"Failed to record dividend payment: {e}")
    
    def get_income_history(self, 
                          portfolio_id: str,
                          start_date: date,
                          end_date: date) -> Dict[str, Any]:
        """
        Get historical dividend income for a portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Start date for history
            end_date: End date for history
            
        Returns:
            Dictionary with income history data
        """
        try:
            logger.info(f"Getting income history for {portfolio_id} from {start_date} to {end_date}")
            
            # This would typically query dividend payments table
            # For now, return mock historical data
            
            months = []
            current_date = start_date.replace(day=1)  # Start of month
            
            while current_date <= end_date:
                # Mock monthly income (would be actual data from database)
                monthly_income = 250.0 + (hash(str(current_date)) % 100)  # Simulate variation
                
                months.append({
                    'date': current_date.isoformat(),
                    'monthly_income': monthly_income,
                    'cumulative_income': sum(m.get('monthly_income', 0) for m in months)
                })
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            total_income = sum(m['monthly_income'] for m in months)
            avg_monthly_income = total_income / len(months) if months else 0
            
            return {
                'portfolio_id': portfolio_id,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_income': total_income,
                'average_monthly_income': avg_monthly_income,
                'monthly_data': months,
                'period_months': len(months)
            }
            
        except Exception as e:
            logger.error(f"Failed to get income history: {e}")
            raise AnalyticsError(f"Failed to get income history: {e}")