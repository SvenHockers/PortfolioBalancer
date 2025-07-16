"""Data validation functions for price data quality checks."""

from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from .models import PriceData


class DataQualityError(Exception):
    """Exception raised for data quality issues."""
    pass


def validate_price_data_quality(data: pd.DataFrame, 
                               tickers: List[str],
                               start_date: Optional[date] = None,
                               end_date: Optional[date] = None) -> Dict[str, List[str]]:
    """
    Validate price data quality and return issues found.
    
    Args:
        data: DataFrame with price data (multi-index: date, symbol)
        tickers: Expected ticker symbols
        start_date: Expected start date (optional)
        end_date: Expected end date (optional)
        
    Returns:
        Dictionary with validation issues by category
        
    Raises:
        DataQualityError: If critical data quality issues are found
    """
    issues = {
        'missing_tickers': [],
        'missing_dates': [],
        'price_anomalies': [],
        'volume_anomalies': [],
        'data_gaps': [],
        'warnings': []
    }
    
    if data.empty:
        raise DataQualityError("Price data is empty")
    
    # Check for missing tickers
    available_tickers = set(data.index.get_level_values('symbol').unique())
    expected_tickers = set(tickers)
    missing_tickers = expected_tickers - available_tickers
    if missing_tickers:
        issues['missing_tickers'] = list(missing_tickers)
    
    # Check date range coverage
    if start_date and end_date:
        available_dates = set(data.index.get_level_values('date').unique())
        expected_dates = set(pd.date_range(start_date, end_date, freq='D').date)
        # Filter out weekends (basic check)
        expected_dates = {d for d in expected_dates if d.weekday() < 5}
        missing_dates = expected_dates - available_dates
        if missing_dates:
            issues['missing_dates'] = sorted(list(missing_dates))
    
    # Check for price anomalies
    for ticker in available_tickers:
        ticker_data = data.xs(ticker, level='symbol')
        
        if ticker_data.empty:
            continue
            
        prices = ticker_data['adjusted_close']
        volumes = ticker_data['volume']
        
        # Check for zero or negative prices
        invalid_prices = prices[prices <= 0]
        if not invalid_prices.empty:
            issues['price_anomalies'].extend([
                f"{ticker}: Zero/negative price on {date}" 
                for date in invalid_prices.index
            ])
        
        # Check for extreme price movements (>50% in one day)
        if len(prices) > 1:
            price_changes = prices.pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.5]
            if not extreme_changes.empty:
                issues['price_anomalies'].extend([
                    f"{ticker}: Extreme price change ({change:.1%}) on {date}"
                    for date, change in extreme_changes.items()
                ])
        
        # Check for zero volume (warning, not critical)
        zero_volume = volumes[volumes == 0]
        if not zero_volume.empty:
            issues['volume_anomalies'].extend([
                f"{ticker}: Zero volume on {date}"
                for date in zero_volume.index
            ])
        
        # Check for data gaps (missing consecutive dates)
        if len(ticker_data) > 1:
            dates = sorted(ticker_data.index)
            for i in range(1, len(dates)):
                gap_days = (dates[i] - dates[i-1]).days
                if gap_days > 7:  # More than a week gap (accounting for weekends)
                    issues['data_gaps'].append(
                        f"{ticker}: {gap_days}-day gap between {dates[i-1]} and {dates[i]}"
                    )
    
    # Add warnings for low data coverage
    total_expected = len(tickers)
    total_available = len(available_tickers)
    coverage = total_available / total_expected if total_expected > 0 else 0
    
    if coverage < 0.8:
        issues['warnings'].append(
            f"Low ticker coverage: {coverage:.1%} ({total_available}/{total_expected})"
        )
    
    return issues


def validate_price_data_completeness(data: pd.DataFrame, 
                                   required_days: int = 252) -> bool:
    """
    Check if price data has sufficient history for analysis.
    
    Args:
        data: DataFrame with price data
        required_days: Minimum number of trading days required
        
    Returns:
        True if data is sufficient, False otherwise
    """
    if data.empty:
        return False
    
    # Count unique dates
    unique_dates = len(data.index.get_level_values('date').unique())
    return unique_dates >= required_days


def validate_individual_price_record(symbol: str, 
                                   price_date: date, 
                                   adjusted_close: float, 
                                   volume: int) -> PriceData:
    """
    Validate and create a PriceData model instance.
    
    Args:
        symbol: Ticker symbol
        price_date: Date of the price data
        adjusted_close: Adjusted closing price
        volume: Trading volume
        
    Returns:
        Validated PriceData instance
        
    Raises:
        ValueError: If validation fails
    """
    return PriceData(
        symbol=symbol,
        date=price_date,
        adjusted_close=adjusted_close,
        volume=volume
    )


def detect_price_outliers(data: pd.DataFrame, 
                         z_threshold: float = 3.0) -> Dict[str, List[Tuple[date, float]]]:
    """
    Detect price outliers using z-score analysis.
    
    Args:
        data: DataFrame with price data
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        Dictionary mapping ticker symbols to list of (date, price) outliers
    """
    outliers = {}
    
    if data.empty:
        return outliers
    
    available_tickers = data.index.get_level_values('symbol').unique()
    
    for ticker in available_tickers:
        ticker_data = data.xs(ticker, level='symbol')
        prices = ticker_data['adjusted_close']
        
        if len(prices) < 10:  # Need sufficient data for statistical analysis
            continue
        
        # Calculate z-scores for price returns
        returns = prices.pct_change().dropna()
        if len(returns) < 5:
            continue
            
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        outlier_dates = z_scores[z_scores > z_threshold].index
        
        if not outlier_dates.empty:
            outliers[ticker] = [
                (date, prices.loc[date]) for date in outlier_dates
            ]
    
    return outliers


def validate_data_freshness(data: pd.DataFrame, 
                          max_age_days: int = 7) -> bool:
    """
    Check if price data is fresh (not too old).
    
    Args:
        data: DataFrame with price data
        max_age_days: Maximum age in days for data to be considered fresh
        
    Returns:
        True if data is fresh, False otherwise
    """
    if data.empty:
        return False
    
    latest_date = data.index.get_level_values('date').max()
    age_days = (date.today() - latest_date).days
    
    return age_days <= max_age_days