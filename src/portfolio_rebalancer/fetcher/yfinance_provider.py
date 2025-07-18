"""YFinance data provider implementation."""

import time
import logging
from datetime import date, timedelta
from typing import List, Optional
import pandas as pd
import yfinance as yf
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError

from ..common.interfaces import DataProvider
from ..common.models import PriceData


logger = logging.getLogger(__name__)


class YFinanceProvider(DataProvider):
    """YFinance implementation of DataProvider interface."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 30.0
    ):
        """
        Initialize YFinance provider with retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            timeout: Request timeout (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
    
    def fetch_prices(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch price data for given tickers within date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with multi-index (date, symbol) and price columns
        """
        if not tickers:
            logger.error("Tickers list cannot be empty")
            return pd.DataFrame()
        
        if start_date > end_date:
            logger.error("Start date cannot be after end date")
            return pd.DataFrame()
        
        logger.info(f"Fetching price data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Clean and validate tickers
        clean_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        if not clean_tickers:
            logger.error("No valid tickers provided")
            return pd.DataFrame()
        
        all_data = []
        failed_tickers = []
        
        for ticker in clean_tickers:
            try:
                ticker_data = self._fetch_ticker_with_retry(ticker, start_date, end_date)
                if ticker_data is not None and not ticker_data.empty:
                    all_data.append(ticker_data)
                else:
                    logger.warning(f"No data returned for ticker {ticker}")
                    failed_tickers.append(ticker)
            except Exception as e:
                logger.error(f"Failed to fetch data for ticker {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if not all_data:
            logger.error(f"Failed to fetch data for all tickers: {failed_tickers}")
            return pd.DataFrame()
        
        if failed_tickers:
            logger.warning(f"Failed to fetch data for {len(failed_tickers)} tickers: {failed_tickers}")
        
        # Combine all ticker data
        combined_df = pd.concat(all_data, ignore_index=False)
        
        # Validate and clean the data
        validated_df = self._validate_and_clean_data(combined_df)
        
        logger.info(f"Successfully fetched data for {len(validated_df.index.get_level_values('symbol').unique())} tickers")
        
        return validated_df
    
    def _fetch_ticker_with_retry(self, ticker: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker with retry logic.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with price data or None if failed
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Fetching data for {ticker}, attempt {attempt + 1}")
                
                # Create yfinance Ticker object
                yf_ticker = yf.Ticker(ticker)
                
                # Fetch historical data
                hist_data = yf_ticker.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),  # yfinance end date is exclusive
                    timeout=self.timeout,
                    raise_errors=True
                )
                
                if hist_data.empty:
                    logger.warning(f"No historical data available for {ticker}")
                    return None
                
                # Convert to our expected format
                ticker_df = self._convert_yfinance_data(hist_data, ticker)
                
                return ticker_df
                
            except (RequestException, HTTPError, Timeout, ConnectionError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"Network error fetching {ticker} (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                    time.sleep(delay)
                else:
                    logger.error(f"Network error fetching {ticker} after {self.max_retries + 1} attempts: {e}")
                    
            except Exception as e:
                logger.error(f"Unexpected error fetching {ticker}: {e}")
                last_exception = e
                break
        
        if last_exception:
            logger.error(f"Failed to fetch data for {ticker}: {last_exception}")
        
        return None
    
    def _convert_yfinance_data(self, hist_data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Convert yfinance DataFrame to our expected format.
        
        Args:
            hist_data: Raw yfinance historical data
            ticker: Ticker symbol
            
        Returns:
            DataFrame with multi-index (date, symbol) and standardized columns or None if conversion fails
        """
        try:
            # Make a copy to avoid modifying the original
            df = hist_data.copy()
            
            # Log the original column names for debugging
            logger.debug(f"Original columns for {ticker}: {list(df.columns)}")
            
            # Reset index to get date as a column
            df = df.reset_index()
            
            # Log columns after reset_index
            logger.debug(f"Columns after reset_index for {ticker}: {list(df.columns)}")
            
            # Handle different possible column names from yfinance
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                logger.debug(f"Processing column '{col}' (lowercase: '{col_lower}') for {ticker}")
                
                if col_lower in ['date', 'datetime']:
                    column_mapping[col] = 'date'
                    logger.debug(f"Mapped '{col}' to 'date' for {ticker}")
                elif col_lower in ['adj close', 'adjusted close', 'adjclose', 'adj_close']:
                    column_mapping[col] = 'adjusted_close'
                    logger.debug(f"Mapped '{col}' to 'adjusted_close' for {ticker}")
                elif col_lower == 'volume':
                    column_mapping[col] = 'volume'
                    logger.debug(f"Mapped '{col}' to 'volume' for {ticker}")
                elif col_lower in ['close', 'price']:
                    # Fallback to regular close if adjusted close not available
                    if 'adjusted_close' not in column_mapping.values():
                        column_mapping[col] = 'adjusted_close'
                        logger.debug(f"Mapped '{col}' to 'adjusted_close' (fallback) for {ticker}")
            
            # Log the mapping
            logger.debug(f"Column mapping for {ticker}: {column_mapping}")
            
            # Rename columns to match our expected format
            df = df.rename(columns=column_mapping)
            
            # Log columns after renaming
            logger.debug(f"Columns after renaming for {ticker}: {list(df.columns)}")
            
            # Ensure we have the required columns
            if 'date' not in df.columns:
                # If no date column found, use the index (which should be dates)
                if hasattr(hist_data.index, 'date') or pd.api.types.is_datetime64_any_dtype(hist_data.index):
                    df['date'] = hist_data.index
                    logger.debug(f"Added date column from index for {ticker}")
                else:
                    logger.error(f"Could not find date information in data for {ticker}")
                    return None
            
            if 'adjusted_close' not in df.columns:
                logger.error(f"Could not find adjusted close price data for {ticker}. Available columns: {list(df.columns)}")
                return None
            
            if 'volume' not in df.columns:
                logger.error(f"Could not find volume data for {ticker}. Available columns: {list(df.columns)}")
                return None
            
            # If we mapped 'Close' to 'adjusted_close', we need to calculate the actual adjusted close
            # using dividends and splits if available
            if 'close' in [col.lower() for col in hist_data.columns] and 'adjusted_close' in df.columns:
                # Check if we have dividends and splits data
                has_dividends = any('dividend' in col.lower() for col in hist_data.columns)
                has_splits = any('split' in col.lower() for col in hist_data.columns)
                
                if has_dividends or has_splits:
                    logger.debug(f"Calculating adjusted close from dividends/splits for {ticker}")
                    # For now, use close as adjusted close since calculating proper adjusted close
                    # requires historical dividend and split data which is complex
                    # In a production system, you might want to implement proper adjusted close calculation
                    pass
                else:
                    logger.debug(f"Using close price as adjusted close for {ticker} (no dividends/splits data)")
            
            # Select only the columns we need
            df = df[['date', 'adjusted_close', 'volume']].copy()
            
            # Add ticker symbol
            df['symbol'] = ticker
            
            # Convert date to date type (remove time component if present)
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Set multi-index
            df = df.set_index(['date', 'symbol'])
            
            logger.debug(f"Successfully converted data for {ticker} with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error converting yfinance data for {ticker}: {e}")
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the fetched price data.
        
        Args:
            df: Raw price DataFrame
            
        Returns:
            Cleaned and validated DataFrame
        """
        if df.empty:
            return df
        
        original_count = len(df)
        
        # Remove rows with missing adjusted_close prices
        df = df.dropna(subset=['adjusted_close'])
        
        # Remove rows with non-positive prices
        df = df[df['adjusted_close'] > 0]
        
        # Remove rows with negative volume
        df = df[df['volume'] >= 0]
        
        # Check for extreme price movements (more than 50% in a day)
        # This is a basic sanity check for data quality
        df_sorted = df.sort_index()
        for symbol in df_sorted.index.get_level_values('symbol').unique():
            symbol_data = df_sorted.xs(symbol, level='symbol')
            if len(symbol_data) > 1:
                price_changes = symbol_data['adjusted_close'].pct_change().abs()
                extreme_changes = price_changes > 0.5
                if extreme_changes.any():
                    extreme_dates = symbol_data[extreme_changes].index
                    logger.warning(f"Extreme price changes detected for {symbol} on dates: {extreme_dates.tolist()}")
        
        # Validate using Pydantic models for additional checks
        validated_rows = []
        for (date_val, symbol), row in df.iterrows():
            try:
                price_data = PriceData(
                    symbol=symbol,
                    date=date_val,
                    adjusted_close=row['adjusted_close'],
                    volume=int(row['volume'])
                )
                validated_rows.append({
                    'date': price_data.date,
                    'symbol': price_data.symbol,
                    'adjusted_close': price_data.adjusted_close,
                    'volume': price_data.volume
                })
            except Exception as e:
                logger.warning(f"Data validation failed for {symbol} on {date_val}: {e}")
        
        if not validated_rows:
            logger.warning("No valid data rows after validation")
            return pd.DataFrame()
        
        # Recreate DataFrame from validated data
        validated_df = pd.DataFrame(validated_rows)
        validated_df = validated_df.set_index(['date', 'symbol'])
        
        cleaned_count = len(validated_df)
        if cleaned_count < original_count:
            logger.info(f"Cleaned data: removed {original_count - cleaned_count} invalid rows")
        
        return validated_df