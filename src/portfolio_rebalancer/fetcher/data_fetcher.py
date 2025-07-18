"""Data fetcher orchestrator that coordinates provider and storage."""

import logging
from datetime import date, timedelta
from typing import List, Optional
import pandas as pd

from ..common.interfaces import DataProvider, DataStorage
from ..common.config import Config, get_config
from .storage import ParquetStorage, SQLiteStorage


logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Orchestrator class that coordinates data provider and storage.
    
    This class is responsible for:
    1. Fetching data from provider
    2. Storing data in the configured storage backend
    3. Handling backfill for missing historical data
    4. Providing a unified interface for data retrieval
    """
    
    def __init__(
        self,
        provider: DataProvider,
        storage: Optional[DataStorage] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize DataFetcher with provider, storage, and configuration.
        
        Args:
            provider: Data provider implementation
            storage: Data storage implementation (if None, will be created based on config)
            config: Configuration object (if None, will use global config)
        """
        self.provider = provider
        self.config = config or get_config()
        
        # Initialize storage if not provided
        if storage is None:
            storage_type = self.config.data.storage_type.lower()
            storage_path = self.config.data.storage_path
            
            if storage_type == "parquet":
                self.storage = ParquetStorage(base_dir=storage_path)
            elif storage_type == "sqlite":
                self.storage = SQLiteStorage(db_path=f"{storage_path}/prices.db")
            else:
                logger.error(f"Unsupported storage type: {storage_type}")
                raise ValueError(f"Unsupported storage type: {storage_type}")
        else:
            self.storage = storage
            
        logger.info(f"Initialized DataFetcher with {type(provider).__name__} provider and {type(self.storage).__name__} storage")
    
    def fetch_and_store_daily_data(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch today's data for specified tickers and store it.
        
        Args:
            tickers: List of ticker symbols (if None, uses tickers from config)
            
        Returns:
            DataFrame with fetched price data
        """
        tickers = tickers or self.config.data.tickers
        if not tickers:
            logger.error("No tickers specified")
            return pd.DataFrame()
        
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        try:
            # Fetch data for yesterday (most recent complete trading day)
            logger.info(f"Fetching daily data for {len(tickers)} tickers")
            data = self.provider.fetch_prices(tickers, yesterday, today)
            
            if data.empty:
                logger.warning("No data returned from provider")
                return pd.DataFrame()
            
            # Store the fetched data
            self.storage.store_prices(data)
            logger.info(f"Successfully stored daily data for {len(data.index.get_level_values('symbol').unique())} tickers")
            
            return data
            
        except Exception as e:
            error_msg = f"Failed to fetch and store daily data: {e}"
            logger.error(error_msg)
            return pd.DataFrame()
    
    def backfill_missing_data(
        self,
        tickers: Optional[List[str]] = None,
        days: Optional[int] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Backfill historical data for specified tickers.
        
        Args:
            tickers: List of ticker symbols (if None, uses tickers from config)
            days: Number of days to backfill (if None, uses backfill_days from config)
            end_date: End date for backfill (if None, uses yesterday)
            
        Returns:
            DataFrame with backfilled price data
        """
        tickers = tickers or self.config.data.tickers
        if not tickers:
            logger.error("No tickers specified")
            return pd.DataFrame()
        
        days = days or self.config.data.backfill_days
        if days <= 0:
            logger.error("Backfill days must be positive")
            return pd.DataFrame()
        
        end_date = end_date or (date.today() - timedelta(days=1))
        start_date = end_date - timedelta(days=days)
        
        try:
            logger.info(f"Backfilling data for {len(tickers)} tickers from {start_date} to {end_date}")
            
            # Fetch historical data
            data = self.provider.fetch_prices(tickers, start_date, end_date)
            
            if data.empty:
                logger.warning("No data returned from provider for backfill")
                return pd.DataFrame()
            
            # Store the fetched data
            self.storage.store_prices(data)
            logger.info(f"Successfully backfilled data for {len(data.index.get_level_values('symbol').unique())} tickers")
            
            return data
            
        except Exception as e:
            error_msg = f"Failed to backfill historical data: {e}"
            logger.error(error_msg)
            return pd.DataFrame()
    
    def get_latest_prices(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get the latest available prices for specified tickers.
        
        Args:
            tickers: List of ticker symbols (if None, uses tickers from config)
            
        Returns:
            DataFrame with latest price data
        """
        tickers = tickers or self.config.data.tickers
        if not tickers:
            logger.error("No tickers specified")
            return pd.DataFrame()
        
        try:
            # Get data for the last 5 days (to account for weekends/holidays)
            data = self.storage.get_prices(tickers, lookback_days=5)
            
            if data.empty:
                logger.warning("No recent price data found in storage")
                return pd.DataFrame()
            
            # Get the latest date for each ticker
            latest_data = []
            for symbol in data.index.get_level_values('symbol').unique():
                symbol_data = data.xs(symbol, level='symbol')
                if not symbol_data.empty:
                    latest_date = symbol_data.index.max()
                    latest_row = symbol_data.loc[latest_date]
                    latest_data.append((latest_date, symbol, latest_row))
            
            if not latest_data:
                return pd.DataFrame()
            
            # Create DataFrame with latest prices
            result = pd.DataFrame([
                {
                    'date': date_val,
                    'symbol': symbol,
                    'adjusted_close': row['adjusted_close'],
                    'volume': row['volume']
                }
                for date_val, symbol, row in latest_data
            ])
            
            # Set multi-index
            result = result.set_index(['date', 'symbol'])
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to get latest prices: {e}"
            logger.error(error_msg)
            return pd.DataFrame()
    
    def get_historical_prices(
        self,
        tickers: Optional[List[str]] = None,
        lookback_days: Optional[int] = None,
        ensure_minimum: bool = True
    ) -> pd.DataFrame:
        """
        Get historical price data for specified tickers.
        
        Args:
            tickers: List of ticker symbols (if None, uses tickers from config)
            lookback_days: Number of days to look back (if None, uses lookback_days from optimization config)
            ensure_minimum: Whether to ensure at least 1 year of data is available (default: True)
            
        Returns:
            DataFrame with historical price data
        """
        tickers = tickers or self.config.data.tickers
        if not tickers:
            logger.error("No tickers specified")
            return pd.DataFrame()
        
        lookback_days = lookback_days or self.config.optimization.lookback_days
        if lookback_days <= 0:
            logger.error("Lookback days must be positive")
            return pd.DataFrame()
        
        try:
            # If ensure_minimum is True, use ensure_data_available to guarantee sufficient data
            if ensure_minimum:
                return self.ensure_data_available(
                    tickers=tickers,
                    lookback_days=lookback_days,
                    force_update=False,
                    min_trading_days=252
                )
            
            # Otherwise, just get data from storage without ensuring minimum
            data = self.storage.get_prices(tickers, lookback_days=lookback_days)
            
            if data.empty:
                logger.warning(f"No historical data found for the past {lookback_days} days")
            else:
                logger.info(f"Retrieved historical data for {len(data.index.get_level_values('symbol').unique())} tickers")
            
            return data
            
        except Exception as e:
            error_msg = f"Failed to get historical prices: {e}"
            logger.error(error_msg)
            return pd.DataFrame()
    
    def ensure_data_available(
        self,
        tickers: Optional[List[str]] = None,
        lookback_days: Optional[int] = None,
        force_update: bool = False,
        min_trading_days: int = 252
    ) -> pd.DataFrame:
        """
        Ensure data is available for specified tickers and lookback period.
        
        This method checks if data is available in storage and backfills if necessary.
        Ensures at least 1 year (252 trading days) of data is available.
        
        Args:
            tickers: List of ticker symbols (if None, uses tickers from config)
            lookback_days: Number of days to ensure (if None, uses lookback_days from optimization config)
            force_update: Whether to force fetching latest data even if already available
            min_trading_days: Minimum number of trading days required (default: 252 for 1 year)
            
        Returns:
            DataFrame with historical price data
        """
        tickers = tickers or self.config.data.tickers
        if not tickers:
            logger.error("No tickers specified")
            return pd.DataFrame()
        
        lookback_days = lookback_days or self.config.optimization.lookback_days
        if lookback_days <= 0:
            logger.error("Lookback days must be positive")
            return pd.DataFrame()
        
        # Ensure we have at least 1 year of data
        required_days = max(lookback_days, min_trading_days)
        logger.info(f"Ensuring at least {required_days} days of data (minimum {min_trading_days} trading days)")
        
        try:
            # First, try to get data from storage
            data = self.storage.get_prices(tickers, lookback_days=required_days)
            
            # Initialize variables for tracking missing/incomplete data
            missing_tickers = set()
            incomplete_tickers = []
            
            # Check if we have complete data for all tickers
            if not data.empty:
                available_tickers = set(data.index.get_level_values('symbol').unique())
                missing_tickers = set(tickers) - available_tickers
                
                # Check if we have enough data for each ticker
                ticker_date_counts = {}
                for symbol in available_tickers:
                    symbol_data = data.xs(symbol, level='symbol')
                    ticker_date_counts[symbol] = len(symbol_data)
                
                # Calculate minimum expected dates (at least 1 year of trading days)
                min_expected_dates = max(min_trading_days, int(required_days * 0.7))  # Account for weekends/holidays
                incomplete_tickers = [
                    symbol for symbol, count in ticker_date_counts.items()
                    if count < min_expected_dates
                ]
                
                if not missing_tickers and not incomplete_tickers and not force_update:
                    logger.info(f"Complete data already available for all {len(tickers)} tickers (at least {min_trading_days} trading days)")
                    return data
                
                if missing_tickers:
                    logger.info(f"Missing data for tickers: {missing_tickers}")
                
                if incomplete_tickers:
                    logger.info(f"Incomplete data for tickers: {incomplete_tickers} (need at least {min_expected_dates} trading days)")
            else:
                logger.info("No data available in storage, will backfill")
            
            # Only backfill if we need to (missing tickers, incomplete data, or force update)
            if data.empty or missing_tickers or incomplete_tickers or force_update:
                # Backfill data - ensure we get enough data for at least 1 year
                backfill_days = max(required_days, self.config.data.backfill_days, min_trading_days * 2)  # Get extra data to ensure coverage
                logger.info(f"Backfilling {backfill_days} days of data to ensure minimum {min_trading_days} trading days")
                
                self.backfill_missing_data(tickers=tickers, days=backfill_days)
                
                # Get the data again after backfill
                updated_data = self.storage.get_prices(tickers, lookback_days=required_days)
                
                if updated_data.empty:
                    logger.warning("Still no data available after backfill")
                else:
                    # Verify we have sufficient data after backfill
                    actual_tickers = set(updated_data.index.get_level_values('symbol').unique())
                    if len(actual_tickers) == len(tickers):
                        # Check data completeness for each ticker
                        all_sufficient = True
                        for symbol in actual_tickers:
                            symbol_data = updated_data.xs(symbol, level='symbol')
                            if len(symbol_data) < min_trading_days:
                                logger.warning(f"Ticker {symbol} still has insufficient data: {len(symbol_data)} days (need {min_trading_days})")
                                all_sufficient = False
                        
                        if all_sufficient:
                            logger.info(f"Successfully ensured data availability for {len(actual_tickers)} tickers with at least {min_trading_days} trading days")
                        else:
                            logger.warning("Some tickers still have insufficient data after backfill")
                    else:
                        logger.warning(f"Missing data for some tickers after backfill: expected {len(tickers)}, got {len(actual_tickers)}")
                
                return updated_data
            
            # If we reach here, we have data but didn't need to backfill
            return data
            
        except Exception as e:
            error_msg = f"Failed to ensure data availability: {e}"
            logger.error(error_msg)
            return pd.DataFrame()
    
    def ensure_one_year_data(self, tickers: Optional[List[str]] = None, force_update: bool = False) -> pd.DataFrame:
        """
        Convenience method to ensure at least 1 year (252 trading days) of data is available.
        
        Args:
            tickers: List of ticker symbols (if None, uses tickers from config)
            force_update: Whether to force fetching latest data even if already available
            
        Returns:
            DataFrame with at least 1 year of historical price data
        """
        return self.ensure_data_available(
            tickers=tickers,
            lookback_days=365,  # Request 365 calendar days to ensure 252 trading days
            force_update=force_update,
            min_trading_days=252
        )