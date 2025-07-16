"""Storage backend implementations for market data."""

import os
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Union
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, Session

from ..common.interfaces import DataStorage
from ..common.models import PriceData


logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Custom exception for storage-related errors."""
    pass


class ParquetStorage(DataStorage):
    """Parquet file-based storage implementation with pandas integration."""
    
    def __init__(self, base_dir: str, filename_template: str = "{symbol}_{date}.parquet"):
        """
        Initialize ParquetStorage with directory configuration.
        
        Args:
            base_dir: Base directory for storing parquet files
            filename_template: Template for parquet filenames
        """
        self.base_dir = Path(base_dir)
        self.filename_template = filename_template
        
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"Initialized ParquetStorage with base directory: {self.base_dir}")
    
    def store_prices(self, data: pd.DataFrame) -> None:
        """
        Store price data to parquet files.
        
        Args:
            data: DataFrame with price data to store (multi-index with date, symbol)
            
        Raises:
            StorageError: If storing data fails
        """
        if data.empty:
            logger.warning("No data to store")
            return
        
        # Ensure data has the expected structure
        if not isinstance(data.index, pd.MultiIndex) or data.index.names != ['date', 'symbol']:
            raise ValueError("Data must have multi-index with (date, symbol)")
        
        if not {'adjusted_close', 'volume'}.issubset(data.columns):
            raise ValueError("Data must contain 'adjusted_close' and 'volume' columns")
        
        try:
            # Group by symbol and store each symbol's data in a separate file
            for symbol in data.index.get_level_values('symbol').unique():
                symbol_data = data.xs(symbol, level='symbol')
                
                # Reset index to get date as a column
                symbol_data = symbol_data.reset_index()
                
                # Convert date objects to datetime for parquet compatibility
                symbol_data['date'] = pd.to_datetime(symbol_data['date'])
                
                # Create filename based on symbol and date range
                min_date = symbol_data['date'].min().strftime('%Y%m%d')
                max_date = symbol_data['date'].max().strftime('%Y%m%d')
                filename = f"{symbol}_{min_date}_to_{max_date}.parquet"
                file_path = self.base_dir / filename
                
                # Store data
                symbol_data.to_parquet(file_path, index=False)
                logger.debug(f"Stored {len(symbol_data)} price records for {symbol} to {file_path}")
            
            logger.info(f"Successfully stored price data for {len(data.index.get_level_values('symbol').unique())} symbols")
        
        except Exception as e:
            error_msg = f"Failed to store price data: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e
    
    def get_prices(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        """
        Retrieve historical price data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            lookback_days: Number of days to look back from current date
            
        Returns:
            DataFrame with historical price data (multi-index with date, symbol)
            
        Raises:
            StorageError: If retrieving data fails
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        if lookback_days <= 0:
            raise ValueError("Lookback days must be positive")
        
        # Calculate start date based on lookback
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        return self.get_prices_date_range(tickers, start_date, end_date)
    
    def get_prices_date_range(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """
        Retrieve historical price data for given tickers within date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with historical price data (multi-index with date, symbol)
            
        Raises:
            StorageError: If retrieving data fails
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        clean_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        if not clean_tickers:
            raise ValueError("No valid tickers provided")
        
        logger.info(f"Retrieving price data for {len(clean_tickers)} tickers from {start_date} to {end_date}")
        
        all_data = []
        missing_tickers = []
        
        try:
            # Find all parquet files in the directory
            parquet_files = list(self.base_dir.glob("*.parquet"))
            
            for ticker in clean_tickers:
                # Find files for this ticker
                ticker_files = [f for f in parquet_files if f.name.startswith(f"{ticker}_")]
                
                if not ticker_files:
                    logger.warning(f"No data files found for ticker {ticker}")
                    missing_tickers.append(ticker)
                    continue
                
                ticker_data = []
                for file_path in ticker_files:
                    try:
                        # Load data from parquet file
                        df = pd.read_parquet(file_path)
                        
                        # Convert date column to date objects
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        
                        # Filter by date range
                        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                        
                        if not df.empty:
                            ticker_data.append(df)
                    
                    except Exception as e:
                        logger.warning(f"Error reading file {file_path}: {e}")
                
                if ticker_data:
                    # Combine data from multiple files
                    combined_df = pd.concat(ticker_data, ignore_index=True)
                    
                    # Remove duplicates (in case date ranges overlap in files)
                    combined_df = combined_df.drop_duplicates(subset=['date'])
                    
                    # Add symbol column
                    combined_df['symbol'] = ticker
                    
                    all_data.append(combined_df)
                else:
                    logger.warning(f"No data found for ticker {ticker} in the specified date range")
                    missing_tickers.append(ticker)
            
            if not all_data:
                logger.warning(f"No data found for any tickers in the specified date range")
                return pd.DataFrame()
            
            # Combine all ticker data
            result_df = pd.concat(all_data, ignore_index=True)
            
            # Set multi-index
            result_df = result_df.set_index(['date', 'symbol'])
            
            logger.info(f"Retrieved {len(result_df)} price records for {len(result_df.index.get_level_values('symbol').unique())} symbols")
            
            if missing_tickers:
                logger.warning(f"Missing data for tickers: {missing_tickers}")
            
            return result_df
        
        except Exception as e:
            error_msg = f"Failed to retrieve price data: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e


# Define SQLAlchemy models
Base = declarative_base()

class PriceRecord(Base):
    """SQLAlchemy model for price records."""
    
    __tablename__ = "price_records"
    
    id = sa.Column(sa.Integer, primary_key=True)
    symbol = sa.Column(sa.String(20), nullable=False, index=True)
    date = sa.Column(sa.Date, nullable=False, index=True)
    adjusted_close = sa.Column(sa.Float, nullable=False)
    volume = sa.Column(sa.BigInteger, nullable=False)
    
    # Composite index for faster lookups
    __table_args__ = (
        sa.Index('idx_symbol_date', 'symbol', 'date', unique=True),
    )
    
    def __repr__(self):
        return f"<PriceRecord(symbol='{self.symbol}', date='{self.date}', adjusted_close={self.adjusted_close})>"


class SQLiteStorage(DataStorage):
    """SQLite database storage implementation with SQLAlchemy."""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLiteStorage with database configuration.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = sa.create_engine(f"sqlite:///{db_path}")
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        logger.info(f"Initialized SQLiteStorage with database: {db_path}")
    
    def store_prices(self, data: pd.DataFrame) -> None:
        """
        Store price data to SQLite database.
        
        Args:
            data: DataFrame with price data to store (multi-index with date, symbol)
            
        Raises:
            StorageError: If storing data fails
        """
        if data.empty:
            logger.warning("No data to store")
            return
        
        # Ensure data has the expected structure
        if not isinstance(data.index, pd.MultiIndex) or data.index.names != ['date', 'symbol']:
            raise ValueError("Data must have multi-index with (date, symbol)")
        
        if not {'adjusted_close', 'volume'}.issubset(data.columns):
            raise ValueError("Data must contain 'adjusted_close' and 'volume' columns")
        
        try:
            # Reset index to get date and symbol as columns
            df = data.reset_index()
            
            # Create session
            with Session(self.engine) as session:
                # Process in batches to avoid memory issues with large datasets
                batch_size = 1000
                total_records = 0
                
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    
                    # Convert batch to list of dictionaries
                    records = batch.to_dict(orient='records')
                    
                    # Prepare records for upsert
                    for record in records:
                        # Check if record already exists
                        existing = session.query(PriceRecord).filter_by(
                            symbol=record['symbol'],
                            date=record['date']
                        ).first()
                        
                        if existing:
                            # Update existing record
                            existing.adjusted_close = record['adjusted_close']
                            existing.volume = record['volume']
                        else:
                            # Create new record
                            new_record = PriceRecord(
                                symbol=record['symbol'],
                                date=record['date'],
                                adjusted_close=record['adjusted_close'],
                                volume=record['volume']
                            )
                            session.add(new_record)
                    
                    # Commit batch
                    session.commit()
                    total_records += len(batch)
                    logger.debug(f"Stored batch of {len(batch)} records (total: {total_records})")
                
                logger.info(f"Successfully stored {total_records} price records to database")
        
        except Exception as e:
            error_msg = f"Failed to store price data: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e
    
    def get_prices(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        """
        Retrieve historical price data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            lookback_days: Number of days to look back from current date
            
        Returns:
            DataFrame with historical price data (multi-index with date, symbol)
            
        Raises:
            StorageError: If retrieving data fails
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        if lookback_days <= 0:
            raise ValueError("Lookback days must be positive")
        
        # Calculate start date based on lookback
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        return self.get_prices_date_range(tickers, start_date, end_date)
    
    def get_prices_date_range(self, tickers: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """
        Retrieve historical price data for given tickers within date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with historical price data (multi-index with date, symbol)
            
        Raises:
            StorageError: If retrieving data fails
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        clean_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        if not clean_tickers:
            raise ValueError("No valid tickers provided")
        
        logger.info(f"Retrieving price data for {len(clean_tickers)} tickers from {start_date} to {end_date}")
        
        try:
            # Create session
            with Session(self.engine) as session:
                # Query price records
                query = (
                    session.query(PriceRecord)
                    .filter(PriceRecord.symbol.in_(clean_tickers))
                    .filter(PriceRecord.date >= start_date)
                    .filter(PriceRecord.date <= end_date)
                    .order_by(PriceRecord.symbol, PriceRecord.date)
                )
                
                # Execute query
                records = query.all()
                
                if not records:
                    logger.warning(f"No data found for tickers in the specified date range")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for record in records:
                    data.append({
                        'date': record.date,
                        'symbol': record.symbol,
                        'adjusted_close': record.adjusted_close,
                        'volume': record.volume
                    })
                
                df = pd.DataFrame(data)
                
                # Set multi-index
                df = df.set_index(['date', 'symbol'])
                
                logger.info(f"Retrieved {len(df)} price records for {len(df.index.get_level_values('symbol').unique())} symbols")
                
                # Check for missing tickers
                retrieved_tickers = set(df.index.get_level_values('symbol').unique())
                missing_tickers = set(clean_tickers) - retrieved_tickers
                if missing_tickers:
                    logger.warning(f"Missing data for tickers: {missing_tickers}")
                
                return df
        
        except Exception as e:
            error_msg = f"Failed to retrieve price data: {e}"
            logger.error(error_msg)
            raise StorageError(error_msg) from e