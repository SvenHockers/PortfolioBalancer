"""Abstract base classes defining core system interfaces."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional
import pandas as pd


class DataProvider(ABC):
    """Abstract interface for market data providers."""
    
    @abstractmethod
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
        pass


class DataStorage(ABC):
    """Abstract interface for data persistence."""
    
    @abstractmethod
    def store_prices(self, data: pd.DataFrame) -> None:
        """
        Store price data to persistent storage.
        
        Args:
            data: DataFrame with price data to store
        """
        pass
    
    @abstractmethod
    def get_prices(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        """
        Retrieve historical price data for given tickers.
        
        Args:
            tickers: List of ticker symbols
            lookback_days: Number of days to look back from current date
            
        Returns:
            DataFrame with historical price data
        """
        pass


class OptimizationStrategy(ABC):
    """Abstract interface for portfolio optimization strategies."""
    
    @abstractmethod
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights.
        
        Args:
            returns: DataFrame with historical returns data
            constraints: Dictionary with optimization constraints
            
        Returns:
            Dictionary mapping ticker symbols to optimal weights
        """
        pass


class BrokerInterface(ABC):
    """Abstract interface for broker API integration."""
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """
        Get current portfolio positions.
        
        Returns:
            Dictionary mapping ticker symbols to position quantities
        """
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, quantity: float, order_type: str) -> Optional[str]:
        """
        Place a trade order.
        
        Args:
            symbol: Ticker symbol
            quantity: Order quantity (positive for buy, negative for sell)
            order_type: Order type ('market' or 'limit')
            
        Returns:
            Order ID string or None if failed
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """
        Get status of a placed order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status string
        """
        pass