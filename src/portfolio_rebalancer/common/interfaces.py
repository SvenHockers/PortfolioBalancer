"""Common interfaces for the portfolio rebalancer system."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional, Any
import pandas as pd
from enum import Enum


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


class PipelineStep(Enum):
    """Enumeration of pipeline steps."""
    DATA_FETCH = "data_fetch"
    OPTIMIZATION = "optimization"
    EXECUTION = "execution"


class PipelineStatus(Enum):
    """Enumeration of pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepResult:
    """Result of a pipeline step execution."""
    
    def __init__(self, 
                 step: PipelineStep, 
                 status: PipelineStatus, 
                 result: Any = None, 
                 error: str = None):
        """
        Initialize step result.
        
        Args:
            step: Pipeline step
            status: Execution status
            result: Step execution result (if any)
            error: Error message that occurred (if any)
        """
        self.step = step
        self.status = status
        self.result = result
        self.error = error
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def set_timing(self, start_time, end_time) -> None:
        """Set timing information for the step."""
        self.start_time = start_time
        self.end_time = end_time
        self.duration = (end_time - start_time).total_seconds()


class PipelineOrchestratorInterface(ABC):
    """
    Abstract base class for pipeline orchestrators.
    
    This interface defines the contract that all orchestrators must implement,
    whether they coordinate services in-process or via HTTP calls.
    """
    
    @abstractmethod
    def execute_pipeline(self) -> Dict[PipelineStep, StepResult]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dictionary mapping pipeline steps to their results
        """
        pass
    
    @abstractmethod
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current status of the pipeline.
        
        Returns:
            Dictionary with pipeline status information
        """
        pass