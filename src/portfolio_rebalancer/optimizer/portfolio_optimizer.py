"""Portfolio optimizer service that orchestrates risk model, optimization, and glide path."""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ..common.interfaces import DataStorage, OptimizationStrategy
from ..common.models import TargetAllocation
from ..common.metrics import (
    timed, record_portfolio_metrics, record_optimization_convergence,
    record_optimization_performance, record_data_quality
)
from .risk_model import RiskModel
from .optimization import SharpeOptimizer
from .glide_path import GlidePath, GlidePathConfig


logger = logging.getLogger(__name__)


class PortfolioOptimizerError(Exception):
    """Custom exception for portfolio optimizer errors."""
    pass


class PortfolioOptimizer:
    """
    Portfolio optimizer service that combines risk model, optimization, and glide path.
    
    Orchestrates the complete portfolio optimization process from historical data
    to target allocation with age-based constraints and persistence.
    """
    
    def __init__(self,
                 data_storage: DataStorage,
                 optimization_strategy: Optional[OptimizationStrategy] = None,
                 glide_path: Optional[GlidePath] = None,
                 risk_model: Optional[RiskModel] = None,
                 allocation_storage_path: str = "data/allocations"):
        """
        Initialize portfolio optimizer service.
        
        Args:
            data_storage: Data storage interface for retrieving price data
            optimization_strategy: Optimization strategy (defaults to SharpeOptimizer)
            glide_path: Glide path implementation (defaults to GlidePath)
            risk_model: Risk model implementation (defaults to RiskModel)
            allocation_storage_path: Path for storing target allocations
        """
        self.data_storage = data_storage
        self.optimization_strategy = optimization_strategy or SharpeOptimizer()
        self.glide_path = glide_path or GlidePath()
        self.risk_model = risk_model or RiskModel()
        
        # Set up allocation storage
        self.allocation_storage_path = Path(allocation_storage_path)
        self.allocation_storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized PortfolioOptimizer with allocation storage: {self.allocation_storage_path}")
    
    @timed("optimizer", "optimize_portfolio")
    def optimize_portfolio(self,
                          tickers: List[str],
                          lookback_days: int,
                          investor_age: int,
                          safe_tickers: Optional[List[str]] = None,
                          constraints: Optional[Dict] = None) -> TargetAllocation:
        """
        Perform complete portfolio optimization with age-based glide path.
        
        Args:
            tickers: List of ticker symbols for aggressive portfolio
            lookback_days: Number of days of historical data to use
            investor_age: Age of investor for glide path calculation
            safe_tickers: List of safe asset tickers (bonds, cash equivalents)
            constraints: Additional optimization constraints
            
        Returns:
            TargetAllocation object with optimal weights and metrics
            
        Raises:
            PortfolioOptimizerError: If optimization fails
        """
        # Validate inputs
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        if lookback_days <= 0:
            raise ValueError("Lookback days must be positive")
        
        if investor_age < 0:
            raise ValueError("Investor age cannot be negative")
        
        # Additional validation
        if not self.validate_optimization_inputs(tickers, lookback_days, investor_age):
            raise ValueError("Invalid optimization inputs")
        
        logger.info(f"Starting portfolio optimization for {len(tickers)} tickers, "
                   f"lookback: {lookback_days} days, age: {investor_age}")
        
        try:
            # Step 1: Retrieve historical price data
            try:
                price_data = self._get_price_data(tickers, lookback_days)
                logger.info(f"Successfully retrieved price data for {len(tickers)} tickers")
            except Exception as e:
                logger.error(f"Failed to retrieve price data: {str(e)}")
                raise PortfolioOptimizerError(f"Price data retrieval failed: {str(e)}") from e
            
            # Step 2: Calculate risk metrics
            try:
                expected_returns, cov_matrix = self.risk_model.calculate_risk_metrics(price_data)
                logger.info(f"Successfully calculated risk metrics for {len(expected_returns)} assets")
            except Exception as e:
                logger.error(f"Risk metrics calculation failed: {str(e)}")
                raise PortfolioOptimizerError(f"Risk model calculation failed: {str(e)}") from e
            
            # Step 3: Optimize aggressive portfolio
            try:
                returns_data = self.risk_model.calculate_returns(price_data)
                optimization_constraints = constraints or {}
                aggressive_allocation = self.optimization_strategy.optimize(returns_data, optimization_constraints)
                logger.info(f"Successfully optimized aggressive portfolio with {len(aggressive_allocation)} assets")
            except Exception as e:
                logger.error(f"Aggressive portfolio optimization failed: {str(e)}")
                # Attempt fallback to equal-weight allocation
                logger.warning("Attempting fallback to equal-weight allocation")
                try:
                    aggressive_allocation = {ticker: 1.0/len(tickers) for ticker in tickers}
                    logger.info("Successfully created equal-weight fallback allocation")
                except Exception as fallback_error:
                    logger.error(f"Equal-weight fallback failed: {str(fallback_error)}")
                    raise PortfolioOptimizerError(f"Optimization failed with no fallback: {str(e)}") from e
            
            # Step 4: Apply age-based glide path
            try:
                final_allocation = self._apply_glide_path(
                    aggressive_allocation, investor_age, safe_tickers or []
                )
                logger.info(f"Successfully applied glide path for age {investor_age}")
            except Exception as e:
                logger.error(f"Glide path application failed: {str(e)}")
                raise PortfolioOptimizerError(f"Glide path application failed: {str(e)}") from e
            
            # Step 5: Calculate portfolio metrics for final allocation
            try:
                portfolio_metrics = self._calculate_final_metrics(
                    final_allocation, expected_returns, cov_matrix
                )
                logger.info("Successfully calculated final portfolio metrics")
                
                # Record portfolio metrics for Prometheus
                record_portfolio_metrics(
                    expected_return=portfolio_metrics[0],
                    volatility=portfolio_metrics[1],
                    sharpe_ratio=portfolio_metrics[2]
                )
                
                # Record optimization performance metrics
                optimization_perf = {
                    "expected_return": portfolio_metrics[0],
                    "volatility": portfolio_metrics[1],
                    "sharpe_ratio": portfolio_metrics[2],
                    "asset_count": len(final_allocation)
                }
                record_optimization_performance("portfolio", optimization_perf)
                
            except Exception as e:
                logger.error(f"Portfolio metrics calculation failed: {str(e)}")
                # Use default metrics as fallback
                logger.warning("Using default metrics as fallback")
                portfolio_metrics = (0.0, 0.0, 0.0)
            
            # Step 6: Create target allocation object
            try:
                target_allocation = TargetAllocation(
                    timestamp=datetime.now(),
                    allocations=final_allocation,
                    expected_return=portfolio_metrics[0],
                    expected_volatility=portfolio_metrics[1],
                    sharpe_ratio=portfolio_metrics[2]
                )
                logger.info("Successfully created target allocation object")
            except Exception as e:
                logger.error(f"Target allocation creation failed: {str(e)}")
                raise PortfolioOptimizerError(f"Failed to create target allocation: {str(e)}") from e
            
            # Step 7: Persist target allocation
            try:
                self._store_target_allocation(target_allocation)
                logger.info("Successfully persisted target allocation")
            except Exception as e:
                logger.error(f"Target allocation persistence failed: {str(e)}")
                # Continue despite persistence failure
                logger.warning("Continuing despite persistence failure")
            
            logger.info(f"Portfolio optimization completed successfully. "
                       f"Sharpe ratio: {target_allocation.sharpe_ratio:.4f}, "
                       f"Expected return: {target_allocation.expected_return:.4f}, "
                       f"Volatility: {target_allocation.expected_volatility:.4f}")
            
            return target_allocation
            
        except PortfolioOptimizerError as e:
            # Re-raise specific optimizer errors
            raise
        except Exception as e:
            error_msg = f"Portfolio optimization failed: {str(e)}"
            logger.error(error_msg)
            raise PortfolioOptimizerError(error_msg) from e
    
    def _get_price_data(self, tickers: List[str], lookback_days: int) -> pd.DataFrame:
        """
        Retrieve and validate price data for optimization.
        
        Args:
            tickers: List of ticker symbols
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with price data suitable for optimization
            
        Raises:
            PortfolioOptimizerError: If data retrieval or validation fails
        """
        try:
            # Retrieve price data
            price_data = self.data_storage.get_prices(tickers, lookback_days)
            
            if price_data.empty:
                raise PortfolioOptimizerError("No price data available for optimization")
            
            # Pivot to get tickers as columns
            if isinstance(price_data.index, pd.MultiIndex):
                price_data = price_data.reset_index().pivot(
                    index='date', columns='symbol', values='adjusted_close'
                )
            
            # Validate data quality
            if not self.risk_model.validate_data_quality(
                self.risk_model.calculate_returns(price_data)
            ):
                raise PortfolioOptimizerError(f"Price data quality validation failed: {self.risk_model.calculate_returns(price_data)} < 252")
            
            logger.debug(f"Retrieved price data: {price_data.shape[0]} days, {price_data.shape[1]} tickers")
            
            return price_data
            
        except Exception as e:
            if isinstance(e, PortfolioOptimizerError):
                raise
            raise PortfolioOptimizerError(f"Failed to retrieve price data: {str(e)}") from e
    
    def _apply_glide_path(self,
                         aggressive_allocation: Dict[str, float],
                         investor_age: int,
                         safe_tickers: List[str]) -> Dict[str, float]:
        """
        Apply age-based glide path to blend aggressive and safe allocations.
        
        Args:
            aggressive_allocation: Optimized aggressive portfolio weights
            investor_age: Age of investor
            safe_tickers: List of safe asset tickers
            
        Returns:
            Dictionary with final blended allocation
        """
        try:
            # If no safe tickers provided, return aggressive allocation
            if not safe_tickers:
                logger.info("No safe tickers provided, using pure aggressive allocation")
                return aggressive_allocation
            
            # Create safe portfolio allocation
            safe_allocation = self.glide_path.get_safe_portfolio_allocation(safe_tickers)
            
            # Blend portfolios based on age
            blended_allocation = self.glide_path.blend_portfolios(
                aggressive_allocation, safe_allocation, investor_age
            )
            
            # Validate final allocation
            if not self.glide_path.validate_allocation(blended_allocation):
                raise PortfolioOptimizerError("Final allocation validation failed")
            
            # Log glide path application
            aggressive_weight, safe_weight = self.glide_path.get_allocation_blend(investor_age)
            logger.info(f"Applied glide path for age {investor_age}: "
                       f"{aggressive_weight:.1%} aggressive, {safe_weight:.1%} safe")
            
            return blended_allocation
            
        except Exception as e:
            if isinstance(e, PortfolioOptimizerError):
                raise
            raise PortfolioOptimizerError(f"Failed to apply glide path: {str(e)}") from e
    
    def _calculate_final_metrics(self,
                               allocation: Dict[str, float],
                               expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate portfolio metrics for the final allocation.
        
        Args:
            allocation: Final portfolio allocation
            expected_returns: Expected returns for all assets
            cov_matrix: Covariance matrix for all assets
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        try:
            # Filter expected returns and covariance matrix to match allocation
            allocation_tickers = list(allocation.keys())
            
            # Get expected returns for allocated assets
            filtered_returns = expected_returns.reindex(allocation_tickers, fill_value=0.0)
            
            # Get covariance matrix for allocated assets
            filtered_cov = cov_matrix.reindex(
                index=allocation_tickers, columns=allocation_tickers, fill_value=0.0
            )
            
            # Calculate metrics using the optimization strategy
            if hasattr(self.optimization_strategy, 'calculate_portfolio_metrics'):
                return self.optimization_strategy.calculate_portfolio_metrics(
                    allocation, filtered_returns, filtered_cov
                )
            else:
                # Fallback calculation using risk model
                import numpy as np
                weights = np.array([allocation.get(ticker, 0.0) for ticker in filtered_returns.index])
                return self.risk_model.calculate_portfolio_metrics(
                    weights, filtered_returns, filtered_cov
                )
                
        except Exception as e:
            logger.warning(f"Failed to calculate final metrics: {str(e)}")
            # Return default metrics if calculation fails
            return 0.0, 0.0, 0.0
    
    def _store_target_allocation(self, target_allocation: TargetAllocation) -> None:
        """
        Store target allocation to persistent storage.
        
        Args:
            target_allocation: TargetAllocation object to store
        """
        try:
            # Ensure storage directory exists
            self.allocation_storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp_str = target_allocation.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"target_allocation_{timestamp_str}.json"
            file_path = self.allocation_storage_path / filename
            
            # Convert to dictionary for JSON serialization
            allocation_dict = {
                "timestamp": target_allocation.timestamp.isoformat(),
                "allocations": target_allocation.allocations,
                "expected_return": target_allocation.expected_return,
                "expected_volatility": target_allocation.expected_volatility,
                "sharpe_ratio": target_allocation.sharpe_ratio
            }
            
            # Write to temporary file first to avoid corruption
            temp_path = self.allocation_storage_path / f"temp_{timestamp_str}.json"
            try:
                with open(temp_path, 'w') as f:
                    json.dump(allocation_dict, f, indent=2)
                
                # Rename to final filename (atomic operation)
                temp_path.rename(file_path)
                logger.info(f"Stored target allocation to {file_path}")
            except Exception as e:
                logger.error(f"Failed to write allocation file: {str(e)}")
                if temp_path.exists():
                    try:
                        temp_path.unlink()  # Clean up temp file
                    except:
                        pass
                raise
            
            # Also store as latest allocation with atomic write
            latest_path = self.allocation_storage_path / "latest_allocation.json"
            latest_temp = self.allocation_storage_path / "latest_allocation.temp.json"
            try:
                with open(latest_temp, 'w') as f:
                    json.dump(allocation_dict, f, indent=2)
                
                # Rename to final filename (atomic operation)
                latest_temp.rename(latest_path)
                logger.debug("Updated latest allocation file")
            except Exception as e:
                logger.error(f"Failed to update latest allocation: {str(e)}")
                if latest_temp.exists():
                    try:
                        latest_temp.unlink()  # Clean up temp file
                    except:
                        pass
            
            # Cleanup old allocations if there are too many
            self._cleanup_old_allocations()
            
        except Exception as e:
            logger.error(f"Failed to store target allocation: {str(e)}")
            # Don't raise exception as this is not critical for optimization
    
    def _cleanup_old_allocations(self, max_files: int = 100) -> None:
        """
        Clean up old allocation files if there are too many.
        
        Args:
            max_files: Maximum number of allocation files to keep
        """
        try:
            allocation_files = list(self.allocation_storage_path.glob("target_allocation_*.json"))
            
            if len(allocation_files) <= max_files:
                return
            
            # Sort by modification time (oldest first)
            allocation_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Delete oldest files
            files_to_delete = allocation_files[:-max_files]
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    logger.debug(f"Deleted old allocation file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old allocation file {file_path.name}: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Failed to clean up old allocation files: {str(e)}")
    
    def get_latest_allocation(self) -> Optional[TargetAllocation]:
        """
        Retrieve the most recent target allocation.
        
        Returns:
            TargetAllocation object or None if no allocation found
        """
        try:
            latest_path = self.allocation_storage_path / "latest_allocation.json"
            
            if not latest_path.exists():
                logger.info("No latest allocation file found")
                return None
            
            with open(latest_path, 'r') as f:
                allocation_dict = json.load(f)
            
            # Convert back to TargetAllocation object
            target_allocation = TargetAllocation(
                timestamp=datetime.fromisoformat(allocation_dict["timestamp"]),
                allocations=allocation_dict["allocations"],
                expected_return=allocation_dict["expected_return"],
                expected_volatility=allocation_dict["expected_volatility"],
                sharpe_ratio=allocation_dict["sharpe_ratio"]
            )
            
            logger.info(f"Retrieved latest allocation from {allocation_dict['timestamp']}")
            
            return target_allocation
            
        except Exception as e:
            logger.error(f"Failed to retrieve latest allocation: {str(e)}")
            return None
    
    def get_allocation_history(self, limit: Optional[int] = None) -> List[TargetAllocation]:
        """
        Retrieve historical target allocations.
        
        Args:
            limit: Maximum number of allocations to return (most recent first)
            
        Returns:
            List of TargetAllocation objects sorted by timestamp (newest first)
        """
        try:
            # Find all allocation files
            allocation_files = list(self.allocation_storage_path.glob("target_allocation_*.json"))
            
            if not allocation_files:
                logger.info("No allocation history found")
                return []
            
            # Sort by filename (which includes timestamp)
            allocation_files.sort(reverse=True)
            
            # Apply limit if specified
            if limit:
                allocation_files = allocation_files[:limit]
            
            allocations = []
            for file_path in allocation_files:
                try:
                    with open(file_path, 'r') as f:
                        allocation_dict = json.load(f)
                    
                    target_allocation = TargetAllocation(
                        timestamp=datetime.fromisoformat(allocation_dict["timestamp"]),
                        allocations=allocation_dict["allocations"],
                        expected_return=allocation_dict["expected_return"],
                        expected_volatility=allocation_dict["expected_volatility"],
                        sharpe_ratio=allocation_dict["sharpe_ratio"]
                    )
                    
                    allocations.append(target_allocation)
                    
                except Exception as e:
                    logger.warning(f"Failed to load allocation from {file_path}: {str(e)}")
            
            logger.info(f"Retrieved {len(allocations)} historical allocations")
            
            return allocations
            
        except Exception as e:
            logger.error(f"Failed to retrieve allocation history: {str(e)}")
            return []
    
    def validate_optimization_inputs(self,
                                   tickers: List[str],
                                   lookback_days: int,
                                   investor_age: int) -> bool:
        """
        Validate inputs for portfolio optimization.
        
        Args:
            tickers: List of ticker symbols
            lookback_days: Number of days of historical data
            investor_age: Age of investor
            
        Returns:
            True if inputs are valid
        """
        try:
            # Validate tickers
            if not tickers:
                logger.error("Tickers list cannot be empty")
                return False
            
            clean_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
            if len(clean_tickers) != len(tickers):
                logger.error("Invalid ticker symbols found")
                return False
            
            # Validate lookback days
            if lookback_days <= 0:
                logger.error("Lookback days must be positive")
                return False
            
            if lookback_days < 252:  # Less than 1 year
                logger.warning(f"Lookback period of {lookback_days} days may be insufficient for reliable optimization")
            
            # Validate investor age
            if investor_age < 0:
                logger.error("Investor age cannot be negative")
                return False
            
            if investor_age > 120:
                logger.warning(f"Investor age of {investor_age} seems unusually high")
            
            logger.info("Optimization inputs validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            return False