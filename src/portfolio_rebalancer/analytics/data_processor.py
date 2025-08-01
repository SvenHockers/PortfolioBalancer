"""Optimized data processing for analytics operations."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import multiprocessing as mp
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Data processing performance metrics."""
    operation: str
    input_size: int
    processing_time: float
    memory_usage: float
    parallel_workers: int = 1


class DataProcessor:
    """Optimized data processing for analytics operations."""
    
    def __init__(self, max_workers: Optional[int] = None, enable_caching: bool = True):
        """
        Initialize data processor.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_caching: Enable result caching
        """
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.enable_caching = enable_caching
        self.processing_metrics: List[ProcessingMetrics] = []
        
        logger.info(f"Data processor initialized with {self.max_workers} workers")
    
    def _track_processing_time(self, operation: str):
        """Decorator to track processing time and metrics."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    processing_time = time.time() - start_time
                    memory_usage = self._get_memory_usage() - start_memory
                    
                    # Estimate input size
                    input_size = 0
                    for arg in args[1:]:  # Skip self
                        if isinstance(arg, (pd.DataFrame, pd.Series)):
                            input_size += len(arg)
                        elif isinstance(arg, (list, dict)):
                            input_size += len(arg)
                    
                    metric = ProcessingMetrics(
                        operation=operation,
                        input_size=input_size,
                        processing_time=processing_time,
                        memory_usage=memory_usage
                    )
                    
                    self.processing_metrics.append(metric)
                    
                    if processing_time > 1.0:
                        logger.info(f"Slow processing: {operation} took {processing_time:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Processing failed for {operation}: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    @_track_processing_time("calculate_returns")
    def calculate_returns(self, prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
        """
        Calculate returns with optimized computation.
        
        Args:
            prices: DataFrame with price data
            method: Return calculation method ('simple' or 'log')
            
        Returns:
            DataFrame with calculated returns
        """
        if prices.empty:
            return pd.DataFrame()
        
        # Vectorized calculation
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        # Remove infinite and NaN values
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        return returns
    
    @_track_processing_time("calculate_rolling_metrics")
    def calculate_rolling_metrics(self, 
                                returns: pd.DataFrame,
                                window: int = 252,
                                metrics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling metrics efficiently.
        
        Args:
            returns: DataFrame with return data
            window: Rolling window size
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if returns.empty:
            return pd.DataFrame()
        
        metrics = metrics or ['volatility', 'sharpe', 'max_drawdown']
        results = pd.DataFrame(index=returns.index)
        
        # Parallel calculation of different metrics
        with ThreadPoolExecutor(max_workers=min(len(metrics), self.max_workers)) as executor:
            futures = {}
            
            for metric in metrics:
                if metric == 'volatility':
                    future = executor.submit(self._calculate_rolling_volatility, returns, window)
                elif metric == 'sharpe':
                    future = executor.submit(self._calculate_rolling_sharpe, returns, window)
                elif metric == 'max_drawdown':
                    future = executor.submit(self._calculate_rolling_max_drawdown, returns, window)
                elif metric == 'correlation':
                    future = executor.submit(self._calculate_rolling_correlation, returns, window)
                
                futures[metric] = future
            
            # Collect results
            for metric, future in futures.items():
                try:
                    result = future.result()
                    if isinstance(result, pd.Series):
                        results[metric] = result
                    elif isinstance(result, pd.DataFrame):
                        for col in result.columns:
                            results[f"{metric}_{col}"] = result[col]
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric}: {e}")
        
        return results
    
    def _calculate_rolling_volatility(self, returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling volatility."""
        return returns.rolling(window=window, min_periods=window//2).std() * np.sqrt(252)
    
    def _calculate_rolling_sharpe(self, returns: pd.DataFrame, window: int, risk_free_rate: float = 0.02) -> pd.DataFrame:
        """Calculate rolling Sharpe ratio."""
        rolling_returns = returns.rolling(window=window, min_periods=window//2)
        mean_returns = rolling_returns.mean() * 252
        volatility = rolling_returns.std() * np.sqrt(252)
        
        return (mean_returns - risk_free_rate) / volatility
    
    def _calculate_rolling_max_drawdown(self, returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling maximum drawdown."""
        def max_drawdown(series):
            cumulative = (1 + series).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        
        return returns.rolling(window=window, min_periods=window//2).apply(max_drawdown)
    
    def _calculate_rolling_correlation(self, returns: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling correlation matrix."""
        if returns.shape[1] < 2:
            return pd.DataFrame()
        
        # Calculate pairwise correlations
        correlations = {}
        columns = returns.columns
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                corr_key = f"{col1}_{col2}"
                correlations[corr_key] = returns[col1].rolling(
                    window=window, min_periods=window//2
                ).corr(returns[col2])
        
        return pd.DataFrame(correlations, index=returns.index)
    
    @_track_processing_time("optimize_portfolio_weights")
    def optimize_portfolio_weights(self,
                                 returns: pd.DataFrame,
                                 method: str = "sharpe",
                                 constraints: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights using efficient algorithms.
        
        Args:
            returns: DataFrame with asset returns
            method: Optimization method
            constraints: Optimization constraints
            
        Returns:
            Dictionary with optimized weights
        """
        if returns.empty or returns.shape[1] < 2:
            return {}
        
        constraints = constraints or {}
        
        try:
            # Calculate covariance matrix efficiently
            cov_matrix = returns.cov().values
            mean_returns = returns.mean().values
            
            if method == "sharpe":
                weights = self._optimize_sharpe_ratio(mean_returns, cov_matrix, constraints)
            elif method == "min_variance":
                weights = self._optimize_min_variance(cov_matrix, constraints)
            elif method == "risk_parity":
                weights = self._optimize_risk_parity(cov_matrix, constraints)
            else:
                # Equal weight as fallback
                n_assets = len(returns.columns)
                weights = np.ones(n_assets) / n_assets
            
            # Convert to dictionary
            return dict(zip(returns.columns, weights))
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Return equal weights as fallback
            n_assets = len(returns.columns)
            equal_weights = 1.0 / n_assets
            return {col: equal_weights for col in returns.columns}
    
    def _optimize_sharpe_ratio(self, mean_returns: np.ndarray, 
                             cov_matrix: np.ndarray, 
                             constraints: Dict[str, Any]) -> np.ndarray:
        """Optimize for maximum Sharpe ratio."""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(mean_returns)
            
            def objective(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_std = np.sqrt(portfolio_variance)
                
                if portfolio_std == 0:
                    return -np.inf
                
                sharpe_ratio = portfolio_return / portfolio_std
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            # Constraints
            constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                return result.x
            else:
                logger.warning("Sharpe optimization failed, using equal weights")
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            logger.warning("scipy not available, using equal weights")
            return np.ones(len(mean_returns)) / len(mean_returns)
    
    def _optimize_min_variance(self, cov_matrix: np.ndarray, 
                             constraints: Dict[str, Any]) -> np.ndarray:
        """Optimize for minimum variance."""
        try:
            from scipy.optimize import minimize
            
            n_assets = cov_matrix.shape[0]
            
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                return result.x
            else:
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
    
    def _optimize_risk_parity(self, cov_matrix: np.ndarray, 
                            constraints: Dict[str, Any]) -> np.ndarray:
        """Optimize for risk parity."""
        try:
            from scipy.optimize import minimize
            
            n_assets = cov_matrix.shape[0]
            
            def objective(weights):
                # Risk contribution of each asset
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                marginal_contrib = np.dot(cov_matrix, weights)
                contrib = weights * marginal_contrib / portfolio_variance
                
                # Minimize sum of squared deviations from equal risk contribution
                target_contrib = 1.0 / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # Constraints
            constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Bounds
            bounds = tuple((0.01, 1) for _ in range(n_assets))  # Minimum 1% allocation
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                return result.x
            else:
                return np.ones(n_assets) / n_assets
                
        except ImportError:
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
    
    @_track_processing_time("batch_process_portfolios")
    def batch_process_portfolios(self,
                               portfolios_data: List[Dict[str, Any]],
                               processing_func: callable,
                               **kwargs) -> List[Any]:
        """
        Process multiple portfolios in parallel.
        
        Args:
            portfolios_data: List of portfolio data dictionaries
            processing_func: Function to apply to each portfolio
            **kwargs: Additional arguments for processing function
            
        Returns:
            List of processing results
        """
        if not portfolios_data:
            return []
        
        results = []
        
        # Use ProcessPoolExecutor for CPU-intensive tasks
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(processing_func, portfolio_data, **kwargs): i
                for i, portfolio_data in enumerate(portfolios_data)
            }
            
            # Collect results in order
            results = [None] * len(portfolios_data)
            
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Portfolio processing failed for index {index}: {e}")
                    results[index] = None
        
        return results
    
    @lru_cache(maxsize=128)
    def _cached_correlation_matrix(self, returns_hash: str, window: int) -> pd.DataFrame:
        """Cached correlation matrix calculation."""
        # This would be called with a hash of the returns data
        # Implementation would depend on how we hash the DataFrame
        pass
    
    @_track_processing_time("preprocess_price_data")
    def preprocess_price_data(self, 
                            price_data: pd.DataFrame,
                            fill_method: str = "forward",
                            outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Preprocess price data for analytics.
        
        Args:
            price_data: Raw price data
            fill_method: Method for filling missing values
            outlier_threshold: Standard deviations for outlier detection
            
        Returns:
            Preprocessed price data
        """
        if price_data.empty:
            return price_data
        
        processed_data = price_data.copy()
        
        # Handle missing values
        if fill_method == "forward":
            processed_data = processed_data.fillna(method='ffill')
        elif fill_method == "backward":
            processed_data = processed_data.fillna(method='bfill')
        elif fill_method == "interpolate":
            processed_data = processed_data.interpolate(method='linear')
        
        # Remove outliers
        if outlier_threshold > 0:
            returns = processed_data.pct_change()
            
            for column in returns.columns:
                mean_return = returns[column].mean()
                std_return = returns[column].std()
                
                # Identify outliers
                outliers = np.abs(returns[column] - mean_return) > (outlier_threshold * std_return)
                
                # Replace outliers with median
                if outliers.any():
                    median_return = returns[column].median()
                    returns.loc[outliers, column] = median_return
                    
                    # Reconstruct prices
                    processed_data[column] = (1 + returns[column]).cumprod() * processed_data[column].iloc[0]
        
        return processed_data
    
    @_track_processing_time("aggregate_time_series")
    def aggregate_time_series(self,
                            data: pd.DataFrame,
                            frequency: str = "M",
                            aggregation_method: str = "last") -> pd.DataFrame:
        """
        Aggregate time series data efficiently.
        
        Args:
            data: Time series data
            frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')
            aggregation_method: Aggregation method ('last', 'mean', 'sum')
            
        Returns:
            Aggregated data
        """
        if data.empty:
            return data
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Perform aggregation
        if aggregation_method == "last":
            aggregated = data.resample(frequency).last()
        elif aggregation_method == "mean":
            aggregated = data.resample(frequency).mean()
        elif aggregation_method == "sum":
            aggregated = data.resample(frequency).sum()
        else:
            aggregated = data.resample(frequency).last()
        
        # Remove NaN rows
        aggregated = aggregated.dropna()
        
        return aggregated
    
    def get_processing_performance_report(self) -> Dict[str, Any]:
        """
        Generate processing performance report.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.processing_metrics:
            return {'message': 'No processing metrics available'}
        
        # Calculate statistics
        total_operations = len(self.processing_metrics)
        avg_processing_time = sum(m.processing_time for m in self.processing_metrics) / total_operations
        total_processing_time = sum(m.processing_time for m in self.processing_metrics)
        
        # Group by operation
        by_operation = {}
        for metric in self.processing_metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = []
            by_operation[metric.operation].append(metric)
        
        operation_stats = {}
        for operation, metrics in by_operation.items():
            operation_stats[operation] = {
                'count': len(metrics),
                'avg_time': sum(m.processing_time for m in metrics) / len(metrics),
                'max_time': max(m.processing_time for m in metrics),
                'total_time': sum(m.processing_time for m in metrics),
                'avg_input_size': sum(m.input_size for m in metrics) / len(metrics),
                'avg_memory_usage': sum(m.memory_usage for m in metrics) / len(metrics)
            }
        
        # Find bottlenecks
        slow_operations = [
            (op, stats['avg_time']) 
            for op, stats in operation_stats.items() 
            if stats['avg_time'] > 1.0
        ]
        slow_operations.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_operations': total_operations,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': total_processing_time,
            'max_workers': self.max_workers,
            'by_operation': operation_stats,
            'bottlenecks': slow_operations[:5],  # Top 5 slowest operations
            'recommendations': self._generate_performance_recommendations(operation_stats)
        }
    
    def _generate_performance_recommendations(self, operation_stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check for slow operations
        slow_ops = [op for op, stats in operation_stats.items() if stats['avg_time'] > 2.0]
        if slow_ops:
            recommendations.append(f"Consider optimizing slow operations: {', '.join(slow_ops)}")
        
        # Check for memory-intensive operations
        memory_intensive = [
            op for op, stats in operation_stats.items() 
            if stats['avg_memory_usage'] > 100  # MB
        ]
        if memory_intensive:
            recommendations.append(f"High memory usage detected in: {', '.join(memory_intensive)}")
        
        # Check for frequently called operations
        frequent_ops = [
            op for op, stats in operation_stats.items() 
            if stats['count'] > 100 and stats['avg_time'] > 0.1
        ]
        if frequent_ops:
            recommendations.append(f"Consider caching results for frequent operations: {', '.join(frequent_ops)}")
        
        return recommendations