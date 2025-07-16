"""Portfolio optimization algorithms for maximum Sharpe ratio."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize
import logging

from ..common.interfaces import OptimizationStrategy

logger = logging.getLogger(__name__)


class SharpeOptimizer(OptimizationStrategy):
    """
    Portfolio optimizer that maximizes Sharpe ratio using scipy.optimize.
    
    Implements Modern Portfolio Theory optimization with configurable constraints
    for minimum and maximum asset weights, with fallback to equal-weight allocation.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 max_iterations: int = 1000):
        """
        Initialize Sharpe optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            min_weight: Minimum weight constraint for any asset
            max_weight: Maximum weight constraint for any asset
            max_iterations: Maximum iterations for optimization
        """
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_iterations = max_iterations
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights that maximize Sharpe ratio.
        
        Args:
            returns: DataFrame with historical returns data
            constraints: Dictionary with optimization constraints
            
        Returns:
            Dictionary mapping ticker symbols to optimal weights
        """
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        
        # Extract constraint parameters
        min_weight = constraints.get('min_weight', self.min_weight)
        max_weight = constraints.get('max_weight', self.max_weight)
        risk_free_rate = constraints.get('risk_free_rate', self.risk_free_rate)
        
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean() * 252  # Annualize
            cov_matrix = returns.cov() * 252  # Annualize
            
            # Perform optimization
            optimal_weights = self._optimize_sharpe_ratio(
                expected_returns, cov_matrix, min_weight, max_weight, risk_free_rate
            )
            
            # Convert to dictionary format
            result = dict(zip(returns.columns, optimal_weights))
            
            # Log optimization results
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
            
            logger.info(f"Optimization successful - Sharpe ratio: {sharpe_ratio:.4f}, "
                       f"Expected return: {portfolio_return:.4f}, Volatility: {portfolio_vol:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            logger.info("Falling back to equal-weight allocation")
            return self._equal_weight_fallback(returns.columns.tolist())
    
    def _optimize_sharpe_ratio(self, 
                              expected_returns: pd.Series, 
                              cov_matrix: pd.DataFrame,
                              min_weight: float,
                              max_weight: float,
                              risk_free_rate: float) -> np.ndarray:
        """
        Perform Sharpe ratio optimization using scipy.optimize.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            risk_free_rate: Risk-free rate
            
        Returns:
            Array of optimal weights
        """
        n_assets = len(expected_returns)
        
        # Objective function: negative Sharpe ratio (minimize)
        def negative_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Avoid division by zero
            if portfolio_vol == 0:
                return -np.inf
            
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # Negative because we minimize
        
        # Constraints
        constraints = [
            # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Bounds for each weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimization options
        options = {
            'maxiter': self.max_iterations,
            'ftol': 1e-9,
            'disp': False
        }
        
        # Perform optimization
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        # Normalize weights to ensure they sum to 1
        optimal_weights = result.x / np.sum(result.x)
        
        return optimal_weights
    
    def _equal_weight_fallback(self, tickers: list) -> Dict[str, float]:
        """
        Generate equal-weight allocation as fallback.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary with equal weights for all assets
        """
        n_assets = len(tickers)
        equal_weight = 1.0 / n_assets
        
        result = {ticker: equal_weight for ticker in tickers}
        
        logger.info(f"Equal-weight fallback applied: {equal_weight:.4f} per asset")
        
        return result
    
    def validate_weights(self, weights: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """
        Validate that weights are properly normalized and within constraints.
        
        Args:
            weights: Dictionary of asset weights
            tolerance: Tolerance for weight sum validation
            
        Returns:
            True if weights are valid
        """
        if not weights:
            return False
        
        weight_values = list(weights.values())
        
        # Check if weights sum to approximately 1
        weight_sum = sum(weight_values)
        if abs(weight_sum - 1.0) > tolerance:
            logger.error(f"Weights sum to {weight_sum:.6f}, expected 1.0")
            return False
        
        # Check if all weights are non-negative
        if any(w < -tolerance for w in weight_values):
            logger.error("Negative weights detected")
            return False
        
        # Check weight bounds
        if any(w < self.min_weight - tolerance or w > self.max_weight + tolerance 
               for w in weight_values):
            logger.error(f"Weights outside bounds [{self.min_weight}, {self.max_weight}]")
            return False
        
        return True
    
    def calculate_portfolio_metrics(self, 
                                  weights: Dict[str, float],
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights: Dictionary of asset weights
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        # Convert weights to array in same order as expected_returns
        weight_array = np.array([weights.get(ticker, 0.0) for ticker in expected_returns.index])
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weight_array, expected_returns)
        portfolio_variance = np.dot(weight_array, np.dot(cov_matrix, weight_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio


class MinimumVarianceOptimizer(OptimizationStrategy):
    """
    Alternative optimizer that minimizes portfolio variance.
    
    Useful as a conservative optimization strategy or for comparison purposes.
    """
    
    def __init__(self, 
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 max_iterations: int = 1000):
        """
        Initialize minimum variance optimizer.
        
        Args:
            min_weight: Minimum weight constraint for any asset
            max_weight: Maximum weight constraint for any asset
            max_iterations: Maximum iterations for optimization
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_iterations = max_iterations
    
    def optimize(self, returns: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights that minimize variance.
        
        Args:
            returns: DataFrame with historical returns data
            constraints: Dictionary with optimization constraints
            
        Returns:
            Dictionary mapping ticker symbols to optimal weights
        """
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        
        # Extract constraint parameters
        min_weight = constraints.get('min_weight', self.min_weight)
        max_weight = constraints.get('max_weight', self.max_weight)
        
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualize
            
            # Perform optimization
            optimal_weights = self._optimize_minimum_variance(cov_matrix, min_weight, max_weight)
            
            # Convert to dictionary format
            result = dict(zip(returns.columns, optimal_weights))
            
            # Log optimization results
            portfolio_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
            logger.info(f"Minimum variance optimization successful - Volatility: {portfolio_vol:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Minimum variance optimization failed: {str(e)}")
            logger.info("Falling back to equal-weight allocation")
            return self._equal_weight_fallback(returns.columns.tolist())
    
    def _optimize_minimum_variance(self, 
                                  cov_matrix: pd.DataFrame,
                                  min_weight: float,
                                  max_weight: float) -> np.ndarray:
        """
        Perform minimum variance optimization.
        
        Args:
            cov_matrix: Covariance matrix
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            
        Returns:
            Array of optimal weights
        """
        n_assets = len(cov_matrix)
        
        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Bounds for each weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimization options
        options = {
            'maxiter': self.max_iterations,
            'ftol': 1e-9,
            'disp': False
        }
        
        # Perform optimization
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        if not result.success:
            raise ValueError(f"Minimum variance optimization failed: {result.message}")
        
        # Normalize weights to ensure they sum to 1
        optimal_weights = result.x / np.sum(result.x)
        
        return optimal_weights
    
    def _equal_weight_fallback(self, tickers: list) -> Dict[str, float]:
        """Generate equal-weight allocation as fallback."""
        n_assets = len(tickers)
        equal_weight = 1.0 / n_assets
        return {ticker: equal_weight for ticker in tickers}