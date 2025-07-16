"""Risk model calculations for portfolio optimization."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


class RiskModel:
    """
    Risk model for portfolio optimization using Modern Portfolio Theory.
    
    Calculates expected returns, covariance matrix, and Sharpe ratios
    from historical price data with numerical stability enhancements.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk model.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Args:
            prices: DataFrame with price data (index: date, columns: symbols)
            
        Returns:
            DataFrame with daily returns
        """
        if prices.empty:
            raise ValueError("Price data cannot be empty")
        
        # Calculate daily returns using log returns for better statistical properties
        returns = np.log(prices / prices.shift(1)).dropna()
        
        # Check for any infinite or NaN values
        if returns.isnull().any().any():
            logger.warning("NaN values found in returns calculation")
            returns = returns.dropna()
        
        if (returns == np.inf).any().any() or (returns == -np.inf).any().any():
            logger.warning("Infinite values found in returns calculation")
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        return returns
    
    def calculate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate expected annual returns using historical mean.
        
        Args:
            returns: DataFrame with daily returns
            
        Returns:
            Series with expected annual returns for each asset
        """
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        
        # Calculate mean daily returns and annualize (252 trading days)
        daily_mean_returns = returns.mean()
        annual_expected_returns = daily_mean_returns * 252
        
        logger.info(f"Calculated expected returns for {len(annual_expected_returns)} assets")
        
        return annual_expected_returns
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame, 
                                  regularization_factor: float = 1e-5) -> pd.DataFrame:
        """
        Calculate covariance matrix with regularization for numerical stability.
        
        Args:
            returns: DataFrame with daily returns
            regularization_factor: Factor for regularization to ensure positive definiteness
            
        Returns:
            DataFrame with annualized covariance matrix
        """
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        
        # Calculate sample covariance matrix
        cov_matrix = returns.cov()
        
        # Annualize the covariance matrix (252 trading days)
        annual_cov_matrix = cov_matrix * 252
        
        # Apply regularization to ensure positive definiteness
        # Add small value to diagonal elements
        regularized_cov = annual_cov_matrix + regularization_factor * np.eye(len(annual_cov_matrix))
        
        # Verify positive definiteness
        try:
            # Attempt Cholesky decomposition to verify positive definiteness
            linalg.cholesky(regularized_cov.values)
            logger.info(f"Covariance matrix is positive definite ({regularized_cov.shape[0]}x{regularized_cov.shape[1]})")
        except linalg.LinAlgError:
            logger.warning("Covariance matrix is not positive definite, applying stronger regularization")
            # Apply stronger regularization
            eigenvals, eigenvecs = linalg.eigh(annual_cov_matrix.values)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < regularization_factor:
                regularization_factor = abs(min_eigenval) + 1e-4
            regularized_cov = annual_cov_matrix + regularization_factor * np.eye(len(annual_cov_matrix))
        
        return pd.DataFrame(regularized_cov, index=annual_cov_matrix.index, columns=annual_cov_matrix.columns)
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                  expected_returns: pd.Series, 
                                  cov_matrix: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate portfolio expected return, volatility, and Sharpe ratio.
        
        Args:
            weights: Array of portfolio weights
            expected_returns: Series of expected returns for each asset
            cov_matrix: Covariance matrix
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        if len(weights) != len(expected_returns):
            raise ValueError("Weights and expected returns must have same length")
        
        if len(weights) != cov_matrix.shape[0]:
            raise ValueError("Weights and covariance matrix dimensions must match")
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate portfolio expected return
        portfolio_return = np.dot(weights, expected_returns.values)
        
        # Calculate portfolio volatility
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe ratio
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def validate_data_quality(self, returns: pd.DataFrame, 
                            min_observations: int = 252) -> bool:
        """
        Validate that returns data is suitable for optimization.
        
        Args:
            returns: DataFrame with daily returns
            min_observations: Minimum number of observations required
            
        Returns:
            True if data quality is acceptable
        """
        if returns.empty:
            logger.error("Returns data is empty")
            return False
        
        if len(returns) < min_observations:
            logger.warning(f"Insufficient data: {len(returns)} observations, minimum {min_observations} required")
            return False
        
        # Check for excessive missing data
        missing_ratio = returns.isnull().sum() / len(returns)
        if (missing_ratio > 0.1).any():
            logger.warning(f"High missing data ratio detected: {missing_ratio.max():.2%}")
            return False
        
        # Check for extreme returns (potential data errors)
        extreme_returns = (np.abs(returns) > 0.5).any()  # 50% daily return threshold
        if extreme_returns.any():
            logger.warning("Extreme returns detected, possible data quality issues")
            return False
        
        logger.info("Data quality validation passed")
        return True
    
    def calculate_risk_metrics(self, prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate expected returns and covariance matrix from price data.
        
        Args:
            prices: DataFrame with historical price data
            
        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        # Calculate returns
        returns = self.calculate_returns(prices)
        
        # Validate data quality
        if not self.validate_data_quality(returns):
            raise ValueError("Data quality validation failed")
        
        # Calculate expected returns and covariance matrix
        expected_returns = self.calculate_expected_returns(returns)
        cov_matrix = self.calculate_covariance_matrix(returns)
        
        logger.info(f"Risk metrics calculated for {len(expected_returns)} assets")
        
        return expected_returns, cov_matrix