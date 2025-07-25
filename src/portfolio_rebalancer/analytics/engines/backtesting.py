"""Backtesting engine for portfolio strategy validation."""

import logging
import hashlib
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ..models import (
    BacktestConfig, BacktestResult, OptimizationStrategy, 
    RebalanceFrequency, StrategyComparison
)
from ..exceptions import BacktestError, InsufficientDataError
from ...common.interfaces import DataStorage

logger = logging.getLogger(__name__)


class OptimizationStrategyRegistry:
    """Registry for portfolio optimization strategies."""
    
    def __init__(self):
        """Initialize strategy registry with default strategies."""
        self._strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register built-in optimization strategies."""
        # Sharpe ratio optimizer
        self._strategies[OptimizationStrategy.SHARPE] = SharpeStrategy()
        
        # Minimum variance optimizer
        self._strategies[OptimizationStrategy.MIN_VARIANCE] = MinimumVarianceStrategy()
        
        # Equal weight strategy
        self._strategies[OptimizationStrategy.EQUAL_WEIGHT] = EqualWeightStrategy()
        
        # Risk parity strategy
        self._strategies[OptimizationStrategy.RISK_PARITY] = RiskParityStrategy()
    
    def get_strategy(self, strategy_type: OptimizationStrategy):
        """Get strategy implementation by type."""
        if strategy_type not in self._strategies:
            logger.warning(f"Strategy {strategy_type} not found, using equal weight")
            return self._strategies[OptimizationStrategy.EQUAL_WEIGHT]
        
        return self._strategies[strategy_type]
    
    def register_strategy(self, strategy_type: OptimizationStrategy, strategy_impl):
        """Register a new strategy implementation."""
        self._strategies[strategy_type] = strategy_impl


class OptimizationStrategy:
    """Base class for optimization strategies."""
    
    def optimize(self, returns_data: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """
        Optimize portfolio allocation.
        
        Args:
            returns_data: Historical returns data
            constraints: Optimization constraints
            
        Returns:
            Dictionary mapping tickers to weights
        """
        raise NotImplementedError


class SharpeStrategy(OptimizationStrategy):
    """Sharpe ratio optimization strategy."""
    
    def optimize(self, returns_data: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """Optimize for maximum Sharpe ratio."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov().values
            
            # Calculate mean returns
            mean_returns = returns_data.mean().values
            
            n_assets = len(returns_data.columns)
            
            # Objective function: minimize negative Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                
                # Avoid division by zero
                if portfolio_vol == 0:
                    return -np.inf
                
                return -(portfolio_return / portfolio_vol)  # Negative for minimization
            
            # Constraints: weights sum to 1
            constraints_opt = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Bounds: weights between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints_opt
            )
            
            if result.success:
                weights = result.x
                return {ticker: weight for ticker, weight in zip(returns_data.columns, weights)}
            else:
                # Fallback to equal weights
                logger.warning("Sharpe optimization failed, using equal weights")
                return {ticker: 1.0 / n_assets for ticker in returns_data.columns}
                
        except Exception as e:
            logger.warning(f"Sharpe optimization failed: {e}")
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            return {ticker: 1.0 / n_assets for ticker in returns_data.columns}


class MinimumVarianceStrategy(OptimizationStrategy):
    """Minimum variance optimization strategy."""
    
    def optimize(self, returns_data: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """Optimize for minimum portfolio variance."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov().values
            
            n_assets = len(returns_data.columns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Constraints: weights sum to 1
            constraints_opt = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Bounds: weights between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints_opt
            )
            
            if result.success:
                weights = result.x
                return {ticker: weight for ticker, weight in zip(returns_data.columns, weights)}
            else:
                # Fallback to equal weights
                logger.warning("Minimum variance optimization failed, using equal weights")
                return {ticker: 1.0 / n_assets for ticker in returns_data.columns}
                
        except Exception as e:
            logger.warning(f"Minimum variance optimization failed: {e}")
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            return {ticker: 1.0 / n_assets for ticker in returns_data.columns}


class EqualWeightStrategy(OptimizationStrategy):
    """Equal weight allocation strategy."""
    
    def optimize(self, returns_data: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """Return equal weights for all assets."""
        tickers = returns_data.columns.tolist()
        weight = 1.0 / len(tickers)
        return {ticker: weight for ticker in tickers}


class RiskParityStrategy(OptimizationStrategy):
    """Risk parity allocation strategy."""
    
    def optimize(self, returns_data: pd.DataFrame, constraints: Dict) -> Dict[str, float]:
        """Optimize for risk parity allocation."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov().values
            
            n_assets = len(returns_data.columns)
            
            # Risk parity objective: minimize sum of squared deviations from equal risk contribution
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # Constraints: weights sum to 1
            constraints_opt = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Bounds: weights between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess: equal weights
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints_opt
            )
            
            if result.success:
                weights = result.x
                return {ticker: weight for ticker, weight in zip(returns_data.columns, weights)}
            else:
                # Fallback to equal weights
                logger.warning("Risk parity optimization failed, using equal weights")
                return {ticker: 1.0 / n_assets for ticker in returns_data.columns}
                
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")
            # Fallback to equal weights
            n_assets = len(returns_data.columns)
            return {ticker: 1.0 / n_assets for ticker in returns_data.columns}


class BacktestingEngine:
    """Historical portfolio performance simulation engine."""
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize backtesting engine.
        
        Args:
            data_storage: Data storage interface for retrieving historical data
        """
        self.data_storage = data_storage
        self.strategy_registry = OptimizationStrategyRegistry()
        
        logger.info("Initialized BacktestingEngine")
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run historical backtest simulation.
        
        Args:
            config: Backtest configuration
            
        Returns:
            BacktestResult with performance metrics and data
            
        Raises:
            BacktestError: If backtest execution fails
            InsufficientDataError: If insufficient historical data
        """
        logger.info(f"Starting backtest: {config.strategy} strategy, "
                   f"{config.start_date} to {config.end_date}, "
                   f"rebalance: {config.rebalance_frequency}")
        
        try:
            # Get historical price data
            price_data = self._get_historical_data(config)
            
            # Run simulation
            simulation_results = self._simulate_portfolio(config, price_data)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                simulation_results, config
            )
            
            # Create result object
            result = BacktestResult(
                config=config,
                total_return=performance_metrics['total_return'],
                annualized_return=performance_metrics['annualized_return'],
                volatility=performance_metrics['volatility'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                max_drawdown=performance_metrics['max_drawdown'],
                calmar_ratio=performance_metrics['calmar_ratio'],
                transaction_costs=performance_metrics['transaction_costs'],
                num_rebalances=performance_metrics['num_rebalances'],
                final_value=performance_metrics['final_value'],
                returns_data=simulation_results['returns_data'],
                allocation_data=simulation_results['allocation_data']
            )
            
            logger.info(f"Backtest completed successfully. "
                       f"Total return: {result.total_return:.2%}, "
                       f"Sharpe ratio: {result.sharpe_ratio:.3f}")
            
            return result
            
        except (BacktestError, InsufficientDataError):
            raise
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise BacktestError(f"Backtest execution failed: {e}") from e
    
    def compare_strategies(self, 
                          strategies: List[OptimizationStrategy],
                          base_config: BacktestConfig) -> StrategyComparison:
        """
        Compare multiple strategies side-by-side.
        
        Args:
            strategies: List of optimization strategies to compare
            base_config: Base configuration (strategy will be overridden)
            
        Returns:
            StrategyComparison with results and statistical analysis
            
        Raises:
            BacktestError: If strategy comparison fails
        """
        if len(strategies) < 2:
            raise BacktestError("At least 2 strategies required for comparison")
        
        logger.info(f"Comparing {len(strategies)} strategies")
        
        try:
            results = {}
            
            # Run backtest for each strategy
            for strategy in strategies:
                config = BacktestConfig(
                    tickers=base_config.tickers,
                    start_date=base_config.start_date,
                    end_date=base_config.end_date,
                    strategy=strategy,
                    rebalance_frequency=base_config.rebalance_frequency,
                    transaction_cost=base_config.transaction_cost,
                    initial_capital=base_config.initial_capital
                )
                
                result = self.run_backtest(config)
                results[strategy.value] = result
            
            # Perform statistical significance tests
            statistical_tests = self._perform_statistical_tests(results)
            
            # Determine best strategy
            best_strategy = max(results.keys(), 
                              key=lambda s: results[s].sharpe_ratio)
            
            # Create ranking
            ranking = sorted(results.keys(), 
                           key=lambda s: results[s].sharpe_ratio, 
                           reverse=True)
            
            comparison = StrategyComparison(
                strategies=[s.value for s in strategies],
                comparison_period=(base_config.start_date, base_config.end_date),
                results=results,
                statistical_significance=statistical_tests,
                best_strategy=best_strategy,
                ranking=ranking
            )
            
            logger.info(f"Strategy comparison completed. Best: {best_strategy}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            raise BacktestError(f"Strategy comparison failed: {e}") from e
    
    def _get_historical_data(self, config: BacktestConfig) -> pd.DataFrame:
        """
        Retrieve and prepare historical price data.
        
        Args:
            config: Backtest configuration
            
        Returns:
            DataFrame with historical price data
            
        Raises:
            InsufficientDataError: If insufficient data available
        """
        try:
            # Calculate required lookback period
            period_days = (config.end_date - config.start_date).days
            lookback_days = period_days + 365  # Extra year for calculations
            min_required_days = max(30, period_days)  # At least 30 days
            
            # Add buffer for preparation and warm-up period
            lookback_days = period_days + lookback_days
            
            # Get price data from storage
            price_data = self.data_storage.get_prices(
                config.tickers, 
                lookback_days
            )
            
            logger.info(f"Retrieved {len(price_data)} days of price data "
                       f"for {len(config.tickers)} tickers")
            
            # Check data completeness
            if price_data.empty:
                raise InsufficientDataError(
                    f"No price data available for tickers: {config.tickers}"
                )
            
            # Check for missing tickers
            missing_tickers = set(config.tickers) - set(price_data.columns)
            if missing_tickers:
                raise InsufficientDataError(
                    f"Missing price data for tickers: {missing_tickers}"
                )
            
            # Check for sufficient data coverage (at least 70%)
            min_required_days = max(30, period_days * 0.7)  # At least 70% coverage
            for ticker in config.tickers:
                valid_days = price_data[ticker].notna().sum()
                if valid_days < min_required_days:
                    raise InsufficientDataError(
                        f"Insufficient data for {ticker}: {valid_days} days "
                        f"(minimum required: {min_required_days})"
                    )
            
            # Filter to backtest period
            start_ts = pd.Timestamp(config.start_date)
            end_ts = pd.Timestamp(config.end_date)
            
            price_data = price_data[
                (price_data.index >= start_ts) & 
                (price_data.index <= end_ts)
            ]
            
            # Convert to pivot format if needed
            if isinstance(price_data.index, pd.MultiIndex):
                price_data = price_data.reset_index().pivot(
                    index='date', 
                    columns='symbol', 
                    values='adjusted_close'
                )
            
            if price_data.empty:
                raise InsufficientDataError(
                    f"No price data available for period {config.start_date} to {config.end_date}"
                )
            
            return price_data
            
        except (InsufficientDataError, BacktestError):
            raise
        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            raise BacktestError(f"Error retrieving historical data: {e}")
    
    def _simulate_portfolio(self, config: BacktestConfig, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate portfolio performance over time.
        
        Args:
            config: Backtest configuration
            price_data: Historical price data
            
        Returns:
            Dictionary with simulation results
        """
        logger.debug(f"Simulating portfolio with {len(config.tickers)} tickers")
        
        # Initialize portfolio state
        portfolio_value = config.initial_capital
        positions = {ticker: 0.0 for ticker in config.tickers}  # shares
        total_transaction_costs = 0.0
        num_rebalances = 0
        
        # Track results
        daily_values = []
        daily_returns = []
        allocation_history = []
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(
            config.start_date, config.end_date, config.rebalance_frequency
        )
        
        # Initialize portfolio on first date
        first_date = price_data.index[0]
        if first_date not in rebalance_dates:
            rebalance_dates.insert(0, first_date)
        
        # Simulate day by day
        for current_date in price_data.index:
            current_prices = price_data.loc[current_date]
            
            # Check if rebalancing is needed
            if current_date in rebalance_dates:
                logger.debug(f"Rebalancing on {current_date}")
                
                # Calculate target allocation
                if len(price_data.loc[:current_date]) >= 30:  # Minimum data for optimization
                    lookback_data = price_data.loc[:current_date].tail(252)  # 1 year lookback
                    target_allocation = self._calculate_target_allocation(
                        config.strategy, lookback_data, config.tickers
                    )
                else:
                    # Use equal weights if insufficient data
                    equal_weight = 1.0 / len(config.tickers)
                    target_allocation = {ticker: equal_weight for ticker in config.tickers}
                
                # Rebalance portfolio
                transaction_cost = self._rebalance_portfolio(
                    positions, 
                    target_allocation, 
                    portfolio_value, 
                    current_prices, 
                    config.transaction_cost
                )
                
                total_transaction_costs += transaction_cost
                num_rebalances += 1
                
                # Record allocation
                allocation_history.append({
                    'date': current_date,
                    'allocation': target_allocation.copy(),
                    'transaction_cost': transaction_cost
                })
            
            # Calculate current portfolio value
            current_value = sum(
                positions.get(ticker, 0) * current_prices.get(ticker, np.nan)
                for ticker in config.tickers
                if not pd.isna(current_prices.get(ticker, np.nan)) and positions.get(ticker, 0) != 0
            )
            
            # Ensure we have a valid portfolio value
            if current_value <= 0 and daily_values:
                # Use previous value if calculation fails
                current_value = daily_values[-1]
            elif current_value <= 0:
                # Use initial capital if no previous value
                current_value = config.initial_capital
            
            daily_values.append(current_value)
            
            # Calculate daily return
            if daily_values:
                if len(daily_values) > 1:
                    prev_value = daily_values[-2]
                    daily_return = max(-0.5, min(0.5, (current_value - prev_value) / prev_value))  # Cap extreme returns to avoid numerical issues
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0.0)
        
        # Prepare returns data for storage
        returns_series = pd.Series(daily_returns, index=price_data.index[1:])
        
        # Prepare data for storage
        returns_data = {
            'dates': [date.strftime('%Y-%m-%d') for date in returns_series.index],
            'returns': returns_series.tolist(),
            'cumulative_returns': (1 + returns_series).cumprod().tolist()
        }
        
        allocation_data = {
            'rebalance_dates': [item['date'].strftime('%Y-%m-%d') for item in allocation_history],
            'allocations': [item['allocation'] for item in allocation_history],
            'transaction_costs': [item['transaction_cost'] for item in allocation_history]
        }
        
        return {
            'daily_returns': returns_series,
            'daily_values': daily_values,
            'total_transaction_costs': total_transaction_costs,
            'num_rebalances': num_rebalances,
            'returns_data': returns_data,
            'allocation_data': allocation_data
        }
    
    def _get_rebalance_dates(self, start_date: date, end_date: date, 
                           frequency: RebalanceFrequency) -> List[date]:
        """
        Generate rebalancing dates based on frequency.
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Rebalancing frequency
            
        Returns:
            List of rebalancing dates
        """
        dates = []
        current_date = start_date
        
        if frequency == RebalanceFrequency.DAILY:
            while current_date <= end_date:
                dates.append(current_date)
                current_date += timedelta(days=1)
        
        elif frequency == RebalanceFrequency.WEEKLY:
            # Rebalance every Monday
            while current_date <= end_date:
                if current_date.weekday() == 0:  # Monday
                    dates.append(current_date)
                current_date += timedelta(days=1)
        
        elif frequency == RebalanceFrequency.MONTHLY:
            # Rebalance on first trading day of each month
            current_month = start_date.month
            current_year = start_date.year
            
            while current_date <= end_date:
                if current_date.month != current_month or current_date.year != current_year:
                    dates.append(current_date)
                    current_month = current_date.month
                    current_year = current_date.year
                current_date += timedelta(days=1)
        
        elif frequency == RebalanceFrequency.QUARTERLY:
            # Rebalance every quarter
            quarter_months = [1, 4, 7, 10]
            current_quarter = None
            
            while current_date <= end_date:
                current_quarter_month = ((current_date.month - 1) // 3) * 3 + 1
                if current_quarter != current_quarter_month:
                    dates.append(current_date)
                    current_quarter = current_quarter_month
                current_date += timedelta(days=1)
        
        # Always include start date if not already included
        if start_date not in dates:
            dates.insert(0, start_date)
        
        return sorted(dates)
    
    def _calculate_target_allocation(self, strategy: OptimizationStrategy, 
                                   price_data: pd.DataFrame, 
                                   tickers: List[str]) -> Dict[str, float]:
        """
        Calculate target portfolio allocation using optimization strategy.
        
        Args:
            strategy: Optimization strategy
            price_data: Historical price data
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping tickers to target weights
        """
        try:
            # Calculate returns
            returns_data = price_data.pct_change().dropna()
            
            # Get strategy optimizer
            optimizer = self.strategy_registry.get_strategy(strategy)
            
            # Use optimizer to get allocation
            if hasattr(optimizer, 'optimize'):
                allocation = optimizer.optimize(returns_data, {})
            else:
                # Fallback to equal weights
                logger.warning(f"Optimizer {strategy} not found, using equal weights")
                equal_weight = 1.0 / len(tickers)
                allocation = {ticker: equal_weight for ticker in tickers}
            
            # Ensure all tickers are included
            for ticker in tickers:
                if ticker not in allocation:
                    allocation[ticker] = 0.0
            
            # Normalize weights to sum to 1
            total_weight = sum(allocation.values())
            if total_weight > 0:
                allocation = {k: v / total_weight for k, v in allocation.items()}
            else:
                # Fallback to equal weights
                equal_weight = 1.0 / len(tickers)
                allocation = {ticker: equal_weight for ticker in tickers}
            
            return allocation
            
        except Exception as e:
            logger.warning(f"Allocation calculation failed: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / len(tickers)
            return {ticker: equal_weight for ticker in tickers}
    
    def _rebalance_portfolio(self, positions: Dict[str, float], 
                           target_allocation: Dict[str, float],
                           portfolio_value: float, 
                           current_prices: pd.Series,
                           transaction_cost_rate: float) -> float:
        """
        Rebalance portfolio to target allocation.
        
        Args:
            positions: Current positions (shares)
            target_allocation: Target allocation weights
            portfolio_value: Current portfolio value
            current_prices: Current asset prices
            transaction_cost_rate: Transaction cost rate
            
        Returns:
            Total transaction costs incurred
        """
        total_transaction_cost = 0.0
        
        for ticker in target_allocation:
            current_price = current_prices.get(ticker, 0.0)
            
            if current_price <= 0:
                continue
            
            # Calculate target value and shares
            target_value = portfolio_value * target_allocation[ticker]
            target_shares = target_value / current_price
            
            # Calculate current shares
            current_shares = positions.get(ticker, 0.0)
            
            # Calculate shares to trade
            shares_to_trade = target_shares - current_shares
            
            if abs(shares_to_trade) > 0.001:  # Minimum trade threshold: $0.001
                # Calculate transaction cost
                trade_value = abs(shares_to_trade) * current_price
                transaction_cost = trade_value * transaction_cost_rate
                total_transaction_cost += transaction_cost
                
                # Update positions
                positions[ticker] = target_shares
        
        return total_transaction_cost
    
    def _calculate_performance_metrics(self, simulation_results: Dict[str, Any], 
                                     config: BacktestConfig) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            simulation_results: Results from portfolio simulation
            config: Backtest configuration
            
        Returns:
            Dictionary with performance metrics
        """
        daily_returns = simulation_results['daily_returns']
        
        # Remove NaN or infinite values
        returns_series = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns_series) == 0:
            return self._get_default_metrics(config)
        
        # Basic metrics
        total_return = (1 + returns_series).prod() - 1
        trading_days = len(returns_series)
        years = trading_days / 252.0  # 252 trading days per year
        
        # Annualized metrics
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        # Volatility
        volatility = returns_series.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        try:
            cumulative_returns = (1 + returns_series).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
        except:
            max_drawdown = 0.0
        
        # Calmar ratio
        if max_drawdown < 0 and not pd.isna(max_drawdown):
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        final_value = config.initial_capital * (1 + total_return)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'transaction_costs': simulation_results['total_transaction_costs'],
            'num_rebalances': simulation_results['num_rebalances'],
            'final_value': final_value
        }
    
    def _get_default_metrics(self, config: BacktestConfig) -> Dict[str, float]:
        """Return default metrics when calculation fails."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'transaction_costs': 0.0,
            'num_rebalances': 0,
            'final_value': config.initial_capital
        }
    
    def _perform_statistical_tests(self, results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """
        Perform statistical significance tests on strategy returns.
        
        Args:
            results: Dictionary of strategy results
            
        Returns:
            Dictionary with statistical test results
        """
        if len(results) < 2:
            return {}
        
        try:
            # Extract returns for each strategy
            strategy_returns = {}
            for strategy_name, result in results.items():
                if result.returns_data and 'returns' in result.returns_data:
                    strategy_returns[strategy_name] = np.array(result.returns_data['returns'])
            
            if len(strategy_returns) < 2:
                return {}
            
            # Perform pairwise t-tests
            statistical_tests = {}
            strategy_names = list(strategy_returns.keys())
            
            for i, strategy1 in enumerate(strategy_names):
                for strategy2 in strategy_names[i+1:]:
                    returns1 = strategy_returns[strategy1]
                    returns2 = strategy_returns[strategy2]
                    
                    # Ensure same length
                    min_length = min(len(returns1), len(returns2))
                    if min_length > 1:
                        returns1 = returns1[:min_length]
                        returns2 = returns2[:min_length]
                        
                        # Perform t-test
                        t_statistic, p_value = stats.ttest_rel(returns1, returns2)
                        
                        statistical_tests[f"{strategy1}_vs_{strategy2}"] = {
                            't_statistic': float(t_statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
            
            return statistical_tests
            
        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")
            return {}