"""Backtesting engine for portfolio strategy validation."""

import logging
import hashlib
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats

from ..models import (
    BacktestConfig, BacktestResult, OptimizationStrategy, 
    RebalanceFrequency, StrategyComparison
)
from ..exceptions import BacktestError, InsufficientDataError
from ...common.interfaces import DataStorage
from ...optimizer.optimization import SharpeOptimizer
from ...optimizer.risk_model import RiskModel

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """Historical portfolio performance simulation engine."""
    
    def __init__(self, data_storage: DataStorage):
        """
        Initialize backtesting engine.
        
        Args:
            data_storage: Data storage interface for retrieving historical data
        """
        self.data_storage = data_storage
        self.risk_model = RiskModel()
        self.strategy_registry = StrategyRegistry()
        
        # Initialize new components for task 3.2
        self.walk_forward_analyzer = WalkForwardAnalysis(data_storage)
        self.walk_forward_analyzer.backtesting_engine = self
        self.performance_attribution = PerformanceAttribution(data_storage)
        
        logger.info("Initialized BacktestingEngine with strategy comparison and validation")
    
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

    def run_walk_forward_analysis(self, 
                                 config: BacktestConfig,
                                 in_sample_months: int = 12,
                                 out_sample_months: int = 3,
                                 step_months: int = 1) -> Dict[str, Any]:
        """
        Run walk-forward analysis for strategy robustness testing.
        
        Args:
            config: Base backtest configuration
            in_sample_months: Months of data for optimization
            out_sample_months: Months of data for out-of-sample testing
            step_months: Step size in months between windows
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        return self.walk_forward_analyzer.run_walk_forward_analysis(
            config, in_sample_months, out_sample_months, step_months
        )
    
    def analyze_performance_attribution(self, 
                                      backtest_result: BacktestResult,
                                      benchmark_tickers: Optional[List[str]] = None,
                                      benchmark_weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze performance attribution for backtest results.
        
        Args:
            backtest_result: Backtest result to analyze
            benchmark_tickers: Benchmark ticker symbols
            benchmark_weights: Benchmark weights
            
        Returns:
            Dictionary with attribution analysis results
        """
        return self.performance_attribution.analyze_attribution(
            backtest_result, benchmark_tickers, benchmark_weights
        )