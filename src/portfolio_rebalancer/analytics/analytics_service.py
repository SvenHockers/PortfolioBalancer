"""Main analytics service class."""

import logging
from typing import Dict, List, Optional, Any
from datetime import date, datetime

from ..common.interfaces import DataStorage
from ..common.logging import create_structured_logger, correlation_context
from .storage import AnalyticsStorage
from .models import (
    BacktestConfig, BacktestResult, MonteCarloConfig, MonteCarloResult,
    RiskAnalysis, PerformanceMetrics, DividendAnalysis
)
from .exceptions import (
    AnalyticsError, BacktestError, SimulationError, RiskAnalysisError,
    PerformanceTrackingError, DividendAnalysisError, StorageError,
    wrap_exception
)
from .config import get_analytics_config
from .error_recovery import (
    with_retry, with_circuit_breaker, with_fallback, 
    error_context, recovery_manager, RetryConfig
)

logger = create_structured_logger('analytics_service')


class AnalyticsService:
    """Main analytics service orchestrating all analytics components."""
    
    def __init__(self, data_storage: DataStorage, analytics_storage: AnalyticsStorage):
        """
        Initialize analytics service.
        
        Args:
            data_storage: Data storage interface for historical data
            analytics_storage: Analytics storage interface for results
        """
        self.data_storage = data_storage
        self.analytics_storage = analytics_storage
        self.config = get_analytics_config()
        
        # Analytics engines will be initialized when needed
        self._backtesting_engine = None
        self._monte_carlo_engine = None
        self._risk_analyzer = None
        self._performance_tracker = None
        self._attribution_analyzer = None
        self._dividend_analyzer = None
        
        # Configure retry settings
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0
        )
        
        logger.info("Analytics service initialized", extra={
            'mode': self.config.mode,
            'max_workers': self.config.max_workers
        })
    
    @property
    def backtesting_engine(self):
        """Lazy initialization of backtesting engine."""
        if self._backtesting_engine is None:
            from .engines.backtesting import BacktestingEngine
            self._backtesting_engine = BacktestingEngine(self.data_storage)
        return self._backtesting_engine
    
    @property
    def monte_carlo_engine(self):
        """Lazy initialization of Monte Carlo engine."""
        if self._monte_carlo_engine is None:
            from .engines.monte_carlo import MonteCarloEngine
            self._monte_carlo_engine = MonteCarloEngine(self.data_storage)
        return self._monte_carlo_engine
    
    @property
    def risk_analyzer(self):
        """Lazy initialization of risk analyzer."""
        if self._risk_analyzer is None:
            from .engines.risk_analysis import RiskAnalyzer
            self._risk_analyzer = RiskAnalyzer(self.data_storage)
        return self._risk_analyzer
    
    @property
    def performance_tracker(self):
        """Lazy initialization of performance tracker."""
        if self._performance_tracker is None:
            from .engines.performance import PerformanceTracker
            self._performance_tracker = PerformanceTracker(self.data_storage, self.analytics_storage)
        return self._performance_tracker
    
    @property
    def attribution_analyzer(self):
        """Lazy initialization of attribution analyzer."""
        if self._attribution_analyzer is None:
            from .engines.attribution import AttributionAnalyzer
            self._attribution_analyzer = AttributionAnalyzer(self.data_storage, self.analytics_storage)
        return self._attribution_analyzer
    
    @property
    def dividend_analyzer(self):
        """Lazy initialization of dividend analyzer."""
        if self._dividend_analyzer is None:
            from .engines.dividend_analysis import DividendAnalyzer
            self._dividend_analyzer = DividendAnalyzer(self.data_storage)
        return self._dividend_analyzer
    
    @with_retry()
    @with_circuit_breaker('backtest')
    @with_fallback(operation_name='backtest')
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run backtesting simulation.
        
        Args:
            config: Backtest configuration
            
        Returns:
            Backtest results
            
        Raises:
            BacktestError: If backtesting fails
        """
        with error_context('backtest', 
                          tickers=config.tickers, 
                          start_date=str(config.start_date),
                          end_date=str(config.end_date),
                          strategy=config.strategy):
            try:
                logger.info("Starting backtest", extra={
                    'ticker_count': len(config.tickers),
                    'date_range_days': (config.end_date - config.start_date).days,
                    'strategy': config.strategy
                })
                
                # Run backtest using the engine
                result = self.backtesting_engine.run_backtest(config)
                
                # Store result
                result_id = self.analytics_storage.store_backtest_result(result)
                logger.info("Backtest completed successfully", extra={
                    'result_id': result_id,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio
                })
                
                return result
                
            except BacktestError:
                raise
            except Exception as e:
                raise wrap_exception(
                    e,
                    message="Backtest operation failed",
                    context={
                        'config': config.model_dump() if hasattr(config, 'model_dump') else str(config)
                    }
                )
    
    @with_retry()
    @with_circuit_breaker('monte_carlo')
    def run_monte_carlo(self, config: MonteCarloConfig) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            config: Monte Carlo configuration
            
        Returns:
            Monte Carlo results
            
        Raises:
            SimulationError: If simulation fails
        """
        with error_context('monte_carlo_simulation',
                          time_horizon=config.time_horizon_years,
                          num_simulations=config.num_simulations,
                          portfolio_size=len(config.portfolio_tickers)):
            try:
                logger.info("Starting Monte Carlo simulation", extra={
                    'time_horizon_years': config.time_horizon_years,
                    'num_simulations': config.num_simulations,
                    'portfolio_tickers': config.portfolio_tickers
                })
                
                # Run simulation using the engine
                result = self.monte_carlo_engine.run_simulation(config)
                
                # Store result
                result_id = self.analytics_storage.store_monte_carlo_result(result)
                logger.info("Monte Carlo simulation completed successfully", extra={
                    'result_id': result_id,
                    'expected_value': result.expected_value,
                    'probability_of_loss': result.probability_of_loss
                })
                
                return result
                
            except SimulationError:
                raise
            except Exception as e:
                raise wrap_exception(
                    e,
                    message="Monte Carlo simulation failed",
                    context={
                        'config': config.model_dump() if hasattr(config, 'model_dump') else str(config)
                    }
                )
    
    @with_retry()
    @with_circuit_breaker('risk_analysis')
    @with_fallback(operation_name='risk_analysis')
    def analyze_risk(self, portfolio_id: str, tickers: List[str], weights: List[float]) -> RiskAnalysis:
        """
        Analyze portfolio risk.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            
        Returns:
            Risk analysis results
            
        Raises:
            RiskAnalysisError: If risk analysis fails
        """
        with error_context('risk_analysis',
                          portfolio_id=portfolio_id,
                          ticker_count=len(tickers),
                          portfolio_size=len(weights)):
            try:
                logger.info("Starting risk analysis", extra={
                    'portfolio_id': portfolio_id,
                    'tickers': tickers,
                    'weights': weights
                })
                
                # Run risk analysis using the engine
                analysis = self.risk_analyzer.analyze_portfolio_risk(portfolio_id, tickers, weights)
                
                # Store result
                self.analytics_storage.store_risk_analysis(analysis)
                logger.info("Risk analysis completed successfully", extra={
                    'portfolio_id': portfolio_id,
                    'var_95': analysis.var_95,
                    'portfolio_beta': analysis.portfolio_beta
                })
                
                return analysis
                
            except RiskAnalysisError:
                raise
            except Exception as e:
                raise wrap_exception(
                    e,
                    message="Risk analysis failed",
                    context={
                        'portfolio_id': portfolio_id,
                        'tickers': tickers,
                        'weights': weights
                    }
                )
    
    def track_performance(self, portfolio_id: str, tickers: List[str], weights: List[float]) -> PerformanceMetrics:
        """
        Track portfolio performance.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            
        Returns:
            Performance metrics
            
        Raises:
            AnalyticsError: If performance tracking fails
        """
        try:
            logger.info(f"Tracking performance for portfolio {portfolio_id}")
            
            # Create portfolio object
            from ..common.models import Portfolio
            portfolio = Portfolio(
                id=portfolio_id,
                tickers=tickers,
                weights=weights,
                target_allocation=dict(zip(tickers, weights))
            )
            
            # Track performance using the engine
            metrics = self.performance_tracker.track_performance(portfolio)
            
            # Store result
            self.analytics_storage.store_performance_metrics(metrics)
            logger.info(f"Performance tracking completed for portfolio {portfolio_id}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance tracking failed: {e}")
            raise AnalyticsError(f"Performance tracking failed: {e}")
    
    def calculate_attribution(self, portfolio_id: str, tickers: List[str], weights: List[float],
                            benchmark_tickers: List[str], benchmark_weights: List[float],
                            start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Calculate performance attribution analysis.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: Portfolio ticker symbols
            weights: Portfolio weights
            benchmark_tickers: Benchmark ticker symbols
            benchmark_weights: Benchmark weights
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Attribution analysis results
            
        Raises:
            AnalyticsError: If attribution analysis fails
        """
        try:
            logger.info(f"Calculating attribution for portfolio {portfolio_id}")
            
            # Create portfolio objects
            from ..common.models import Portfolio
            portfolio = Portfolio(
                id=portfolio_id,
                tickers=tickers,
                weights=weights,
                target_allocation=dict(zip(tickers, weights))
            )
            
            benchmark_portfolio = Portfolio(
                id=f"{portfolio_id}_benchmark",
                tickers=benchmark_tickers,
                weights=benchmark_weights,
                target_allocation=dict(zip(benchmark_tickers, benchmark_weights))
            )
            
            # Calculate attribution using the engine
            attribution = self.attribution_analyzer.calculate_brinson_attribution(
                portfolio, benchmark_portfolio, start_date, end_date
            )
            
            logger.info(f"Attribution calculation completed for portfolio {portfolio_id}")
            return attribution
            
        except Exception as e:
            logger.error(f"Attribution calculation failed: {e}")
            raise AnalyticsError(f"Attribution calculation failed: {e}")
    
    def calculate_multi_period_attribution(self, portfolio_id: str, tickers: List[str], weights: List[float],
                                         benchmark_tickers: List[str], benchmark_weights: List[float],
                                         periods: List[tuple]) -> Dict[str, Any]:
        """
        Calculate attribution analysis for multiple time periods.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: Portfolio ticker symbols
            weights: Portfolio weights
            benchmark_tickers: Benchmark ticker symbols
            benchmark_weights: Benchmark weights
            periods: List of (start_date, end_date) tuples
            
        Returns:
            Multi-period attribution results
            
        Raises:
            AnalyticsError: If attribution analysis fails
        """
        try:
            logger.info(f"Calculating multi-period attribution for portfolio {portfolio_id}")
            
            # Create portfolio objects
            from ..common.models import Portfolio
            portfolio = Portfolio(
                id=portfolio_id,
                tickers=tickers,
                weights=weights,
                target_allocation=dict(zip(tickers, weights))
            )
            
            benchmark_portfolio = Portfolio(
                id=f"{portfolio_id}_benchmark",
                tickers=benchmark_tickers,
                weights=benchmark_weights,
                target_allocation=dict(zip(benchmark_tickers, benchmark_weights))
            )
            
            # Calculate multi-period attribution using the engine
            attribution = self.attribution_analyzer.calculate_multi_period_attribution(
                portfolio, benchmark_portfolio, periods
            )
            
            logger.info(f"Multi-period attribution calculation completed for portfolio {portfolio_id}")
            return attribution
            
        except Exception as e:
            logger.error(f"Multi-period attribution calculation failed: {e}")
            raise AnalyticsError(f"Multi-period attribution calculation failed: {e}")
    
    def analyze_dividends(self, portfolio_id: str, tickers: List[str], weights: List[float]) -> DividendAnalysis:
        """
        Analyze dividend income.
        
        Args:
            portfolio_id: Portfolio identifier
            tickers: List of ticker symbols
            weights: Portfolio weights
            
        Returns:
            Dividend analysis results
            
        Raises:
            AnalyticsError: If dividend analysis fails
        """
        try:
            logger.info(f"Starting dividend analysis for portfolio {portfolio_id}")
            
            # Run dividend analysis using the engine
            analysis = self.dividend_analyzer.analyze_dividend_income(portfolio_id, tickers, weights)
            
            # Store result
            self.analytics_storage.store_dividend_analysis(analysis)
            logger.info(f"Dividend analysis completed for portfolio {portfolio_id}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Dividend analysis failed: {e}")
            raise AnalyticsError(f"Dividend analysis failed: {e}")
    
    def get_backtest_result(self, result_id: str) -> Optional[BacktestResult]:
        """
        Retrieve stored backtest result.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Backtest result or None if not found
        """
        try:
            return self.analytics_storage.get_backtest_result(result_id)
        except Exception as e:
            logger.error(f"Failed to retrieve backtest result {result_id}: {e}")
            raise AnalyticsError(f"Failed to retrieve backtest result: {e}")
    
    def get_monte_carlo_result(self, result_id: str) -> Optional[MonteCarloResult]:
        """
        Retrieve stored Monte Carlo result.
        
        Args:
            result_id: Result identifier
            
        Returns:
            Monte Carlo result or None if not found
        """
        try:
            return self.analytics_storage.get_monte_carlo_result(result_id)
        except Exception as e:
            logger.error(f"Failed to retrieve Monte Carlo result {result_id}: {e}")
            raise AnalyticsError(f"Failed to retrieve Monte Carlo result: {e}")
    
    def get_risk_analysis(self, portfolio_id: str, analysis_date: date) -> Optional[RiskAnalysis]:
        """
        Retrieve stored risk analysis.
        
        Args:
            portfolio_id: Portfolio identifier
            analysis_date: Analysis date
            
        Returns:
            Risk analysis or None if not found
        """
        try:
            return self.analytics_storage.get_risk_analysis(portfolio_id, analysis_date)
        except Exception as e:
            logger.error(f"Failed to retrieve risk analysis: {e}")
            raise AnalyticsError(f"Failed to retrieve risk analysis: {e}")
    
    def get_performance_metrics(self, portfolio_id: str, calculation_date: date) -> Optional[PerformanceMetrics]:
        """
        Retrieve stored performance metrics.
        
        Args:
            portfolio_id: Portfolio identifier
            calculation_date: Calculation date
            
        Returns:
            Performance metrics or None if not found
        """
        try:
            return self.analytics_storage.get_performance_metrics(portfolio_id, calculation_date)
        except Exception as e:
            logger.error(f"Failed to retrieve performance metrics: {e}")
            raise AnalyticsError(f"Failed to retrieve performance metrics: {e}")
    
    def get_performance_history(self, portfolio_id: str, start_date: date, end_date: date) -> List[PerformanceMetrics]:
        """
        Retrieve performance history.
        
        Args:
            portfolio_id: Portfolio identifier
            start_date: Start date
            end_date: End date
            
        Returns:
            List of performance metrics
        """
        try:
            return self.analytics_storage.get_performance_history(portfolio_id, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to retrieve performance history: {e}")
            raise AnalyticsError(f"Failed to retrieve performance history: {e}")
    
    def get_dividend_analysis(self, portfolio_id: str, analysis_date: date) -> Optional[DividendAnalysis]:
        """
        Retrieve stored dividend analysis.
        
        Args:
            portfolio_id: Portfolio identifier
            analysis_date: Analysis date
            
        Returns:
            Dividend analysis or None if not found
        """
        try:
            return self.analytics_storage.get_dividend_analysis(portfolio_id, analysis_date)
        except Exception as e:
            logger.error(f"Failed to retrieve dividend analysis: {e}")
            raise AnalyticsError(f"Failed to retrieve dividend analysis: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of analytics service.
        
        Returns:
            Health status information
        """
        try:
            # Check analytics storage
            storage_healthy = self.analytics_storage.health_check()
            
            # Check data storage (basic check)
            data_storage_healthy = True
            try:
                # Try to get some data to verify data storage is working
                test_data = self.data_storage.get_prices(['AAPL'], 30)
                data_storage_healthy = test_data is not None
            except Exception:
                data_storage_healthy = False
            
            overall_healthy = storage_healthy and data_storage_healthy
            
            return {
                'status': 'healthy' if overall_healthy else 'unhealthy',
                'analytics_storage': 'healthy' if storage_healthy else 'unhealthy',
                'data_storage': 'healthy' if data_storage_healthy else 'unhealthy',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }