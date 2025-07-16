"""Pipeline orchestrator for coordinating fetcher, optimizer, and executor services."""

import logging
import time
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import functools

from ..common.config import get_config
from ..common.logging import correlation_context, log_execution_time
from ..common.metrics import (
    timed, record_pipeline_execution, record_pipeline_step_duration,
    update_system_health, start_metrics_server
)
from ..common.models import TargetAllocation
from ..fetcher.data_fetcher import DataFetcher
from ..optimizer.portfolio_optimizer import PortfolioOptimizer
from ..executor.trade_executor import TradeExecutor


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
                 error: Optional[Exception] = None):
        """
        Initialize step result.
        
        Args:
            step: Pipeline step
            status: Execution status
            result: Step execution result (if any)
            error: Exception that occurred (if any)
        """
        self.step = step
        self.status = status
        self.result = result
        self.error = error
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def set_timing(self, start_time: datetime, end_time: datetime) -> None:
        """Set timing information for the step."""
        self.start_time = start_time
        self.end_time = end_time
        self.duration = (end_time - start_time).total_seconds()


class PipelineOrchestrator:
    """
    Orchestrator that coordinates fetcher, optimizer, and executor services.
    
    This class is responsible for:
    1. Managing dependencies between services
    2. Sequencing execution of pipeline steps
    3. Handling errors and implementing retry logic
    4. Tracking execution status and results
    """
    
    def __init__(self, 
                 data_fetcher: Optional[DataFetcher] = None,
                 portfolio_optimizer: Optional[PortfolioOptimizer] = None,
                 trade_executor: Optional[TradeExecutor] = None):
        """
        Initialize pipeline orchestrator.
        
        Args:
            data_fetcher: Data fetcher service (if None, will be created)
            portfolio_optimizer: Portfolio optimizer service (if None, will be created)
            trade_executor: Trade executor service (if None, will be created)
        """
        self.config = get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize services if not provided
        self.data_fetcher = data_fetcher
        self.portfolio_optimizer = portfolio_optimizer
        self.trade_executor = trade_executor
        
        # Track execution results
        self.results = {}
        
        self.logger.info("PipelineOrchestrator initialized")
    
    def _ensure_services(self) -> None:
        """Ensure all required services are initialized."""
        from ..fetcher.yfinance_provider import YFinanceProvider
        from ..fetcher.storage import ParquetStorage, SQLiteStorage
        
        # Initialize data fetcher if needed
        if self.data_fetcher is None:
            self.logger.info("Initializing data fetcher")
            provider = YFinanceProvider()
            
            # Initialize storage based on config
            storage_type = self.config.data.storage_type.lower()
            storage_path = self.config.data.storage_path
            
            if storage_type == "parquet":
                storage = ParquetStorage(base_dir=storage_path)
            elif storage_type == "sqlite":
                storage = SQLiteStorage(db_path=f"{storage_path}/prices.db")
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
            
            self.data_fetcher = DataFetcher(provider=provider, storage=storage)
        
        # Initialize portfolio optimizer if needed
        if self.portfolio_optimizer is None:
            self.logger.info("Initializing portfolio optimizer")
            # We need a data storage instance, which should be the same as used by data fetcher
            if hasattr(self.data_fetcher, 'storage'):
                storage = self.data_fetcher.storage
                self.portfolio_optimizer = PortfolioOptimizer(data_storage=storage)
            else:
                raise ValueError("Data fetcher does not have a storage attribute")
        
        # Initialize trade executor if needed
        if self.trade_executor is None:
            self.logger.info("Initializing trade executor")
            self.trade_executor = TradeExecutor()
    
    def _with_retry(self, func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (result, exception) where exception is None if successful
        """
        max_attempts = self.config.scheduler.retry_attempts
        retry_delay = self.config.scheduler.retry_delay
        
        for attempt in range(1, max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                return result, None
            except Exception as e:
                self.logger.error(
                    f"Attempt {attempt}/{max_attempts} failed: {str(e)}",
                    exc_info=True
                )
                
                if attempt < max_attempts:
                    self.logger.info(f"Retrying in {retry_delay} seconds")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"All {max_attempts} attempts failed")
                    return None, e
    
    @log_execution_time(logging.getLogger(__name__), "Data fetch step")
    @timed("orchestrator", "data_fetch")
    def _execute_data_fetch(self) -> StepResult:
        """
        Execute data fetch step.
        
        Returns:
            StepResult with execution status and result
        """
        step = PipelineStep.DATA_FETCH
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting data fetch step")
            
            # Ensure data fetcher is initialized
            if self.data_fetcher is None:
                self._ensure_services()
            
            # Execute data fetch with retry
            tickers = self.config.data.tickers
            lookback_days = self.config.optimization.lookback_days
            
            self.logger.info(f"Fetching data for {len(tickers)} tickers with {lookback_days} days lookback")
            result, error = self._with_retry(
                self.data_fetcher.ensure_data_available,
                tickers=tickers,
                lookback_days=lookback_days,
                force_update=True
            )
            
            end_time = datetime.now()
            
            if error:
                step_result = StepResult(step, PipelineStatus.FAILED, None, error)
                self.logger.error(f"Data fetch step failed: {str(error)}")
                update_system_health("data_fetcher", False)
            else:
                step_result = StepResult(step, PipelineStatus.COMPLETED, result)
                self.logger.info(f"Data fetch step completed successfully")
                update_system_health("data_fetcher", True)
            
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            
            # Record metrics
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            
            return step_result
            
        except Exception as e:
            end_time = datetime.now()
            step_result = StepResult(step, PipelineStatus.FAILED, None, e)
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            self.logger.error(f"Unexpected error in data fetch step: {str(e)}", exc_info=True)
            
            # Record metrics for failure
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            update_system_health("data_fetcher", False)
            
            return step_result
    
    @log_execution_time(logging.getLogger(__name__), "Portfolio optimization step")
    @timed("orchestrator", "optimization")
    def _execute_optimization(self) -> StepResult:
        """
        Execute portfolio optimization step.
        
        Returns:
            StepResult with execution status and result
        """
        step = PipelineStep.OPTIMIZATION
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting portfolio optimization step")
            
            # Check if data fetch step was successful
            data_fetch_result = self.results.get(PipelineStep.DATA_FETCH)
            if data_fetch_result is None or data_fetch_result.status != PipelineStatus.COMPLETED:
                self.logger.error("Cannot execute optimization: data fetch step not completed successfully")
                step_result = StepResult(step, PipelineStatus.SKIPPED)
                step_result.set_timing(start_time, datetime.now())
                self.results[step] = step_result
                
                # Record metrics for skipped step
                record_pipeline_step_duration(step.value, 0)
                update_system_health("optimizer", False)
                
                return step_result
            
            # Ensure portfolio optimizer is initialized
            if self.portfolio_optimizer is None:
                self._ensure_services()
            
            # Execute optimization with retry
            tickers = self.config.data.tickers
            lookback_days = self.config.optimization.lookback_days
            investor_age = self.config.optimization.user_age
            
            # Identify safe tickers (bonds) - this is a simplified approach
            # In a real implementation, this would be more sophisticated
            safe_tickers = [ticker for ticker in tickers if "BOND" in ticker or "TLT" in ticker or "AGG" in ticker]
            
            # Set up optimization constraints
            constraints = {
                "min_weight": self.config.optimization.min_weight,
                "max_weight": self.config.optimization.max_weight
            }
            
            self.logger.info(
                f"Optimizing portfolio for {len(tickers)} tickers, age {investor_age}, "
                f"with {len(safe_tickers)} safe tickers"
            )
            
            result, error = self._with_retry(
                self.portfolio_optimizer.optimize_portfolio,
                tickers=tickers,
                lookback_days=lookback_days,
                investor_age=investor_age,
                safe_tickers=safe_tickers,
                constraints=constraints
            )
            
            end_time = datetime.now()
            
            if error:
                step_result = StepResult(step, PipelineStatus.FAILED, None, error)
                self.logger.error(f"Portfolio optimization step failed: {str(error)}")
                update_system_health("optimizer", False)
            else:
                step_result = StepResult(step, PipelineStatus.COMPLETED, result)
                self.logger.info(
                    f"Portfolio optimization step completed successfully with "
                    f"Sharpe ratio: {result.sharpe_ratio:.4f}"
                )
                
                # Record portfolio metrics
                from ..common.metrics import record_portfolio_metrics, record_portfolio_allocation
                record_portfolio_metrics(
                    result.expected_return,
                    result.expected_volatility,
                    result.sharpe_ratio
                )
                
                # Record portfolio allocation
                record_portfolio_allocation(result.allocations)
                
                update_system_health("optimizer", True)
            
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            
            # Record metrics
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            
            return step_result
            
        except Exception as e:
            end_time = datetime.now()
            step_result = StepResult(step, PipelineStatus.FAILED, None, e)
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            self.logger.error(f"Unexpected error in portfolio optimization step: {str(e)}", exc_info=True)
            
            # Record metrics for failure
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            update_system_health("optimizer", False)
            
            return step_result
    
    @log_execution_time(logging.getLogger(__name__), "Trade execution step")
    @timed("orchestrator", "trade_execution")
    def _execute_trade(self) -> StepResult:
        """
        Execute trade execution step.
        
        Returns:
            StepResult with execution status and result
        """
        step = PipelineStep.EXECUTION
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting trade execution step")
            
            # Check if optimization step was successful
            optimization_result = self.results.get(PipelineStep.OPTIMIZATION)
            if optimization_result is None or optimization_result.status != PipelineStatus.COMPLETED:
                self.logger.error("Cannot execute trades: optimization step not completed successfully")
                step_result = StepResult(step, PipelineStatus.SKIPPED)
                step_result.set_timing(start_time, datetime.now())
                self.results[step] = step_result
                
                # Record metrics for skipped step
                record_pipeline_step_duration(step.value, 0)
                update_system_health("trade_executor", False)
                
                return step_result
            
            # Get target allocation from optimization result
            target_allocation = optimization_result.result
            if not isinstance(target_allocation, TargetAllocation):
                self.logger.error("Invalid optimization result: not a TargetAllocation object")
                step_result = StepResult(step, PipelineStatus.FAILED, None, ValueError("Invalid optimization result"))
                step_result.set_timing(start_time, datetime.now())
                self.results[step] = step_result
                
                # Record metrics for failure
                record_pipeline_step_duration(step.value, 0)
                update_system_health("trade_executor", False)
                
                return step_result
            
            # Ensure trade executor is initialized
            if self.trade_executor is None:
                self._ensure_services()
            
            # First check if rebalancing is needed
            self.logger.info("Checking if rebalancing is needed")
            rebalancing_needed, drift = self.trade_executor.check_rebalancing_needed(target_allocation)
            
            # Record portfolio drift metrics
            from ..common.metrics import record_portfolio_drift
            record_portfolio_drift(drift)
            
            if not rebalancing_needed:
                self.logger.info("No rebalancing needed, skipping trade execution")
                step_result = StepResult(step, PipelineStatus.COMPLETED, {"rebalancing_needed": False, "drift": drift})
                step_result.set_timing(start_time, datetime.now())
                self.results[step] = step_result
                
                # Record metrics for completion without trades
                duration = (datetime.now() - start_time).total_seconds()
                record_pipeline_step_duration(step.value, duration)
                update_system_health("trade_executor", True)
                
                return step_result
            
            # Execute trades with retry
            self.logger.info("Rebalancing needed, executing trades")
            result, error = self._with_retry(
                self.trade_executor.execute_rebalancing,
                target_allocation=target_allocation
            )
            
            end_time = datetime.now()
            
            if error:
                step_result = StepResult(step, PipelineStatus.FAILED, None, error)
                self.logger.error(f"Trade execution step failed: {str(error)}")
                update_system_health("trade_executor", False)
            else:
                step_result = StepResult(
                    step, 
                    PipelineStatus.COMPLETED, 
                    {"rebalancing_needed": True, "orders": result, "drift": drift}
                )
                self.logger.info(f"Trade execution step completed successfully with {len(result)} orders")
                update_system_health("trade_executor", True)
            
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            
            # Record metrics
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            
            return step_result
            
        except Exception as e:
            end_time = datetime.now()
            step_result = StepResult(step, PipelineStatus.FAILED, None, e)
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            self.logger.error(f"Unexpected error in trade execution step: {str(e)}", exc_info=True)
            
            # Record metrics for failure
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            update_system_health("trade_executor", False)
            
            return step_result
    
    @log_execution_time(logging.getLogger(__name__), "Complete pipeline execution")
    @timed("orchestrator", "pipeline_execution")
    def execute_pipeline(self) -> Dict[PipelineStep, StepResult]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dictionary mapping pipeline steps to their results
        """
        with correlation_context():
            self.logger.info("Starting pipeline execution")
            
            # Reset results
            self.results = {}
            
            # Start metrics server if not already running
            start_metrics_server()
            
            # Step 1: Data fetch
            data_fetch_result = self._execute_data_fetch()
            if data_fetch_result.status != PipelineStatus.COMPLETED:
                self.logger.error("Pipeline execution stopped due to data fetch failure")
                record_pipeline_execution("failure")
                return self.results
            
            # Step 2: Portfolio optimization
            optimization_result = self._execute_optimization()
            if optimization_result.status != PipelineStatus.COMPLETED:
                self.logger.error("Pipeline execution stopped due to optimization failure")
                record_pipeline_execution("failure")
                return self.results
            
            # Step 3: Trade execution
            trade_result = self._execute_trade()
            
            # Log overall pipeline status
            all_completed = all(
                result.status == PipelineStatus.COMPLETED 
                for result in self.results.values()
            )
            
            if all_completed:
                self.logger.info("Pipeline execution completed successfully")
                record_pipeline_execution("success")
            else:
                self.logger.error("Pipeline execution completed with errors")
                record_pipeline_execution("partial")
            
            return self.results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current status of the pipeline.
        
        Returns:
            Dictionary with pipeline status information
        """
        status = {
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        for step, result in self.results.items():
            step_status = {
                "status": result.status.value,
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration": result.duration,
                "error": str(result.error) if result.error else None
            }
            
            # Add step-specific result information
            if result.status == PipelineStatus.COMPLETED and result.result is not None:
                if step == PipelineStep.OPTIMIZATION and isinstance(result.result, TargetAllocation):
                    step_status["allocation_count"] = len(result.result.allocations)
                    step_status["sharpe_ratio"] = result.result.sharpe_ratio
                elif step == PipelineStep.EXECUTION and isinstance(result.result, dict):
                    step_status["rebalancing_needed"] = result.result.get("rebalancing_needed", False)
                    if "orders" in result.result:
                        step_status["order_count"] = len(result.result["orders"])
            
            status["steps"][step.value] = step_status
        
        return status