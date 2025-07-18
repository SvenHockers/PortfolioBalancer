"""Container-based pipeline orchestrator for coordinating services via HTTP."""

import logging
import time
import requests
from datetime import datetime
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass

from .config import get_config
from .logging import correlation_context, log_execution_time
from .metrics import (
    timed, record_pipeline_execution, record_pipeline_step_duration,
    update_system_health
)
from .interfaces import (
    PipelineOrchestratorInterface, PipelineStep, PipelineStatus, StepResult
)


class ContainerOrchestrator(PipelineOrchestratorInterface):
    """
    Container-based orchestrator that coordinates services via HTTP calls.
    
    This class is responsible for:
    1. Making HTTP calls to other service containers
    2. Sequencing execution of pipeline steps
    3. Handling errors and implementing retry logic
    4. Tracking execution status and results
    """
    
    def __init__(self):
        """Initialize container orchestrator."""
        self.config = get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Service URLs
        self.fetcher_url = f"http://data-fetcher:8080"
        self.optimizer_url = f"http://optimizer:8081"
        self.executor_url = f"http://executor:8082"
        
        # Track execution results
        self.results: Dict[PipelineStep, StepResult] = {}
        
        self.logger.info("ContainerOrchestrator initialized")
    
    def _make_http_call(self, url: str, endpoint: str, method: str = "POST", 
                       timeout: int = 300) -> Dict[str, Any]:
        """
        Make HTTP call to a service.
        
        Args:
            url: Base URL of the service
            endpoint: Endpoint to call
            method: HTTP method
            timeout: Request timeout in seconds
            
        Returns:
            Response data as dictionary
            
        Raises:
            Exception: If the HTTP call fails
        """
        full_url = f"{url}/{endpoint}"
        
        try:
            if method.upper() == "POST":
                response = requests.post(full_url, timeout=timeout)
            else:
                response = requests.get(full_url, timeout=timeout)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP call to {full_url} failed: {str(e)}")
            raise Exception(f"Service call failed: {str(e)}")
    
    def _with_retry(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, Optional[Exception]]:
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
                    f"Attempt {attempt}/{max_attempts} failed: {str(e)}"
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
        Execute data fetch step via HTTP call.
        
        Returns:
            StepResult with execution status and result
        """
        step = PipelineStep.DATA_FETCH
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting data fetch step")
            
            # Make HTTP call to data fetcher service
            result, error = self._with_retry(
                self._make_http_call,
                self.fetcher_url,
                "fetch"
            )
            
            end_time = datetime.now()
            
            if error:
                step_result = StepResult(step, PipelineStatus.FAILED, None, str(error))
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
            step_result = StepResult(step, PipelineStatus.FAILED, None, str(e))
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
        Execute portfolio optimization step via HTTP call.
        
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
            
            # Make HTTP call to optimizer service
            result, error = self._with_retry(
                self._make_http_call,
                self.optimizer_url,
                "optimize"
            )
            
            end_time = datetime.now()
            
            if error:
                step_result = StepResult(step, PipelineStatus.FAILED, None, str(error))
                self.logger.error(f"Portfolio optimization step failed: {str(error)}")
                update_system_health("optimizer", False)
            else:
                step_result = StepResult(step, PipelineStatus.COMPLETED, result)
                self.logger.info(f"Portfolio optimization step completed successfully")
                
                # Record portfolio metrics if available
                if result and 'result' in result:
                    from .metrics import record_portfolio_metrics
                    opt_result = result['result']
                    record_portfolio_metrics(
                        opt_result.get('expected_return', 0),
                        opt_result.get('expected_volatility', 0),
                        opt_result.get('sharpe_ratio', 0)
                    )
                
                update_system_health("optimizer", True)
            
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            
            # Record metrics
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            
            return step_result
            
        except Exception as e:
            end_time = datetime.now()
            step_result = StepResult(step, PipelineStatus.FAILED, None, str(e))
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
        Execute trade execution step via HTTP call.
        
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
            
            # Make HTTP call to executor service
            result, error = self._with_retry(
                self._make_http_call,
                self.executor_url,
                "execute"
            )
            
            end_time = datetime.now()
            
            if error:
                step_result = StepResult(step, PipelineStatus.FAILED, None, str(error))
                self.logger.error(f"Trade execution step failed: {str(error)}")
                update_system_health("trade_executor", False)
            else:
                step_result = StepResult(step, PipelineStatus.COMPLETED, result)
                self.logger.info(f"Trade execution step completed successfully")
                update_system_health("trade_executor", True)
            
            step_result.set_timing(start_time, end_time)
            self.results[step] = step_result
            
            # Record metrics
            duration = (end_time - start_time).total_seconds()
            record_pipeline_step_duration(step.value, duration)
            
            return step_result
            
        except Exception as e:
            end_time = datetime.now()
            step_result = StepResult(step, PipelineStatus.FAILED, None, str(e))
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
        Execute the complete pipeline via HTTP calls to services.
        
        Returns:
            Dictionary mapping pipeline steps to their results
        """
        with correlation_context():
            self.logger.info("Starting container-based pipeline execution")
            
            # Reset results
            self.results = {}
            
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
                "error": result.error
            }
            
            # Add step-specific result information
            if result.status == PipelineStatus.COMPLETED and result.result is not None:
                if step == PipelineStep.OPTIMIZATION and 'result' in result.result:
                    opt_result = result.result['result']
                    step_status["allocation_count"] = opt_result.get('allocation_count', 0)
                    step_status["sharpe_ratio"] = opt_result.get('sharpe_ratio', 0)
                elif step == PipelineStep.EXECUTION and 'result' in result.result:
                    exec_result = result.result['result']
                    step_status["rebalancing_needed"] = exec_result.get('rebalancing_needed', False)
                    step_status["trades_executed"] = exec_result.get('trades_executed', 0)
            
            status["steps"][step.value] = step_status
        
        return status 