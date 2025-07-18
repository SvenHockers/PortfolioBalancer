"""Scheduler service with cron-like scheduling capabilities."""

import logging
import time
import threading
import schedule
import datetime
import pytz
import requests
from typing import Dict, List, Optional, Callable, Any, Union
import traceback

from .config import get_config
from .logging import correlation_context
from .orchestration import PipelineOrchestrator


class SchedulerError(Exception):
    """Custom exception for scheduler errors."""
    pass


class ServiceStatus:
    """Status of a service health check."""
    
    def __init__(self, service_name: str, is_healthy: bool, message: str = ""):
        """
        Initialize service status.
        
        Args:
            service_name: Name of the service
            is_healthy: Whether the service is healthy
            message: Additional status message
        """
        self.service_name = service_name
        self.is_healthy = is_healthy
        self.message = message
        self.timestamp = datetime.datetime.now()


class Scheduler:
    """
    Scheduler service with cron-like scheduling capabilities.
    
    This class is responsible for:
    1. Scheduling pipeline execution at specified times
    2. Checking health of dependent services before execution
    3. Sending notifications for pipeline failures
    4. Managing scheduled jobs
    """
    
    def __init__(self, orchestrator: Optional[PipelineOrchestrator] = None):
        """
        Initialize scheduler service.
        
        Args:
            orchestrator: Pipeline orchestrator (if None, will be created)
        """
        self.config = get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize orchestrator if not provided
        self.orchestrator = orchestrator or PipelineOrchestrator()
        
        # Initialize scheduler
        self.scheduler = schedule
        
        # Track scheduled jobs
        self.jobs = {}
        
        # Track service health
        self.service_health = {}
        
        # Thread for running the scheduler
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
        self.logger.info("Scheduler initialized")
    
    def schedule_daily(self, time_str: Optional[str] = None) -> None:
        """
        Schedule daily pipeline execution.
        
        Args:
            time_str: Time string in HH:MM format (if None, uses execution_time from config)
        """
        time_str = time_str or self.config.scheduler.execution_time
        
        try:
            # Parse time string
            hour, minute = map(int, time_str.split(':'))
            
            # Schedule job
            job = self.scheduler.every().day.at(time_str).do(self._execute_pipeline)
            
            # Track job
            job_id = f"daily_{hour:02d}_{minute:02d}"
            self.jobs[job_id] = job
            
            self.logger.info(f"Scheduled daily pipeline execution at {time_str}")
            
        except Exception as e:
            error_msg = f"Failed to schedule daily execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise SchedulerError(error_msg) from e
    
    def schedule_interval(self, interval_minutes: int) -> None:
        """
        Schedule pipeline execution at regular intervals.
        
        Args:
            interval_minutes: Interval in minutes
        """
        if interval_minutes <= 0:
            raise ValueError("Interval must be positive")
        
        try:
            # Schedule job
            job = self.scheduler.every(interval_minutes).minutes.do(self._execute_pipeline)
            
            # Track job
            job_id = f"interval_{interval_minutes}"
            self.jobs[job_id] = job
            
            self.logger.info(f"Scheduled pipeline execution every {interval_minutes} minutes")
            
        except Exception as e:
            error_msg = f"Failed to schedule interval execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise SchedulerError(error_msg) from e
    
    def schedule_custom(self, schedule_func: Callable[[], None], job_id: str) -> None:
        """
        Schedule custom pipeline execution.
        
        Args:
            schedule_func: Function that schedules the job using the schedule library
            job_id: Unique identifier for the job
        """
        try:
            # Call scheduling function
            job = schedule_func()
            
            # Track job
            self.jobs[job_id] = job
            
            self.logger.info(f"Scheduled custom pipeline execution with ID {job_id}")
            
        except Exception as e:
            error_msg = f"Failed to schedule custom execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise SchedulerError(error_msg) from e
    
    def start(self) -> None:
        """Start the scheduler in a background thread."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler is already running")
            return
        
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler is not running")
            return
        
        self.stop_event.set()
        self.scheduler_thread.join(timeout=5.0)
        
        if self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler thread did not stop gracefully")
        else:
            self.logger.info("Scheduler stopped")
    
    def clear_jobs(self) -> None:
        """Clear all scheduled jobs."""
        self.scheduler.clear()
        self.jobs.clear()
        self.logger.info("All scheduled jobs cleared")
    
    def execute_now(self) -> Dict[str, Any]:
        """
        Execute pipeline immediately.
        
        Returns:
            Pipeline execution results
        """
        self.logger.info("Executing pipeline immediately")
        return self._execute_pipeline()
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        self.logger.info("Scheduler loop started")
        
        while not self.stop_event.is_set():
            try:
                self.scheduler.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}", exc_info=True)
                # Continue running despite errors
    
    def _execute_pipeline(self) -> Dict[str, Any]:
        """
        Execute the pipeline with health checks and notifications.
        
        Returns:
            Pipeline execution results
        """
        with correlation_context():
            self.logger.info("Starting scheduled pipeline execution")
            
            try:
                # Check service health
                if not self._check_service_health():
                    self.logger.error("Pipeline execution aborted due to service health issues")
                    self._send_notification("Pipeline execution aborted due to service health issues")
                    return {"status": "aborted", "reason": "service_health"}
                
                # Execute pipeline
                results = self.orchestrator.execute_pipeline()
                
                # Check for failures
                has_failures = any(
                    result.status.value == "failed" 
                    for result in results.values()
                )
                
                if has_failures:
                    self.logger.error("Pipeline execution completed with failures")
                    self._send_notification("Pipeline execution completed with failures")
                else:
                    self.logger.info("Pipeline execution completed successfully")
                
                return {"status": "completed", "results": self.orchestrator.get_pipeline_status()}
                
            except Exception as e:
                error_msg = f"Pipeline execution failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self._send_notification(f"Pipeline execution failed: {str(e)}")
                return {"status": "failed", "error": str(e)}
    
    def _check_service_health(self) -> bool:
        """
        Check health of dependent services.
        
        Returns:
            True if all services are healthy
        """
        self.logger.info("Checking service health")
        
        # Reset service health
        self.service_health = {}
        
        # Check data provider health
        data_provider_health = self._check_data_provider_health()
        self.service_health["data_provider"] = data_provider_health
        
        # Check broker health
        broker_health = self._check_broker_health()
        self.service_health["broker"] = broker_health
        
        # Check if all services are healthy
        all_healthy = all(status.is_healthy for status in self.service_health.values())
        
        if all_healthy:
            self.logger.info("All services are healthy")
        else:
            unhealthy_services = [
                f"{name}: {status.message}"
                for name, status in self.service_health.items()
                if not status.is_healthy
            ]
            self.logger.error(f"Unhealthy services: {', '.join(unhealthy_services)}")
        
        return all_healthy
    
    def _check_data_provider_health(self) -> ServiceStatus:
        """
        Check health of data provider.
        
        Returns:
            ServiceStatus object
        """
        try:
            # For yfinance, we can check by making a simple request
            import yfinance as yf
            
            # Try to fetch a single data point for a well-known ticker
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            
            if info and "symbol" in info:
                return ServiceStatus("data_provider", True, "yfinance API is accessible")
            else:
                return ServiceStatus("data_provider", False, "yfinance API returned incomplete data")
                
        except Exception as e:
            return ServiceStatus("data_provider", False, f"yfinance API check failed: {str(e)}")
    
    def _check_broker_health(self) -> ServiceStatus:
        """
        Check health of broker API.
        
        Returns:
            ServiceStatus object
        """
        broker_type = self.config.executor.broker_type
        
        try:
            if broker_type == "alpaca":
                # Check Alpaca API health
                api_key = self.config.broker.alpaca_api_key
                secret_key = self.config.broker.alpaca_secret_key
                base_url = self.config.broker.alpaca_base_url
                
                if not api_key or not secret_key:
                    return ServiceStatus("broker", False, "Alpaca API credentials not configured")
                
                # Make a simple request to check API health
                headers = {
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": secret_key
                }
                
                response = requests.get(f"{base_url}/v2/account", headers=headers)
                
                if response.status_code == 200:
                    return ServiceStatus("broker", True, "Alpaca API is accessible")
                else:
                    return ServiceStatus("broker", False, f"Alpaca API returned status {response.status_code}")
                    
            elif broker_type == "ib":
                # For Interactive Brokers, we would need to check the connection
                # This is a simplified check that just verifies the configuration
                host = self.config.broker.ib_host
                port = self.config.broker.ib_port
                
                if not host or not port:
                    return ServiceStatus("broker", False, "Interactive Brokers configuration incomplete")
                
                # In a real implementation, we would try to connect to the TWS API
                # For now, just return a placeholder status
                return ServiceStatus("broker", True, "Interactive Brokers configuration is valid")

            elif broker_type == "t212":
                api_key = self.config.broker.t212_api_key
                demo = self.config.broker.t212_demo

                if not api_key or demo:
                    return ServiceStatus("broker", False, "Interactive Brokers configuration incomplete")
                return ServiceStatus("broker", True, "Trading212 configuration is complete")
                
            else:
                return ServiceStatus("broker", False, f"Unknown broker type: {broker_type}")
                
        except Exception as e:
            return ServiceStatus("broker", False, f"Broker health check failed: {str(e)}")
    
    def _send_notification(self, message: str) -> None:
        """
        Send notification for pipeline events.
        
        Args:
            message: Notification message
        """
        self.logger.info(f"Sending notification: {message}")
        
        # In a real implementation, this would send an email, Slack message, etc.
        # For now, just log the notification
        
        # Example webhook notification (commented out)
        # webhook_url = os.environ.get("NOTIFICATION_WEBHOOK_URL")
        # if webhook_url:
        #     try:
        #         payload = {
        #             "text": f"Portfolio Rebalancer: {message}",
        #             "timestamp": datetime.datetime.now().isoformat()
        #         }
        #         requests.post(webhook_url, json=payload)
        #     except Exception as e:
        #         self.logger.error(f"Failed to send webhook notification: {str(e)}")
    
    def get_next_run_time(self) -> Optional[datetime.datetime]:
        """
        Get the next scheduled run time.
        
        Returns:
            Datetime of next run or None if no jobs scheduled
        """
        next_run = self.scheduler.next_run()
        
        if next_run:
            # Convert to timezone-aware datetime
            timezone = pytz.timezone(self.config.scheduler.timezone)
            return timezone.localize(next_run)
        
        return None
    
    def get_service_health(self) -> Dict[str, ServiceStatus]:
        """
        Get current service health status.
        
        Returns:
            Dictionary mapping service names to ServiceStatus objects
        """
        # Update service health if empty
        if not self.service_health:
            self._check_service_health()
        
        return self.service_health