"""Enhanced asynchronous processing for long-running analytics operations."""

import logging
import os
import time
import psutil
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
from contextlib import contextmanager

from celery import Celery, Task
from celery.result import AsyncResult
from celery.exceptions import Retry, WorkerLostError, SoftTimeLimitExceeded
from celery.signals import task_prerun, task_postrun, task_failure, task_retry

from .models import BacktestConfig, MonteCarloConfig, BacktestResult, MonteCarloResult
from .exceptions import AnalyticsError, BacktestError, SimulationError

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Enhanced task status."""
    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    RETRY = "RETRY"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"
    TIMEOUT = "TIMEOUT"


@dataclass
class TaskMetrics:
    """Task execution metrics."""
    task_id: str
    task_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None
    input_size: int = 0
    output_size: int = 0


class ResourceMonitor:
    """Monitor resource usage during task execution."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _monitor_resources(self):
        """Monitor resources in background thread."""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                self.metrics.append({
                    'timestamp': datetime.utcnow(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent
                })
                
                # Keep only recent metrics (last 100 points)
                if len(self.metrics) > 100:
                    self.metrics = self.metrics[-100:]
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage."""
        if not self.metrics:
            return {'memory_mb': 0.0, 'cpu_percent': 0.0}
        
        peak_memory = max(m['memory_mb'] for m in self.metrics)
        peak_cpu = max(m['cpu_percent'] for m in self.metrics)
        
        return {'memory_mb': peak_memory, 'cpu_percent': peak_cpu}

# Enhanced Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')

# Create Celery app with enhanced configuration
celery_app = Celery(
    'analytics_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['portfolio_rebalancer.analytics.async_processing']
)

# Enhanced Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task tracking and limits
    task_track_started=True,
    task_time_limit=7200,  # 2 hours max for complex analytics
    task_soft_time_limit=6600,  # 110 minutes soft limit
    
    # Worker configuration
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    worker_disable_rate_limits=False,
    
    # Task acknowledgment and retry
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Result backend
    result_expires=172800,  # 48 hours
    result_compression='gzip',
    
    # Task routing and priorities
    task_routes={
        'analytics.run_backtest_async': {'queue': 'analytics_high'},
        'analytics.run_monte_carlo_async': {'queue': 'analytics_high'},
        'analytics.batch_process_portfolios': {'queue': 'analytics_batch'},
        'analytics.cleanup_old_results': {'queue': 'analytics_maintenance'},
    },
    
    # Priority queues
    task_default_queue='analytics_normal',
    task_create_missing_queues=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Memory and resource management
    worker_max_memory_per_child=1024 * 1024,  # 1GB per worker
)


class EnhancedAnalyticsTask(Task):
    """Enhanced base task class with comprehensive monitoring and error handling."""
    
    def __init__(self):
        super().__init__()
        self.resource_monitor = None
        self.task_metrics = None
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
            peak_usage = self.resource_monitor.get_peak_usage()
            logger.info(f"Task {task_id} completed successfully. Peak usage: {peak_usage}")
        else:
            logger.info(f"Task {task_id} completed successfully")
        
        # Store task metrics
        if self.task_metrics:
            self.task_metrics.completed_at = datetime.utcnow()
            self.task_metrics.duration = (
                self.task_metrics.completed_at - self.task_metrics.started_at
            ).total_seconds()
            
            if self.resource_monitor:
                peak_usage = self.resource_monitor.get_peak_usage()
                self.task_metrics.memory_usage_mb = peak_usage['memory_mb']
                self.task_metrics.cpu_usage_percent = peak_usage['cpu_percent']
            
            # Log performance metrics
            logger.info(f"Task metrics: {asdict(self.task_metrics)}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        error_msg = str(exc)
        logger.error(f"Task {task_id} failed: {error_msg}")
        logger.error(f"Traceback: {einfo.traceback}")
        
        # Store failure metrics
        if self.task_metrics:
            self.task_metrics.completed_at = datetime.utcnow()
            self.task_metrics.error_message = error_msg
            if self.task_metrics.started_at:
                self.task_metrics.duration = (
                    self.task_metrics.completed_at - self.task_metrics.started_at
                ).total_seconds()
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is being retried."""
        if self.task_metrics:
            self.task_metrics.retry_count += 1
        
        logger.warning(f"Task {task_id} retrying (attempt {self.task_metrics.retry_count if self.task_metrics else 'unknown'}): {exc}")
    
    def apply_async(self, args=None, kwargs=None, task_id=None, producer=None,
                   link=None, link_error=None, shadow=None, **options):
        """Override apply_async to add priority and resource management."""
        
        # Set priority-based routing
        priority = options.get('priority', TaskPriority.NORMAL)
        if priority == TaskPriority.HIGH:
            options['queue'] = 'analytics_high'
        elif priority == TaskPriority.LOW:
            options['queue'] = 'analytics_low'
        else:
            options['queue'] = 'analytics_normal'
        
        # Add resource limits based on task type
        if 'monte_carlo' in self.name:
            options['time_limit'] = 7200  # 2 hours for Monte Carlo
            options['soft_time_limit'] = 6600
        elif 'backtest' in self.name:
            options['time_limit'] = 3600  # 1 hour for backtests
            options['soft_time_limit'] = 3300
        
        return super().apply_async(args, kwargs, task_id, producer, link, link_error, shadow, **options)
    
    def _setup_monitoring(self, task_id: str, task_name: str):
        """Setup resource monitoring and metrics tracking."""
        self.resource_monitor = ResourceMonitor(task_id)
        self.resource_monitor.start_monitoring()
        
        self.task_metrics = TaskMetrics(
            task_id=task_id,
            task_name=task_name,
            started_at=datetime.utcnow()
        )
    
    def _handle_soft_timeout(self):
        """Handle soft timeout gracefully."""
        logger.warning(f"Task {self.request.id} approaching time limit, attempting graceful shutdown")
        
        # Update task state
        self.update_state(
            state='TIMEOUT',
            meta={
                'status': 'Task timed out',
                'progress': -1,
                'error': 'Task exceeded time limit'
            }
        )
        
        # Cleanup resources
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        raise SoftTimeLimitExceeded("Task exceeded soft time limit")
    
    def _calculate_retry_delay(self, retry_count: int, base_delay: int = 60) -> int:
        """Calculate exponential backoff retry delay."""
        # Exponential backoff with jitter
        import random
        delay = min(base_delay * (2 ** retry_count), 300)  # Max 5 minutes
        jitter = random.uniform(0.8, 1.2)  # Add 20% jitter
        return int(delay * jitter)


@celery_app.task(bind=True, base=EnhancedAnalyticsTask, name='analytics.run_backtest_async')
def run_backtest_async(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced asynchronous backtest execution with comprehensive monitoring.
    
    Args:
        config_dict: Backtest configuration as dictionary
        
    Returns:
        Backtest result as dictionary
    """
    # Setup monitoring and metrics
    self._setup_monitoring(self.request.id, 'run_backtest_async')
    
    try:
        # Update task state to indicate processing has started
        self.update_state(
            state=TaskStatus.PROCESSING,
            meta={
                'status': 'Starting backtest',
                'progress': 0,
                'started_at': datetime.utcnow().isoformat(),
                'estimated_duration': '5-30 minutes',
                'resource_usage': {'memory_mb': 0, 'cpu_percent': 0}
            }
        )
        
        # Validate input size for metrics
        if self.task_metrics:
            self.task_metrics.input_size = len(str(config_dict))
        
        # Import here to avoid circular imports
        from .analytics_service import AnalyticsService
        from ..common.config import get_config
        from ..fetcher.storage import ParquetStorage, SQLiteStorage
        from .storage import PostgreSQLAnalyticsStorage
        
        # Initialize services (similar to main service)
        config = get_config()
        
        if config.data.storage_type == "parquet":
            data_storage = ParquetStorage(config.data.storage_path)
        else:
            data_storage = SQLiteStorage(config.data.storage_path)
        
        # Get analytics DB URL
        analytics_db_url = os.getenv('ANALYTICS_DB_URL')
        if not analytics_db_url:
            db_host = os.getenv('POSTGRES_HOST', 'localhost')
            db_port = os.getenv('POSTGRES_PORT', '5432')
            db_name = os.getenv('POSTGRES_DB', 'portfolio_rebalancer')
            db_user = os.getenv('POSTGRES_USER', 'portfolio_user')
            db_password = os.getenv('POSTGRES_PASSWORD', 'portfolio_pass')
            analytics_db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        analytics_storage = PostgreSQLAnalyticsStorage(analytics_db_url)
        analytics_service = AnalyticsService(data_storage, analytics_storage)
        
        # Parse configuration
        backtest_config = BacktestConfig.model_validate(config_dict)
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Configuration validated, starting backtest',
                'progress': 10,
                'config': config_dict
            }
        )
        
        # Run backtest with progress updates
        class ProgressCallback:
            def __init__(self, task):
                self.task = task
                self.last_progress = 10
            
            def update(self, message: str, progress: int):
                if progress > self.last_progress:
                    self.task.update_state(
                        state='PROCESSING',
                        meta={
                            'status': message,
                            'progress': min(progress, 90),  # Reserve 90-100 for finalization
                        }
                    )
                    self.last_progress = progress
        
        progress_callback = ProgressCallback(self)
        
        # Update progress for data retrieval
        progress_callback.update('Retrieving historical data', 20)
        
        # Run the actual backtest
        result = analytics_service.run_backtest(backtest_config)
        
        # Update progress for completion
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Backtest completed, storing results',
                'progress': 95
            }
        )
        
        # Convert result to dictionary for serialization
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
        
        logger.info(f"Async backtest completed for task {self.request.id}")
        
        return {
            'status': 'completed',
            'result': result_dict,
            'completed_at': datetime.utcnow().isoformat(),
            'task_id': self.request.id
        }
        
    except BacktestError as e:
        logger.error(f"Backtest failed in async task: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=2)
    except Exception as e:
        logger.error(f"Unexpected error in async backtest: {e}")
        raise AnalyticsError(f"Async backtest failed: {e}")


@celery_app.task(bind=True, base=EnhancedAnalyticsTask, name='analytics.run_monte_carlo_async')
def run_monte_carlo_async(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation asynchronously.
    
    Args:
        config_dict: Monte Carlo configuration as dictionary
        
    Returns:
        Monte Carlo result as dictionary
    """
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Starting Monte Carlo simulation',
                'progress': 0,
                'started_at': datetime.utcnow().isoformat()
            }
        )
        
        # Import here to avoid circular imports
        from .analytics_service import AnalyticsService
        from ..common.config import get_config
        from ..fetcher.storage import ParquetStorage, SQLiteStorage
        from .storage import PostgreSQLAnalyticsStorage
        
        # Initialize services
        config = get_config()
        
        if config.data.storage_type == "parquet":
            data_storage = ParquetStorage(config.data.storage_path)
        else:
            data_storage = SQLiteStorage(config.data.storage_path)
        
        # Get analytics DB URL
        analytics_db_url = os.getenv('ANALYTICS_DB_URL')
        if not analytics_db_url:
            db_host = os.getenv('POSTGRES_HOST', 'localhost')
            db_port = os.getenv('POSTGRES_PORT', '5432')
            db_name = os.getenv('POSTGRES_DB', 'portfolio_rebalancer')
            db_user = os.getenv('POSTGRES_USER', 'portfolio_user')
            db_password = os.getenv('POSTGRES_PASSWORD', 'portfolio_pass')
            analytics_db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        analytics_storage = PostgreSQLAnalyticsStorage(analytics_db_url)
        analytics_service = AnalyticsService(data_storage, analytics_storage)
        
        # Parse configuration
        monte_carlo_config = MonteCarloConfig.model_validate(config_dict)
        
        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Configuration validated, starting simulation',
                'progress': 10,
                'config': config_dict
            }
        )
        
        # Progress callback for simulation
        class SimulationProgressCallback:
            def __init__(self, task):
                self.task = task
                self.last_progress = 10
            
            def update(self, message: str, progress: int):
                if progress > self.last_progress:
                    self.task.update_state(
                        state='PROCESSING',
                        meta={
                            'status': message,
                            'progress': min(progress, 90),
                        }
                    )
                    self.last_progress = progress
        
        progress_callback = SimulationProgressCallback(self)
        
        # Update progress for data preparation
        progress_callback.update('Preparing simulation data', 20)
        
        # Run the Monte Carlo simulation
        result = analytics_service.run_monte_carlo(monte_carlo_config)
        
        # Update progress for completion
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Simulation completed, storing results',
                'progress': 95
            }
        )
        
        # Convert result to dictionary
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
        
        logger.info(f"Async Monte Carlo simulation completed for task {self.request.id}")
        
        return {
            'status': 'completed',
            'result': result_dict,
            'completed_at': datetime.utcnow().isoformat(),
            'task_id': self.request.id
        }
        
    except SimulationError as e:
        logger.error(f"Monte Carlo simulation failed in async task: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=2)
    except Exception as e:
        logger.error(f"Unexpected error in async Monte Carlo: {e}")
        raise AnalyticsError(f"Async Monte Carlo failed: {e}")


class AsyncAnalyticsProcessor:
    """Enhanced manager for asynchronous analytics processing with comprehensive monitoring."""
    
    def __init__(self):
        """Initialize async processor."""
        self.celery_app = celery_app
        self.task_queue = task_queue
        self.task_metrics_history = []
        logger.info("Enhanced async analytics processor initialized")
    
    def submit_backtest(self, config: BacktestConfig, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit backtest for asynchronous processing with priority.
        
        Args:
            config: Backtest configuration
            priority: Task priority level
            
        Returns:
            Task ID for tracking progress
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            
            # Estimate duration based on configuration
            estimated_duration = self._estimate_backtest_duration(config)
            
            # Submit task with priority
            task_id = self.task_queue.submit_task_with_priority(
                run_backtest_async,
                args=(config_dict,),
                kwargs={},
                priority=priority,
                estimated_duration=estimated_duration
            )
            
            logger.info(f"Submitted async backtest with task ID: {task_id}, priority: {priority.value}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit async backtest: {e}")
            raise AnalyticsError(f"Failed to submit async backtest: {e}")
    
    def submit_monte_carlo(self, config: MonteCarloConfig, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit Monte Carlo simulation for asynchronous processing with priority.
        
        Args:
            config: Monte Carlo configuration
            priority: Task priority level
            
        Returns:
            Task ID for tracking progress
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            
            # Estimate duration based on configuration
            estimated_duration = self._estimate_monte_carlo_duration(config)
            
            # Submit task with priority
            task_id = self.task_queue.submit_task_with_priority(
                run_monte_carlo_async,
                args=(config_dict,),
                kwargs={},
                priority=priority,
                estimated_duration=estimated_duration
            )
            
            logger.info(f"Submitted async Monte Carlo with task ID: {task_id}, priority: {priority.value}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit async Monte Carlo: {e}")
            raise AnalyticsError(f"Failed to submit async Monte Carlo: {e}")
    
    def submit_batch_processing(self, portfolios_data: List[Dict[str, Any]], 
                              operation: str, priority: TaskPriority = TaskPriority.LOW) -> str:
        """
        Submit batch portfolio processing.
        
        Args:
            portfolios_data: List of portfolio data
            operation: Operation to perform
            priority: Task priority level
            
        Returns:
            Task ID for tracking progress
        """
        try:
            estimated_duration = len(portfolios_data) * 30  # 30 seconds per portfolio estimate
            
            task_id = self.task_queue.submit_task_with_priority(
                batch_process_portfolios_async,
                args=(portfolios_data, operation),
                kwargs={},
                priority=priority,
                estimated_duration=estimated_duration
            )
            
            logger.info(f"Submitted batch processing with task ID: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit batch processing: {e}")
            raise AnalyticsError(f"Failed to submit batch processing: {e}")
    
    def submit_cleanup_task(self, days_to_keep: int = 90) -> str:
        """
        Submit cleanup task for old analytics results.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Task ID for tracking progress
        """
        try:
            task_id = self.task_queue.submit_task_with_priority(
                cleanup_old_results_async,
                args=(days_to_keep,),
                kwargs={},
                priority=TaskPriority.LOW,
                estimated_duration=600  # 10 minutes estimate
            )
            
            logger.info(f"Submitted cleanup task with task ID: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit cleanup task: {e}")
            raise AnalyticsError(f"Failed to submit cleanup task: {e}")
    
    def _estimate_backtest_duration(self, config: BacktestConfig) -> int:
        """Estimate backtest duration in seconds."""
        # Base duration
        base_duration = 300  # 5 minutes
        
        # Factor in number of assets
        asset_factor = len(config.tickers) * 30  # 30 seconds per asset
        
        # Factor in time period
        days = (config.end_date - config.start_date).days
        time_factor = days * 0.5  # 0.5 seconds per day
        
        # Factor in rebalancing frequency
        freq_multiplier = {
            'daily': 2.0,
            'weekly': 1.5,
            'monthly': 1.0,
            'quarterly': 0.8
        }
        
        frequency = config.rebalance_frequency
        if isinstance(frequency, str):
            multiplier = freq_multiplier.get(frequency, 1.0)
        else:
            multiplier = freq_multiplier.get(frequency.value, 1.0)
        
        estimated = int((base_duration + asset_factor + time_factor) * multiplier)
        return min(estimated, 3600)  # Cap at 1 hour
    
    def _estimate_monte_carlo_duration(self, config: MonteCarloConfig) -> int:
        """Estimate Monte Carlo simulation duration in seconds."""
        # Base duration
        base_duration = 600  # 10 minutes
        
        # Factor in number of simulations
        sim_factor = config.num_simulations * 0.01  # 0.01 seconds per simulation
        
        # Factor in time horizon
        horizon_factor = config.time_horizon_years * 10  # 10 seconds per year
        
        # Factor in number of assets
        asset_factor = len(config.portfolio_tickers) * 20  # 20 seconds per asset
        
        estimated = int(base_duration + sim_factor + horizon_factor + asset_factor)
        return min(estimated, 7200)  # Cap at 2 hours
    
    def submit_monte_carlo(self, config: MonteCarloConfig) -> str:
        """
        Submit Monte Carlo simulation for asynchronous processing.
        
        Args:
            config: Monte Carlo configuration
            
        Returns:
            Task ID for tracking progress
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            
            # Submit task
            result = run_monte_carlo_async.delay(config_dict)
            
            logger.info(f"Submitted async Monte Carlo with task ID: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit async Monte Carlo: {e}")
            raise AnalyticsError(f"Failed to submit async Monte Carlo: {e}")
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of async task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            status_info = {
                'task_id': task_id,
                'state': result.state,
                'ready': result.ready(),
                'successful': result.successful() if result.ready() else None,
                'failed': result.failed() if result.ready() else None,
            }
            
            if result.state == 'PENDING':
                status_info.update({
                    'status': 'Task is waiting to be processed',
                    'progress': 0
                })
            elif result.state == 'PROCESSING':
                # Get progress information
                info = result.info or {}
                status_info.update({
                    'status': info.get('status', 'Processing'),
                    'progress': info.get('progress', 0),
                    'started_at': info.get('started_at'),
                    'config': info.get('config')
                })
            elif result.state == 'SUCCESS':
                # Task completed successfully
                info = result.info or {}
                status_info.update({
                    'status': 'Completed successfully',
                    'progress': 100,
                    'result': info.get('result'),
                    'completed_at': info.get('completed_at')
                })
            elif result.state == 'FAILURE':
                # Task failed
                status_info.update({
                    'status': 'Task failed',
                    'progress': 0,
                    'error': str(result.info) if result.info else 'Unknown error',
                    'traceback': result.traceback
                })
            elif result.state == 'RETRY':
                # Task is being retried
                status_info.update({
                    'status': 'Task is being retried',
                    'progress': 0,
                    'retry_count': getattr(result.info, 'retries', 0) if result.info else 0
                })
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return {
                'task_id': task_id,
                'state': 'UNKNOWN',
                'error': str(e)
            }
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get result of completed task.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            Task result or None if not ready/failed
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            if not result.ready():
                return None
            
            if result.successful():
                return result.get(timeout=timeout)
            else:
                logger.error(f"Task {task_id} failed: {result.info}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get task result for {task_id}: {e}")
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancellation was successful
        """
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Cancelled task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get list of active tasks.
        
        Returns:
            List of active task information
        """
        try:
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if not active_tasks:
                return []
            
            # Flatten tasks from all workers
            all_tasks = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task_info = {
                        'task_id': task['id'],
                        'name': task['name'],
                        'worker': worker,
                        'args': task.get('args', []),
                        'kwargs': task.get('kwargs', {}),
                        'time_start': task.get('time_start')
                    }
                    all_tasks.append(task_info)
            
            return all_tasks
            
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return []
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of async processing system.
        
        Returns:
            Comprehensive status information
        """
        try:
            # Get basic queue status
            queue_status = self.task_queue.get_queue_status()
            
            # Get task metrics
            task_metrics = self._get_task_performance_metrics()
            
            # Get resource usage
            resource_usage = self._get_system_resource_usage()
            
            # Get error rates
            error_rates = self._calculate_error_rates()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'queue_status': queue_status,
                'task_metrics': task_metrics,
                'resource_usage': resource_usage,
                'error_rates': error_rates,
                'health_score': self._calculate_health_score(queue_status, error_rates)
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def _get_task_performance_metrics(self) -> Dict[str, Any]:
        """Get task performance metrics."""
        if not self.task_metrics_history:
            return {'message': 'No task metrics available'}
        
        # Calculate averages
        total_tasks = len(self.task_metrics_history)
        avg_duration = sum(m.duration for m in self.task_metrics_history if m.duration) / total_tasks
        avg_memory = sum(m.memory_usage_mb for m in self.task_metrics_history) / total_tasks
        
        # Group by task type
        by_type = {}
        for metric in self.task_metrics_history:
            task_type = metric.task_name
            if task_type not in by_type:
                by_type[task_type] = []
            by_type[task_type].append(metric)
        
        type_metrics = {}
        for task_type, metrics in by_type.items():
            type_metrics[task_type] = {
                'count': len(metrics),
                'avg_duration': sum(m.duration for m in metrics if m.duration) / len(metrics),
                'avg_memory_mb': sum(m.memory_usage_mb for m in metrics) / len(metrics),
                'success_rate': len([m for m in metrics if not m.error_message]) / len(metrics)
            }
        
        return {
            'total_tasks': total_tasks,
            'avg_duration': avg_duration,
            'avg_memory_mb': avg_memory,
            'by_task_type': type_metrics
        }
    
    def _get_system_resource_usage(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': round(memory_available_gb, 2),
                'disk_percent': disk_percent,
                'disk_free_gb': round(disk_free_gb, 2),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
        except ImportError:
            return {'message': 'psutil not available for resource monitoring'}
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates for different task types."""
        if not self.task_metrics_history:
            return {}
        
        # Group by task type
        by_type = {}
        for metric in self.task_metrics_history:
            task_type = metric.task_name
            if task_type not in by_type:
                by_type[task_type] = {'total': 0, 'errors': 0}
            
            by_type[task_type]['total'] += 1
            if metric.error_message:
                by_type[task_type]['errors'] += 1
        
        # Calculate error rates
        error_rates = {}
        for task_type, counts in by_type.items():
            error_rates[task_type] = counts['errors'] / counts['total'] if counts['total'] > 0 else 0
        
        # Overall error rate
        total_tasks = sum(counts['total'] for counts in by_type.values())
        total_errors = sum(counts['errors'] for counts in by_type.values())
        error_rates['overall'] = total_errors / total_tasks if total_tasks > 0 else 0
        
        return error_rates
    
    def _calculate_health_score(self, queue_status: Dict[str, Any], error_rates: Dict[str, float]) -> float:
        """Calculate overall health score (0-100)."""
        try:
            score = 100.0
            
            # Deduct for inactive workers
            active_workers = queue_status.get('active_workers', 0)
            if active_workers == 0:
                score -= 50
            elif active_workers < 2:
                score -= 20
            
            # Deduct for high queue lengths
            total_queued = (
                queue_status.get('total_reserved', 0) + 
                queue_status.get('total_scheduled', 0)
            )
            if total_queued > 100:
                score -= 30
            elif total_queued > 50:
                score -= 15
            
            # Deduct for high error rates
            overall_error_rate = error_rates.get('overall', 0)
            if overall_error_rate > 0.2:  # 20% error rate
                score -= 40
            elif overall_error_rate > 0.1:  # 10% error rate
                score -= 20
            
            return max(0, score)
            
        except Exception:
            return 50.0  # Default moderate health score
    
    def optimize_task_distribution(self) -> Dict[str, Any]:
        """
        Optimize task distribution based on current load and performance.
        
        Returns:
            Optimization results
        """
        try:
            # Get current status
            status = self.get_comprehensive_status()
            
            recommendations = []
            actions_taken = []
            
            # Check if workers are overloaded
            resource_usage = status.get('resource_usage', {})
            cpu_percent = resource_usage.get('cpu_percent', 0)
            memory_percent = resource_usage.get('memory_percent', 0)
            
            if cpu_percent > 80:
                recommendations.append("Consider adding more workers or reducing task complexity")
                
                # Cancel low priority tasks if system is overloaded
                cancelled = self.task_queue.cancel_task_by_priority(TaskPriority.LOW)
                if cancelled > 0:
                    actions_taken.append(f"Cancelled {cancelled} low priority tasks")
            
            if memory_percent > 85:
                recommendations.append("High memory usage detected, consider memory optimization")
            
            # Check error rates
            error_rates = status.get('error_rates', {})
            overall_error_rate = error_rates.get('overall', 0)
            
            if overall_error_rate > 0.15:
                recommendations.append("High error rate detected, investigate task failures")
            
            # Check queue lengths
            queue_status = status.get('queue_status', {})
            total_queued = (
                queue_status.get('total_reserved', 0) + 
                queue_status.get('total_scheduled', 0)
            )
            
            if total_queued > 50:
                recommendations.append("High queue length, consider scaling workers")
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'health_score': status.get('health_score', 0),
                'recommendations': recommendations,
                'actions_taken': actions_taken,
                'current_load': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'queue_length': total_queued,
                    'error_rate': overall_error_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Task optimization failed: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of async processing system.
        
        Returns:
            Health status information
        """
        try:
            status = self.get_comprehensive_status()
            health_score = status.get('health_score', 0)
            
            return {
                'status': 'healthy' if health_score > 70 else 'degraded' if health_score > 30 else 'unhealthy',
                'health_score': health_score,
                'active_workers': status.get('queue_status', {}).get('active_workers', 0),
                'total_queued_tasks': (
                    status.get('queue_status', {}).get('total_reserved', 0) +
                    status.get('queue_status', {}).get('total_scheduled', 0)
                ),
                'error_rate': status.get('error_rates', {}).get('overall', 0),
                'broker_url': CELERY_BROKER_URL,
                'result_backend': CELERY_RESULT_BACKEND,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Async processing health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Enh
anced task functions with better monitoring and error handling

@celery_app.task(bind=True, base=EnhancedAnalyticsTask, name='analytics.batch_process_portfolios')
def batch_process_portfolios_async(self, portfolios_data: List[Dict[str, Any]], 
                                 operation: str) -> Dict[str, Any]:
    """
    Process multiple portfolios in batch asynchronously.
    
    Args:
        portfolios_data: List of portfolio data
        operation: Operation to perform ('performance', 'risk', 'dividend')
        
    Returns:
        Batch processing results
    """
    self._setup_monitoring(self.request.id, 'batch_process_portfolios')
    
    try:
        total_portfolios = len(portfolios_data)
        
        self.update_state(
            state=TaskStatus.PROCESSING,
            meta={
                'status': f'Starting batch {operation} processing',
                'progress': 0,
                'total_portfolios': total_portfolios,
                'completed_portfolios': 0,
                'started_at': datetime.utcnow().isoformat()
            }
        )
        
        # Import services
        from .analytics_service import AnalyticsService
        from ..common.config import get_config
        from ..fetcher.storage import ParquetStorage, SQLiteStorage
        from .storage import PostgreSQLAnalyticsStorage
        
        # Initialize services
        config = get_config()
        
        if config.data.storage_type == "parquet":
            data_storage = ParquetStorage(config.data.storage_path)
        else:
            data_storage = SQLiteStorage(config.data.storage_path)
        
        analytics_db_url = os.getenv('ANALYTICS_DB_URL')
        if not analytics_db_url:
            db_host = os.getenv('POSTGRES_HOST', 'localhost')
            db_port = os.getenv('POSTGRES_PORT', '5432')
            db_name = os.getenv('POSTGRES_DB', 'portfolio_rebalancer')
            db_user = os.getenv('POSTGRES_USER', 'portfolio_user')
            db_password = os.getenv('POSTGRES_PASSWORD', 'portfolio_pass')
            analytics_db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        analytics_storage = PostgreSQLAnalyticsStorage(analytics_db_url)
        analytics_service = AnalyticsService(data_storage, analytics_storage)
        
        # Process portfolios
        results = {}
        completed = 0
        
        for i, portfolio_data in enumerate(portfolios_data):
            try:
                portfolio_id = portfolio_data.get('portfolio_id')
                
                # Check for soft timeout
                if hasattr(self, '_handle_soft_timeout'):
                    # Simple timeout check - in real implementation would be more sophisticated
                    pass
                
                # Process based on operation type
                if operation == 'performance':
                    result = analytics_service.get_performance_metrics(portfolio_id)
                elif operation == 'risk':
                    result = analytics_service.get_risk_analysis(portfolio_id)
                elif operation == 'dividend':
                    result = analytics_service.get_dividend_analysis(portfolio_id)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                results[portfolio_id] = result.model_dump() if hasattr(result, 'model_dump') else result
                completed += 1
                
                # Update progress
                progress = int((completed / total_portfolios) * 90)  # Reserve 90-100 for finalization
                self.update_state(
                    state=TaskStatus.PROCESSING,
                    meta={
                        'status': f'Processed {completed}/{total_portfolios} portfolios',
                        'progress': progress,
                        'completed_portfolios': completed,
                        'current_portfolio': portfolio_id
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to process portfolio {portfolio_data.get('portfolio_id', 'unknown')}: {e}")
                results[portfolio_data.get('portfolio_id', f'portfolio_{i}')] = {'error': str(e)}
                completed += 1
        
        # Finalization
        self.update_state(
            state=TaskStatus.PROCESSING,
            meta={
                'status': 'Finalizing batch processing',
                'progress': 95,
                'completed_portfolios': completed
            }
        )
        
        return {
            'status': 'completed',
            'operation': operation,
            'total_portfolios': total_portfolios,
            'completed_portfolios': completed,
            'results': results,
            'completed_at': datetime.utcnow().isoformat(),
            'task_id': self.request.id
        }
        
    except SoftTimeLimitExceeded:
        return {
            'status': 'timeout',
            'message': 'Batch processing exceeded time limit',
            'completed_portfolios': completed if 'completed' in locals() else 0,
            'partial_results': results if 'results' in locals() else {}
        }
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise AnalyticsError(f"Batch processing failed: {e}")


@celery_app.task(bind=True, base=EnhancedAnalyticsTask, name='analytics.cleanup_old_results')
def cleanup_old_results_async(self, days_to_keep: int = 90) -> Dict[str, Any]:
    """
    Clean up old analytics results asynchronously.
    
    Args:
        days_to_keep: Number of days of data to keep
        
    Returns:
        Cleanup statistics
    """
    self._setup_monitoring(self.request.id, 'cleanup_old_results')
    
    try:
        self.update_state(
            state=TaskStatus.PROCESSING,
            meta={
                'status': 'Starting cleanup of old analytics results',
                'progress': 0,
                'days_to_keep': days_to_keep,
                'started_at': datetime.utcnow().isoformat()
            }
        )
        
        # Import query optimizer for cleanup
        from .query_optimizer import QueryOptimizer
        
        # Get database URL
        analytics_db_url = os.getenv('ANALYTICS_DB_URL')
        if not analytics_db_url:
            db_host = os.getenv('POSTGRES_HOST', 'localhost')
            db_port = os.getenv('POSTGRES_PORT', '5432')
            db_name = os.getenv('POSTGRES_DB', 'portfolio_rebalancer')
            db_user = os.getenv('POSTGRES_USER', 'portfolio_user')
            db_password = os.getenv('POSTGRES_PASSWORD', 'portfolio_pass')
            analytics_db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        query_optimizer = QueryOptimizer(analytics_db_url)
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING,
            meta={
                'status': 'Cleaning up old data',
                'progress': 50
            }
        )
        
        # Perform cleanup
        cleanup_stats = query_optimizer.cleanup_old_data(days_to_keep)
        
        # Update progress
        self.update_state(
            state=TaskStatus.PROCESSING,
            meta={
                'status': 'Running database maintenance',
                'progress': 80
            }
        )
        
        # Run VACUUM ANALYZE for performance
        query_optimizer.vacuum_analyze_tables()
        
        return {
            'status': 'completed',
            'cleanup_stats': cleanup_stats,
            'days_kept': days_to_keep,
            'completed_at': datetime.utcnow().isoformat(),
            'task_id': self.request.id
        }
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise AnalyticsError(f"Cleanup task failed: {e}")


# Task signal handlers for enhanced monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task prerun for monitoring."""
    logger.info(f"Task {task_id} ({task.name}) starting with args: {args[:2] if args else []}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                        retval=None, state=None, **kwds):
    """Handle task postrun for monitoring."""
    logger.info(f"Task {task_id} ({task.name}) completed with state: {state}")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handle task failure for monitoring."""
    logger.error(f"Task {task_id} failed with exception: {exception}")


@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwds):
    """Handle task retry for monitoring."""
    logger.warning(f"Task {task_id} retrying due to: {reason}")


class TaskQueue:
    """Task queue management with priority and resource control."""
    
    def __init__(self):
        self.celery_app = celery_app
        self.active_tasks = {}
        self.task_history = []
    
    def submit_task_with_priority(self, task_func: callable, args: tuple, kwargs: dict,
                                priority: TaskPriority = TaskPriority.NORMAL,
                                estimated_duration: Optional[int] = None) -> str:
        """
        Submit task with priority and resource management.
        
        Args:
            task_func: Task function to execute
            args: Task arguments
            kwargs: Task keyword arguments
            priority: Task priority level
            estimated_duration: Estimated duration in seconds
            
        Returns:
            Task ID
        """
        try:
            # Add priority to options
            options = {
                'priority': priority,
                'estimated_duration': estimated_duration
            }
            
            # Submit task
            result = task_func.apply_async(args=args, kwargs=kwargs, **options)
            
            # Track active task
            self.active_tasks[result.id] = {
                'task_name': task_func.name,
                'priority': priority,
                'submitted_at': datetime.utcnow(),
                'estimated_duration': estimated_duration
            }
            
            logger.info(f"Submitted task {result.id} with priority {priority.value}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise AnalyticsError(f"Failed to submit task: {e}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.
        
        Returns:
            Queue status information
        """
        try:
            inspect = self.celery_app.control.inspect()
            
            # Get queue lengths
            reserved = inspect.reserved() or {}
            active = inspect.active() or {}
            scheduled = inspect.scheduled() or {}
            
            # Calculate totals
            total_reserved = sum(len(tasks) for tasks in reserved.values())
            total_active = sum(len(tasks) for tasks in active.values())
            total_scheduled = sum(len(tasks) for tasks in scheduled.values())
            
            # Get worker stats
            stats = inspect.stats() or {}
            
            return {
                'active_workers': len(stats),
                'total_reserved': total_reserved,
                'total_active': total_active,
                'total_scheduled': total_scheduled,
                'worker_stats': stats,
                'queue_lengths': {
                    'analytics_high': self._get_queue_length('analytics_high'),
                    'analytics_normal': self._get_queue_length('analytics_normal'),
                    'analytics_low': self._get_queue_length('analytics_low'),
                    'analytics_batch': self._get_queue_length('analytics_batch')
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {'error': str(e)}
    
    def _get_queue_length(self, queue_name: str) -> int:
        """Get length of specific queue."""
        try:
            # This would require additional Redis connection to check queue lengths
            # For now, return 0 as placeholder
            return 0
        except:
            return 0
    
    def cancel_task_by_priority(self, priority: TaskPriority) -> int:
        """
        Cancel all tasks with specific priority.
        
        Args:
            priority: Priority level to cancel
            
        Returns:
            Number of tasks cancelled
        """
        try:
            cancelled_count = 0
            
            # Get tasks to cancel
            tasks_to_cancel = [
                task_id for task_id, task_info in self.active_tasks.items()
                if task_info.get('priority') == priority
            ]
            
            # Cancel tasks
            for task_id in tasks_to_cancel:
                self.celery_app.control.revoke(task_id, terminate=True)
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                cancelled_count += 1
            
            logger.info(f"Cancelled {cancelled_count} tasks with priority {priority.value}")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Failed to cancel tasks by priority: {e}")
            return 0


# Global task queue instance
task_queue = TaskQueue()