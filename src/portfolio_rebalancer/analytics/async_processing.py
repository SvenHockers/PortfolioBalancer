"""Asynchronous processing for long-running analytics operations."""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from celery import Celery, Task
from celery.result import AsyncResult
from celery.exceptions import Retry

from .models import BacktestConfig, MonteCarloConfig, BacktestResult, MonteCarloResult
from .exceptions import AnalyticsError, BacktestError, SimulationError

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')

# Create Celery app
celery_app = Celery(
    'analytics_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['portfolio_rebalancer.analytics.async_processing']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_default_retry_delay=60,
    task_max_retries=3,
    result_expires=86400,  # 24 hours
)


class CallbackTask(Task):
    """Base task class with callback support."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"Task {task_id} retrying: {exc}")


@celery_app.task(bind=True, base=CallbackTask, name='analytics.run_backtest_async')
def run_backtest_async(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run backtest asynchronously.
    
    Args:
        config_dict: Backtest configuration as dictionary
        
    Returns:
        Backtest result as dictionary
    """
    try:
        # Update task state to indicate processing has started
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Starting backtest',
                'progress': 0,
                'started_at': datetime.utcnow().isoformat()
            }
        )
        
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


@celery_app.task(bind=True, base=CallbackTask, name='analytics.run_monte_carlo_async')
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
    """Manager for asynchronous analytics processing."""
    
    def __init__(self):
        """Initialize async processor."""
        self.celery_app = celery_app
        logger.info("Async analytics processor initialized")
    
    def submit_backtest(self, config: BacktestConfig) -> str:
        """
        Submit backtest for asynchronous processing.
        
        Args:
            config: Backtest configuration
            
        Returns:
            Task ID for tracking progress
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            
            # Submit task
            result = run_backtest_async.delay(config_dict)
            
            logger.info(f"Submitted async backtest with task ID: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit async backtest: {e}")
            raise AnalyticsError(f"Failed to submit async backtest: {e}")
    
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
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of async processing system.
        
        Returns:
            Health status information
        """
        try:
            inspect = self.celery_app.control.inspect()
            
            # Check if workers are available
            stats = inspect.stats()
            active_workers = len(stats) if stats else 0
            
            # Get queue lengths
            reserved = inspect.reserved()
            active = inspect.active()
            
            total_reserved = sum(len(tasks) for tasks in (reserved or {}).values())
            total_active = sum(len(tasks) for tasks in (active or {}).values())
            
            return {
                'status': 'healthy' if active_workers > 0 else 'unhealthy',
                'active_workers': active_workers,
                'total_reserved_tasks': total_reserved,
                'total_active_tasks': total_active,
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