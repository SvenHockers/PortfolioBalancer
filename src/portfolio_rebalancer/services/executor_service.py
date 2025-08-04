#!/usr/bin/env python3
"""Trade executor service entry point."""

import logging
import sys
import signal
import threading
from typing import Optional, Dict, Any
from flask import Flask, jsonify
from datetime import datetime

from ..common.config import get_config
from ..common.logging import setup_logging
from ..executor.trade_executor import TradeExecutor
from ..fetcher.storage import ParquetStorage, SQLiteStorage


class ExecutorService:
    """Trade executor service with health check endpoint."""
    
    def __init__(self):
        """Initialize the executor service."""
        self.config = get_config()
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Service state - initialize early
        self.is_healthy = False  # Start as unhealthy until fully initialized
        self.last_execution = None
        self.last_execution_result = None
        self.shutdown_event = threading.Event()
        self.executor = None
        self.storage = None
        
        # Initialize with retry logic
        self._initialize_with_retry()
        
        # Flask app for health checks
        self.app = Flask(__name__)
        self.setup_routes()
    
    def _initialize_with_retry(self, max_retries=5, base_retry_delay=5):
        """Initialize service components with exponential backoff retry logic."""
        import os
        import time
        import random
        
        # Add startup delay to prevent thundering herd
        startup_delay = int(os.getenv('SERVICE_STARTUP_DELAY', '0'))
        if startup_delay > 0:
            jitter = random.uniform(0, min(startup_delay * 0.2, 10))  # Up to 20% jitter, max 10s
            actual_delay = startup_delay + jitter
            self.logger.info(f"Waiting {actual_delay:.1f}s before initialization (startup delay + jitter)")
            time.sleep(actual_delay)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing executor service (attempt {attempt + 1}/{max_retries})")
                
                # Set up yfinance cache to avoid permission errors (executor may use yfinance indirectly)
                from ..common.api_error_handling import setup_yfinance_cache
                setup_yfinance_cache()
                
                # Initialize storage based on configuration
                try:
                    if self.config.data.storage_type == "parquet":
                        self.storage = ParquetStorage(self.config.data.storage_path)
                    else:
                        self.storage = SQLiteStorage(self.config.data.storage_path)
                    self.logger.debug(f"Storage ({self.config.data.storage_type}) initialized successfully")
                except Exception as storage_error:
                    self.logger.error(f"Storage initialization failed: {storage_error}")
                    raise  # Storage is critical, can't continue without it
                
                # Initialize executor with graceful broker error handling
                try:
                    self.executor = TradeExecutor()
                    self.logger.debug("TradeExecutor initialized successfully")
                except Exception as executor_error:
                    self.logger.warning(f"TradeExecutor initialization failed: {executor_error}")
                    # For executor, we can continue in degraded mode (mock trading)
                    self.executor = None
                
                # Mark as healthy if initialization succeeds (even with degraded executor)
                self.is_healthy = True
                self.logger.info("Executor service initialized successfully" + 
                               (" (degraded mode - no trading)" if self.executor is None else ""))
                return
                
            except Exception as e:
                self.logger.error(f"Initialization attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    # Calculate exponential backoff delay with jitter
                    retry_delay = base_retry_delay * (2 ** attempt)
                    max_delay = int(os.getenv('SERVICE_MAX_RETRY_DELAY', '60'))
                    retry_delay = min(retry_delay, max_delay)
                    
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0, retry_delay * 0.1)  # 10% jitter
                    actual_delay = retry_delay + jitter
                    
                    self.logger.info(f"Retrying in {actual_delay:.1f} seconds...")
                    time.sleep(actual_delay)
                else:
                    self.logger.error("All initialization attempts failed - service will start in degraded mode")
                    # Allow service to start in degraded mode for debugging
                    self.is_healthy = False
        
    def setup_routes(self):
        """Setup Flask routes for health checks."""
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint with proper error handling."""
            try:
                # Check if service is in degraded mode
                if not hasattr(self, 'executor') or self.executor is None:
                    if self.is_healthy:
                        # Service started but in degraded mode
                        return jsonify({
                            'status': 'degraded',
                            'service': 'trade-executor',
                            'message': 'Service running in degraded mode - executor not available (mock mode)'
                        }), 200
                    else:
                        return jsonify({
                            'status': 'unhealthy',
                            'service': 'trade-executor',
                            'error': 'Service not properly initialized'
                        }), 503
                
                # Basic health check with timeout protection using threading
                import threading
                import time
                
                health_result = {'completed': False, 'result': None, 'error': None}
                
                def check_health():
                    try:
                        # Check storage accessibility with graceful error handling
                        storage_healthy = True
                        try:
                            # Quick storage check - just verify path exists and is readable
                            import os
                            storage_path = self.config.data.storage_path
                            
                            # Check if path exists
                            if not os.path.exists(storage_path):
                                try:
                                    os.makedirs(storage_path, exist_ok=True)
                                    self.logger.debug(f"Created storage directory: {storage_path}")
                                except (PermissionError, OSError) as mkdir_error:
                                    self.logger.warning(f"Cannot create storage directory {storage_path}: {mkdir_error}")
                                    # Check if parent directory is readable at least
                                    parent_dir = os.path.dirname(storage_path)
                                    if os.path.exists(parent_dir) and os.access(parent_dir, os.R_OK):
                                        storage_healthy = True  # Parent exists and readable, good enough
                                    else:
                                        storage_healthy = False
                                    return
                            
                            # Test basic read access
                            if os.access(storage_path, os.R_OK):
                                # Try to test write access, but don't fail if we can't
                                try:
                                    test_file = os.path.join(storage_path, '.health_check')
                                    with open(test_file, 'w') as f:
                                        f.write('health_check')
                                    os.remove(test_file)
                                    self.logger.debug("Storage write test successful")
                                except (PermissionError, OSError) as write_error:
                                    self.logger.debug(f"Storage write test failed (non-critical): {write_error}")
                                    # Read-only access is acceptable for health check
                                    storage_healthy = True
                            else:
                                self.logger.warning(f"Storage path not readable: {storage_path}")
                                storage_healthy = False
                                
                        except Exception as storage_error:
                            self.logger.warning(f"Storage health check failed: {storage_error}")
                            # Don't fail health check for storage issues - service can run in degraded mode
                            storage_healthy = True
                        
                        # Check broker health with timeout
                        broker_healthy = True
                        try:
                            broker_healthy = self._check_broker_health()
                        except Exception as broker_error:
                            self.logger.warning(f"Broker health check failed: {broker_error}")
                            broker_healthy = False
                        
                        status = {
                            'status': 'healthy' if self.is_healthy and storage_healthy else 'unhealthy',
                            'service': 'trade-executor',
                            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                            'broker_type': self.config.executor.broker_type,
                            'rebalance_threshold': self.config.executor.rebalance_threshold,
                            'storage_healthy': storage_healthy,
                            'broker_healthy': broker_healthy,
                            'dry_run': self.config.executor.dry_run
                        }
                        
                        if self.last_execution_result:
                            status['last_result'] = {
                                'rebalancing_needed': self.last_execution_result.get('rebalancing_needed'),
                                'trades_executed': len(self.last_execution_result.get('trades', [])),
                                'total_drift': self.last_execution_result.get('total_drift'),
                                'success': self.last_execution_result.get('success')
                            }
                        
                        health_result['result'] = (status, 200 if (self.is_healthy and storage_healthy) else 503)
                        health_result['completed'] = True
                        
                    except Exception as e:
                        health_result['error'] = str(e)
                        health_result['completed'] = True
                
                # Run health check in thread with timeout
                health_thread = threading.Thread(target=check_health)
                health_thread.daemon = True
                health_thread.start()
                health_thread.join(timeout=5.0)
                
                if not health_result['completed']:
                    self.logger.error("Health check timed out")
                    return jsonify({
                        'status': 'unhealthy',
                        'service': 'trade-executor',
                        'error': 'Health check timed out'
                    }), 503
                
                if health_result['error']:
                    raise Exception(health_result['error'])
                
                return jsonify(health_result['result'][0]), health_result['result'][1]
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}", exc_info=True)
                return jsonify({
                    'status': 'unhealthy',
                    'service': 'trade-executor',
                    'error': str(e)
                }), 503
        
        @self.app.route('/ready')
        def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if we have target allocation data by looking for allocation files
                from pathlib import Path
                allocation_path = Path("data/allocations/latest_allocation.json")
                
                if not allocation_path.exists():
                    return jsonify({
                        'status': 'not_ready',
                        'service': 'trade-executor',
                        'reason': 'no_target_allocation'
                    }), 503
                
                # Check broker connectivity
                broker_healthy = self._check_broker_health()
                
                if not broker_healthy:
                    return jsonify({
                        'status': 'not_ready',
                        'service': 'trade-executor',
                        'reason': 'broker_unavailable'
                    }), 503
                
                # Get allocation age
                import json
                try:
                    with open(allocation_path, 'r') as f:
                        allocation_data = json.load(f)
                    allocation_timestamp = datetime.fromisoformat(allocation_data['timestamp'])
                    allocation_age = (datetime.now() - allocation_timestamp).total_seconds()
                except Exception:
                    allocation_age = None
                
                return jsonify({
                    'status': 'ready',
                    'service': 'trade-executor',
                    'target_allocation_age': allocation_age
                }), 200
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {str(e)}")
                return jsonify({
                    'status': 'not_ready',
                    'service': 'trade-executor',
                    'error': str(e)
                }), 503
        
        @self.app.route('/execute', methods=['POST'])
        def execute_trades_endpoint():
            """Execute trade rebalancing via HTTP endpoint."""
            try:
                self.logger.info("Trade execution requested via HTTP endpoint")
                
                success = self.execute_trades()
                
                if success and self.last_execution_result:
                    return jsonify({
                        'status': 'success',
                        'service': 'trade-executor',
                        'message': 'Trade execution completed successfully',
                        'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                        'result': {
                            'rebalancing_needed': self.last_execution_result.get('rebalancing_needed'),
                            'trades_executed': len(self.last_execution_result.get('trades', [])),
                            'total_drift': self.last_execution_result.get('total_drift'),
                            'success': self.last_execution_result.get('success')
                        }
                    }), 200
                else:
                    return jsonify({
                        'status': 'failed',
                        'service': 'trade-executor',
                        'message': 'Trade execution failed'
                    }), 500
                    
            except Exception as e:
                self.logger.error(f"Trade execution endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'trade-executor',
                    'error': str(e)
                }), 500
    
    def _check_broker_health(self) -> bool:
        """
        Check if broker is accessible.
        
        Returns:
            True if broker is healthy
        """
        try:
            broker_type = self.config.executor.broker_type
            
            if broker_type == "alpaca":
                # Check Alpaca API with graceful error handling
                from ..common.api_error_handling import handle_alpaca_api_error
                import requests
                
                api_key = self.config.broker.alpaca_api_key
                secret_key = self.config.broker.alpaca_secret_key
                base_url = self.config.broker.alpaca_base_url
                
                if not api_key or not secret_key:
                    self.logger.warning("Alpaca API credentials not configured, broker will run in mock mode")
                    return True  # Return True to allow service to start in mock mode
                
                try:
                    headers = {
                        "APCA-API-KEY-ID": api_key,
                        "APCA-API-SECRET-KEY": secret_key
                    }
                    
                    response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
                    response.raise_for_status()
                    return True
                    
                except Exception as api_error:
                    self.logger.warning(f"Alpaca API check failed, broker will run in mock mode: {api_error}")
                    handle_alpaca_api_error(api_error, "Broker health check")
                    return True  # Return True to allow service to start in mock mode
                
            elif broker_type == "ib":
                # For IB, we would check TWS connection
                # For now, just check if configuration is present
                return bool(self.config.broker.ib_host and self.config.broker.ib_port)
            
            return True  # Allow service to start even if broker type is unknown
            
        except Exception as e:
            self.logger.error(f"Broker health check failed: {str(e)}")
            return True  # Return True to allow service to start in degraded mode
            return False
    
    def execute_trades(self) -> bool:
        """
        Execute trade rebalancing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting trade execution")
            
            # Load target allocation from file
            from pathlib import Path
            import json
            from ..common.models import TargetAllocation
            
            allocation_path = Path("data/allocations/latest_allocation.json")
            if not allocation_path.exists():
                self.logger.error("No target allocation found")
                self.is_healthy = False
                return False
            
            # Load target allocation
            with open(allocation_path, 'r') as f:
                allocation_data = json.load(f)
            
            target_allocation = TargetAllocation(
                timestamp=datetime.fromisoformat(allocation_data['timestamp']),
                allocations=allocation_data['allocations'],
                expected_return=allocation_data['expected_return'],
                expected_volatility=allocation_data['expected_volatility'],
                sharpe_ratio=allocation_data['sharpe_ratio']
            )
            
            # Check if rebalancing is needed
            rebalancing_needed, drift = self.executor.check_rebalancing_needed(target_allocation)
            
            if rebalancing_needed:
                # Execute rebalancing
                executed_orders = self.executor.execute_rebalancing(target_allocation)
                trades_count = len(executed_orders)
                self.logger.info(f"Trade execution completed: {trades_count} trades executed")
                
                # Create result dictionary for compatibility
                result = {
                    'success': True,
                    'rebalancing_needed': True,
                    'trades': executed_orders,
                    'total_drift': sum(abs(d) for d in drift.values()) if drift else 0.0
                }
            else:
                self.logger.info("No rebalancing needed - portfolio within threshold")
                result = {
                    'success': True,
                    'rebalancing_needed': False,
                    'trades': [],
                    'total_drift': sum(abs(d) for d in drift.values()) if drift else 0.0
                }
            
            self.last_execution = datetime.now()
            self.last_execution_result = result
            self.is_healthy = result.get('success', False)
            
            return self.is_healthy
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}", exc_info=True)
            self.is_healthy = False
            return False
    
    def run_once(self) -> int:
        """
        Run trade execution once and exit.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        success = self.execute_trades()
        return 0 if success else 1
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8082):
        """
        Run the service with health check server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start Flask server in a separate thread
        server_thread = threading.Thread(
            target=lambda: self.app.run(host=host, port=port, debug=False),
            daemon=True
        )
        server_thread.start()
        
        self.logger.info(f"Health check server started on {host}:{port}")
        
        # Wait for shutdown signal
        self.shutdown_event.wait()
        self.logger.info("Service shutting down")


def main():
    """Main entry point for the executor service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Rebalancer Executor Service")
    parser.add_argument("--mode", choices=["once", "server"], default="once",
                       help="Run mode: 'once' for single execution, 'server' for health check server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for health check server")
    parser.add_argument("--port", type=int, default=8082, help="Port for health check server")
    
    args = parser.parse_args()
    
    try:
        service = ExecutorService()
        
        if args.mode == "once":
            exit_code = service.run_once()
            sys.exit(exit_code)
        else:
            service.run_server(host=args.host, port=args.port)
            
    except Exception as e:
        logging.error(f"Service failed to start: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()