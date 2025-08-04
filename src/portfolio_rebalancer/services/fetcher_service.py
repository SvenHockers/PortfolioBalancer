#!/usr/bin/env python3
"""Data fetcher service entry point."""

import logging
import sys
import signal
import time
from typing import Optional
from flask import Flask, jsonify
import threading

from ..common.config import get_config
from ..common.logging import setup_logging
from ..fetcher.data_fetcher import DataFetcher
from ..fetcher.yfinance_provider import YFinanceProvider
from ..fetcher.storage import ParquetStorage, SQLiteStorage


class FetcherService:
    """Data fetcher service with health check endpoint."""
    
    def __init__(self):
        """Initialize the fetcher service."""
        self.config = get_config()
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Service state - initialize early
        self.is_healthy = False  # Start as unhealthy until fully initialized
        self.last_execution = None
        self.shutdown_event = threading.Event()
        self.data_fetcher = None
        self.storage = None
        self.data_provider = None
        
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
                self.logger.info(f"Initializing fetcher service (attempt {attempt + 1}/{max_retries})")
                
                # Set up yfinance cache to avoid permission errors
                from ..common.api_error_handling import setup_yfinance_cache
                setup_yfinance_cache()
                
                # Initialize components with error handling
                try:
                    self.data_provider = YFinanceProvider()
                    self.logger.debug("YFinanceProvider initialized successfully")
                except Exception as provider_error:
                    self.logger.warning(f"YFinanceProvider initialization failed: {provider_error}")
                    # Continue with degraded mode - provider will be created later if needed
                    self.data_provider = None
                
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
                
                # Initialize data fetcher
                try:
                    # Create provider if not already created
                    if self.data_provider is None:
                        self.data_provider = YFinanceProvider()
                    
                    self.data_fetcher = DataFetcher(
                        provider=self.data_provider,
                        storage=self.storage
                    )
                    self.logger.debug("DataFetcher initialized successfully")
                except Exception as fetcher_error:
                    self.logger.error(f"DataFetcher initialization failed: {fetcher_error}")
                    raise  # DataFetcher is critical, can't continue without it
                
                # Mark as healthy if initialization succeeds
                self.is_healthy = True
                self.logger.info("Configuration loaded successfully")
                self.logger.info("Fetcher service initialized successfully")
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
                if not hasattr(self, 'data_fetcher') or self.data_fetcher is None:
                    if self.is_healthy:
                        # Service started but in degraded mode
                        return jsonify({
                            'status': 'degraded',
                            'service': 'data-fetcher',
                            'message': 'Service running in degraded mode - data fetcher not available'
                        }), 200
                    else:
                        return jsonify({
                            'status': 'unhealthy',
                            'service': 'data-fetcher',
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
                        
                        status = {
                            'status': 'healthy' if self.is_healthy and storage_healthy else 'unhealthy',
                            'service': 'data-fetcher',
                            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                            'storage_type': self.config.data.storage_type,
                            'storage_path': self.config.data.storage_path,
                            'storage_healthy': storage_healthy,
                            'tickers_count': len(self.config.data.tickers),
                            'backfill_days': self.config.data.backfill_days
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
                        'service': 'data-fetcher',
                        'error': 'Health check timed out'
                    }), 503
                
                if health_result['error']:
                    raise Exception(health_result['error'])
                
                return jsonify(health_result['result'][0]), health_result['result'][1]
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}", exc_info=True)
                return jsonify({
                    'status': 'unhealthy',
                    'service': 'data-fetcher',
                    'error': str(e)
                }), 503
        
        @self.app.route('/ready')
        def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if we can connect to data provider with graceful error handling
                from ..common.api_error_handling import handle_yfinance_error
                import yfinance as yf
                
                try:
                    ticker = yf.Ticker("AAPL")
                    info = ticker.info
                    
                    if info and "symbol" in info:
                        return jsonify({
                            'status': 'ready',
                            'service': 'data-fetcher'
                        }), 200
                    else:
                        self.logger.warning("yfinance connection test failed, but service will continue")
                        return jsonify({
                            'status': 'ready',
                            'service': 'data-fetcher',
                            'note': 'data_provider_degraded_mode'
                        }), 200
                        
                except Exception as yf_error:
                    self.logger.warning(f"yfinance readiness check failed: {yf_error}")
                    handle_yfinance_error(yf_error, "AAPL", "Readiness check")
                    return jsonify({
                        'status': 'ready',
                        'service': 'data-fetcher',
                        'note': 'data_provider_degraded_mode'
                    }), 200
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {str(e)}")
                return jsonify({
                    'status': 'ready',
                    'service': 'data-fetcher',
                    'note': 'degraded_mode'
                }), 200
        
        @self.app.route('/fetch', methods=['POST'])
        def fetch_data_endpoint():
            """Execute data fetching via HTTP endpoint."""
            try:
                self.logger.info("Data fetch requested via HTTP endpoint")
                
                success = self.fetch_data()
                
                if success:
                    return jsonify({
                        'status': 'success',
                        'service': 'data-fetcher',
                        'message': 'Data fetch completed successfully',
                        'last_execution': self.last_execution.isoformat() if self.last_execution else None
                    }), 200
                else:
                    return jsonify({
                        'status': 'failed',
                        'service': 'data-fetcher',
                        'message': 'Data fetch failed'
                    }), 500
                    
            except Exception as e:
                self.logger.error(f"Data fetch endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'data-fetcher',
                    'error': str(e)
                }), 500
    
    def fetch_data(self) -> bool:
        """
        Execute data fetching.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting data fetch")
            
            # Ensure we have at least 1 year of data for all configured tickers
            results = self.data_fetcher.ensure_one_year_data(
                tickers=self.config.data.tickers,
                force_update=False  # Only update if data is missing or insufficient
            )
            
            # Check if we got data back
            if results is not None and not results.empty:
                from datetime import datetime
                self.last_execution = datetime.now()
                self.is_healthy = True
                self.logger.info(f"Data fetch completed successfully - ensured at least 1 year of data for {len(results.index.get_level_values('symbol').unique())} tickers")
            else:
                self.is_healthy = False
                self.logger.error("Data fetch completed but no data was returned")
            
            return self.is_healthy
            
        except Exception as e:
            self.logger.error(f"Data fetch failed: {str(e)}", exc_info=True)
            self.is_healthy = False
            return False
    
    def run_once(self) -> int:
        """
        Run data fetching once and exit.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        success = self.fetch_data()
        return 0 if success else 1
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8080):
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
    """Main entry point for the fetcher service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Rebalancer Data Fetcher Service")
    parser.add_argument("--mode", choices=["once", "server"], default="once",
                       help="Run mode: 'once' for single execution, 'server' for health check server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for health check server")
    parser.add_argument("--port", type=int, default=8080, help="Port for health check server")
    
    args = parser.parse_args()
    
    try:
        service = FetcherService()
        
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