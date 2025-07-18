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
        
        # Initialize storage based on configuration
        if self.config.data.storage_type == "parquet":
            self.storage = ParquetStorage(self.config.data.storage_path)
        else:
            self.storage = SQLiteStorage(self.config.data.storage_path)
        
        # Initialize executor
        self.executor = TradeExecutor()
        
        # Flask app for health checks
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Service state
        self.is_healthy = True
        self.last_execution = None
        self.last_execution_result = None
        self.shutdown_event = threading.Event()
        
    def setup_routes(self):
        """Setup Flask routes for health checks."""
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                status = {
                    'status': 'healthy' if self.is_healthy else 'unhealthy',
                    'service': 'trade-executor',
                    'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                    'broker_type': self.config.executor.broker_type,
                    'rebalance_threshold': self.config.executor.rebalance_threshold
                }
                
                if self.last_execution_result:
                    status['last_result'] = {
                        'rebalancing_needed': self.last_execution_result.get('rebalancing_needed'),
                        'trades_executed': len(self.last_execution_result.get('trades', [])),
                        'total_drift': self.last_execution_result.get('total_drift'),
                        'success': self.last_execution_result.get('success')
                    }
                
                return jsonify(status), 200 if self.is_healthy else 503
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                return jsonify({
                    'status': 'unhealthy',
                    'service': 'trade-executor',
                    'error': str(e)
                }), 503
        
        @self.app.route('/ready')
        def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if we have target allocation data
                target_allocation = self.storage.get_latest_target_allocation()
                
                if target_allocation is None:
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
                
                return jsonify({
                    'status': 'ready',
                    'service': 'trade-executor',
                    'target_allocation_age': (datetime.now() - target_allocation['timestamp']).total_seconds()
                }), 200
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {str(e)}")
                return jsonify({
                    'status': 'not_ready',
                    'service': 'trade-executor',
                    'error': str(e)
                }), 503
    
    def _check_broker_health(self) -> bool:
        """
        Check if broker is accessible.
        
        Returns:
            True if broker is healthy
        """
        try:
            broker_type = self.config.executor.broker_type
            
            if broker_type == "alpaca":
                # Check Alpaca API
                import requests
                api_key = self.config.broker.alpaca_api_key
                secret_key = self.config.broker.alpaca_secret_key
                base_url = self.config.broker.alpaca_base_url
                
                if not api_key or not secret_key:
                    return False
                
                headers = {
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": secret_key
                }
                
                response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
                return response.status_code == 200
                
            elif broker_type == "ib":
                # For IB, we would check TWS connection
                # For now, just check if configuration is present
                return bool(self.config.broker.ib_host and self.config.broker.ib_port)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Broker health check failed: {str(e)}")
            return False
    
    def execute_trades(self) -> bool:
        """
        Execute trade rebalancing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting trade execution")
            
            # Execute rebalancing
            result = self.executor.execute_rebalancing(
                rebalance_threshold=self.config.executor.rebalance_threshold
            )
            
            self.last_execution = datetime.now()
            self.last_execution_result = result
            self.is_healthy = result.get('success', False)
            
            if result.get('rebalancing_needed', False):
                trades_count = len(result.get('trades', []))
                self.logger.info(f"Trade execution completed: {trades_count} trades executed")
            else:
                self.logger.info("No rebalancing needed - portfolio within threshold")
            
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