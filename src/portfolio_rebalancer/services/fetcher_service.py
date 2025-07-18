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
        
        # Initialize components
        self.data_provider = YFinanceProvider()
        
        # Initialize storage based on configuration
        if self.config.data.storage_type == "parquet":
            self.storage = ParquetStorage(self.config.data.storage_path)
        else:
            self.storage = SQLiteStorage(self.config.data.storage_path)
        
        self.data_fetcher = DataFetcher(
            provider=self.data_provider,
            storage=self.storage
        )
        
        # Flask app for health checks
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Service state
        self.is_healthy = True
        self.last_execution = None
        self.shutdown_event = threading.Event()
        
    def setup_routes(self):
        """Setup Flask routes for health checks."""
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                # Basic health check
                status = {
                    'status': 'healthy' if self.is_healthy else 'unhealthy',
                    'service': 'data-fetcher',
                    'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                    'storage_type': self.config.data.storage_type,
                    'tickers_count': len(self.config.data.tickers)
                }
                
                return jsonify(status), 200 if self.is_healthy else 503
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                return jsonify({
                    'status': 'unhealthy',
                    'service': 'data-fetcher',
                    'error': str(e)
                }), 503
        
        @self.app.route('/ready')
        def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if we can connect to data provider
                import yfinance as yf
                ticker = yf.Ticker("AAPL")
                info = ticker.info
                
                if info and "symbol" in info:
                    return jsonify({
                        'status': 'ready',
                        'service': 'data-fetcher'
                    }), 200
                else:
                    return jsonify({
                        'status': 'not_ready',
                        'service': 'data-fetcher',
                        'reason': 'data_provider_unavailable'
                    }), 503
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {str(e)}")
                return jsonify({
                    'status': 'not_ready',
                    'service': 'data-fetcher',
                    'error': str(e)
                }), 503
        
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
            
            # Fetch data for all configured tickers
            results = self.data_fetcher.backfill_missing_data(
                tickers=self.config.data.tickers,
                days=self.config.data.backfill_days
            )
            
            # Check if we got data back
            if results is not None and not results.empty:
                from datetime import datetime
                self.last_execution = datetime.now()
                self.is_healthy = True
                self.logger.info(f"Data fetch completed successfully - fetched {len(results)} records")
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