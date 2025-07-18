#!/usr/bin/env python3
"""Portfolio optimizer service entry point."""

import logging
import sys
import signal
import threading
from typing import Optional
from flask import Flask, jsonify
from datetime import datetime

from ..common.config import get_config
from ..common.logging import setup_logging
from ..optimizer.portfolio_optimizer import PortfolioOptimizer
from ..fetcher.storage import ParquetStorage, SQLiteStorage


class OptimizerService:
    """Portfolio optimizer service with health check endpoint."""
    
    def __init__(self):
        """Initialize the optimizer service."""
        self.config = get_config()
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage based on configuration
        if self.config.data.storage_type == "parquet":
            self.storage = ParquetStorage(self.config.data.storage_path)
        else:
            self.storage = SQLiteStorage(self.config.data.storage_path)
        
        # Initialize optimizer
        self.optimizer = PortfolioOptimizer(data_storage=self.storage)
        
        # Flask app for health checks
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Service state
        self.is_healthy = True
        self.last_execution = None
        self.last_optimization_result = None
        self.shutdown_event = threading.Event()
        
    def setup_routes(self):
        """Setup Flask routes for health checks."""
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                status = {
                    'status': 'healthy' if self.is_healthy else 'unhealthy',
                    'service': 'portfolio-optimizer',
                    'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                    'user_age': self.config.optimization.user_age,
                    'lookback_days': self.config.optimization.lookback_days
                }
                
                if self.last_optimization_result:
                    status['last_result'] = {
                        'expected_return': self.last_optimization_result.get('expected_return'),
                        'expected_volatility': self.last_optimization_result.get('expected_volatility'),
                        'sharpe_ratio': self.last_optimization_result.get('sharpe_ratio'),
                        'allocation_count': len(self.last_optimization_result.get('allocations', {}))
                    }
                
                return jsonify(status), 200 if self.is_healthy else 503
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                return jsonify({
                    'status': 'unhealthy',
                    'service': 'portfolio-optimizer',
                    'error': str(e)
                }), 503
        
        @self.app.route('/ready')
        def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if we have sufficient historical data
                tickers = self.config.tickers
                lookback_days = self.config.optimizer.lookback_days
                
                # Try to get price data
                price_data = self.storage.get_prices(tickers, lookback_days)
                
                if price_data is not None and not price_data.empty:
                    data_days = len(price_data.index.get_level_values('date').unique())
                    min_required_days = max(30, lookback_days // 2)  # At least 30 days or half lookback
                    
                    if data_days >= min_required_days:
                        return jsonify({
                            'status': 'ready',
                            'service': 'portfolio-optimizer',
                            'data_days': data_days
                        }), 200
                    else:
                        return jsonify({
                            'status': 'not_ready',
                            'service': 'portfolio-optimizer',
                            'reason': 'insufficient_data',
                            'data_days': data_days,
                            'required_days': min_required_days
                        }), 503
                else:
                    return jsonify({
                        'status': 'not_ready',
                        'service': 'portfolio-optimizer',
                        'reason': 'no_data_available'
                    }), 503
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {str(e)}")
                return jsonify({
                    'status': 'not_ready',
                    'service': 'portfolio-optimizer',
                    'error': str(e)
                }), 503
        
        @self.app.route('/optimize', methods=['POST'])
        def optimize_portfolio_endpoint():
            """Execute portfolio optimization via HTTP endpoint."""
            try:
                self.logger.info("Portfolio optimization requested via HTTP endpoint")
                
                success = self.optimize_portfolio()
                
                if success and self.last_optimization_result:
                    return jsonify({
                        'status': 'success',
                        'service': 'portfolio-optimizer',
                        'message': 'Portfolio optimization completed successfully',
                        'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                        'result': {
                            'expected_return': self.last_optimization_result.get('expected_return'),
                            'expected_volatility': self.last_optimization_result.get('expected_volatility'),
                            'sharpe_ratio': self.last_optimization_result.get('sharpe_ratio'),
                            'allocation_count': len(self.last_optimization_result.get('allocations', {}))
                        }
                    }), 200
                else:
                    return jsonify({
                        'status': 'failed',
                        'service': 'portfolio-optimizer',
                        'message': 'Portfolio optimization failed'
                    }), 500
                    
            except Exception as e:
                self.logger.error(f"Portfolio optimization endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'portfolio-optimizer',
                    'error': str(e)
                }), 500
    
    def optimize_portfolio(self) -> bool:
        """
        Execute portfolio optimization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting portfolio optimization")
            
            # Run optimization
            result = self.optimizer.optimize_portfolio(
                tickers=self.config.tickers,
                user_age=self.config.optimizer.user_age,
                lookback_days=self.config.optimizer.lookback_days
            )
            
            self.last_execution = datetime.now()
            self.last_optimization_result = result
            self.is_healthy = result is not None
            
            if self.is_healthy:
                self.logger.info(f"Portfolio optimization completed successfully")
                self.logger.info(f"Expected return: {result.get('expected_return', 'N/A'):.4f}")
                self.logger.info(f"Expected volatility: {result.get('expected_volatility', 'N/A'):.4f}")
                self.logger.info(f"Sharpe ratio: {result.get('sharpe_ratio', 'N/A'):.4f}")
            else:
                self.logger.error("Portfolio optimization failed")
            
            return self.is_healthy
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {str(e)}", exc_info=True)
            self.is_healthy = False
            return False
    
    def run_once(self) -> int:
        """
        Run portfolio optimization once and exit.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        success = self.optimize_portfolio()
        return 0 if success else 1
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8081):
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
    """Main entry point for the optimizer service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Rebalancer Optimizer Service")
    parser.add_argument("--mode", choices=["once", "server"], default="once",
                       help="Run mode: 'once' for single execution, 'server' for health check server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for health check server")
    parser.add_argument("--port", type=int, default=8081, help="Port for health check server")
    
    args = parser.parse_args()
    
    try:
        service = OptimizerService()
        
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