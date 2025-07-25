#!/usr/bin/env python3
"""Analytics service entry point."""

import logging
import sys
import signal
import threading
import os
import asyncio
from typing import Optional, Dict, Any
from flask import Flask, jsonify, request
from datetime import datetime, date

from ..common.config import get_config
from ..common.logging import setup_logging
from ..analytics.analytics_service import AnalyticsService
from ..analytics.storage import PostgreSQLAnalyticsStorage
from ..analytics.caching import AnalyticsCache
from ..analytics.async_processing import AsyncAnalyticsProcessor
from ..analytics.websocket_handler import WebSocketServer
from ..analytics.models import (
    BacktestConfig, MonteCarloConfig, AnalyticsError,
    OptimizationStrategy, RebalanceFrequency
)
from ..fetcher.storage import ParquetStorage, SQLiteStorage


class AnalyticsServiceRunner:
    """Analytics service with health check endpoint."""
    
    def __init__(self):
        """Initialize the analytics service."""
        self.config = get_config()
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize data storage based on configuration
        if self.config.data.storage_type == "parquet":
            self.data_storage = ParquetStorage(self.config.data.storage_path)
        else:
            self.data_storage = SQLiteStorage(self.config.data.storage_path)
        
        # Initialize analytics storage (PostgreSQL)
        analytics_db_url = self._get_analytics_db_url()
        self.analytics_storage = PostgreSQLAnalyticsStorage(analytics_db_url)
        
        # Initialize caching layer
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.cache = AnalyticsCache(redis_url)
        
        # Initialize async processor
        self.async_processor = AsyncAnalyticsProcessor()
        
        # Initialize analytics service
        self.analytics_service = AnalyticsService(
            data_storage=self.data_storage,
            analytics_storage=self.analytics_storage
        )
        
        # Initialize WebSocket server
        ws_host = os.getenv('WEBSOCKET_HOST', 'localhost')
        ws_port = int(os.getenv('WEBSOCKET_PORT', '8085'))
        self.websocket_server = WebSocketServer(self.async_processor, ws_host, ws_port)
        
        # Flask app for API endpoints
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Service state
        self.is_healthy = True
        self.last_execution = None
        self.shutdown_event = threading.Event()
        
    def _get_analytics_db_url(self) -> str:
        """Get PostgreSQL database URL from environment or config."""
        # Try environment variables first
        db_url = os.getenv('ANALYTICS_DB_URL')
        if db_url:
            return db_url
        
        # Build from individual components
        db_host = os.getenv('POSTGRES_HOST', 'localhost')
        db_port = os.getenv('POSTGRES_PORT', '5432')
        db_name = os.getenv('POSTGRES_DB', 'portfolio_rebalancer')
        db_user = os.getenv('POSTGRES_USER', 'portfolio_user')
        db_password = os.getenv('POSTGRES_PASSWORD', 'portfolio_pass')
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def setup_routes(self):
        """Setup Flask routes for API endpoints."""
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                health_status = self.analytics_service.health_check()
                
                status_code = 200 if health_status['status'] == 'healthy' else 503
                
                return jsonify({
                    'status': health_status['status'],
                    'service': 'analytics',
                    'last_execution': self.last_execution.isoformat() if self.last_execution else None,
                    'components': {
                        'analytics_storage': health_status.get('analytics_storage'),
                        'data_storage': health_status.get('data_storage')
                    },
                    'timestamp': health_status['timestamp']
                }), status_code
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                return jsonify({
                    'status': 'unhealthy',
                    'service': 'analytics',
                    'error': str(e)
                }), 503
        
        @self.app.route('/ready')
        def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if analytics storage is ready
                storage_ready = self.analytics_storage.health_check()
                
                # Check if we have some historical data
                tickers = self.config.data.tickers or ['AAPL']  # Fallback for testing
                try:
                    price_data = self.data_storage.get_prices(tickers[:1], 30)  # Test with one ticker
                    data_ready = price_data is not None and not price_data.empty
                except Exception:
                    data_ready = False
                
                if storage_ready and data_ready:
                    return jsonify({
                        'status': 'ready',
                        'service': 'analytics'
                    }), 200
                else:
                    return jsonify({
                        'status': 'not_ready',
                        'service': 'analytics',
                        'storage_ready': storage_ready,
                        'data_ready': data_ready
                    }), 503
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {str(e)}")
                return jsonify({
                    'status': 'not_ready',
                    'service': 'analytics',
                    'error': str(e)
                }), 503
        
        @self.app.route('/api/v1/analytics/backtest', methods=['POST'])
        def run_backtest():
            """Execute backtesting via HTTP endpoint."""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Request body must be JSON'}), 400
                
                # Parse backtest configuration
                config = BacktestConfig(
                    tickers=data.get('tickers', []),
                    start_date=date.fromisoformat(data['start_date']),
                    end_date=date.fromisoformat(data['end_date']),
                    strategy=OptimizationStrategy(data.get('strategy', 'sharpe')),
                    rebalance_frequency=RebalanceFrequency(data.get('rebalance_frequency', 'monthly')),
                    transaction_cost=data.get('transaction_cost', 0.001),
                    initial_capital=data.get('initial_capital', 100000.0)
                )
                
                # Check if async processing is requested
                async_mode = data.get('async', False)
                
                if async_mode:
                    # Submit for async processing
                    task_id = self.async_processor.submit_backtest(config)
                    return jsonify({
                        'status': 'submitted',
                        'service': 'analytics',
                        'message': 'Backtest submitted for async processing',
                        'task_id': task_id,
                        'websocket_url': f"ws://{self.websocket_server.host}:{self.websocket_server.port}/ws/backtest/{task_id}"
                    }), 202
                
                # Check cache first
                cached_result = self.cache.get_cached_backtest(config)
                if cached_result:
                    self.logger.info(f"Returning cached backtest result for {config.tickers}")
                    return jsonify({
                        'status': 'success',
                        'service': 'analytics',
                        'message': 'Backtest completed successfully (cached)',
                        'cached': True,
                        'result': {
                            'total_return': cached_result.total_return,
                            'annualized_return': cached_result.annualized_return,
                            'volatility': cached_result.volatility,
                            'sharpe_ratio': cached_result.sharpe_ratio,
                            'max_drawdown': cached_result.max_drawdown,
                            'calmar_ratio': cached_result.calmar_ratio,
                            'final_value': cached_result.final_value,
                            'num_rebalances': cached_result.num_rebalances,
                            'transaction_costs': cached_result.transaction_costs
                        }
                    }), 200
                
                self.logger.info(f"Backtest requested via HTTP endpoint: {config.tickers}")
                
                # Run backtest synchronously
                result = self.analytics_service.run_backtest(config)
                self.last_execution = datetime.now()
                
                # Cache the result
                self.cache.cache_backtest_result(config, result)
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'message': 'Backtest completed successfully',
                    'cached': False,
                    'result': {
                        'total_return': result.total_return,
                        'annualized_return': result.annualized_return,
                        'volatility': result.volatility,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'calmar_ratio': result.calmar_ratio,
                        'final_value': result.final_value,
                        'num_rebalances': result.num_rebalances,
                        'transaction_costs': result.transaction_costs
                    }
                }), 200
                
            except AnalyticsError as e:
                self.logger.error(f"Backtest failed: {str(e)}")
                return jsonify({
                    'status': 'failed',
                    'service': 'analytics',
                    'error': str(e)
                }), 400
            except Exception as e:
                self.logger.error(f"Backtest endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/monte-carlo', methods=['POST'])
        def run_monte_carlo():
            """Execute Monte Carlo simulation via HTTP endpoint."""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Request body must be JSON'}), 400
                
                # Parse Monte Carlo configuration
                config = MonteCarloConfig(
                    portfolio_tickers=data.get('portfolio_tickers', []),
                    portfolio_weights=data.get('portfolio_weights', []),
                    time_horizon_years=data.get('time_horizon_years', 10),
                    num_simulations=data.get('num_simulations', 10000),
                    confidence_levels=data.get('confidence_levels', [0.05, 0.25, 0.5, 0.75, 0.95]),
                    initial_value=data.get('initial_value', 100000.0)
                )
                
                # Check if async processing is requested
                async_mode = data.get('async', False)
                
                if async_mode:
                    # Submit for async processing
                    task_id = self.async_processor.submit_monte_carlo(config)
                    return jsonify({
                        'status': 'submitted',
                        'service': 'analytics',
                        'message': 'Monte Carlo simulation submitted for async processing',
                        'task_id': task_id,
                        'websocket_url': f"ws://{self.websocket_server.host}:{self.websocket_server.port}/ws/monte-carlo/{task_id}"
                    }), 202
                
                # Check cache first
                cached_result = self.cache.get_cached_monte_carlo(config)
                if cached_result:
                    self.logger.info(f"Returning cached Monte Carlo result for {config.portfolio_tickers}")
                    return jsonify({
                        'status': 'success',
                        'service': 'analytics',
                        'message': 'Monte Carlo simulation completed successfully (cached)',
                        'cached': True,
                        'result': {
                            'expected_value': cached_result.expected_value,
                            'probability_of_loss': cached_result.probability_of_loss,
                            'value_at_risk_95': cached_result.value_at_risk_95,
                            'conditional_var_95': cached_result.conditional_var_95
                        }
                    }), 200
                
                self.logger.info(f"Monte Carlo simulation requested: {config.portfolio_tickers}")
                
                # Run simulation synchronously
                result = self.analytics_service.run_monte_carlo(config)
                self.last_execution = datetime.now()
                
                # Cache the result
                self.cache.cache_monte_carlo_result(config, result)
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'message': 'Monte Carlo simulation completed successfully',
                    'cached': False,
                    'result': {
                        'expected_value': result.expected_value,
                        'probability_of_loss': result.probability_of_loss,
                        'value_at_risk_95': result.value_at_risk_95,
                        'conditional_var_95': result.conditional_var_95
                    }
                }), 200
                
            except AnalyticsError as e:
                self.logger.error(f"Monte Carlo simulation failed: {str(e)}")
                return jsonify({
                    'status': 'failed',
                    'service': 'analytics',
                    'error': str(e)
                }), 400
            except Exception as e:
                self.logger.error(f"Monte Carlo endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/risk/<portfolio_id>', methods=['POST'])
        def analyze_risk(portfolio_id: str):
            """Execute risk analysis via HTTP endpoint."""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Request body must be JSON'}), 400
                
                tickers = data.get('tickers', [])
                weights = data.get('weights', [])
                
                if not tickers or not weights:
                    return jsonify({'error': 'tickers and weights are required'}), 400
                
                if len(tickers) != len(weights):
                    return jsonify({'error': 'tickers and weights must have same length'}), 400
                
                self.logger.info(f"Risk analysis requested for portfolio {portfolio_id}")
                
                # Run risk analysis
                result = self.analytics_service.analyze_risk(portfolio_id, tickers, weights)
                self.last_execution = datetime.now()
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'message': 'Risk analysis completed successfully',
                    'result': {
                        'portfolio_beta': result.portfolio_beta,
                        'tracking_error': result.tracking_error,
                        'information_ratio': result.information_ratio,
                        'var_95': result.var_95,
                        'cvar_95': result.cvar_95,
                        'max_drawdown': result.max_drawdown,
                        'concentration_risk': result.concentration_risk
                    }
                }), 200
                
            except AnalyticsError as e:
                self.logger.error(f"Risk analysis failed: {str(e)}")
                return jsonify({
                    'status': 'failed',
                    'service': 'analytics',
                    'error': str(e)
                }), 400
            except Exception as e:
                self.logger.error(f"Risk analysis endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/performance/<portfolio_id>', methods=['POST'])
        def track_performance(portfolio_id: str):
            """Execute performance tracking via HTTP endpoint."""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Request body must be JSON'}), 400
                
                tickers = data.get('tickers', [])
                weights = data.get('weights', [])
                
                if not tickers or not weights:
                    return jsonify({'error': 'tickers and weights are required'}), 400
                
                if len(tickers) != len(weights):
                    return jsonify({'error': 'tickers and weights must have same length'}), 400
                
                self.logger.info(f"Performance tracking requested for portfolio {portfolio_id}")
                
                # Track performance
                result = self.analytics_service.track_performance(portfolio_id, tickers, weights)
                self.last_execution = datetime.now()
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'message': 'Performance tracking completed successfully',
                    'result': {
                        'total_return': result.total_return,
                        'annualized_return': result.annualized_return,
                        'volatility': result.volatility,
                        'sharpe_ratio': result.sharpe_ratio,
                        'sortino_ratio': result.sortino_ratio,
                        'alpha': result.alpha,
                        'beta': result.beta,
                        'r_squared': result.r_squared,
                        'tracking_error': result.tracking_error,
                        'information_ratio': result.information_ratio
                    }
                }), 200
                
            except AnalyticsError as e:
                self.logger.error(f"Performance tracking failed: {str(e)}")
                return jsonify({
                    'status': 'failed',
                    'service': 'analytics',
                    'error': str(e)
                }), 400
            except Exception as e:
                self.logger.error(f"Performance tracking endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/dividends/<portfolio_id>', methods=['POST'])
        def analyze_dividends(portfolio_id: str):
            """Execute dividend analysis via HTTP endpoint."""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Request body must be JSON'}), 400
                
                tickers = data.get('tickers', [])
                weights = data.get('weights', [])
                
                if not tickers or not weights:
                    return jsonify({'error': 'tickers and weights are required'}), 400
                
                if len(tickers) != len(weights):
                    return jsonify({'error': 'tickers and weights must have same length'}), 400
                
                self.logger.info(f"Dividend analysis requested for portfolio {portfolio_id}")
                
                # Analyze dividends
                result = self.analytics_service.analyze_dividends(portfolio_id, tickers, weights)
                self.last_execution = datetime.now()
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'message': 'Dividend analysis completed successfully',
                    'result': {
                        'current_yield': result.current_yield,
                        'projected_annual_income': result.projected_annual_income,
                        'dividend_growth_rate': result.dividend_growth_rate,
                        'payout_ratio': result.payout_ratio,
                        'dividend_coverage': result.dividend_coverage,
                        'income_sustainability_score': result.income_sustainability_score
                    }
                }), 200
                
            except AnalyticsError as e:
                self.logger.error(f"Dividend analysis failed: {str(e)}")
                return jsonify({
                    'status': 'failed',
                    'service': 'analytics',
                    'error': str(e)
                }), 400
            except Exception as e:
                self.logger.error(f"Dividend analysis endpoint failed: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/backtest/<backtest_id>', methods=['GET'])
        def get_backtest_result(backtest_id: str):
            """Retrieve stored backtest result."""
            try:
                result = self.analytics_service.get_backtest_result(backtest_id)
                
                if not result:
                    return jsonify({
                        'status': 'not_found',
                        'message': f'Backtest result {backtest_id} not found'
                    }), 404
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'result': {
                        'total_return': result.total_return,
                        'annualized_return': result.annualized_return,
                        'volatility': result.volatility,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'calmar_ratio': result.calmar_ratio,
                        'final_value': result.final_value,
                        'num_rebalances': result.num_rebalances,
                        'transaction_costs': result.transaction_costs,
                        'config': {
                            'tickers': result.config.tickers,
                            'start_date': result.config.start_date.isoformat(),
                            'end_date': result.config.end_date.isoformat(),
                            'strategy': result.config.strategy,
                            'rebalance_frequency': result.config.rebalance_frequency,
                            'transaction_cost': result.config.transaction_cost,
                            'initial_capital': result.config.initial_capital
                        }
                    }
                }), 200
                
            except Exception as e:
                self.logger.error(f"Failed to retrieve backtest result: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/tasks/<task_id>/status', methods=['GET'])
        def get_task_status(task_id: str):
            """Get status of async task."""
            try:
                status = self.async_processor.get_task_status(task_id)
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'task_status': status
                }), 200
                
            except Exception as e:
                self.logger.error(f"Failed to get task status: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/tasks/<task_id>/result', methods=['GET'])
        def get_task_result(task_id: str):
            """Get result of completed async task."""
            try:
                result = self.async_processor.get_task_result(task_id)
                
                if result is None:
                    return jsonify({
                        'status': 'not_ready',
                        'message': 'Task result not ready or task failed'
                    }), 404
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'result': result
                }), 200
                
            except Exception as e:
                self.logger.error(f"Failed to get task result: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/tasks/<task_id>', methods=['DELETE'])
        def cancel_task(task_id: str):
            """Cancel a running async task."""
            try:
                success = self.async_processor.cancel_task(task_id)
                
                return jsonify({
                    'status': 'success' if success else 'failed',
                    'service': 'analytics',
                    'message': f'Task {task_id} {"cancelled" if success else "could not be cancelled"}',
                    'cancelled': success
                }), 200 if success else 400
                
            except Exception as e:
                self.logger.error(f"Failed to cancel task: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/tasks', methods=['GET'])
        def get_active_tasks():
            """Get list of active async tasks."""
            try:
                tasks = self.async_processor.get_active_tasks()
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'active_tasks': tasks,
                    'count': len(tasks)
                }), 200
                
            except Exception as e:
                self.logger.error(f"Failed to get active tasks: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/cache/stats', methods=['GET'])
        def get_cache_stats():
            """Get cache statistics."""
            try:
                stats = self.cache.get_cache_stats()
                
                return jsonify({
                    'status': 'success',
                    'service': 'analytics',
                    'cache_stats': stats
                }), 200
                
            except Exception as e:
                self.logger.error(f"Failed to get cache stats: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/v1/analytics/cache/clear', methods=['POST'])
        def clear_cache():
            """Clear analytics cache."""
            try:
                data = request.get_json() or {}
                pattern = data.get('pattern', '*')
                
                if pattern == '*':
                    # Clear all cache
                    success = self.cache.clear_all_cache()
                    message = 'All cache cleared' if success else 'Failed to clear cache'
                    deleted_count = 'all' if success else 0
                else:
                    # Clear specific pattern
                    deleted_count = self.cache.invalidate_cache(pattern)
                    success = deleted_count >= 0
                    message = f'Cleared {deleted_count} cache entries'
                
                return jsonify({
                    'status': 'success' if success else 'failed',
                    'service': 'analytics',
                    'message': message,
                    'deleted_count': deleted_count
                }), 200 if success else 500
                
            except Exception as e:
                self.logger.error(f"Failed to clear cache: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'service': 'analytics',
                    'error': str(e)
                }), 500
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8084):
        """
        Run the service with API server.
        
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
        
        # Start WebSocket server in a separate thread
        def run_websocket_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.websocket_server.start_server())
                loop.run_forever()
            except Exception as e:
                self.logger.error(f"WebSocket server error: {e}")
            finally:
                loop.close()
        
        websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
        websocket_thread.start()
        
        # Start Flask server in a separate thread
        server_thread = threading.Thread(
            target=lambda: self.app.run(host=host, port=port, debug=False),
            daemon=True
        )
        server_thread.start()
        
        self.logger.info(f"Analytics service started on {host}:{port}")
        self.logger.info(f"WebSocket server started on {self.websocket_server.host}:{self.websocket_server.port}")
        
        # Wait for shutdown signal
        self.shutdown_event.wait()
        
        # Shutdown WebSocket server
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.websocket_server.stop_server())
            loop.close()
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")
        
        self.logger.info("Analytics service shutting down")


def main():
    """Main entry point for the analytics service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Rebalancer Analytics Service")
    parser.add_argument("--mode", choices=["server"], default="server",
                       help="Run mode: 'server' for API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host for API server")
    parser.add_argument("--port", type=int, default=8084, help="Port for API server")
    
    args = parser.parse_args()
    
    try:
        service = AnalyticsServiceRunner()
        service.run_server(host=args.host, port=args.port)
            
    except Exception as e:
        logging.error(f"Analytics service failed to start: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()