#!/usr/bin/env python3
"""Scheduler service entry point."""

import logging
import os
import sys
import signal
import threading
from typing import Optional
from flask import Flask, jsonify
from datetime import datetime

from ..common.config import get_config
from ..common.logging import setup_logging
from ..common.scheduler import Scheduler
from ..common.container_orchestration import ContainerOrchestrator


class SchedulerService:
    """Scheduler service with health check endpoint."""
    
    def __init__(self):
        """Initialize the scheduler service."""
        self.config = get_config()
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize orchestrator and scheduler
        self.orchestrator = ContainerOrchestrator()
        self.scheduler = Scheduler(orchestrator=self.orchestrator)
        
        # Flask app for health checks
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Service state
        self.is_healthy = True
        self.shutdown_event = threading.Event()
        
    def setup_routes(self):
        """Setup Flask routes for health checks."""
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                next_run = self.scheduler.get_next_run_time()
                service_health = self.scheduler.get_service_health()
                
                status = {
                    'status': 'healthy' if self.is_healthy else 'unhealthy',
                    'service': 'scheduler',
                    'next_run': next_run.isoformat() if next_run else None,
                    'scheduled_jobs': len(self.scheduler.jobs),
                    'services_healthy': all(s.is_healthy for s in service_health.values()),
                    'service_status': {
                        name: {
                            'healthy': status.is_healthy,
                            'message': status.message,
                            'timestamp': status.timestamp.isoformat()
                        }
                        for name, status in service_health.items()
                    }
                }
                
                return jsonify(status), 200 if self.is_healthy else 503
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
                return jsonify({
                    'status': 'unhealthy',
                    'service': 'scheduler',
                    'error': str(e)
                }), 503
        
        @self.app.route('/ready')
        def readiness_check():
            """Readiness check endpoint."""
            try:
                # Check if scheduler has jobs scheduled
                if not self.scheduler.jobs:
                    return jsonify({
                        'status': 'not_ready',
                        'service': 'scheduler',
                        'reason': 'no_jobs_scheduled'
                    }), 503
                
                return jsonify({
                    'status': 'ready',
                    'service': 'scheduler',
                    'scheduled_jobs': len(self.scheduler.jobs)
                }), 200
                    
            except Exception as e:
                self.logger.error(f"Readiness check failed: {str(e)}")
                return jsonify({
                    'status': 'not_ready',
                    'service': 'scheduler',
                    'error': str(e)
                }), 503
        
        @self.app.route('/execute')
        def execute_now():
            """Execute pipeline immediately."""
            try:
                self.logger.info("Manual pipeline execution requested")
                result = self.scheduler.execute_now()
                
                return jsonify({
                    'status': 'executed',
                    'result': result
                }), 200
                
            except Exception as e:
                self.logger.error(f"Manual execution failed: {str(e)}")
                return jsonify({
                    'status': 'failed',
                    'error': str(e)
                }), 500
    
    def setup_schedule(self):
        """Setup the scheduled jobs."""
        try:
            # Check if interval scheduling is requested
            interval_minutes = self.config.scheduler.schedule_interval_minutes
            
            if interval_minutes:
                # Use interval scheduling for testing
                interval_minutes = int(interval_minutes)
                self.scheduler.schedule_interval(interval_minutes)
                self.logger.info(f"Scheduled pipeline execution every {interval_minutes} minutes")
            else:
                # Use daily scheduling (default)
                execution_time = self.config.scheduler.execution_time
                self.scheduler.schedule_daily(execution_time)
                self.logger.info(f"Scheduled daily execution at {execution_time}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup schedule: {str(e)}")
            self.is_healthy = False
            raise
    
    def run_scheduler(self, host: str = "0.0.0.0", port: int = 8083):
        """
        Run the scheduler service.
        
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
        
        try:
            # Setup schedule
            self.setup_schedule()
            
            # Start scheduler
            self.scheduler.start()
            
            # Start Flask server in a separate thread
            server_thread = threading.Thread(
                target=lambda: self.app.run(host=host, port=port, debug=False),
                daemon=True
            )
            server_thread.start()
            
            self.logger.info(f"Scheduler service started on {host}:{port}")
            
            # Wait for shutdown signal
            self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Scheduler service failed: {str(e)}", exc_info=True)
            self.is_healthy = False
            raise
        finally:
            # Stop scheduler
            self.scheduler.stop()
            self.logger.info("Scheduler service shutting down")
    
    def run_once(self) -> int:
        """
        Execute pipeline once and exit.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            result = self.orchestrator.execute_pipeline()
            
            # Check if any service failed
            has_failures = any(
                service_result.status.value == "failed"
                for service_result in result.values()
            )
            
            return 1 if has_failures else 0
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            return 1


def main():
    """Main entry point for the scheduler service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Rebalancer Scheduler Service")
    parser.add_argument("--mode", choices=["once", "scheduler"], default="scheduler",
                       help="Run mode: 'once' for single execution, 'scheduler' for continuous scheduling")
    parser.add_argument("--host", default="0.0.0.0", help="Host for health check server")
    parser.add_argument("--port", type=int, default=8083, help="Port for health check server")
    
    args = parser.parse_args()
    
    try:
        service = SchedulerService()
        
        if args.mode == "once":
            exit_code = service.run_once()
            sys.exit(exit_code)
        else:
            service.run_scheduler(host=args.host, port=args.port)
            
    except Exception as e:
        logging.error(f"Service failed to start: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()