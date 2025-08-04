#!/usr/bin/env python3
"""
Service startup wrapper with enhanced error recovery and restart loop prevention.
This script wraps service startup to provide better error handling and prevent rapid restart loops.
"""

import os
import sys
import time
import signal
import logging
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List


class ServiceStartupWrapper:
    """Wrapper for service startup with enhanced error recovery."""
    
    def __init__(self, service_name: str, command: List[str]):
        """
        Initialize the startup wrapper.
        
        Args:
            service_name: Name of the service for logging
            command: Command to execute
        """
        self.service_name = service_name
        self.command = command
        self.process: Optional[subprocess.Popen] = None
        self.shutdown_requested = False
        
        # Configuration from environment
        self.max_startup_attempts = int(os.getenv('SERVICE_MAX_STARTUP_ATTEMPTS', '5'))
        self.startup_timeout = int(os.getenv('SERVICE_STARTUP_TIMEOUT', '120'))
        self.restart_delay = int(os.getenv('SERVICE_RESTART_DELAY', '30'))
        self.health_check_url = os.getenv('SERVICE_HEALTH_CHECK_URL', '')
        self.health_check_timeout = int(os.getenv('SERVICE_HEALTH_CHECK_TIMEOUT', '10'))
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def setup_logging(self):
        """Setup logging for the wrapper."""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'/app/logs/{self.service_name}-wrapper.log')
            ]
        )
        
        self.logger = logging.getLogger(f'{self.service_name}-wrapper')
    
    def _signal_handler(self, signum: int, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        if self.process and self.process.poll() is None:
            self.logger.info("Terminating service process...")
            try:
                self.process.terminate()
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                    self.logger.info("Service terminated gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning("Service did not terminate gracefully, forcing kill...")
                    self.process.kill()
                    self.process.wait()
                    self.logger.info("Service killed")
            except Exception as e:
                self.logger.error(f"Error during service shutdown: {e}")
    
    def check_health(self) -> bool:
        """
        Check if the service is healthy using HTTP health check.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.health_check_url:
            return True  # Skip health check if URL not configured
        
        try:
            import requests
            response = requests.get(
                self.health_check_url,
                timeout=self.health_check_timeout
            )
            
            # Accept both healthy and degraded status as successful startup
            if response.status_code == 200:
                response_data = response.json()
                status = response_data.get('status', '').lower()
                if status in ['healthy', 'degraded']:
                    self.logger.info(f"Service health check passed: {status}")
                    return True
                else:
                    self.logger.warning(f"Service health check returned unexpected status: {status}")
                    return False
            else:
                self.logger.warning(f"Service health check failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Service health check failed: {e}")
            return False
    
    def wait_for_startup(self) -> bool:
        """
        Wait for service to start up successfully.
        
        Returns:
            True if startup successful, False otherwise
        """
        if not self.health_check_url:
            # If no health check URL, just wait a bit and assume success
            time.sleep(10)
            return True
        
        start_time = time.time()
        
        while time.time() - start_time < self.startup_timeout:
            if self.shutdown_requested:
                return False
            
            if self.process and self.process.poll() is not None:
                # Process has exited
                return_code = self.process.returncode
                if return_code == 0:
                    self.logger.info("Service process completed successfully")
                    return True
                else:
                    self.logger.error(f"Service process exited with code {return_code}")
                    return False
            
            if self.check_health():
                self.logger.info("Service startup completed successfully")
                return True
            
            time.sleep(5)  # Check every 5 seconds
        
        self.logger.error(f"Service startup timed out after {self.startup_timeout} seconds")
        return False
    
    def start_service(self) -> bool:
        """
        Start the service process.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            self.logger.info(f"Starting service: {' '.join(self.command)}")
            
            # Set up environment for the service
            env = os.environ.copy()
            
            # Start the process
            self.process = subprocess.Popen(
                self.command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            self.logger.info(f"Service process started with PID {self.process.pid}")
            
            # Start a thread to log output
            import threading
            
            def log_output():
                try:
                    for line in iter(self.process.stdout.readline, ''):
                        if line.strip():
                            self.logger.info(f"[{self.service_name}] {line.strip()}")
                except Exception as e:
                    self.logger.error(f"Error reading service output: {e}")
            
            output_thread = threading.Thread(target=log_output, daemon=True)
            output_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            return False
    
    def run(self) -> int:
        """
        Run the service with restart logic.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        attempt = 0
        
        while attempt < self.max_startup_attempts and not self.shutdown_requested:
            attempt += 1
            self.logger.info(f"Service startup attempt {attempt}/{self.max_startup_attempts}")
            
            # Start the service
            if not self.start_service():
                self.logger.error(f"Failed to start service on attempt {attempt}")
                if attempt < self.max_startup_attempts:
                    self.logger.info(f"Waiting {self.restart_delay} seconds before retry...")
                    time.sleep(self.restart_delay)
                continue
            
            # Wait for successful startup
            if self.wait_for_startup():
                self.logger.info("Service started successfully")
                
                # Monitor the service
                try:
                    while not self.shutdown_requested:
                        if self.process and self.process.poll() is not None:
                            return_code = self.process.returncode
                            if return_code == 0:
                                self.logger.info("Service completed successfully")
                                return 0
                            else:
                                self.logger.error(f"Service exited unexpectedly with code {return_code}")
                                break
                        
                        time.sleep(5)  # Check every 5 seconds
                        
                except KeyboardInterrupt:
                    self.logger.info("Received interrupt signal")
                    self.shutdown_requested = True
                
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested, exiting...")
                    return 0
                
                # Service crashed, try to restart if we have attempts left
                if attempt < self.max_startup_attempts:
                    self.logger.warning(f"Service crashed, waiting {self.restart_delay} seconds before restart...")
                    time.sleep(self.restart_delay)
                
            else:
                self.logger.error(f"Service startup failed on attempt {attempt}")
                if self.process and self.process.poll() is None:
                    self.logger.info("Terminating failed service process...")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                
                if attempt < self.max_startup_attempts:
                    self.logger.info(f"Waiting {self.restart_delay} seconds before retry...")
                    time.sleep(self.restart_delay)
        
        self.logger.error(f"Service failed to start after {self.max_startup_attempts} attempts")
        return 1


def main():
    """Main entry point for the startup wrapper."""
    parser = argparse.ArgumentParser(description="Service startup wrapper with error recovery")
    parser.add_argument("--service-name", required=True, help="Name of the service")
    parser.add_argument("--health-check-url", help="URL for health check")
    parser.add_argument("command", nargs="+", help="Command to execute")
    
    args = parser.parse_args()
    
    # Set health check URL if provided
    if args.health_check_url:
        os.environ['SERVICE_HEALTH_CHECK_URL'] = args.health_check_url
    
    wrapper = ServiceStartupWrapper(args.service_name, args.command)
    exit_code = wrapper.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()