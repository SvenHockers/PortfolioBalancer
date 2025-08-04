"""API error handling utilities with retry logic and graceful fallbacks."""

import time
import logging
import os
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List
from functools import wraps
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError
import requests

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class AuthenticationError(APIError):
    """Exception for API authentication failures."""
    pass


class RateLimitError(APIError):
    """Exception for API rate limit errors."""
    pass


class APIErrorHandler:
    """Handles API errors with retry logic and graceful fallbacks."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize API error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_multiplier: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays to avoid thundering herd
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
    
    def with_retry(self, operation_name: str = "API operation"):
        """
        Decorator to add retry logic to API operations.
        
        Args:
            operation_name: Name of the operation for logging
            
        Returns:
            Decorated function with retry logic
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                        
                    except (AuthenticationError, requests.exceptions.HTTPError) as e:
                        # Check for authentication errors (403, 401)
                        if hasattr(e, 'response') and e.response is not None:
                            status_code = e.response.status_code
                            if status_code in [401, 403]:
                                logger.error(f"{operation_name} authentication failed (HTTP {status_code}): {e}")
                                raise AuthenticationError(f"Authentication failed: {e}") from e
                            elif status_code == 429:
                                # Rate limiting - use longer delay
                                last_exception = RateLimitError(f"Rate limited: {e}")
                                if attempt < self.max_retries:
                                    delay = self._calculate_delay(attempt) * 2  # Double delay for rate limits
                                    logger.warning(f"{operation_name} rate limited (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                                    time.sleep(delay)
                                else:
                                    logger.error(f"{operation_name} rate limited after {self.max_retries + 1} attempts: {e}")
                                continue
                            elif status_code >= 500:
                                # Server errors - retry with normal delay
                                last_exception = e
                                if attempt < self.max_retries:
                                    delay = self._calculate_delay(attempt)
                                    logger.warning(f"{operation_name} server error (HTTP {status_code}, attempt {attempt + 1}): {e}. Retrying in {delay}s")
                                    time.sleep(delay)
                                else:
                                    logger.error(f"{operation_name} server error after {self.max_retries + 1} attempts: {e}")
                                continue
                        
                        # For other HTTP errors, retry if not the last attempt
                        last_exception = e
                        if attempt < self.max_retries:
                            delay = self._calculate_delay(attempt)
                            logger.warning(f"{operation_name} failed (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                            time.sleep(delay)
                        else:
                            logger.error(f"{operation_name} failed after {self.max_retries + 1} attempts: {e}")
                            
                    except (RequestException, ConnectionError, Timeout) as e:
                        # Network-related errors - retry with exponential backoff
                        last_exception = e
                        if attempt < self.max_retries:
                            delay = self._calculate_delay(attempt)
                            logger.warning(f"{operation_name} network error (attempt {attempt + 1}): {e}. Retrying in {delay}s")
                            time.sleep(delay)
                        else:
                            logger.error(f"{operation_name} network error after {self.max_retries + 1} attempts: {e}")
                            
                    except Exception as e:
                        # Unexpected errors - don't retry
                        logger.error(f"{operation_name} unexpected error: {e}")
                        last_exception = e
                        break
                
                # If we get here, all retries failed
                if last_exception:
                    raise last_exception
                else:
                    raise APIError(f"{operation_name} failed after {self.max_retries + 1} attempts")
                    
            return wrapper
        return decorator
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with optional jitter."""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd problem
        if self.jitter:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure minimum delay
        
        return delay


def handle_alpaca_api_error(error: Exception, operation: str = "Alpaca API operation") -> Optional[Dict]:
    """
    Handle Alpaca API errors gracefully.
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        
    Returns:
        None for authentication errors, empty dict for other errors to allow fallback
    """
    if isinstance(error, requests.exceptions.HTTPError):
        status_code = error.response.status_code if error.response else None
        
        if status_code == 403:
            logger.error(f"{operation} failed: 403 Forbidden - Invalid API credentials or insufficient permissions")
            return None
        elif status_code == 401:
            logger.error(f"{operation} failed: 401 Unauthorized - Invalid API credentials")
            return None
        elif status_code == 429:
            logger.warning(f"{operation} failed: 429 Rate Limited - Too many requests")
            raise RateLimitError(f"Rate limited: {error}")
        else:
            logger.error(f"{operation} failed: HTTP {status_code} - {error}")
            return {}
    
    elif isinstance(error, (ConnectionError, Timeout)):
        logger.warning(f"{operation} failed: Network error - {error}")
        return {}
    
    else:
        logger.error(f"{operation} failed: Unexpected error - {error}")
        return {}


def setup_yfinance_cache() -> None:
    """
    Configure yfinance cache location to avoid permission errors.
    Sets up alternative cache locations if default is not writable.
    """
    # Check if cache is already configured via environment variable
    if "YFINANCE_CACHE_DIR" in os.environ:
        configured_cache = Path(os.environ["YFINANCE_CACHE_DIR"])
        try:
            configured_cache.mkdir(parents=True, exist_ok=True)
            test_file = configured_cache / "test_write"
            test_file.touch()
            test_file.unlink()
            logger.info(f"Using pre-configured yfinance cache location: {configured_cache}")
            return
        except (PermissionError, OSError) as e:
            logger.warning(f"Pre-configured cache location not writable {configured_cache}: {e}")
            # Continue to try other locations
        except Exception as e:
            logger.warning(f"Error testing pre-configured cache location {configured_cache}: {e}")
            # Continue to try other locations
    
    try:
        # Try to use default cache location first
        default_cache = Path.home() / ".cache" / "py-yfinance"
        
        # Test if we can write to the default location
        try:
            default_cache.mkdir(parents=True, exist_ok=True)
            test_file = default_cache / "test_write"
            test_file.touch()
            test_file.unlink()
            # Set environment variable even for default location for consistency
            os.environ["YFINANCE_CACHE_DIR"] = str(default_cache)
            logger.info(f"Using default yfinance cache location: {default_cache}")
            return
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot write to default yfinance cache location {default_cache}: {e}")
    
    except Exception as e:
        logger.warning(f"Error checking default cache location: {e}")
    
    # Try alternative cache locations in order of preference
    alternative_locations = [
        Path("/app/cache/yfinance"),  # Container-friendly location (highest priority)
        Path.cwd() / "cache" / "yfinance",  # Current working directory
        Path("/tmp/yfinance_cache"),  # Temporary directory
        Path("/var/tmp/yfinance_cache"),  # Alternative temp directory
        Path.home() / "yfinance_cache",  # Home directory fallback
        Path.home() / ".cache" / "yfinance_fallback"  # Alternative home cache
    ]
    
    for cache_dir in alternative_locations:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            test_file = cache_dir / "test_write"
            test_file.touch()
            test_file.unlink()
            
            # Set environment variable to redirect yfinance cache
            os.environ["YFINANCE_CACHE_DIR"] = str(cache_dir)
            logger.info(f"Using alternative yfinance cache location: {cache_dir}")
            return
            
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot write to alternative cache location {cache_dir}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error with cache location {cache_dir}: {e}")
            continue
    
    # If all locations fail, try to disable caching gracefully
    logger.warning("Could not find writable cache location, attempting to disable yfinance caching")
    try:
        # Try multiple approaches to disable caching
        os.environ["YFINANCE_CACHE_DISABLE"] = "1"
        os.environ["YFINANCE_CACHE_DIR"] = ""
        
        # Also try yfinance-specific environment variables
        os.environ["YF_CACHE_DISABLE"] = "1"
        
        logger.info("yfinance caching disabled due to permission issues")
        return
    except Exception as e:
        logger.error(f"Failed to disable yfinance caching: {e}")
    
    # As a last resort, try to use a memory-based cache location
    try:
        import tempfile
        temp_cache = Path(tempfile.gettempdir()) / f"yfinance_fallback_{os.getpid()}"
        temp_cache.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = temp_cache / "test_write"
        test_file.touch()
        test_file.unlink()
        
        os.environ["YFINANCE_CACHE_DIR"] = str(temp_cache)
        logger.info(f"Using temporary fallback cache location: {temp_cache}")
        return
    except Exception as temp_error:
        logger.error(f"Failed to set up temporary cache: {temp_error}")
    
    # Final fallback - try to use in-memory caching by setting a non-existent path
    # This should force yfinance to skip file caching
    try:
        os.environ["YFINANCE_CACHE_DIR"] = "/dev/null/yfinance_disabled"
        logger.warning("Set yfinance cache to /dev/null to disable file caching")
    except Exception as final_error:
        logger.error(f"Final cache fallback failed: {final_error}")
        logger.error("yfinance may encounter permission errors - consider running with elevated permissions or fixing directory permissions")


def handle_yfinance_error(error: Exception, ticker: str = "", operation: str = "yfinance operation") -> Optional[Any]:
    """
    Handle yfinance API errors gracefully.
    
    Args:
        error: The exception that occurred
        ticker: Ticker symbol being processed
        operation: Description of the operation that failed
        
    Returns:
        None to indicate failure, allowing fallback to cached data or mock mode
    """
    error_msg = str(error).lower()
    
    if "permission denied" in error_msg or "errno 13" in error_msg:
        logger.error(f"{operation} for {ticker} failed: Permission denied - trying to set up alternative cache")
        setup_yfinance_cache()
        return None
    
    elif "could not find" in error_msg and ("adjusted close" in error_msg or "price data" in error_msg):
        logger.warning(f"{operation} for {ticker} failed: No price data available - ticker may be delisted or invalid")
        return None
    
    elif "possibly delisted" in error_msg or "no timezone found" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: Ticker appears to be delisted or invalid")
        return None
    
    elif "403" in error_msg or "forbidden" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: Access forbidden - possible rate limiting or API restrictions")
        return None
    
    elif "429" in error_msg or "too many requests" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: Rate limited - too many requests")
        raise RateLimitError(f"Rate limited for {ticker}: {error}")
    
    elif "timeout" in error_msg or "timed out" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: Request timeout - network or server issues")
        return None
    
    elif "connection" in error_msg or "network" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: Connection error - network issues")
        return None
    
    elif "ssl" in error_msg or "certificate" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: SSL/Certificate error")
        return None
    
    elif "http error 404" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: Data not found (HTTP 404) - ticker may not exist")
        return None
    
    elif "http error 500" in error_msg or "internal server error" in error_msg:
        logger.warning(f"{operation} for {ticker} failed: Server error (HTTP 500) - temporary issue")
        return None
    
    elif "json" in error_msg and ("decode" in error_msg or "parse" in error_msg):
        logger.warning(f"{operation} for {ticker} failed: Invalid response format - API may be down")
        return None
    
    else:
        logger.error(f"{operation} for {ticker} failed: {error}")
        return None


class MockDataProvider:
    """Provides mock data when real APIs are unavailable."""
    
    @staticmethod
    def get_mock_positions() -> Dict[str, float]:
        """Return mock portfolio positions for testing."""
        return {
            "AAPL": 100.0,
            "GOOGL": 50.0,
            "MSFT": 75.0,
            "TSLA": 25.0
        }
    
    @staticmethod
    def get_mock_account_info() -> Dict[str, Any]:
        """Return mock account information for testing."""
        return {
            "account_id": "mock_account",
            "buying_power": 10000.0,
            "cash": 5000.0,
            "portfolio_value": 50000.0,
            "status": "ACTIVE"
        }
    
    @staticmethod
    def get_mock_order_response(symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        """Return mock order response for testing."""
        return {
            "id": f"mock_order_{symbol}_{int(time.time())}",
            "symbol": symbol,
            "qty": str(abs(quantity)),
            "side": side,
            "type": "market",
            "status": "filled",
            "created_at": "2024-01-01T12:00:00Z",
            "filled_avg_price": "100.00"
        }


# Global error handler instance
api_error_handler = APIErrorHandler()