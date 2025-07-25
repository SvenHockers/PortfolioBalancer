"""Redis caching layer for analytics results."""

import json
import logging
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError

from .models import BacktestResult, MonteCarloResult, BacktestConfig, MonteCarloConfig
from .exceptions import AnalyticsError

logger = logging.getLogger(__name__)


class AnalyticsCache:
    """Redis-based caching layer for analytics results."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 default_ttl: int = 3600):
        """
        Initialize analytics cache.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise AnalyticsError(f"Redis connection failed: {e}")
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration to use as cache key."""
        # Sort config to ensure consistent hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _serialize_result(self, result: Any) -> str:
        """Serialize result for storage in Redis."""
        if hasattr(result, 'model_dump'):
            # Pydantic model
            return json.dumps(result.model_dump(), default=str)
        else:
            return json.dumps(result, default=str)
    
    def _deserialize_result(self, data: str, result_type: type) -> Any:
        """Deserialize result from Redis storage."""
        try:
            result_dict = json.loads(data)
            if hasattr(result_type, 'model_validate'):
                # Pydantic model
                return result_type.model_validate(result_dict)
            else:
                return result_dict
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to deserialize cached result: {e}")
            return None
    
    def cache_backtest_result(self, config: BacktestConfig, result: BacktestResult, 
                            ttl: Optional[int] = None) -> str:
        """
        Cache backtest result.
        
        Args:
            config: Backtest configuration
            result: Backtest result
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            Cache key used for storage
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            cache_key = f"backtest:{self._generate_config_hash(config_dict)}"
            
            # Store result with metadata
            cache_data = {
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'cached_at': datetime.utcnow().isoformat(),
                'config': config_dict
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            ttl = ttl or self.default_ttl
            
            self.redis_client.setex(cache_key, ttl, serialized_data)
            logger.info(f"Cached backtest result with key: {cache_key}")
            
            return cache_key
            
        except RedisError as e:
            logger.error(f"Failed to cache backtest result: {e}")
            # Don't raise exception - caching failure shouldn't break the operation
            return ""
    
    def get_cached_backtest(self, config: BacktestConfig) -> Optional[BacktestResult]:
        """
        Retrieve cached backtest result.
        
        Args:
            config: Backtest configuration
            
        Returns:
            Cached backtest result or None if not found
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            cache_key = f"backtest:{self._generate_config_hash(config_dict)}"
            
            cached_data = self.redis_client.get(cache_key)
            if not cached_data:
                return None
            
            cache_dict = json.loads(cached_data)
            result_data = cache_dict.get('result')
            
            if result_data:
                logger.info(f"Retrieved cached backtest result: {cache_key}")
                return BacktestResult.model_validate(result_data)
            
            return None
            
        except (RedisError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to retrieve cached backtest result: {e}")
            return None
    
    def cache_monte_carlo_result(self, config: MonteCarloConfig, result: MonteCarloResult,
                               ttl: Optional[int] = None) -> str:
        """
        Cache Monte Carlo result.
        
        Args:
            config: Monte Carlo configuration
            result: Monte Carlo result
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            Cache key used for storage
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            cache_key = f"monte_carlo:{self._generate_config_hash(config_dict)}"
            
            # Store result with metadata
            cache_data = {
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'cached_at': datetime.utcnow().isoformat(),
                'config': config_dict
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            ttl = ttl or self.default_ttl
            
            self.redis_client.setex(cache_key, ttl, serialized_data)
            logger.info(f"Cached Monte Carlo result with key: {cache_key}")
            
            return cache_key
            
        except RedisError as e:
            logger.error(f"Failed to cache Monte Carlo result: {e}")
            return ""
    
    def get_cached_monte_carlo(self, config: MonteCarloConfig) -> Optional[MonteCarloResult]:
        """
        Retrieve cached Monte Carlo result.
        
        Args:
            config: Monte Carlo configuration
            
        Returns:
            Cached Monte Carlo result or None if not found
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            cache_key = f"monte_carlo:{self._generate_config_hash(config_dict)}"
            
            cached_data = self.redis_client.get(cache_key)
            if not cached_data:
                return None
            
            cache_dict = json.loads(cached_data)
            result_data = cache_dict.get('result')
            
            if result_data:
                logger.info(f"Retrieved cached Monte Carlo result: {cache_key}")
                return MonteCarloResult.model_validate(result_data)
            
            return None
            
        except (RedisError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to retrieve cached Monte Carlo result: {e}")
            return None
    
    def cache_analysis_result(self, key: str, result: Any, ttl: Optional[int] = None) -> bool:
        """
        Cache generic analysis result.
        
        Args:
            key: Cache key
            result: Analysis result
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            cache_data = {
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            ttl = ttl or self.default_ttl
            
            self.redis_client.setex(key, ttl, serialized_data)
            logger.info(f"Cached analysis result with key: {key}")
            
            return True
            
        except RedisError as e:
            logger.error(f"Failed to cache analysis result: {e}")
            return False
    
    def get_cached_analysis(self, key: str, result_type: type = dict) -> Optional[Any]:
        """
        Retrieve cached analysis result.
        
        Args:
            key: Cache key
            result_type: Expected result type for deserialization
            
        Returns:
            Cached analysis result or None if not found
        """
        try:
            cached_data = self.redis_client.get(key)
            if not cached_data:
                return None
            
            cache_dict = json.loads(cached_data)
            result_data = cache_dict.get('result')
            
            if result_data:
                logger.info(f"Retrieved cached analysis result: {key}")
                if hasattr(result_type, 'model_validate'):
                    return result_type.model_validate(result_data)
                else:
                    return result_data
            
            return None
            
        except (RedisError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to retrieve cached analysis result: {e}")
            return None
    
    def invalidate_cache(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Redis key pattern (supports wildcards)
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching pattern: {pattern}")
                return deleted
            return 0
            
        except RedisError as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            info = self.redis_client.info()
            
            # Count keys by type
            backtest_keys = len(self.redis_client.keys("backtest:*"))
            monte_carlo_keys = len(self.redis_client.keys("monte_carlo:*"))
            analysis_keys = len(self.redis_client.keys("analysis:*"))
            
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'total_keys': info.get('db0', {}).get('keys', 0),
                'backtest_keys': backtest_keys,
                'monte_carlo_keys': monte_carlo_keys,
                'analysis_keys': analysis_keys,
                'hit_rate': self._calculate_hit_rate(),
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }
            
        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        try:
            info = self.redis_client.info()
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            
            if hits + misses == 0:
                return 0.0
            
            return hits / (hits + misses)
            
        except (RedisError, ZeroDivisionError):
            return 0.0
    
    def health_check(self) -> bool:
        """
        Check if Redis is healthy.
        
        Returns:
            True if Redis is responding, False otherwise
        """
        try:
            self.redis_client.ping()
            return True
        except RedisError:
            return False
    
    def clear_all_cache(self) -> bool:
        """
        Clear all cached data (use with caution).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.redis_client.flushdb()
            logger.warning("Cleared all cached data")
            return True
        except RedisError as e:
            logger.error(f"Failed to clear cache: {e}")
            return False