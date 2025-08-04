"""Redis caching layer for analytics results."""

import json
import logging
import hashlib
import asyncio
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import redis
from redis.exceptions import RedisError
import time

from .models import (
    BacktestResult, MonteCarloResult, BacktestConfig, MonteCarloConfig,
    RiskAnalysis, PerformanceMetrics, DividendAnalysis
)
from .exceptions import AnalyticsError

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategy types."""
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    CACHE_ASIDE = "cache_aside"


class CachePriority(str, Enum):
    """Cache priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    errors: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate."""
        return 1.0 - self.hit_rate


@dataclass
class CacheConfig:
    """Cache configuration."""
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "256mb"
    eviction_policy: str = "allkeys-lru"
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    enable_warming: bool = True
    warming_interval: int = 300  # 5 minutes
    monitoring_enabled: bool = True


class AnalyticsCache:
    """Intelligent Redis-based caching layer for analytics results."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 config: Optional[CacheConfig] = None):
        """
        Initialize analytics cache.
        
        Args:
            redis_url: Redis connection URL
            config: Cache configuration
        """
        self.redis_url = redis_url
        self.config = config or CacheConfig()
        self.metrics = CacheMetrics()
        self._warming_tasks: Dict[str, Callable] = {}
        self._warming_thread: Optional[threading.Thread] = None
        self._stop_warming = threading.Event()
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
            
            # Configure Redis settings
            self._configure_redis()
            
            # Start cache warming if enabled
            if self.config.enable_warming:
                self._start_cache_warming()
                
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise AnalyticsError(f"Redis connection failed: {e}")
    
    def _configure_redis(self):
        """Configure Redis settings for optimal performance."""
        try:
            # Set memory policy
            self.redis_client.config_set('maxmemory-policy', self.config.eviction_policy)
            
            # Set max memory if specified
            if self.config.max_memory:
                self.redis_client.config_set('maxmemory', self.config.max_memory)
                
            logger.info("Redis configuration applied successfully")
            
        except RedisError as e:
            logger.warning(f"Failed to configure Redis: {e}")
    
    def _start_cache_warming(self):
        """Start background cache warming thread."""
        if self._warming_thread and self._warming_thread.is_alive():
            return
            
        self._warming_thread = threading.Thread(
            target=self._cache_warming_worker,
            daemon=True,
            name="CacheWarmingWorker"
        )
        self._warming_thread.start()
        logger.info("Cache warming thread started")
    
    def _cache_warming_worker(self):
        """Background worker for cache warming."""
        while not self._stop_warming.wait(self.config.warming_interval):
            try:
                for key, warming_func in self._warming_tasks.items():
                    try:
                        warming_func()
                        logger.debug(f"Cache warming completed for: {key}")
                    except Exception as e:
                        logger.error(f"Cache warming failed for {key}: {e}")
                        
            except Exception as e:
                logger.error(f"Cache warming worker error: {e}")
    
    def register_warming_task(self, key: str, warming_func: Callable):
        """
        Register a cache warming task.
        
        Args:
            key: Unique key for the warming task
            warming_func: Function to execute for warming
        """
        self._warming_tasks[key] = warming_func
        logger.info(f"Registered cache warming task: {key}")
    
    def unregister_warming_task(self, key: str):
        """
        Unregister a cache warming task.
        
        Args:
            key: Key of the warming task to remove
        """
        if key in self._warming_tasks:
            del self._warming_tasks[key]
            logger.info(f"Unregistered cache warming task: {key}")
    
    def _compress_data(self, data: str) -> str:
        """Compress data if it exceeds threshold."""
        if not self.config.enable_compression or len(data) < self.config.compression_threshold:
            return data
            
        try:
            import gzip
            import base64
            
            compressed = gzip.compress(data.encode('utf-8'))
            encoded = base64.b64encode(compressed).decode('utf-8')
            return f"COMPRESSED:{encoded}"
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, data: str) -> str:
        """Decompress data if it was compressed."""
        if not data.startswith("COMPRESSED:"):
            return data
            
        try:
            import gzip
            import base64
            
            encoded_data = data[11:]  # Remove "COMPRESSED:" prefix
            compressed = base64.b64decode(encoded_data.encode('utf-8'))
            return gzip.decompress(compressed).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data
    
    def _update_metrics(self, operation: str, success: bool = True):
        """Update cache metrics."""
        if not self.config.monitoring_enabled:
            return
            
        self.metrics.total_requests += 1
        
        if operation == "hit":
            self.metrics.hits += 1
        elif operation == "miss":
            self.metrics.misses += 1
        elif operation == "write":
            self.metrics.writes += 1
        elif operation == "eviction":
            self.metrics.evictions += 1
            
        if not success:
            self.metrics.errors += 1
    
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
                            ttl: Optional[int] = None, priority: CachePriority = CachePriority.MEDIUM) -> str:
        """
        Cache backtest result with intelligent caching strategy.
        
        Args:
            config: Backtest configuration
            result: Backtest result
            ttl: Time-to-live in seconds (uses default if None)
            priority: Cache priority level
            
        Returns:
            Cache key used for storage
        """
        try:
            config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
            cache_key = f"backtest:{self._generate_config_hash(config_dict)}"
            
            # Determine TTL based on priority and complexity
            if ttl is None:
                ttl = self._calculate_intelligent_ttl(
                    data_type="backtest",
                    priority=priority,
                    complexity_factor=len(config.tickers) * (config.end_date - config.start_date).days
                )
            
            # Store result with enhanced metadata
            cache_data = {
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'cached_at': datetime.utcnow().isoformat(),
                'config': config_dict,
                'priority': priority.value,
                'access_count': 0,
                'last_accessed': datetime.utcnow().isoformat(),
                'computation_time': getattr(result, 'computation_time', None),
                'data_freshness': self._calculate_data_freshness(config.end_date)
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            compressed_data = self._compress_data(serialized_data)
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.setex(cache_key, ttl, compressed_data)
            
            # Add to priority index for intelligent eviction
            priority_key = f"priority:{priority.value}"
            pipe.zadd(priority_key, {cache_key: time.time()})
            
            # Add to access tracking
            pipe.hset(f"access:{cache_key}", mapping={
                'count': 0,
                'last_access': datetime.utcnow().isoformat()
            })
            
            pipe.execute()
            
            self._update_metrics("write", True)
            logger.info(f"Cached backtest result with key: {cache_key}, priority: {priority.value}")
            
            return cache_key
            
        except RedisError as e:
            logger.error(f"Failed to cache backtest result: {e}")
            self._update_metrics("write", False)
            return ""
    
    def get_cached_backtest(self, config: BacktestConfig) -> Optional[BacktestResult]:
        """
        Retrieve cached backtest result with intelligent access tracking.
        
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
                self._update_metrics("miss")
                return None
            
            # Decompress if needed
            decompressed_data = self._decompress_data(cached_data)
            cache_dict = json.loads(decompressed_data)
            
            # Check data freshness
            if not self._is_data_fresh(cache_dict):
                logger.info(f"Cache data is stale for key: {cache_key}")
                self._update_metrics("miss")
                return None
            
            result_data = cache_dict.get('result')
            
            if result_data:
                # Update access tracking
                self._update_access_tracking(cache_key)
                
                # Extend TTL for frequently accessed items
                self._extend_ttl_if_popular(cache_key)
                
                self._update_metrics("hit")
                logger.info(f"Retrieved cached backtest result: {cache_key}")
                return BacktestResult.model_validate(result_data)
            
            self._update_metrics("miss")
            return None
            
        except (RedisError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to retrieve cached backtest result: {e}")
            self._update_metrics("miss")
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
    
    def cache_analysis_result(self, key: str, result: Any, ttl: Optional[int] = None, 
                            priority: CachePriority = CachePriority.MEDIUM) -> bool:
        """
        Cache generic analysis result with intelligent caching.
        
        Args:
            key: Cache key
            result: Analysis result
            ttl: Time-to-live in seconds (uses default if None)
            priority: Cache priority level
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            # Determine data type from key
            data_type = key.split(':')[0] if ':' in key else 'analysis'
            
            if ttl is None:
                ttl = self._calculate_intelligent_ttl(
                    data_type=data_type,
                    priority=priority,
                    complexity_factor=1
                )
            
            cache_data = {
                'result': result.model_dump() if hasattr(result, 'model_dump') else result,
                'cached_at': datetime.utcnow().isoformat(),
                'priority': priority.value,
                'access_count': 0,
                'last_accessed': datetime.utcnow().isoformat(),
                'data_type': data_type
            }
            
            serialized_data = json.dumps(cache_data, default=str)
            compressed_data = self._compress_data(serialized_data)
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.setex(key, ttl, compressed_data)
            
            # Add to priority index
            priority_key = f"priority:{priority.value}"
            pipe.zadd(priority_key, {key: time.time()})
            
            # Add to access tracking
            pipe.hset(f"access:{key}", mapping={
                'count': 0,
                'last_access': datetime.utcnow().isoformat()
            })
            
            pipe.execute()
            
            self._update_metrics("write", True)
            logger.info(f"Cached analysis result with key: {key}, priority: {priority.value}")
            
            return True
            
        except RedisError as e:
            logger.error(f"Failed to cache analysis result: {e}")
            self._update_metrics("write", False)
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
    
    def _calculate_intelligent_ttl(self, data_type: str, priority: CachePriority, 
                                 complexity_factor: int = 1) -> int:
        """
        Calculate intelligent TTL based on data type, priority, and complexity.
        
        Args:
            data_type: Type of cached data
            priority: Cache priority level
            complexity_factor: Factor indicating computation complexity
            
        Returns:
            TTL in seconds
        """
        base_ttl = self.config.default_ttl
        
        # Priority multipliers
        priority_multipliers = {
            CachePriority.LOW: 0.5,
            CachePriority.MEDIUM: 1.0,
            CachePriority.HIGH: 2.0,
            CachePriority.CRITICAL: 4.0
        }
        
        # Data type multipliers
        type_multipliers = {
            "backtest": 2.0,  # Expensive to compute
            "monte_carlo": 3.0,  # Very expensive
            "risk_analysis": 1.5,
            "performance": 1.0,
            "dividend": 1.2
        }
        
        # Complexity factor (more complex = longer cache)
        complexity_multiplier = min(1.0 + (complexity_factor / 1000), 3.0)
        
        ttl = int(
            base_ttl * 
            priority_multipliers.get(priority, 1.0) * 
            type_multipliers.get(data_type, 1.0) * 
            complexity_multiplier
        )
        
        # Ensure reasonable bounds
        return max(300, min(ttl, 86400))  # 5 minutes to 24 hours
    
    def _calculate_data_freshness(self, end_date) -> float:
        """Calculate data freshness score based on how recent the data is."""
        if isinstance(end_date, str):
            from datetime import datetime
            end_date = datetime.fromisoformat(end_date).date()
        
        days_old = (datetime.now().date() - end_date).days
        
        # Fresher data gets higher score
        if days_old <= 1:
            return 1.0
        elif days_old <= 7:
            return 0.8
        elif days_old <= 30:
            return 0.6
        elif days_old <= 90:
            return 0.4
        else:
            return 0.2
    
    def _is_data_fresh(self, cache_dict: Dict[str, Any]) -> bool:
        """Check if cached data is still fresh enough to use."""
        freshness = cache_dict.get('data_freshness', 1.0)
        cached_at = datetime.fromisoformat(cache_dict.get('cached_at', datetime.utcnow().isoformat()))
        
        # Data is considered stale if:
        # 1. Freshness score is too low and cache is old
        # 2. Cache is older than maximum allowed age
        
        cache_age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600
        
        if freshness < 0.3 and cache_age_hours > 2:
            return False
        
        if cache_age_hours > 24:  # Maximum cache age
            return False
        
        return True
    
    def _update_access_tracking(self, cache_key: str):
        """Update access tracking for cache key."""
        try:
            access_key = f"access:{cache_key}"
            pipe = self.redis_client.pipeline()
            pipe.hincrby(access_key, 'count', 1)
            pipe.hset(access_key, 'last_access', datetime.utcnow().isoformat())
            pipe.expire(access_key, 86400)  # Expire access tracking after 24 hours
            pipe.execute()
        except RedisError as e:
            logger.warning(f"Failed to update access tracking: {e}")
    
    def _extend_ttl_if_popular(self, cache_key: str):
        """Extend TTL for popular cache entries."""
        try:
            access_info = self.redis_client.hgetall(f"access:{cache_key}")
            access_count = int(access_info.get('count', 0))
            
            # If accessed frequently, extend TTL
            if access_count > 5:
                current_ttl = self.redis_client.ttl(cache_key)
                if current_ttl > 0:
                    extension = min(current_ttl * 0.5, 3600)  # Extend by 50% or 1 hour max
                    self.redis_client.expire(cache_key, int(current_ttl + extension))
                    logger.debug(f"Extended TTL for popular key: {cache_key}")
                    
        except (RedisError, ValueError) as e:
            logger.warning(f"Failed to extend TTL: {e}")
    
    def warm_cache_for_portfolio(self, portfolio_id: str, analytics_service):
        """
        Warm cache for a specific portfolio.
        
        Args:
            portfolio_id: Portfolio identifier
            analytics_service: Analytics service instance for data generation
        """
        try:
            logger.info(f"Starting cache warming for portfolio: {portfolio_id}")
            
            # Warm performance metrics
            try:
                performance = analytics_service.get_performance_metrics(portfolio_id)
                if performance:
                    self.cache_analysis_result(
                        f"performance:{portfolio_id}",
                        performance,
                        priority=CachePriority.HIGH
                    )
            except Exception as e:
                logger.warning(f"Failed to warm performance cache: {e}")
            
            # Warm risk analysis
            try:
                risk_analysis = analytics_service.get_risk_analysis(portfolio_id)
                if risk_analysis:
                    self.cache_analysis_result(
                        f"risk:{portfolio_id}",
                        risk_analysis,
                        priority=CachePriority.HIGH
                    )
            except Exception as e:
                logger.warning(f"Failed to warm risk cache: {e}")
            
            # Warm dividend analysis
            try:
                dividend_analysis = analytics_service.get_dividend_analysis(portfolio_id)
                if dividend_analysis:
                    self.cache_analysis_result(
                        f"dividend:{portfolio_id}",
                        dividend_analysis,
                        priority=CachePriority.MEDIUM
                    )
            except Exception as e:
                logger.warning(f"Failed to warm dividend cache: {e}")
            
            logger.info(f"Cache warming completed for portfolio: {portfolio_id}")
            
        except Exception as e:
            logger.error(f"Cache warming failed for portfolio {portfolio_id}: {e}")
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """
        Generate cache efficiency report.
        
        Returns:
            Dictionary with cache efficiency metrics
        """
        try:
            stats = self.get_cache_stats()
            
            # Calculate efficiency metrics
            efficiency_score = self.metrics.hit_rate * 100
            
            # Memory efficiency
            memory_info = self.redis_client.info('memory')
            memory_usage = memory_info.get('used_memory', 0)
            memory_peak = memory_info.get('used_memory_peak', 1)
            memory_efficiency = (1 - memory_usage / memory_peak) * 100 if memory_peak > 0 else 0
            
            # Access pattern analysis
            access_patterns = self._analyze_access_patterns()
            
            return {
                'efficiency_score': efficiency_score,
                'hit_rate': self.metrics.hit_rate,
                'miss_rate': self.metrics.miss_rate,
                'memory_efficiency': memory_efficiency,
                'total_requests': self.metrics.total_requests,
                'cache_size': stats.get('total_keys', 0),
                'access_patterns': access_patterns,
                'recommendations': self._generate_cache_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate efficiency report: {e}")
            return {'error': str(e)}
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze cache access patterns."""
        try:
            # Get all access tracking keys
            access_keys = self.redis_client.keys("access:*")
            
            if not access_keys:
                return {'total_tracked': 0}
            
            total_accesses = 0
            hot_keys = []
            cold_keys = []
            
            for access_key in access_keys[:100]:  # Limit to avoid performance issues
                access_info = self.redis_client.hgetall(access_key)
                count = int(access_info.get('count', 0))
                total_accesses += count
                
                cache_key = access_key.replace('access:', '')
                
                if count > 10:
                    hot_keys.append({'key': cache_key, 'count': count})
                elif count <= 2:
                    cold_keys.append({'key': cache_key, 'count': count})
            
            return {
                'total_tracked': len(access_keys),
                'total_accesses': total_accesses,
                'hot_keys': sorted(hot_keys, key=lambda x: x['count'], reverse=True)[:10],
                'cold_keys': len(cold_keys),
                'average_accesses': total_accesses / len(access_keys) if access_keys else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze access patterns: {e}")
            return {'error': str(e)}
    
    def _generate_cache_recommendations(self) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        
        if self.metrics.hit_rate < 0.5:
            recommendations.append("Consider increasing cache TTL or warming more data")
        
        if self.metrics.hit_rate > 0.9:
            recommendations.append("Cache is performing well, consider reducing TTL to save memory")
        
        if self.metrics.errors > self.metrics.total_requests * 0.1:
            recommendations.append("High error rate detected, check Redis connectivity")
        
        # Check memory usage
        try:
            memory_info = self.redis_client.info('memory')
            used_memory = memory_info.get('used_memory', 0)
            max_memory = memory_info.get('maxmemory', 0)
            
            if max_memory > 0 and used_memory / max_memory > 0.8:
                recommendations.append("Memory usage is high, consider implementing more aggressive eviction")
        except:
            pass
        
        return recommendations
    
    def optimize_cache(self):
        """Perform cache optimization based on access patterns."""
        try:
            logger.info("Starting cache optimization")
            
            # Remove cold keys (rarely accessed)
            access_keys = self.redis_client.keys("access:*")
            cold_keys_removed = 0
            
            for access_key in access_keys:
                access_info = self.redis_client.hgetall(access_key)
                count = int(access_info.get('count', 0))
                last_access = access_info.get('last_access')
                
                if count <= 1 and last_access:
                    try:
                        last_access_time = datetime.fromisoformat(last_access)
                        if (datetime.utcnow() - last_access_time).total_seconds() > 3600:  # 1 hour
                            cache_key = access_key.replace('access:', '')
                            self.redis_client.delete(cache_key, access_key)
                            cold_keys_removed += 1
                    except:
                        continue
            
            logger.info(f"Cache optimization completed, removed {cold_keys_removed} cold keys")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    def clear_all_cache(self) -> bool:
        """
        Clear all cached data (use with caution).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.redis_client.flushdb()
            
            # Reset metrics
            self.metrics = CacheMetrics()
            
            logger.warning("Cleared all cached data")
            return True
        except RedisError as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        try:
            # Stop warming thread
            if self._warming_thread and self._warming_thread.is_alive():
                self._stop_warming.set()
                self._warming_thread.join(timeout=5)
            
            # Close Redis connection
            if hasattr(self.redis_client, 'close'):
                self.redis_client.close()
            
            logger.info("Analytics cache shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cache shutdown: {e}")