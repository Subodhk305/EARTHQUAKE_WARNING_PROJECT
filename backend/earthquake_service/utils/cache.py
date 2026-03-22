"""Redis async cache layer."""
import json
import logging
from typing import Optional, Any
from earthquake_service.config import settings

logger = logging.getLogger(__name__)

# Only import redis if we're using it
if settings.USE_REDIS:
    import redis.asyncio as redis
    _redis: Optional[redis.Redis] = None
else:
    _redis = None
    logger.info("Redis is disabled (USE_REDIS=false)")


async def init_redis():
    """Initialize Redis connection pool if enabled."""
    if not settings.USE_REDIS:
        logger.info("Redis cache disabled")
        return
        
    global _redis
    try:
        import redis.asyncio as redis
        _redis = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await _redis.ping()
        logger.info("✅ Redis connected successfully to %s", settings.REDIS_URL)
    except Exception as e:
        logger.error("❌ Failed to connect to Redis: %s", e)
        logger.info("Continuing without Redis cache...")
        _redis = None


def get_redis():
    """Get the Redis client instance."""
    if not settings.USE_REDIS:
        return None
    if _redis is None:
        return None
    return _redis


async def cache_set(key: str, value: Any, ttl: int = 300):
    """Set a value in cache with TTL."""
    if not settings.USE_REDIS:
        return
        
    try:
        r = get_redis()
        if r:
            await r.setex(key, ttl, json.dumps(value))
            logger.debug("Cache set for key: %s (TTL: %d)", key, ttl)
    except Exception as e:
        logger.error("Failed to set cache for key %s: %s", key, e)


async def cache_get(key: str) -> Optional[Any]:
    """Get a value from cache."""
    if not settings.USE_REDIS:
        return None
        
    try:
        r = get_redis()
        if r:
            data = await r.get(key)
            if data:
                logger.debug("Cache hit for key: %s", key)
                return json.loads(data)
            logger.debug("Cache miss for key: %s", key)
        return None
    except Exception as e:
        logger.error("Failed to get cache for key %s: %s", key, e)
        return None


async def cache_delete(key: str):
    """Delete a key from cache."""
    if not settings.USE_REDIS:
        return
        
    try:
        r = get_redis()
        if r:
            await r.delete(key)
            logger.debug("Cache deleted for key: %s", key)
    except Exception as e:
        logger.error("Failed to delete cache for key %s: %s", key, e)


async def close_redis():
    """Close Redis connection pool."""
    global _redis
    if _redis and settings.USE_REDIS:
        await _redis.close()
        _redis = None
        logger.info("Redis connection closed")


# Context manager for Redis operations
class RedisCache:
    """Context manager for Redis cache operations."""
    
    def __init__(self):
        self.redis = None
    
    async def __aenter__(self):
        if settings.USE_REDIS:
            self.redis = get_redis()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def get(self, key: str) -> Optional[Any]:
        return await cache_get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        await cache_set(key, value, ttl)
    
    async def delete(self, key: str):
        await cache_delete(key)


# Decorator for caching function results
def cached(ttl: int = 300):
    """
    Decorator to cache function results in Redis.
    
    Args:
        ttl: Time to live in seconds
        
    Example:
        @cached(ttl=600)
        async def get_expensive_data(param):
            # Expensive operation
            return result
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not settings.USE_REDIS:
                return await func(*args, **kwargs)
                
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = await cache_get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator    