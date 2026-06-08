"""
Redis-backed caching layer.

Usage::

    from medgraphia.cache import get_redis, close_redis
"""

from medgraphia.cache.redis_client import close_redis, get_redis, reset_for_testing

__all__ = ["get_redis", "close_redis", "reset_for_testing"]
