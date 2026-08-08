"""
Redis-backed daily request caps: per-IP and global, for public demo deployments.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request, status

from medgraphia.api.auth import require_api_key
from medgraphia.cache.redis_client import get_redis
from medgraphia.config import get_settings
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_SECONDS_IN_DAY = 86400


def _today() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


async def enforce_daily_rate_limit(
    request: Request,
    principal: dict = Depends(require_api_key),
) -> dict:
    """
    FastAPI dependency: rejects the request once the per-IP or global daily
    quota is exhausted. Admin-role keys are exempt. Fails open (no limiting)
    if rate limiting is disabled or Redis is unreachable.
    """
    cfg = get_settings()
    if not cfg.rate_limit_enabled or principal.get("role") == "admin":
        return principal

    redis = await get_redis()
    if redis is None:
        return principal

    day = _today()
    client_ip = request.client.host if request.client else "unknown"

    global_key = f"ratelimit:global:{day}"
    ip_key = f"ratelimit:ip:{client_ip}:{day}"

    global_count = await redis.incr(global_key)
    if global_count == 1:
        await redis.expire(global_key, _SECONDS_IN_DAY)

    ip_count = await redis.incr(ip_key)
    if ip_count == 1:
        await redis.expire(ip_key, _SECONDS_IN_DAY)

    if global_count > cfg.rate_limit_global_daily:
        logger.warning("rate_limit_global_exceeded", count=global_count)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="This demo has reached its daily request limit. Please try again tomorrow.",
        )

    if ip_count > cfg.rate_limit_ip_daily:
        logger.warning("rate_limit_ip_exceeded", ip=client_ip, count=ip_count)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You've reached today's request limit for this demo. Please try again tomorrow.",
        )

    return principal
