"""
Redis-backed daily request caps: per-visitor and global, for public demo deployments.
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
    FastAPI dependency: rejects the request once the per-visitor or global
    daily quota is exhausted. Admin-role keys are exempt. Fails open (no
    limiting) if rate limiting is disabled or Redis is unreachable.

    "Per-visitor" keys on X-Client-ID (the UI's guest cookie) when present,
    falling back to the raw client IP otherwise. The UI proxies every request
    server-side, so request.client.host is always the UI container's own IP,
    not the visitor's — X-Client-ID is the only signal that actually varies
    per browser in that deployment shape. A direct API caller with no
    X-Client-ID (curl, another client) still gets a real per-IP limit.

    A key can also carry its own daily_limit (set via /admin/keys), enforced
    as a fourth bucket keyed by key prefix — e.g. a shared "public_test" key
    can be capped in aggregate across every visitor using it, independent of
    each visitor's own per-client cap.
    """
    cfg = get_settings()
    if not cfg.rate_limit_enabled or principal.get("role") == "admin":
        return principal

    redis = await get_redis()
    if redis is None:
        return principal

    day = _today()
    client_ip = request.client.host if request.client else "unknown"
    client_key = request.headers.get("X-Client-ID") or client_ip

    global_key = f"ratelimit:global:{day}"
    client_bucket_key = f"ratelimit:client:{client_key}:{day}"

    global_count = await redis.incr(global_key)
    if global_count == 1:
        await redis.expire(global_key, _SECONDS_IN_DAY)

    client_count = await redis.incr(client_bucket_key)
    if client_count == 1:
        await redis.expire(client_bucket_key, _SECONDS_IN_DAY)

    if global_count > cfg.rate_limit_global_daily:
        logger.warning("rate_limit_global_exceeded", count=global_count)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="This demo has reached its daily request limit. Please try again tomorrow.",
        )

    if client_count > cfg.rate_limit_ip_daily:
        logger.warning("rate_limit_client_exceeded", client_key=client_key, count=client_count)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You've reached today's request limit for this demo. Please try again tomorrow.",
        )

    key_daily_limit = principal.get("daily_limit")
    if key_daily_limit is not None:
        key_prefix = principal.get("prefix", "")
        key_bucket_key = f"ratelimit:key:{key_prefix}:{day}"
        key_count = await redis.incr(key_bucket_key)
        if key_count == 1:
            await redis.expire(key_bucket_key, _SECONDS_IN_DAY)
        if key_count > key_daily_limit:
            logger.warning("rate_limit_key_exceeded", key_prefix=key_prefix, count=key_count)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="This API key has reached its daily request limit. Please try again tomorrow.",
            )

    return principal
