"""
Authentication layer.
Supports multiple strategies: none, apikey, oidc.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from medgraphia.config import get_settings
from medgraphia.graph.queries import create_api_key_node, get_api_key_node
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _hash_key(key: str) -> str:
    """Return a SHA-256 hash of the API key for secure storage."""
    return hashlib.sha256(key.encode()).hexdigest()


async def _load_bootstrap_key() -> None:
    """
    Populate the database with the admin bootstrap key from config
    if it doesn't already exist.
    """
    cfg = get_settings()
    bootstrap = cfg.admin_bootstrap_key.get_secret_value()
    if bootstrap:
        if bootstrap == "change-me":
            logger.warning(
                "auth_insecure_default_key",
                message="Using default 'change-me' admin key. CHANGE THIS IN PRODUCTION!",
            )

        key_hash = _hash_key(bootstrap)
        # Check if already exists to avoid unnecessary writes
        existing = await get_api_key_node(key_hash)
        if not existing:
            await create_api_key_node(key_hash=key_hash, prefix=bootstrap[:8], role="admin")
            logger.info("auth_bootstrap_key_persisted", prefix=bootstrap[:8])


def generate_api_key() -> str:
    """Generate a cryptographically secure random API key."""
    return secrets.token_urlsafe(32)


async def register_key(key: str, role: str = "user") -> None:
    """Add a key to the database (admin only)."""
    await create_api_key_node(key_hash=_hash_key(key), prefix=key[:8], role=role)


async def require_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> dict:
    """
    FastAPI dependency: validates auth based on settings.auth_strategy.
    Returns metadata about the authenticated principal.
    """
    cfg = get_settings()
    # Per-browser guest ID (set by the UI, see ui/components/guest_id.py) so
    # multiple visitors sharing one API key still get isolated chat sessions.
    client_id = request.headers.get("X-Client-ID", "")

    # Strategy: none (Permissive)
    if cfg.auth_strategy == "none":
        return {
            "role": "admin",
            "id": f"anonymous:{client_id}" if client_id else "anonymous",
            "strategy": "none",
        }

    # Strategy: oidc (Keycloak/SSO)
    if cfg.auth_strategy == "oidc":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OIDC strategy not implemented. Use 'apikey'.",
        )

    # Strategy: apikey (Default)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    # Verify key hash against database
    key_hash = _hash_key(api_key)
    meta = await get_api_key_node(key_hash)

    if not meta:
        logger.warning("auth_invalid_key", key_prefix=api_key[:8])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or revoked API key",
        )

    # Isolate by key first, then by guest ID — so distinct keys never share
    # data, and distinct visitors sharing one key don't see each other's either.
    prefix = meta.get("prefix", "")
    meta["id"] = f"{prefix}:{client_id}" if client_id else prefix

    return meta


async def require_admin(
    meta: dict = Depends(require_api_key),
) -> dict:
    """FastAPI dependency: ensures the caller has the admin role."""
    if meta.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return meta
