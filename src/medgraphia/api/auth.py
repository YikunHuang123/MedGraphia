"""
Authentication layer.
Supports multiple strategies: none, apikey, oidc.
"""
from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from medgraphia.config import get_settings
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory key store for apikey strategy.
# Key: raw key string; Value: {"role": "admin"|"user", "active": True}
# Bootstrap with ADMIN_BOOTSTRAP_KEY from config on startup.
_key_store: dict[str, dict] = {}


def _load_bootstrap_key() -> None:
    """Populate the key store with the admin bootstrap key from config."""
    cfg = get_settings()
    bootstrap = cfg.admin_bootstrap_key.get_secret_value()
    if bootstrap and bootstrap != "change-me":
        _key_store[bootstrap] = {"role": "admin", "active": True}


def generate_api_key() -> str:
    """Generate a cryptographically secure random API key."""
    return secrets.token_urlsafe(32)


def register_key(key: str, role: str = "user") -> None:
    """Add a key to the in-memory store (admin only, called from admin router)."""
    _key_store[key] = {"role": role, "active": True}


def revoke_key(key: str) -> None:
    _key_store.pop(key, None)


async def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> dict:
    """
    FastAPI dependency: validates auth based on settings.auth_strategy.
    Returns metadata about the authenticated principal.
    """
    cfg = get_settings()

    # Strategy: none (Permissive)
    if cfg.auth_strategy == "none":
        return {"role": "admin", "id": "anonymous", "strategy": "none"}

    # Strategy: oidc (Keycloak/SSO)
    if cfg.auth_strategy == "oidc":
        # TODO: Implement actual JWT validation against Keycloak in Phase 12
        # For now, this is a placeholder that enforces OIDC logic.
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OIDC authentication strategy is not yet implemented. Use 'apikey' for now.",
        )

    # Strategy: apikey (Default)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    meta = _key_store.get(api_key)
    if not meta or not meta.get("active"):
        logger.warning("auth_invalid_key", key_prefix=api_key[:8])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or revoked API key",
        )
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
