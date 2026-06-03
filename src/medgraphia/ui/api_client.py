"""
Thin HTTP/SSE client for the MedGraphia FastAPI backend.

All UI pages access the backend exclusively through `MedGraphiaClient`.
No business logic should ever be re-implemented on the UI side.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=120.0)
STREAM_TIMEOUT = httpx.Timeout(30.0, read=300.0)


@dataclass
class APIError(Exception):
    """Raised when the backend returns a non-2xx response."""

    status_code: int
    detail: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.status_code}] {self.detail}"


class MedGraphiaClient:
    """Synchronous HTTP wrapper with connection pooling."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        admin_key: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("API_BASE_URL", "http://localhost:8058")).rstrip("/")
        self.api_key = api_key or os.getenv("MEDGRAPHIA_API_KEY", "")
        self.admin_key = admin_key or os.getenv("MEDGRAPHIA_ADMIN_KEY", "")
        
        # Persistent client for connection pooling
        self.client = httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def __enter__(self) -> "MedGraphiaClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Header helpers
    # ------------------------------------------------------------------

    def _headers(self, admin: bool = False) -> dict[str, str]:
        """
        Build headers. If admin=True, use admin_key.
        If admin=False, use api_key, falling back to admin_key if api_key is missing.
        """
        if admin:
            key = self.admin_key
        else:
            # Admin key can encompass all user operations
            key = self.api_key or self.admin_key

        headers = {"Accept": "application/json"}
        if key:
            headers["X-API-Key"] = key
        return headers

    # ------------------------------------------------------------------
    # Generic request wrapper
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        admin: bool = False,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self.client.request(
                method,
                url,
                headers=self._headers(admin=admin),
                json=json_body,
                params=params,
            )
        except httpx.RequestError as exc:
            raise APIError(status_code=0, detail=f"Network error: {exc}") from exc

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(status_code=resp.status_code, detail=str(detail))

        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_live(self) -> dict[str, str]:
        return self._request("GET", "/health/live")

    def health_ready(self) -> dict[str, str]:
        return self._request("GET", "/health/ready")

    # ------------------------------------------------------------------
    # Knowledge graph
    # ------------------------------------------------------------------

    def search_entities(
        self, q: str, limit: int = 10, entity_type: str | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"q": q, "limit": limit}
        if entity_type:
            params["entity_type"] = entity_type
        return self._request("GET", "/graph/entity/search", params=params) or []

    def get_entity_subgraph(
        self, name: str, hops: int = 1, lang: str = "en"
    ) -> dict[str, Any]:
        params = {"name": name, "hops": hops, "lang": lang}
        return self._request("GET", "/graph/entity", params=params)

    def graph_stats(self) -> dict[str, int]:
        return self._request("GET", "/graph/stats")

    # ------------------------------------------------------------------
    # Chat (synchronous)
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        language: str = "en",
        domain: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "message": message,
            "session_id": session_id,
            "language": language,
            "domain": domain,
            "stream": False,
        }
        return self._request("POST", "/chat", json_body=body)

    # ------------------------------------------------------------------
    # Chat (SSE streaming) — synchronous generator
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        message: str,
        session_id: str | None = None,
        language: str = "en",
        domain: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield parsed SSE events. Each event is a dict with `type` key:
        `chunk` | `citations` | `done` | `error`.
        """
        body = {
            "message": message,
            "session_id": session_id,
            "language": language,
            "domain": domain,
            "stream": True,
        }
        url = f"{self.base_url}/chat/stream"
        try:
            with self.client.stream(
                "POST", url, headers=self._headers(), json=body, timeout=STREAM_TIMEOUT
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise APIError(
                        status_code=resp.status_code,
                        detail=resp.text or "Stream rejected",
                    )
                
                # Buffer for multi-line data concatenation
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    
                    # Robust SSE parsing: check for data: prefix
                    if raw_line.startswith("data:"):
                        payload = raw_line[len("data:"):].lstrip()
                        if not payload:
                            continue
                        try:
                            yield json.loads(payload)
                        except json.JSONDecodeError:
                            # Fallback if the data isn't pure JSON
                            yield {"type": "chunk", "content": payload}
                    elif raw_line.startswith(":"):
                        # SSE Comment / Heartbeat (ignore)
                        continue
        except httpx.RequestError as exc:
            yield {"type": "error", "detail": f"Network error: {exc}"}

    # ------------------------------------------------------------------
    # Admin — Pipeline
    # ------------------------------------------------------------------

    def trigger_pipeline(
        self,
        domain: str,
        pubmed_query: str | None = None,
        pubmed_limit: int = 200,
        include_drugbank: bool = False,
        include_ema_smpc: bool = False,
    ) -> dict[str, str]:
        body = {
            "domain": domain,
            "pubmed_query": pubmed_query,
            "pubmed_limit": pubmed_limit,
            "include_drugbank": include_drugbank,
            "include_ema_smpc": include_ema_smpc,
        }
        return self._request(
            "POST", "/admin/pipeline/trigger", admin=True, json_body=body
        )

    def pipeline_status(self, domain: str = "t2dm") -> dict[str, Any]:
        return self._request(
            "GET", "/admin/pipeline/status", admin=True, params={"domain": domain}
        )

    def admin_graph_stats(self) -> dict[str, int]:
        return self._request("GET", "/admin/graph/stats", admin=True)

    # ------------------------------------------------------------------
    # Admin — API Keys
    # ------------------------------------------------------------------

    def create_api_key(self, role: str = "user") -> dict[str, str]:
        return self._request(
            "POST", "/admin/keys", admin=True, params={"role": role}
        )

    def list_api_keys(self) -> list[dict[str, str]]:
        return self._request("GET", "/admin/keys", admin=True) or []

    def revoke_api_key(self, prefix: str) -> dict[str, Any]:
        return self._request("DELETE", f"/admin/keys/{prefix}", admin=True)
