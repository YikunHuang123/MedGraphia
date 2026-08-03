"""
Sleep/wake orchestration for vLLM engines backing the SMALL/MEDIUM LLM tiers.

Each vLLM-backed tier is its own persistent process started with
--enable-sleep-mode. This manager wakes an engine right before it's used
and puts idle engines back to sleep on a background timer, so only the
tier actually being served holds GPU memory.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from medgraphia.logger import get_logger

logger = get_logger(__name__)


class VLLMSleepManager:
    """Tracks last-use time per vLLM base_url and wakes/sleeps engines on demand."""

    def __init__(self, idle_seconds: int = 120) -> None:
        self._idle_seconds = idle_seconds
        self._last_used: dict[str, float] = {}
        self._monitor_task: asyncio.Task | None = None

    async def ensure_awake(self, base_url: str) -> None:
        """Wake the engine at base_url if it's asleep, and mark it as just used."""
        self._last_used[base_url] = time.monotonic()
        root = base_url.removesuffix("/v1")
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{root}/is_sleeping")
                resp.raise_for_status()
                if resp.json().get("is_sleeping"):
                    logger.info("vllm_waking", base_url=base_url)
                    t0 = time.monotonic()
                    await client.post(f"{root}/wake_up")
                    logger.info("vllm_woken", base_url=base_url, seconds=round(time.monotonic() - t0, 2))
            except httpx.HTTPError as exc:
                logger.warning("vllm_sleep_check_failed", base_url=base_url, error=str(exc))

    def ensure_awake_sync(self, base_url: str) -> None:
        """Sync counterpart for callers without an event loop (e.g. DSPy's LM cache path)."""
        self._last_used[base_url] = time.monotonic()
        root = base_url.removesuffix("/v1")
        with httpx.Client(timeout=30.0) as client:
            try:
                resp = client.get(f"{root}/is_sleeping")
                resp.raise_for_status()
                if resp.json().get("is_sleeping"):
                    logger.info("vllm_waking", base_url=base_url)
                    t0 = time.monotonic()
                    client.post(f"{root}/wake_up")
                    logger.info("vllm_woken", base_url=base_url, seconds=round(time.monotonic() - t0, 2))
            except httpx.HTTPError as exc:
                logger.warning("vllm_sleep_check_failed", base_url=base_url, error=str(exc))

    async def _sleep_if_awake(self, base_url: str) -> None:
        root = base_url.removesuffix("/v1")
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(f"{root}/is_sleeping")
                resp.raise_for_status()
                if not resp.json().get("is_sleeping"):
                    logger.info("vllm_sleeping_idle_engine", base_url=base_url)
                    await client.post(f"{root}/sleep?level=1")
            except httpx.HTTPError as exc:
                logger.warning("vllm_sleep_call_failed", base_url=base_url, error=str(exc))

    async def _monitor_loop(self, poll_seconds: int = 30) -> None:
        while True:
            await asyncio.sleep(poll_seconds)
            now = time.monotonic()
            for base_url, last_used in list(self._last_used.items()):
                if now - last_used >= self._idle_seconds:
                    await self._sleep_if_awake(base_url)
                    # Reset the clock so a sleeping engine isn't re-checked every poll.
                    self._last_used[base_url] = now

    def start_idle_monitor(self) -> None:
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("vllm_sleep_monitor_started", idle_seconds=self._idle_seconds)

    def stop_idle_monitor(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            self._monitor_task = None


_manager: VLLMSleepManager | None = None


def get_sleep_manager() -> VLLMSleepManager:
    """Process-wide singleton, sized from Settings on first use."""
    global _manager
    if _manager is None:
        from medgraphia.config import get_settings

        cfg = get_settings()
        _manager = VLLMSleepManager(idle_seconds=cfg.vllm_sleep_idle_seconds)
    return _manager
