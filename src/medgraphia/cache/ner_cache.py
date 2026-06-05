"""
Redis-backed cache for Query NER + Entity Linking results.

Cache key  : ner:{lang}:{hex16_of_sha256(NFC_lower(query))}
TTL        : NER_CACHE_TTL seconds (default 3 600 s = 1 hour)
Serialiser : JSON via orjson (fast, handles bytes)

On any Redis error the helpers are silent no-ops so the calling code
never needs to branch on cache availability.
"""
from __future__ import annotations

import hashlib
import unicodedata
from typing import TYPE_CHECKING

from medgraphia.cache.redis_client import get_redis
from medgraphia.domain.base import EntityType, Language
from medgraphia.domain.medical import Entity
from medgraphia.logger import get_logger

if TYPE_CHECKING:
    from medgraphia.retrieval.query_ner import QueryEntities

logger = get_logger(__name__)

NER_CACHE_TTL: int = 3600  # 1 hour
_PREFIX = "ner"


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _cache_key(query: str, lang: Language) -> str:
    normalised = unicodedata.normalize("NFC", query.strip().lower())
    digest = hashlib.sha256(normalised.encode()).hexdigest()[:16]
    return f"{_PREFIX}:{lang.value}:{digest}"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialise(result: "QueryEntities") -> str:
    import orjson  # already a project dependency

    entities_raw = [
        {
            "cui": e.cui,
            "label": e.label,
            "entity_type": e.entity_type.value,
            "lang_labels": e.lang_labels,
            "source_ids": e.source_ids,
            "confidence": e.confidence,
        }
        for e in result.entities
    ]
    payload = {
        "query": result.query,
        "language": result.language.value,
        "linked_cuis": result.linked_cuis,
        "unlinked_mentions": result.unlinked_mentions,
        "entities": entities_raw,
    }
    return orjson.dumps(payload).decode()


def _deserialise(data: str) -> "QueryEntities | None":
    try:
        import orjson
        from medgraphia.retrieval.query_ner import QueryEntities

        payload = orjson.loads(data)
        lang = Language(payload["language"])

        entities: list[Entity] = []
        for raw in payload.get("entities", []):
            try:
                entities.append(
                    Entity(
                        cui=raw["cui"],
                        label=raw["label"],
                        entity_type=EntityType(raw.get("entity_type", "Unknown")),
                        lang_labels=raw.get("lang_labels", {}),
                        source_ids=raw.get("source_ids", []),
                        confidence=raw.get("confidence", 1.0),
                    )
                )
            except Exception:
                # Skip individual malformed entity; keep the rest
                pass

        return QueryEntities(
            query=payload["query"],
            language=lang,
            entities=entities,
            linked_cuis=payload["linked_cuis"],
            unlinked_mentions=payload["unlinked_mentions"],
        )
    except Exception as exc:
        logger.warning("ner_cache_deserialise_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ner_cache_get(query: str, lang: Language) -> "QueryEntities | None":
    """Return a cached NER result, or None on cache miss / Redis unavailable."""
    redis = await get_redis()
    if redis is None:
        return None

    key = _cache_key(query, lang)
    try:
        data: str | None = await redis.get(key)
        if data is None:
            return None
        result = _deserialise(data)
        if result is not None:
            logger.debug("ner_cache_hit", lang=lang.value, linked=len(result.linked_cuis))
        return result
    except Exception as exc:
        logger.warning("ner_cache_get_error", error=str(exc))
        return None


async def ner_cache_set(query: str, lang: Language, result: "QueryEntities") -> None:
    """Persist a NER result in Redis, silently ignoring any errors."""
    redis = await get_redis()
    if redis is None:
        return

    key = _cache_key(query, lang)
    try:
        payload = _serialise(result)
        await redis.set(key, payload, ex=NER_CACHE_TTL)
        logger.debug("ner_cache_set", lang=lang.value, linked=len(result.linked_cuis))
    except Exception as exc:
        logger.warning("ner_cache_set_error", error=str(exc))
