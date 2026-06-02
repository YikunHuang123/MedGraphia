"""
Entity Linker: maps NER mention entities (cui="MENTION:...") to MeSH IDs.

Two-stage pipeline (architecture doc §2.4):
  Stage 1 — BM25 candidate retrieval (top-K lexical candidates)
  Stage 2 — SapBERT re-ranking (cross-lingual semantic similarity)

Cross-lingual alignment goal:
  "心肌梗死" / "myocardial infarction" / "Myokardinfarkt"  →  MeSH D0027051

Lite mode (SapBERT not installed):
  BM25 score + difflib SequenceMatcher for string similarity.

Enterprise mode (sentence-transformers available):
  BM25 top-K → SapBERT cosine similarity → keep best candidate above threshold.

Degradation:
  - MeSH index empty   → return entities unchanged (provisional MENTION: CUIs kept)
  - SapBERT unavail.  → fall back to BM25 + string distance
  - Neo4j unavail.    → entities are linked in-memory but not written to the graph
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from medgraphia.domain import Chunk, Entity, EntityType
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from rank_bm25 import BM25Okapi as _BM25  # type: ignore[import]
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logger.warning("rank_bm25_not_installed", msg="pip install rank-bm25 for candidate retrieval")

try:
    from sentence_transformers import SentenceTransformer as _ST  # type: ignore[import]
    import numpy as _np
    _SAPBERT_AVAILABLE = True
except ImportError:
    _SAPBERT_AVAILABLE = False
    logger.warning(
        "sentence_transformers_not_installed",
        msg="pip install sentence-transformers for SapBERT re-ranking",
    )

# Prefix used by the NER pipeline for unlinked mentions
_MENTION_PREFIX = "MENTION:"


# ---------------------------------------------------------------------------
# Text tokenizer (BM25 tokenization)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Tokenize for BM25 indexing.
    Western text: split on whitespace + punctuation.
    CJK text:    character-level tokens (each character is a 'term').
    Mixed:       combine both approaches.
    """
    tokens: list[str] = []
    # Western words (lowercase)
    tokens.extend(re.findall(r"[a-zA-ZäöüÄÖÜß]+", text.lower()))
    # CJK characters
    tokens.extend(re.findall(r"[一-鿿㐀-䶿]", text))
    return tokens or [text.lower()]


# ---------------------------------------------------------------------------
# MeSH concept index entry
# ---------------------------------------------------------------------------

class _ConceptEntry:
    __slots__ = ("cui", "label", "synonyms", "entity_type", "lang_labels", "all_tokens")

    def __init__(self, concept: dict[str, Any]) -> None:
        self.cui: str = concept["cui"]
        self.label: str = concept.get("label") or ""
        self.synonyms: list[str] = concept.get("synonyms") or []
        self.entity_type: str = concept.get("entity_type") or "Unknown"
        self.lang_labels: dict[str, str] = concept.get("lang_labels") or {}

        # Pre-tokenized corpus string used by BM25
        all_text = " ".join([self.label] + self.synonyms + list(self.lang_labels.values()))
        self.all_tokens: list[str] = _tokenize(all_text)


# ---------------------------------------------------------------------------
# EntityLinker
# ---------------------------------------------------------------------------

class EntityLinker:
    """
    Links Entity objects (cui="MENTION:...") to real MeSH IDs.

    Typical lifecycle::

        linker = EntityLinker.from_mesh("data/mesh")
        linker.build_index()
        linked_chunk = linker.link_chunk(chunk)

        # or batch:
        linked_chunks = [linker.link_chunk(c) for c in chunks]

        # optional: write entities to Neo4j
        await linker.write_entities_to_neo4j(linked_chunk)
    """

    def __init__(
        self,
        concept_index: dict[str, Any] | None = None,
        bm25_top_k: int = 50,
        link_threshold: float = 0.70,
        sapbert_model: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        sapbert_threshold: float = 0.75,
    ) -> None:

        self._raw_index: dict[str, Any] = concept_index or {}
        self._bm25_top_k = bm25_top_k
        self._link_threshold = link_threshold
        self._sapbert_model_name = sapbert_model
        self._sapbert_threshold = sapbert_threshold

        # Built lazily
        self._entries: list[_ConceptEntry] = []
        self._bm25: Any = None
        self._sapbert: Any = None
        self._built = False

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_mesh(
        cls,
        mesh_dir: str = "data/mesh",
        limit: int | None = None,
        sapbert_model: str | None = None,
        sapbert_threshold: float | None = None,
        **kwargs: Any,
    ) -> "EntityLinker":
        """Load MeSH data and return a ready-to-use EntityLinker."""
        from medgraphia.data.mesh import MeSHLoader
        loader = MeSHLoader(storage_dir=mesh_dir)
        try:
            index = loader.load(limit=limit)
            logger.info("el_mesh_loaded", concepts=len(index))
        except Exception as exc:
            logger.warning("el_mesh_load_failed", error=str(exc))
            index = {}
        
        # Use provided sapbert settings or fall back to defaults
        if sapbert_model: kwargs["sapbert_model"] = sapbert_model
        if sapbert_threshold: kwargs["sapbert_threshold"] = sapbert_threshold
            
        return cls(concept_index=index, **kwargs)

    @classmethod
    def from_settings(cls) -> "EntityLinker":
        from medgraphia.config import get_settings
        cfg = get_settings()
        return cls.from_mesh(
            mesh_dir=cfg.mesh_dir,
            bm25_top_k=cfg.el_bm25_top_k,
            link_threshold=cfg.el_link_threshold,
            sapbert_model=cfg.el_sapbert_model,
            sapbert_threshold=cfg.el_sapbert_threshold,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """
        Build the BM25 (and optionally SapBERT) index from the MeSH concept map.
        Idempotent — does nothing if already built.
        """
        if self._built:
            return

        if not self._raw_index:
            logger.warning("el_index_empty", msg="Entity linker has no concepts; linking disabled")
            self._built = True
            return

        logger.info("el_building_index", concepts=len(self._raw_index))
        self._entries = [_ConceptEntry(c) for c in self._raw_index.values()]

        if _BM25_AVAILABLE and self._entries:
            corpus = [e.all_tokens for e in self._entries]
            self._bm25 = _BM25(corpus)
            logger.info("el_bm25_built", terms=len(corpus))

        self._built = True

    def _load_sapbert(self) -> None:
        if self._sapbert is not None or not _SAPBERT_AVAILABLE:
            return
        try:
            logger.info("el_sapbert_loading", model=self._sapbert_model_name)
            self._sapbert = _ST(self._sapbert_model_name)
            logger.info("el_sapbert_loaded")
        except Exception as exc:
            logger.warning("el_sapbert_load_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Linking
    # ------------------------------------------------------------------

    def link_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        Resolve provisional MENTION: CUIs to real UMLS CUIs.
        Entities already bearing a real CUI (no MENTION: prefix) are kept unchanged.
        """
        if not self._built:
            self.build_index()

        linked: list[Entity] = []
        for entity in entities:
            if not entity.cui.startswith(_MENTION_PREFIX):
                linked.append(entity)
                continue

            mention_text = entity.cui[len(_MENTION_PREFIX):]
            best = self._find_best_match(mention_text, entity.entity_type)
            if best:
                cui, label, lang_labels, confidence = best
                linked.append(
                    Entity(
                        cui=cui,
                        label=label,
                        entity_type=entity.entity_type,
                        lang_labels=lang_labels,
                        confidence=min(entity.confidence, confidence),
                        source_ids=entity.source_ids,
                    )
                )
                logger.debug(
                    "el_linked",
                    mention=mention_text,
                    cui=cui,
                    label=label,
                    score=f"{confidence:.3f}",
                )
            else:
                # Keep provisional CUI — downstream can still index the mention
                linked.append(entity)
                logger.debug("el_unlinked", mention=mention_text)

        return linked

    def link_chunk(self, chunk: Chunk) -> Chunk:
        """Link all entities in a chunk and return a new Chunk (original unchanged)."""
        if not chunk.entities:
            return chunk
        linked = self.link_entities(chunk.entities)
        return chunk.model_copy(update={"entities": linked})

    # ------------------------------------------------------------------
    # Neo4j write (graceful degradation)
    # ------------------------------------------------------------------

    async def write_entities_to_neo4j(self, chunk: Chunk) -> None:
        """
        Write linked entities + MENTIONED_IN edges to Neo4j.
        Skips silently if Neo4j is unavailable.
        Entities with provisional MENTION: CUIs are also written so they can be
        retroactively linked when UMLS data arrives later.
        """
        if not chunk.entities:
            return
        try:
            from medgraphia.graph.queries import link_entity_to_chunk, merge_entity
            for entity in chunk.entities:
                await merge_entity(entity)
                await link_entity_to_chunk(entity.cui, entity.entity_type.value, chunk.chunk_id)
        except Exception as exc:
            logger.warning(
                "el_neo4j_write_failed",
                chunk_id=chunk.chunk_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Candidate retrieval + scoring
    # ------------------------------------------------------------------

    def _find_best_match(
        self,
        mention: str,
        entity_type: EntityType,
    ) -> tuple[str, str, dict[str, str], float] | None:
        """
        Return (cui, label, lang_labels, score) for the best UMLS match, or None.
        """
        if not self._entries:
            return None

        candidates = self._bm25_candidates(mention)
        if not candidates:
            return None

        # Type filter: prefer same-type concepts but fall back to any type if none match
        typed = [e for e in candidates if e.entity_type == entity_type.value]
        pool = typed if typed else candidates

        return self._score_and_rank(mention, pool)

    def _bm25_candidates(self, mention: str) -> list[_ConceptEntry]:
        """Return top-K BM25 candidates for a mention string."""
        query_tokens = _tokenize(mention)

        if self._bm25 is not None and _BM25_AVAILABLE:
            scores = self._bm25.get_scores(query_tokens)
            top_k_idx = _top_k_indices(scores, self._bm25_top_k)
            return [self._entries[i] for i in top_k_idx if scores[i] > 0]

        # Fallback: linear scan with string containment
        mention_lower = mention.lower()
        results = []
        for entry in self._entries:
            if mention_lower in entry.label.lower():
                results.append(entry)
            elif any(mention_lower in syn.lower() for syn in entry.synonyms):
                results.append(entry)
            if len(results) >= self._bm25_top_k:
                break
        return results

    def _score_and_rank(
        self,
        mention: str,
        candidates: list[_ConceptEntry],
    ) -> tuple[str, str, dict[str, str], float] | None:
        """
        Re-rank candidates using SapBERT (if available) or string similarity.
        Returns best (cui, label, lang_labels, score) if score >= threshold, else None.
        """
        if not candidates:
            return None

        if _SAPBERT_AVAILABLE:
            self._load_sapbert()

        if self._sapbert is not None:
            return self._sapbert_rerank(mention, candidates)
        else:
            return self._string_rerank(mention, candidates)

    def _sapbert_rerank(
        self,
        mention: str,
        candidates: list[_ConceptEntry],
    ) -> tuple[str, str, dict[str, str], float] | None:
        """SapBERT cosine similarity re-ranking."""
        try:
            cand_labels = [e.label for e in candidates]
            all_texts = [mention] + cand_labels
            embeddings = self._sapbert.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
            mention_emb = embeddings[0]
            cand_embs = embeddings[1:]
            scores = (cand_embs @ mention_emb).tolist()

            best_idx = int(_np.argmax(scores))
            best_score = float(scores[best_idx])

            if best_score < self._sapbert_threshold:
                # Below SapBERT threshold — try string fallback
                string_result = self._string_rerank(mention, candidates)
                if string_result and string_result[3] >= self._link_threshold:
                    return string_result
                return None

            best = candidates[best_idx]
            return best.cui, best.label, best.lang_labels, best_score

        except Exception as exc:
            logger.warning("el_sapbert_rerank_failed", error=str(exc))
            return self._string_rerank(mention, candidates)

    def _string_rerank(
        self,
        mention: str,
        candidates: list[_ConceptEntry],
    ) -> tuple[str, str, dict[str, str], float] | None:
        """
        Fallback: normalized edit-distance similarity via difflib.SequenceMatcher.
        Also considers all synonyms, not just the primary label.
        """
        mention_lower = mention.lower()
        best_score = 0.0
        best_entry: _ConceptEntry | None = None

        for entry in candidates:
            all_names = [entry.label] + entry.synonyms + list(entry.lang_labels.values())
            for name in all_names:
                if not name:
                    continue
                sim = SequenceMatcher(None, mention_lower, name.lower()).ratio()
                if sim > best_score:
                    best_score = sim
                    best_entry = entry

        if best_entry is None or best_score < self._link_threshold:
            return None
        return best_entry.cui, best_entry.label, best_entry.lang_labels, best_score


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _top_k_indices(scores: Any, k: int) -> list[int]:
    """Return indices of top-k scores (descending order).  Works with numpy arrays."""
    try:
        import numpy as np
        arr = np.array(scores)
        k = min(k, len(arr))
        idx = np.argpartition(arr, -k)[-k:]
        return idx[np.argsort(arr[idx])[::-1]].tolist()
    except Exception:
        # Pure-Python fallback
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])
        return [i for i, _ in indexed[:k]]
