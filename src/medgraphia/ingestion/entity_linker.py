"""
Entity Linker: maps NER mention entities (cui="MENTION:...") to MeSH IDs via
SapBERT dense retrieval over the full multilingual MeSH dictionary.

Cross-lingual alignment goal:
  "心肌梗死" / "myocardial infarction" / "Myokardinfarkt"  →  MeSH D0027051

Degradation:
  - MeSH index empty  → return entities unchanged (provisional MENTION: CUIs kept)
  - SapBERT unavail.  → fall back to difflib string similarity
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
    import numpy as _np
    from sentence_transformers import SentenceTransformer as _ST  # type: ignore[import]

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
# MeSH concept index entry
# ---------------------------------------------------------------------------


class _ConceptEntry:
    __slots__ = ("cui", "label", "synonyms", "entity_type", "lang_labels")

    def __init__(self, concept: dict[str, Any]) -> None:
        self.cui: str = concept["cui"]
        self.label: str = concept.get("label") or ""
        self.synonyms: list[str] = concept.get("synonyms") or []
        self.entity_type: str = concept.get("entity_type") or "Unknown"
        self.lang_labels: dict[str, str] = concept.get("lang_labels") or {}


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
        link_threshold: float = 0.70,
        sapbert_model: str = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        sapbert_threshold: float = 0.75,
    ) -> None:

        self._raw_index: dict[str, Any] = concept_index or {}
        self._link_threshold = link_threshold
        self._sapbert_model_name = sapbert_model
        self._sapbert_threshold = sapbert_threshold

        # Built lazily
        self._entries: list[_ConceptEntry] = []
        self._sapbert: Any = None
        self._concept_embs: Any = None
        self._built = False

        # Global cache to prevent redundant cross-batch computations
        self._global_link_cache: dict[tuple[str, EntityType], tuple[str, str, dict[str, str], float] | None] = {}

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
        translation_file: str | None = None,
        **kwargs: Any,
    ) -> EntityLinker:
        """Load MeSH data and return a ready-to-use EntityLinker.

        Args:
            translation_file: Optional TSV file with multilingual labels.
                Format: <CUI>\\t<lang>\\t<label>  (e.g. D009203\\tzh\\t心肌梗死)
                Generate from UMLS MRCONSO.RRF:
                  awk -F'|' '$2=="CHI"&&$12=="MSH"{print $9"\\tzh\\t"$15}' MRCONSO.RRF > mesh_translations.tsv
                  awk -F'|' '$2=="GER"&&$12=="MSH"{print $9"\\tde\\t"$15}' MRCONSO.RRF >> mesh_translations.tsv
        """
        from medgraphia.data.mesh import MeSHLoader

        loader = MeSHLoader(storage_dir=mesh_dir)
        try:
            index = loader.load(limit=limit)
            logger.info("el_mesh_loaded", concepts=len(index))
        except Exception as exc:
            logger.warning("el_mesh_load_failed", error=str(exc))
            index = {}

        if translation_file:
            loader.load_translations(translation_file)

        # Use provided sapbert settings or fall back to defaults
        if sapbert_model is not None:
            kwargs["sapbert_model"] = sapbert_model
        if sapbert_threshold is not None:
            kwargs["sapbert_threshold"] = sapbert_threshold

        return cls(concept_index=index, **kwargs)

    @classmethod
    def from_settings(cls) -> EntityLinker:
        from medgraphia.config import get_settings

        cfg = get_settings()
        return cls.from_mesh(
            mesh_dir=cfg.mesh_dir,
            link_threshold=cfg.el_link_threshold,
            sapbert_model=cfg.el_sapbert_model,
            sapbert_threshold=cfg.el_sapbert_threshold,
            translation_file=cfg.el_translation_file or None,
        )

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """
        Build the SapBERT index from the MeSH concept map.
        Idempotent — does nothing if already built.
        """
        if self._built:
            return

        if not self._raw_index:
            logger.warning("el_index_empty", msg="Entity linker has no concepts; linking disabled")
            self._built = True
            return

        logger.info("el_building_index", concepts=len(self._raw_index))
        base_entries = [_ConceptEntry(c) for c in self._raw_index.values()]

        # Build a flat (entry, label) list covering English + all available lang_labels.
        # SapBERT all-lang model embeds multilingual text into the same space, so a
        # Chinese mention "心肌梗死" will directly match the Chinese dictionary entry
        # for D009203 without any cross-lingual model inference.
        index_pairs: list[tuple[_ConceptEntry, str]] = []
        for entry in base_entries:
            index_pairs.append((entry, entry.label))          # English always included
            for label in entry.lang_labels.values():
                if label and label != entry.label:
                    index_pairs.append((entry, label))        # zh / de / etc.

        self._entries = [pair[0] for pair in index_pairs]    # parallel list of entries
        all_labels   = [pair[1] for pair in index_pairs]

        n_base = len(base_entries)
        n_total = len(self._entries)
        logger.info("el_index_multilingual", base=n_base, total=n_total,
                    extra=n_total - n_base)

        if _SAPBERT_AVAILABLE and self._entries:
            self._load_sapbert()
            logger.info("el_sapbert_precomputing_dictionary")
            self._concept_embs = self._sapbert.encode(
                all_labels, normalize_embeddings=True, show_progress_bar=True,
                batch_size=256, convert_to_tensor=True
            )
            logger.info("el_sapbert_dictionary_ready")

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

            raw_mention = entity.cui[len(_MENTION_PREFIX) :]
            if ":" in raw_mention:
                _, mention_text = raw_mention.split(":", 1)
            else:
                mention_text = raw_mention
                
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
                        start_char=entity.start_char,
                        end_char=entity.end_char,
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
                linked.append(entity)
                logger.debug("el_unlinked", mention=mention_text)

        return linked

    def link_chunk(self, chunk: Chunk) -> Chunk:
        """Link all entities in a chunk and return a new Chunk (original unchanged)."""
        if not chunk.entities:
            return chunk
        linked = self.link_entities(chunk.entities)
        return chunk.model_copy(update={"entities": linked})

    def link_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Link all entities across all chunks using a single SapBERT encode call.

        Deduplicates by (mention_text, entity_type) so identical surface forms
        appearing in multiple chunks are encoded only once.
        """
        if not self._built:
            self.build_index()
        if not self._entries:
            return chunks

        unique_mentions: set[tuple[str, EntityType]] = set()
        
        for chunk in chunks:
            for entity in chunk.entities:
                if not entity.cui.startswith(_MENTION_PREFIX):
                    continue
                raw = entity.cui[len(_MENTION_PREFIX):]
                _, mention_text = raw.split(":", 1) if ":" in raw else ("", raw)
                key = (mention_text, entity.entity_type)
                
                if key in self._global_link_cache:
                    continue
                    
                unique_mentions.add(key)

        if unique_mentions:
            if _SAPBERT_AVAILABLE and self._sapbert is not None:
                self._batch_sapbert_link(list(unique_mentions), self._global_link_cache)
            else:
                # SapBERT unavailable — fall back to string rerank against full entry list
                for key in unique_mentions:
                    mention_text, entity_type = key
                    sr = self._string_rerank(mention_text, self._entries)
                    self._global_link_cache[key] = sr if (sr and sr[3] >= self._link_threshold) else None

        result: list[Chunk] = []
        for chunk in chunks:
            if not chunk.entities:
                result.append(chunk)
                continue
            linked_entities: list[Entity] = []
            for entity in chunk.entities:
                if not entity.cui.startswith(_MENTION_PREFIX):
                    linked_entities.append(entity)
                    continue
                raw = entity.cui[len(_MENTION_PREFIX):]
                _, mention_text = raw.split(":", 1) if ":" in raw else ("", raw)
                best = self._global_link_cache.get((mention_text, entity.entity_type))
                if best:
                    cui, label, lang_labels, confidence = best
                    linked_entities.append(
                        Entity(
                            cui=cui,
                            label=label,
                            entity_type=entity.entity_type,
                            lang_labels=lang_labels,
                            confidence=min(entity.confidence, confidence),
                            source_ids=entity.source_ids,
                            start_char=entity.start_char,
                            end_char=entity.end_char,
                        )
                    )
                    logger.debug("el_linked", mention=mention_text, cui=cui, score=f"{confidence:.3f}")
                else:
                    linked_entities.append(entity)
                    logger.debug("el_unlinked", mention=mention_text)
            result.append(chunk.model_copy(update={"entities": linked_entities}))

        return result

    def _batch_sapbert_link(
        self,
        unique_mentions: list[tuple[str, EntityType]],
        link_cache: dict[tuple[str, EntityType], tuple[str, str, dict[str, str], float] | None],
    ) -> None:
        """Encode mentions and perform GPU matrix multiplication against pre-computed concept dict."""
        import torch
        mention_texts = [k[0] for k in unique_mentions]

        try:
            mention_embs = self._sapbert.encode(
                mention_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=256, convert_to_tensor=True
            )

            scores = torch.matmul(mention_embs, self._concept_embs.T)
            topk_scores, topk_idxs = torch.topk(scores, k=50, dim=1)

            topk_scores = topk_scores.cpu().tolist()
            topk_idxs = topk_idxs.cpu().tolist()

            for i, key in enumerate(unique_mentions):
                mention_text, required_type = key
                cands = []
                for idx, score in zip(topk_idxs[i], topk_scores[i]):
                    entry = self._entries[idx]
                    if entry.entity_type == required_type.value:
                        cands.append((entry, score))

                if not cands:
                    # Fallback: if no candidates match the required type, use the raw top 50
                    for idx, score in zip(topk_idxs[i], topk_scores[i]):
                        cands.append((self._entries[idx], score))

                if not cands:
                    link_cache[key] = None
                    continue

                best_entry, best_score = cands[0]

                if best_score >= self._sapbert_threshold:
                    link_cache[key] = (best_entry.cui, best_entry.label, best_entry.lang_labels, best_score)
                elif not mention_text.isascii() and best_score >= self._link_threshold:
                    # Cross-script query (Chinese, German umlauts, etc.): string rerank is
                    # meaningless across different writing systems, so accept the SapBERT
                    # match if it clears the relaxed link_threshold.
                    link_cache[key] = (best_entry.cui, best_entry.label, best_entry.lang_labels, best_score)
                else:
                    cand_entries = [c[0] for c in cands]
                    sr = self._string_rerank(mention_text, cand_entries)
                    link_cache[key] = sr if (sr and sr[3] >= self._link_threshold) else None

        except Exception as exc:
            logger.warning("el_batch_sapbert_failed", error=str(exc))
            for key in unique_mentions:
                link_cache[key] = None

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

    def search_concepts(
        self, query: str, limit: int = 10, entity_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Perform an in-memory vector search for concepts matching the query.
        Returns a list of dicts with cui, label, entity_type, and score.
        """
        if not self._entries or self._concept_embs is None:
            return []

        import torch
        emb = self._sapbert.encode([query], normalize_embeddings=True, convert_to_tensor=True)
        scores = torch.matmul(emb, self._concept_embs.T)[0]

        if entity_type:
            # Need to search more broadly to filter by type
            topk_scores, topk_idxs = torch.topk(scores, k=min(200, len(self._entries)))
            topk_scores = topk_scores.cpu().tolist()
            topk_idxs = topk_idxs.cpu().tolist()

            results = []
            for idx, score in zip(topk_idxs, topk_scores):
                entry = self._entries[idx]
                if entry.entity_type == entity_type:
                    results.append({
                        "cui": entry.cui,
                        "label": entry.label,
                        "entity_type": entry.entity_type,
                        "score": score,
                        "lang_labels": entry.lang_labels
                    })
                    if len(results) >= limit:
                        break
            return results
        else:
            topk_scores, topk_idxs = torch.topk(scores, k=min(limit, len(self._entries)))
            topk_scores = topk_scores.cpu().tolist()
            topk_idxs = topk_idxs.cpu().tolist()

            return [
                {
                    "cui": self._entries[idx].cui,
                    "label": self._entries[idx].label,
                    "entity_type": self._entries[idx].entity_type,
                    "score": score,
                    "lang_labels": self._entries[idx].lang_labels
                }
                for idx, score in zip(topk_idxs, topk_scores)
            ]

    def _find_best_match(
        self,
        mention: str,
        entity_type: EntityType,
    ) -> tuple[str, str, dict[str, str], float] | None:
        """
        Return (cui, label, lang_labels, score) for the best UMLS match, or None.
        (Used mainly as a fallback for 1-by-1 linking)
        """
        if not self._entries or self._concept_embs is None:
            return None

        # Dense retrieval for single mention
        import torch
        emb = self._sapbert.encode([mention], normalize_embeddings=True, convert_to_tensor=True)
        scores = torch.matmul(emb, self._concept_embs.T)[0]
        topk_scores, topk_idxs = torch.topk(scores, k=50)

        topk_scores = topk_scores.cpu().tolist()
        topk_idxs = topk_idxs.cpu().tolist()

        cands = []
        for idx, score in zip(topk_idxs, topk_scores):
            entry = self._entries[idx]
            if entry.entity_type == entity_type.value:
                cands.append((entry, score))

        if not cands:
            # Fallback: no type-matched candidates in top-50, use all top-50
            cands = [(self._entries[idx], score) for idx, score in zip(topk_idxs, topk_scores)]

        if not cands:
            return None

        best_entry, best_score = cands[0]
        if best_score >= self._sapbert_threshold:
            return best_entry.cui, best_entry.label, best_entry.lang_labels, float(best_score)

        if not mention.isascii() and best_score >= self._link_threshold:
            # Cross-script query: string rerank is meaningless across different writing
            # systems, so accept the SapBERT match if it clears the relaxed link_threshold.
            return best_entry.cui, best_entry.label, best_entry.lang_labels, float(best_score)

        cand_entries = [c[0] for c in cands]
        return self._string_rerank(mention, cand_entries)

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
