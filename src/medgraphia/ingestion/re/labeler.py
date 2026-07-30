"""
Training data labeler for BERT relation extraction.

DataLabeler orchestrates three steps:
  1. fetch_chunks()       — pull candidate chunks from Neo4j
  2. label_chunk()        — call LLM extractor → positive (src, tgt, rel) triples
  3. build_labeled_pairs() — match pairs against LLM output; sample NONE negatives
"""

from __future__ import annotations

import random

from medgraphia.domain import Chunk, Entity, EntityType, Language, SourceMeta
from medgraphia.domain.base import RelationType
from medgraphia.ingestion.re._types import EntityPair, LabeledPair
from medgraphia.ingestion.re.pair_builder import PairBuilder
from medgraphia.ingestion.re.type_filter import filter_pairs, get_candidate_relations
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_LLM_ALLOWED = ", ".join(
    rt.value for rt in RelationType
    if rt not in {RelationType.MENTIONED_IN, RelationType.FROM_DOC, RelationType.CODED_AS}
)


async def fetch_chunks(lang_filter: str | None, limit: int) -> list[Chunk]:
    """Fetch chunks with ≥2 linked entities from Neo4j."""
    from medgraphia.graph.client import get_session

    lang_clause = "AND c.language = $lang" if lang_filter else ""
    # Collect entity node + MENTIONED_IN edge properties in one pass.
    # r.start_char / r.end_char are set by batch_upsert_entities_and_links() during
    # ingestion so pair_builder can inject markers without re-scanning chunk text.
    cypher = f"""
    MATCH (c:Chunk)
    WHERE (c.re_processed IS NULL OR c.re_processed = false)
      AND c.text IS NOT NULL
    {lang_clause}
    OPTIONAL MATCH (e)-[r:MENTIONED_IN]->(c)
    WITH c, collect({{
        node: e,
        mention_text: r.mention_text,
        start_char:   r.start_char,
        end_char:     r.end_char
    }}) AS entity_data
    WHERE size(entity_data) >= 2
    OPTIONAL MATCH (c)-[:FROM_DOC]->(d:Document)
    RETURN c, d, entity_data
    LIMIT $limit
    """
    params: dict = {"limit": limit}
    if lang_filter:
        params["lang"] = lang_filter

    chunks: list[Chunk] = []
    async with get_session() as session:
        result = await session.run(cypher, **params)
        async for record in result:
            c_node = record["c"]
            d_node = record["d"]
            entity_data = record["entity_data"]

            source = SourceMeta(
                source_id=c_node.get("source_id", "unknown"),
                source_title=d_node.get("source_title", "Unknown") if d_node else "Unknown",
                source_version=c_node.get("source_version", "1.0"),
            )
            entities: list[Entity] = []
            for item in entity_data:
                e = item["node"]
                if e is None:
                    continue
                cui = e.get("cui", "")
                label = e.get("label", "")
                if not cui or not label or cui.startswith("MENTION:"):
                    continue
                raw_labels = set(e.labels) - {"Entity"}
                e_type_str = next(iter(raw_labels), "Unknown")
                try:
                    e_type = EntityType(e_type_str)
                except ValueError:
                    e_type = EntityType.UNKNOWN
                lang_labels: dict[str, str] = {}
                if e.get("lang_zh"):
                    lang_labels["zh"] = e["lang_zh"]
                if e.get("lang_de"):
                    lang_labels["de"] = e["lang_de"]

                # Use span from MENTIONED_IN edge when available (set during ingestion).
                # -1 sentinel means "not stored" (pre-schema-fix data).
                start_char = item.get("start_char")
                end_char = item.get("end_char")
                if start_char == -1:
                    start_char = None
                if end_char == -1:
                    end_char = None

                entities.append(Entity(
                    cui=cui,
                    label=label,
                    entity_type=e_type,
                    lang_labels=lang_labels,
                    start_char=start_char,
                    end_char=end_char,
                ))

            if len(entities) < 2:
                continue
            chunks.append(Chunk(
                chunk_id=c_node["chunk_id"],
                doc_id=c_node.get("doc_id", ""),
                text=c_node["text"],
                language=Language(c_node.get("language", "en")),
                source=source,
                entities=entities,
            ))
    return chunks


class DataLabeler:
    """
    Generates LabeledPairs from chunks using an LLM to assign relation types.

    Usage::

        labeler = DataLabeler(none_ratio=2.0)
        llm_map = await labeler.label_chunk(chunk)
        pairs   = labeler.build_labeled_pairs(chunk, llm_map)
    """

    def __init__(self, none_ratio: float = 2.0, min_confidence: float = 0.5) -> None:
        self._none_ratio = none_ratio
        self._min_confidence = min_confidence
        self._pair_builder = PairBuilder()

    async def label_chunk(self, chunk: Chunk) -> dict[tuple[str, str], tuple[str, float]] | None:
        """
        Call DSPy extractor on a single chunk.
        Returns {(source_cui, target_cui): (relation_type, confidence)}.
        """
        import dspy

        from medgraphia.llm.dspy_setup import get_lm
        from medgraphia.programs.extractor import get_extractor

        entity_info = "\n".join(
            f"- {e.label} (CUI: {e.cui}, Type: {e.entity_type.value})"
            for e in chunk.entities
        )
        lm = get_lm("extractor")
        try:
            with dspy.context(lm=lm):
                prog = get_extractor()
                async_prog = dspy.asyncify(prog)
                prediction = await async_prog(
                    text_content=chunk.text,
                    entities=entity_info,
                    allowed_relations=_LLM_ALLOWED,
                )
            return {
                (r.source_cui, r.target_cui): (r.relation_type, r.confidence)
                for r in prediction.relations
                if r.confidence >= self._min_confidence
            }
        except Exception:
            logger.exception("llm_labeling_failed", chunk_id=chunk.chunk_id[:8])
            return None  # None = exception; {} = LLM ran but found no relations

    def build_labeled_pairs(
        self,
        chunk: Chunk,
        llm_map: dict[tuple[str, str], tuple[str, float]],
    ) -> list[LabeledPair]:
        """
        Match entity pairs from a chunk against LLM output.

        - Pairs with a valid LLM match → positive label
        - Remaining pairs             → NONE label (sampled to self._none_ratio)
        """
        valid_pairs = filter_pairs(self._pair_builder.build_pairs(chunk))
        if not valid_pairs:
            return []

        positives: list[LabeledPair] = []
        none_candidates: list[LabeledPair] = []

        for pair in valid_pairs:
            lp = self._make_labeled_pair(pair, chunk)
            key = (pair.source.cui, pair.target.cui)

            if key in llm_map:
                rel_type_str, conf = llm_map[key]
                valid_rels = {r.value for r in get_candidate_relations(
                    pair.source.entity_type, pair.target.entity_type
                )}
                if rel_type_str in valid_rels:
                    lp.label = rel_type_str
                    lp.confidence = conf
                    positives.append(lp)
                    continue

            none_candidates.append(lp)

        n_none = min(len(none_candidates), max(1, int(len(positives) * self._none_ratio)))
        sampled_none = random.sample(none_candidates, n_none) if none_candidates else []
        return positives + sampled_none

    @staticmethod
    def _make_labeled_pair(pair: EntityPair, chunk: Chunk) -> LabeledPair:
        return LabeledPair(
            marked_text=pair.marked_text,
            label="NONE",
            lang=chunk.language.value,
            source_cui=pair.source.cui,
            source_label=pair.source.label,
            source_type=pair.source.entity_type.value,
            target_cui=pair.target.cui,
            target_label=pair.target.label,
            target_type=pair.target.entity_type.value,
            chunk_id=chunk.chunk_id,
            confidence=0.0,
        )
