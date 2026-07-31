"""
Leiden community detection and LLM-based summarization.

Communities are detected over an entity co-occurrence graph — two entities
are linked if they're mentioned in the same chunk, weighted by how many
chunks they share — rather than typed relation edges (see project notes
"关系抽取阶段/4. 放弃类型化关系抽取..." for the rationale).
"""

from __future__ import annotations

import hashlib

import networkx as nx

from medgraphia.domain import Chunk, Community, Entity
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Try to import leidenalg
try:
    import leidenalg

    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


class CommunityBuilder:
    """
    Groups entities using community detection and summarizes them via DSPy.
    """

    def __init__(
        self,
        min_size: int = 3,
        resolution: float = 1.0,
    ) -> None:
        self._min_size = min_size
        self._resolution = resolution

    async def build_from_chunks(
        self,
        chunks: list[Chunk],
        entity_map: dict[str, Entity],
    ) -> list[Community]:
        """Build co-occurrence graph -> detect communities -> summarize."""
        if not chunks:
            return []

        graph = _build_cooccurrence_graph(chunks)
        if graph.number_of_nodes() == 0:
            return []
        partition = self._partition(graph)
        logger.info(
            "community_partition_done",
            total_nodes=graph.number_of_nodes(),
            communities=len(partition),
        )

        import asyncio

        async def process_community(member_cuis: set[str]) -> Community | None:
            if len(member_cuis) < self._min_size:
                return None

            subgraph_pairs = [
                (a, b, graph[a][b]["weight"])
                for a, b in graph.subgraph(member_cuis).edges()
            ]

            summary_text = await self._summarise(member_cuis, subgraph_pairs, entity_map)

            comm_id = _stable_community_id(member_cuis)
            return Community(
                community_id=comm_id,
                members=list(member_cuis),
                summary=summary_text,
                level=0,
            )

        tasks = [process_community(p) for p in partition]
        results = await asyncio.gather(*tasks)
        return [c for c in results if c is not None]

    def _partition(self, graph: nx.Graph) -> list[set[str]]:
        if HAS_LEIDEN:
            try:
                import igraph as ig

                g_ig = ig.Graph.from_networkx(graph)
                part = leidenalg.find_partition(
                    g_ig,
                    leidenalg.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=self._resolution,
                )
                return [{g_ig.vs[v]["_nx_name"] for v in p} for p in part if p]
            except Exception:
                pass
        from networkx.algorithms.community import louvain_communities

        return louvain_communities(graph, weight="weight", resolution=self._resolution)

    async def _summarise(
        self,
        member_cuis: set[str],
        cooccurrence_pairs: list[tuple[str, str, int]],
        entity_map: dict[str, Entity],
    ) -> str:
        """Use DSPy to generate a clinical summary."""
        import dspy

        from medgraphia.llm.dspy_setup import get_lm
        from medgraphia.programs.summarizer import get_summarizer

        lm = get_lm("summarizer")

        concept_lines = _format_concepts(member_cuis, entity_map)
        relation_lines = _format_cooccurrences(cooccurrence_pairs, entity_map)

        try:
            with dspy.context(lm=lm):
                # Use the centralized summarizer program (supports compiled few-shots)
                program = get_summarizer()
                prediction = await dspy.asyncify(program)(concepts=concept_lines, relations=relation_lines)
            data = prediction.result
            return f"{data.summary} {data.explanation} {data.clinical_relevance}"
        except Exception as exc:
            logger.warning("community_summary_failed", error=str(exc))
            return _auto_summary(member_cuis, entity_map)

    async def write_communities_to_neo4j(self, communities: list[Community]) -> None:
        """Write Community nodes to Neo4j."""
        if not communities:
            return
        try:
            from medgraphia.graph.queries import upsert_community

            for comm in communities:
                await upsert_community(
                    community_id=comm.community_id, summary=comm.summary, member_cuis=comm.members
                )
            logger.info("community_neo4j_written", count=len(communities))
        except Exception as exc:
            logger.warning("community_neo4j_failed", error=str(exc))

    @classmethod
    def from_settings(cls) -> CommunityBuilder:
        from medgraphia.config import get_settings

        cfg = get_settings()
        return cls(min_size=cfg.community_min_size, resolution=cfg.community_resolution)


def _build_cooccurrence_graph(chunks: list[Chunk]) -> nx.Graph:
    """Two entities are linked if they're mentioned in the same chunk; edge
    weight is the number of chunks they co-occur in."""
    graph = nx.Graph()
    for chunk in chunks:
        cuis = sorted({e.cui for e in chunk.entities if not e.cui.startswith("MENTION:")})
        for i in range(len(cuis)):
            for j in range(i + 1, len(cuis)):
                a, b = cuis[i], cuis[j]
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)
    return graph


def _format_concepts(cuis: set[str], entity_map: dict[str, Entity]) -> str:
    lines = []
    for cui in sorted(cuis):
        entity = entity_map.get(cui)
        if entity:
            lines.append(f"  [{cui}] {entity.label} ({entity.entity_type.value})")
        else:
            lines.append(f"  [{cui}]")
    return "\n".join(lines) or "  (no entity details available)"


def _format_cooccurrences(
    pairs: list[tuple[str, str, int]], entity_map: dict[str, Entity]
) -> str:
    if not pairs:
        return "  (no intra-community co-occurrences)"
    lines = []
    for src_cui, tgt_cui, weight in sorted(pairs, key=lambda p: -p[2])[:30]:
        src_label = entity_map.get(src_cui)
        tgt_label = entity_map.get(tgt_cui)
        src = src_label.label if src_label else src_cui
        tgt = tgt_label.label if tgt_label else tgt_cui
        lines.append(f"  {src} co-occurs with {tgt} (in {weight} shared chunk(s))")
    return "\n".join(lines)


def _auto_summary(member_cuis: set[str], entity_map: dict[str, Entity]) -> str:
    labels = [entity_map[c].label for c in sorted(member_cuis) if c in entity_map][:5]
    concept_str = ", ".join(labels) if labels else f"{len(member_cuis)} entities"
    return f"This cluster contains {len(member_cuis)} related medical concepts including {concept_str}."


def _stable_community_id(member_cuis: set[str]) -> str:
    key = "|".join(sorted(member_cuis))
    return "comm_" + hashlib.sha1(key.encode()).hexdigest()[:12]
