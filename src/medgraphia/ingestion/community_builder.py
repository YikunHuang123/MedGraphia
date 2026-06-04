"""
Leiden community detection and LLM-based summarization.
"""
from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx
from medgraphia.domain import Community, Entity, Relation
from medgraphia.logger import get_logger
from medgraphia.prompts import SummarizeMedicalCommunity, CommunitySummaryResult

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

    async def build_from_relations(
        self,
        relations: list[Relation],
        entity_map: dict[str, Entity],
    ) -> list[Community]:
        """Build graph -> detect communities -> summarize."""
        if not relations: return []

        graph = _build_graph(relations)
        partition = self._partition(graph)
        logger.info("community_partition_done", total_nodes=graph.number_of_nodes(), communities=len(partition))

        communities: list[Community] = []
        for member_cuis in partition:
            if len(member_cuis) < self._min_size: continue

            subgraph_relations = [
                r for r in relations
                if r.source_cui in member_cuis and r.target_cui in member_cuis
            ]

            summary_text = await self._summarise(member_cuis, subgraph_relations, entity_map)
            
            comm_id = _stable_community_id(member_cuis)
            communities.append(
                Community(
                    community_id=comm_id,
                    members=list(member_cuis),
                    summary=summary_text,
                    level=0,
                )
            )
        return communities

    def _partition(self, graph: nx.Graph) -> list[set[str]]:
        if HAS_LEIDEN:
            try:
                import igraph as ig
                g_ig = ig.Graph.from_networkx(graph)
                part = leidenalg.find_partition(g_ig, leidenalg.RBConfigurationVertexPartition, resolution_parameter=self._resolution)
                return [set(graph.nodes)[set(p)] for p in part if p]
            except Exception: pass
        from networkx.algorithms.community import louvain_communities
        return louvain_communities(graph, resolution=self._resolution)

    async def _summarise(self, member_cuis: set[str], relations: list[Relation], entity_map: dict[str, Entity]) -> str:
        """Use DSPy to generate a clinical summary."""
        import dspy
        from medgraphia.llm.dspy_setup import get_lm
        lm = get_lm("summarizer")

        concept_lines = _format_concepts(member_cuis, entity_map)
        relation_lines = _format_relations(relations, entity_map)

        try:
            with dspy.context(lm=lm):
                # Use dspy.Predict for Pydantic Signature in DSPy 3.x
                predictor = dspy.Predict(SummarizeMedicalCommunity)
                prediction = predictor(concepts=concept_lines, relations=relation_lines)
            data = prediction.result
            return f"{data.summary} {data.explanation} {data.clinical_relevance}"
        except Exception as exc:
            logger.warning("community_summary_failed", error=str(exc))
            return _auto_summary(member_cuis, relations, entity_map)

    async def write_communities_to_neo4j(self, communities: list[Community]) -> None:
        """Write Community nodes to Neo4j."""
        if not communities: return
        try:
            from medgraphia.graph.queries import upsert_community
            for comm in communities:
                await upsert_community(community_id=comm.community_id, summary=comm.summary, member_cuis=comm.members)
            logger.info("community_neo4j_written", count=len(communities))
        except Exception as exc:
            logger.warning("community_neo4j_failed", error=str(exc))

    @classmethod
    def from_settings(cls) -> "CommunityBuilder":
        from medgraphia.config import get_settings
        cfg = get_settings()
        return cls(min_size=cfg.community_min_size, resolution=cfg.community_resolution)

def _build_graph(relations: list[Relation]) -> nx.Graph:
    graph = nx.Graph()
    for rel in relations:
        graph.add_edge(rel.source_cui, rel.target_cui, relation_type=rel.relation_type.value, confidence=rel.confidence)
    return graph

def _format_concepts(cuis: set[str], entity_map: dict[str, Entity]) -> str:
    lines = []
    for cui in sorted(cuis):
        entity = entity_map.get(cui)
        if entity: lines.append(f"  [{cui}] {entity.label} ({entity.entity_type.value})")
        else: lines.append(f"  [{cui}]")
    return "\n".join(lines) or "  (no entity details available)"

def _format_relations(relations: list[Relation], entity_map: dict[str, Entity]) -> str:
    if not relations: return "  (no intra-community relations)"
    lines = []
    for rel in relations[:30]:
        src_label = entity_map.get(rel.source_cui)
        tgt_label = entity_map.get(rel.target_cui)
        src = src_label.label if src_label else rel.source_cui
        tgt = tgt_label.label if tgt_label else rel.target_cui
        lines.append(f"  {src} --[{rel.relation_type.value}]--> {tgt}" + (f": \"{rel.evidence_text[:60]}...\"" if rel.evidence_text else ""))
    return "\n".join(lines)

def _auto_summary(member_cuis: set[str], relations: list[Relation], entity_map: dict[str, Entity]) -> str:
    labels = [entity_map[c].label for c in sorted(member_cuis) if c in entity_map][:5]
    rel_types = list({r.relation_type.value for r in relations})[:3]
    concept_str = ", ".join(labels) if labels else f"{len(member_cuis)} entities"
    rel_str = ", ".join(rel_types) if rel_types else "various"
    return f"This cluster contains {len(member_cuis)} related medical concepts including {concept_str}. Key relationships include: {rel_str}."

def _stable_community_id(member_cuis: set[str]) -> str:
    key = "|".join(sorted(member_cuis))
    return "comm_" + hashlib.sha1(key.encode()).hexdigest()[:12]
