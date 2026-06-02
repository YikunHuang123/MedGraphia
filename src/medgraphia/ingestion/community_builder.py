"""
Leiden community detection and LLM-based summarization using pydantic-ai.

Hierarchical GraphRAG support (architecture doc §2.6).
Groups entities into semantic clusters and generates clinical summaries.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Type

import networkx as nx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from medgraphia.domain import Community, Entity, Relation
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Try to import leidenalg, fallback to Louvain
try:
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------

class CommunitySummaryResult(BaseModel):
    """Structured clinical summary of a knowledge graph community."""
    summary: str = Field(description="A 2-3 sentence overview of the clinical focus of this community")
    explanation: str = Field(description="Brief explanation of why these concepts are grouped together")
    clinical_relevance: str = Field(description="Key clinical implications or takeaway")

# ---------------------------------------------------------------------------
# CommunityBuilder
# ---------------------------------------------------------------------------

class CommunityBuilder:
    """
    Groups entities using community detection and summarizes them via LLM.
    """

    def __init__(
        self,
        model: Model,
        min_size: int = 3,
        resolution: float = 1.0,
    ) -> None:
        self._min_size = min_size
        self._resolution = resolution
        
        # Initialize the pydantic-ai Agent for summarization
        self._agent: Agent[None, CommunitySummaryResult] = Agent(
            model,
            output_type=CommunitySummaryResult,
            system_prompt=(
                "You are a medical research assistant. You will be provided with a list of "
                "medical concepts and their relationships within a specific knowledge community. "
                "Your task is to provide a concise, clinically accurate summary of this community. "
                "Focus on the primary medical theme connecting these entities."
            )
        )

    async def build_from_relations(
        self,
        relations: list[Relation],
        entity_map: dict[str, Entity],
    ) -> list[Community]:
        """
        Main entry point: build graph -> detect communities -> summarize.
        """
        if not relations:
            return []

        # 1. Build graph
        graph = _build_graph(relations)
        
        # 2. Partition
        partition = self._partition(graph)
        logger.info("community_partition_done", total_nodes=graph.number_of_nodes(), communities=len(partition))

        # 3. Summarize each community
        communities: list[Community] = []
        for i, member_cuis in enumerate(partition):
            if len(member_cuis) < self._min_size:
                continue

            # Get relations local to this community
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
                    level=0,  # Single-level for now
                )
            )

        return communities

    def _partition(self, graph: nx.Graph) -> list[set[str]]:
        """Partition the graph using Leiden or Louvain."""
        if HAS_LEIDEN:
            try:
                import igraph as ig
                # Convert networkx to igraph
                g_ig = ig.Graph.from_networkx(graph)
                part = leidenalg.find_partition(
                    g_ig, leidenalg.RBConfigurationVertexPartition, resolution_parameter=self._resolution
                )
                return [set(graph.nodes)[set(p)] for p in part if p]
            except Exception as exc:
                logger.warning("leiden_failed_falling_back", error=str(exc))

        # Fallback to Louvain
        try:
            from networkx.algorithms.community import louvain_communities
            return louvain_communities(graph, resolution=self._resolution)
        except Exception:
            # Last resort: connected components
            return [set(c) for c in nx.connected_components(graph)]

    async def _summarise(
        self,
        member_cuis: set[str],
        relations: list[Relation],
        entity_map: dict[str, Entity],
    ) -> str:
        """Use pydantic-ai to generate a clinical summary."""
        concept_lines = _format_concepts(member_cuis, entity_map)
        relation_lines = _format_relations(relations, entity_map)

        user_prompt = (
            f"CONCEPTS IN THIS CLUSTER ({len(member_cuis)}):\n{concept_lines}\n\n"
            f"KEY RELATIONS:\n{relation_lines}\n\n"
            "Provide a concise clinical summary."
        )

        try:
            result = await self._agent.run(user_prompt)
            data = result.output
            # Combine components into a single summary string for the Community domain model
            return f"{data.summary} {data.explanation} {data.clinical_relevance}"
        except Exception as exc:
            logger.warning("community_summary_failed", error=str(exc))
            return _auto_summary(member_cuis, relations, entity_map)

    async def write_communities_to_neo4j(self, communities: list[Community]) -> None:
        """Write Community nodes and member relationships to Neo4j."""
        if not communities:
            return
        try:
            from medgraphia.graph.queries import upsert_community
            for comm in communities:
                await upsert_community(
                    community_id=comm.community_id,
                    summary=comm.summary,
                    member_cuis=comm.members
                )
            logger.info("community_neo4j_written", count=len(communities))
        except Exception as exc:
            logger.warning("community_neo4j_failed", error=str(exc))

    @classmethod
    def from_settings(cls) -> "CommunityBuilder":
        """Construct from the global Settings singleton."""
        from medgraphia.config import get_settings
        from medgraphia.llm.client import get_model

        cfg = get_settings()
        # Use a specific model for summaries if configured, otherwise default
        model = get_model(model_override=cfg.community_summary_llm or None)
        
        return cls(
            model=model,
            min_size=cfg.community_min_size,
            resolution=cfg.community_resolution
        )


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _build_graph(relations: list[Relation]) -> nx.Graph:
    """Build an undirected NetworkX graph from Relation objects."""
    graph = nx.Graph()
    for rel in relations:
        graph.add_edge(
            rel.source_cui,
            rel.target_cui,
            relation_type=rel.relation_type.value,
            confidence=rel.confidence,
        )
    return graph


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_concepts(cuis: set[str], entity_map: dict[str, Entity]) -> str:
    lines = []
    for cui in sorted(cuis):
        entity = entity_map.get(cui)
        if entity:
            lines.append(f"  [{cui}] {entity.label} ({entity.entity_type.value})")
        else:
            lines.append(f"  [{cui}]")
    return "\n".join(lines) or "  (no entity details available)"


def _format_relations(relations: list[Relation], entity_map: dict[str, Entity]) -> str:
    if not relations:
        return "  (no intra-community relations)"
    lines = []
    # Cap relations for prompt length
    for rel in relations[:30]:
        src_label = entity_map.get(rel.source_cui)
        tgt_label = entity_map.get(rel.target_cui)
        src = src_label.label if src_label else rel.source_cui
        tgt = tgt_label.label if tgt_label else rel.target_cui
        lines.append(
            f"  {src} --[{rel.relation_type.value}]--> {tgt}"
            + (f": \"{rel.evidence_text[:60]}...\"" if rel.evidence_text else "")
        )
    return "\n".join(lines)


def _auto_summary(
    member_cuis: set[str],
    relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    """Template-based fallback summary when LLM is unavailable."""
    labels = [
        entity_map[c].label for c in sorted(member_cuis) if c in entity_map
    ][:5]
    rel_types = list({r.relation_type.value for r in relations})[:3]

    concept_str = ", ".join(labels) if labels else f"{len(member_cuis)} entities"
    rel_str = ", ".join(rel_types) if rel_types else "various"
    return (
        f"This cluster contains {len(member_cuis)} related medical concepts "
        f"including {concept_str}. "
        f"Key relationships include: {rel_str}."
    )


def _stable_community_id(member_cuis: set[str]) -> str:
    """Generate a deterministic community ID from sorted member CUIs."""
    key = "|".join(sorted(member_cuis))
    return "comm_" + hashlib.sha1(key.encode()).hexdigest()[:12]
