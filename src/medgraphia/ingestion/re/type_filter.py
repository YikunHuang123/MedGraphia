"""
Entity type pair filter for BERT relation extraction.

Defines which (source_type, target_type) combinations are medically plausible
and which RelationTypes are candidates for each combination.

Pairs NOT in VALID_PAIR_RELATIONS are discarded before BERT inference —
this eliminates ~60-70% of entity pairs and avoids overwhelming the classifier
with semantically impossible combinations (e.g. Symptom→Drug).
"""

from __future__ import annotations

from medgraphia.domain.base import EntityType, RelationType
from medgraphia.ingestion.re._types import EntityPair

# RE-relevant relation types only (excludes structural edges: MENTIONED_IN, FROM_DOC, CODED_AS)
_RE_RELATIONS = frozenset(RelationType) - {
    RelationType.MENTIONED_IN,
    RelationType.FROM_DOC,
    RelationType.CODED_AS,
}

VALID_PAIR_RELATIONS: dict[tuple[EntityType, EntityType], frozenset[RelationType]] = {
    # Drug as source
    (EntityType.DRUG, EntityType.DISEASE): frozenset({
        RelationType.TREATS,
        RelationType.CAUSES,
        RelationType.DOSAGE_FOR,
        RelationType.CONTRAINDICATED_IN,
    }),
    (EntityType.DRUG, EntityType.SYMPTOM): frozenset({
        RelationType.TREATS,
        RelationType.CAUSES,
    }),
    (EntityType.DRUG, EntityType.DRUG): frozenset({
        RelationType.INTERACTS_WITH,
    }),
    # Disease as source
    (EntityType.DISEASE, EntityType.SYMPTOM): frozenset({
        RelationType.CAUSES,
    }),
    (EntityType.DISEASE, EntityType.DISEASE): frozenset({
        RelationType.COMPLICATION_OF,
        RelationType.ASSOCIATED_WITH,
    }),

    # Symptom as source
    (EntityType.SYMPTOM, EntityType.DISEASE): frozenset({
        RelationType.SYMPTOM_OF,
    }),
    (EntityType.SYMPTOM, EntityType.ANATOMY): frozenset({
        RelationType.ASSOCIATED_WITH,
    }),

    # Gene as source/target — now that Anatomy is its own type, GLiNER no longer
    # misclassifies body parts as Gene, so these pairs are safe to enable.
    (EntityType.GENE, EntityType.DISEASE): frozenset({
        RelationType.ASSOCIATED_WITH,
    }),
    (EntityType.DRUG, EntityType.GENE): frozenset({
        RelationType.INTERACTS_WITH,
    }),

    # Anatomy as target — where in the body a disease/procedure/symptom applies
    (EntityType.DISEASE, EntityType.ANATOMY): frozenset({
        RelationType.ASSOCIATED_WITH,
    }),
    (EntityType.PROCEDURE, EntityType.ANATOMY): frozenset({
        RelationType.ASSOCIATED_WITH,
    }),

    # Living being (pathogen) as source — infectious disease relations
    (EntityType.LIVING_BEING, EntityType.DISEASE): frozenset({
        RelationType.CAUSES,
    }),
    (EntityType.DRUG, EntityType.LIVING_BEING): frozenset({
        RelationType.INTERACTS_WITH,
    }),
}

# Flat set of all valid (src_type, tgt_type) keys — used for O(1) lookup
_VALID_KEYS: frozenset[tuple[EntityType, EntityType]] = frozenset(VALID_PAIR_RELATIONS)


def is_valid_pair(src_type: EntityType, tgt_type: EntityType) -> bool:
    return (src_type, tgt_type) in _VALID_KEYS


def get_candidate_relations(
    src_type: EntityType, tgt_type: EntityType
) -> frozenset[RelationType]:
    return VALID_PAIR_RELATIONS.get((src_type, tgt_type), frozenset())


def filter_pairs(pairs: list[EntityPair]) -> list[EntityPair]:
    """Return only pairs whose entity type combination has at least one plausible relation."""
    return [
        p for p in pairs
        if is_valid_pair(p.source.entity_type, p.target.entity_type)
    ]
