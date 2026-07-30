from __future__ import annotations

import itertools
import re

from medgraphia.domain import Chunk
from medgraphia.ingestion.re._types import EntityPair
from medgraphia.logger import get_logger

logger = get_logger(__name__)


class PairBuilder:
    """
    Generates all valid permutation pairs from a chunk's entities 
    and injects position markers into the text.
    """
    def __init__(self, source_marker: str = "[E1]", target_marker: str = "[E2]"):
        self.source_marker_start = source_marker
        self.source_marker_end = source_marker.replace("[", "[/")
        self.target_marker_start = target_marker
        self.target_marker_end = target_marker.replace("[", "[/")

    def build_pairs(self, chunk: Chunk) -> list[EntityPair]:
        pairs = []
        
        # 1. Resolve character offsets
        valid_entities = []
        for e in chunk.entities:
            start_char = e.start_char
            end_char = e.end_char
            
            # Fallback to string match if offsets are missing.
            # Try language-specific label first (zh/de stored as lang_labels on Entity),
            # then fall back to the canonical English label.
            # Both searches are case-insensitive.
            if start_char is None or end_char is None:
                chunk_lang = chunk.language.value if hasattr(chunk, "language") else "en"
                candidates: list[str] = []
                lang_label = e.lang_labels.get(chunk_lang, "") if e.lang_labels else ""
                if lang_label:
                    candidates.append(lang_label)
                candidates.append(e.label)

                text_lower = chunk.text.lower()
                start_char = -1
                for candidate in candidates:
                    candidate_lower = candidate.strip().lower()
                    if not candidate_lower:
                        continue
                    pattern = r'(?<!\w)' + re.escape(candidate_lower) + r'(?!\w)'
                    m = re.search(pattern, text_lower)
                    if m:
                        start_char = m.start()
                        end_char = m.start() + len(candidate)
                        break

                if start_char == -1:
                    logger.debug("re_entity_not_found_in_text", entity=e.label, lang=chunk_lang)
                    continue
            
            valid_entities.append((e, start_char, end_char))
            
        # 2. Generate Permutations (N * (N-1))
        # Important: Order matters for asymmetric relations like TREATS
        for (src_e, src_start, src_end), (tgt_e, tgt_start, tgt_end) in itertools.permutations(valid_entities, 2):
            if src_e.cui == tgt_e.cui:
                continue  # Skip self-loops
                
            # If the entities overlap, we skip (BERT typically struggles with overlapping markers)
            if max(src_start, tgt_start) < min(src_end, tgt_end):
                continue

            # 3. Inject markers from right to left (to avoid offset shifting)
            marks = [
                (src_start, self.source_marker_start),
                (src_end, self.source_marker_end),
                (tgt_start, self.target_marker_start),
                (tgt_end, self.target_marker_end)
            ]
            marks.sort(key=lambda x: x[0], reverse=True)
            
            marked_text = chunk.text
            for pos, mark in marks:
                marked_text = marked_text[:pos] + mark + marked_text[pos:]
                
            pairs.append(EntityPair(
                chunk=chunk,
                source=src_e,
                target=tgt_e,
                marked_text=marked_text
            ))
            
        return pairs
