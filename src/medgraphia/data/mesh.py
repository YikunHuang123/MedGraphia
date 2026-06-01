"""
MeSH (Medical Subject Headings) Open Data Loader.
Downloads and parses the publicly available MeSH ASCII descriptors.

Source: https://www.nlm.nih.gov/mesh/download_mesh.html
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import httpx

from medgraphia.domain import EntityType
from medgraphia.logger import get_logger

logger = get_logger(__name__)

# 2024 MeSH Descriptors (ASCII format)
MESH_ASCII_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/2024/asciimesh/d2024.bin"


class MeSHLoader:
    """
    Handles downloading and parsing of MeSH descriptors for entity linking.
    """

    def __init__(self, storage_dir: str = "data/mesh") -> None:
        self.storage_dir = Path(storage_dir)
        self.data_file = self.storage_dir / "d2024.bin"
        self._index: dict[str, dict[str, Any]] = {}

    async def ensure_data(self) -> None:
        """Download MeSH ASCII data if it doesn't exist locally."""
        if self.data_file.exists():
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("mesh_download_start", url=MESH_ASCII_URL)
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(MESH_ASCII_URL)
            response.raise_for_status()
            self.data_file.write_bytes(response.content)
            
        logger.info("mesh_download_complete", path=str(self.data_file))

    def load(self, limit: int | None = None) -> dict[str, dict[str, Any]]:
        """Parse the ASCII file and build a concept index."""
        if self._index:
            return self._index

        if not self.data_file.exists():
            raise FileNotFoundError(f"MeSH data not found at {self.data_file}. Call ensure_data() first.")

        content = self.data_file.read_text(encoding="utf-8", errors="replace")
        # MeSH ASCII records are separated by *NEWRECORD
        records = content.split("*NEWRECORD")
        
        count = 0
        for rec in records:
            if not rec.strip():
                continue

            # UI = Unique Identifier (e.g., D008687)
            ui_match = re.search(r"UI = (D\d+)", rec)
            # MH = MeSH Header (Preferred Label)
            mh_match = re.search(r"MH = (.+)", rec)
            
            if not ui_match or not mh_match:
                continue

            ui = ui_match.group(1)
            label = mh_match.group(1)
            
            # Entry terms (synonyms)
            # Format: ENTRY = Word|... or PRINT ENTRY = Word|...
            synonyms = re.findall(r"(?:PRINT )?ENTRY = ([^|]+)", rec)
            
            # Tree Numbers (used to determine EntityType)
            # C = Diseases, D = Chemicals and Drugs, F03 = Mental Disorders
            mns = re.findall(r"MN = ([A-Z][^ \n]+)", rec)
            
            entity_type = self._resolve_entity_type(mns)
            
            self._index[ui] = {
                "cui": ui,  # Use 'cui' key for compatibility with existing domain/linker
                "label": label,
                "synonyms": list(set(synonyms)),
                "entity_type": entity_type.value,
                "lang_labels": {},  # Core MeSH is English; multi-lang can be added via translations
            }
            
            count += 1
            if limit and count >= limit:
                break

        logger.info("mesh_loaded", concepts=len(self._index))
        return self._index

    def _resolve_entity_type(self, tree_numbers: list[str]) -> EntityType:
        """Map MeSH tree numbers to MedGraphia EntityTypes."""
        for tn in tree_numbers:
            if tn.startswith("C") or tn.startswith("F03"):
                return EntityType.DISEASE
            if tn.startswith("D"):
                return EntityType.DRUG
            if tn.startswith("G") or tn.startswith("A"): # Some genes/proteins are under G or A
                 if "gen" in tn.lower() or "prot" in tn.lower():
                     return EntityType.GENE
        return EntityType.UNKNOWN

    def iter_concepts(self) -> Iterator[dict[str, Any]]:
        yield from self._index.values()


# ---------------------------------------------------------------------------
# Module-level helper (importable by tests)
# ---------------------------------------------------------------------------

def _resolve_entity_type(tree_numbers: list[str]) -> str:
    """
    Map MeSH tree numbers to EntityType value string.
    Mirrors MeSHLoader._resolve_entity_type but as a standalone function
    so it can be imported and unit-tested without instantiating a loader.
    """
    for tn in tree_numbers:
        if tn.startswith("C") or tn.startswith("F03"):
            return EntityType.DISEASE.value
        if tn.startswith("D"):
            return EntityType.DRUG.value
        if (tn.startswith("G") or tn.startswith("A")) and (
            "gen" in tn.lower() or "prot" in tn.lower()
        ):
            return EntityType.GENE.value
    return EntityType.UNKNOWN.value
