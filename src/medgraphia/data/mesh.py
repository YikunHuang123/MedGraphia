"""
MeSH (Medical Subject Headings) Open Data Loader.
Downloads and parses the publicly available MeSH ASCII descriptors.

Source: https://www.nlm.nih.gov/mesh/download_mesh.html
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
            raise FileNotFoundError(
                f"MeSH data not found at {self.data_file}. Call ensure_data() first."
            )

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
        """Map MeSH tree numbers to MedGraphia EntityTypes.

        Scans ALL tree numbers first, then resolves by priority so that a concept
        with multiple tree numbers (e.g. Insulin: D12 protein + D06 hormone/drug)
        gets the most clinically useful type.

        MeSH tree prefixes:
          C23.888.* → SYMPTOM (Signs and Symptoms only; C23.550=Necrosis → DISEASE)
          C*, F03.* → DISEASE
          D12.*  → GENE      (Proteins as gene products; overridden by DRUG if D* also present)
          D*     → DRUG      (Chemicals and Drugs)
          E*     → PROCEDURE (Diagnostic and Therapeutic Techniques)
          G05.*  → GENE      (Genetic Phenomena)

        Priority: DRUG > SYMPTOM > DISEASE > PROCEDURE > GENE > UNKNOWN
          Rationale: a protein that is also a licensed drug (insulin, antibodies)
          should be findable as DRUG; C23 (Symptoms) is more specific than C (Diseases).
        """
        types_found: set[EntityType] = set()
        for tn in tree_numbers:
            if tn.startswith("C23.888"):
                types_found.add(EntityType.SYMPTOM)
            elif tn.startswith("C") or tn.startswith("F03"):
                types_found.add(EntityType.DISEASE)
            elif tn.startswith("G05"):
                types_found.add(EntityType.GENE)
            elif tn.startswith("D12"):
                types_found.add(EntityType.GENE)
            elif tn.startswith("D"):
                types_found.add(EntityType.DRUG)
            elif tn.startswith("E"):
                types_found.add(EntityType.PROCEDURE)

        for preferred in (
            EntityType.DRUG,
            EntityType.SYMPTOM,
            EntityType.DISEASE,
            EntityType.PROCEDURE,
            EntityType.GENE,
        ):
            if preferred in types_found:
                return preferred
        return EntityType.UNKNOWN

    def load_translations(self, translation_file: str) -> None:
        """Inject multilingual labels from a TSV file into the loaded concept index.

        Expected TSV format (no header):
            <CUI>\\t<lang>\\t<label>
        Example:
            D009203\\tzh\\t心肌梗死
            D009203\\tde\\tHerzinfarkt

        The file can be generated from UMLS MRCONSO.RRF:
            awk -F'|' '$2=="CHI" && $12=="MSH" {print $9"\\tzh\\t"$15}' MRCONSO.RRF > mesh_zh.tsv
            awk -F'|' '$2=="GER" && $12=="MSH" {print $9"\\tde\\t"$15}' MRCONSO.RRF >> mesh_zh.tsv
        """
        path = Path(translation_file)
        if not path.exists():
            logger.warning("mesh_translation_missing", path=str(path))
            return

        loaded = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                cui, lang, label = parts
                label = label.strip()
                if not label or cui not in self._index:
                    continue
                self._index[cui].setdefault("lang_labels", {})[lang] = label
                loaded += 1

        logger.info("mesh_translations_loaded", file=str(path), entries=loaded)

    def iter_concepts(self) -> Iterator[dict[str, Any]]:
        yield from self._index.values()


# ---------------------------------------------------------------------------
# Module-level helper (importable by tests)
# ---------------------------------------------------------------------------


def _resolve_entity_type(tree_numbers: list[str]) -> str:
    """Module-level wrapper — delegates to MeSHLoader._resolve_entity_type logic."""
    types_found: set[EntityType] = set()
    for tn in tree_numbers:
        if tn.startswith("C23"):
            types_found.add(EntityType.SYMPTOM)
        elif tn.startswith("C") or tn.startswith("F03"):
            types_found.add(EntityType.DISEASE)
        elif tn.startswith("G05"):
            types_found.add(EntityType.GENE)
        elif tn.startswith("D12"):
            types_found.add(EntityType.GENE)
        elif tn.startswith("D"):
            types_found.add(EntityType.DRUG)
        elif tn.startswith("E"):
            types_found.add(EntityType.PROCEDURE)
    for preferred in (
        EntityType.DRUG,
        EntityType.SYMPTOM,
        EntityType.DISEASE,
        EntityType.PROCEDURE,
        EntityType.GENE,
    ):
        if preferred in types_found:
            return preferred.value
    return EntityType.UNKNOWN.value
