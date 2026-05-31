"""
UMLS Metathesaurus loader.
Reads the MetamorphoSys RRF export (MRCONSO.RRF + MRSTY.RRF) and loads
concept names into Neo4j for entity-linking candidate retrieval.

Prerequisites:
  1. Obtain a UMLS license: https://uts.nlm.nih.gov/uts/
  2. Run MetamorphoSys to export a subset (SNOMED CT + RxNorm + MeSH, EN+ZH+DE)
  3. Place the RRF files under data/umls/ (e.g. data/umls/META/MRCONSO.RRF)

The loader creates a lightweight in-memory index:
  CUI → {preferred_label, synonyms, semantic_types, lang_labels}
and (optionally) writes Concept nodes to Neo4j for linking.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from medgraphia.logger import get_logger

logger = get_logger(__name__)

# Semantic type groups we care about (UMLS STY abbreviations → EntityType)
_STY_TO_ENTITY_TYPE: dict[str, str] = {
    "T047": "Disease",   # Disease or Syndrome
    "T048": "Disease",   # Mental or Behavioral Dysfunction
    "T191": "Disease",   # Neoplastic Process
    "T121": "Drug",      # Pharmacologic Substance
    "T109": "Drug",      # Organic Chemical (includes many drugs)
    "T184": "Symptom",   # Sign or Symptom
    "T033": "Symptom",   # Finding
    "T116": "Gene",      # Amino Acid, Peptide, or Protein
    "T028": "Gene",      # Gene or Genome
    "T061": "Procedure", # Therapeutic or Preventive Procedure
    "T059": "Procedure", # Laboratory Procedure
}

# Languages we want to load
_WANTED_LANGUAGES = {"ENG", "CHI", "GER"}
_LANG_MAP = {"ENG": "en", "CHI": "zh", "GER": "de"}


class UMLSLoader:
    """
    Streams MRCONSO.RRF and MRSTY.RRF to build an in-memory concept index.
    Suitable for a MetamorphoSys subset (a full load of UMLS is ~4 M concepts).
    """

    def __init__(self, umls_meta_dir: str = "data/umls/META") -> None:
        self._meta_dir = Path(umls_meta_dir)
        self._mrconso = self._meta_dir / "MRCONSO.RRF"
        self._mrsty = self._meta_dir / "MRSTY.RRF"
        self._index: dict[str, dict] = {}   # CUI → concept dict

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------

    def load(self, max_concepts: int | None = None) -> dict[str, dict]:
        """
        Parse MRCONSO + MRSTY and return the concept index.
        Call once; result is cached in self._index.
        """
        if self._index:
            return self._index

        # Step 1: load semantic types for each CUI
        sty_map: dict[str, list[str]] = defaultdict(list)
        if self._mrsty.exists():
            for cui, sty in _stream_mrsty(self._mrsty):
                sty_map[cui].append(sty)
        else:
            logger.warning("umls_mrsty_missing", path=str(self._mrsty))

        # Step 2: stream MRCONSO
        count = 0
        for entry in _stream_mrconso(self._mrconso):
            cui = entry["cui"]
            if cui not in self._index:
                # Resolve entity type from semantic types
                stys = sty_map.get(cui, [])
                entity_type = _resolve_entity_type(stys)
                self._index[cui] = {
                    "cui": cui,
                    "entity_type": entity_type,
                    "label": "",           # filled by English preferred name
                    "synonyms": [],
                    "lang_labels": {},
                }

            lang = entry.get("lang", "")
            text = entry.get("str", "")
            is_preferred = entry.get("ts") == "P" and entry.get("ispref") == "Y"

            if lang == "ENG":
                if is_preferred or not self._index[cui]["label"]:
                    self._index[cui]["label"] = text
                if text not in self._index[cui]["synonyms"]:
                    self._index[cui]["synonyms"].append(text)
            elif lang in ("CHI", "GER"):
                iso = _LANG_MAP[lang]
                if iso not in self._index[cui]["lang_labels"]:
                    self._index[cui]["lang_labels"][iso] = text

            count += 1
            if max_concepts and len(self._index) >= max_concepts:
                break

        logger.info("umls_loaded", concepts=len(self._index), rows_processed=count)
        return self._index

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_concept(self, cui: str) -> dict | None:
        return self._index.get(cui)

    def search_by_label(self, label: str, lang: str = "en") -> list[dict]:
        """Simple linear scan — replace with BM25 index in entity_linker.py."""
        label_lower = label.lower()
        results = []
        for concept in self._index.values():
            if lang == "en" and label_lower in (concept.get("label", "") or "").lower():
                results.append(concept)
            elif concept.get("lang_labels", {}).get(lang, "").lower() == label_lower:
                results.append(concept)
        return results[:50]

    def iter_concepts(self) -> Iterator[dict]:
        yield from self._index.values()


# ---------------------------------------------------------------------------
# RRF streaming helpers
# ---------------------------------------------------------------------------

def _stream_mrconso(path: Path) -> Iterator[dict]:
    """Stream MRCONSO.RRF rows, yielding dicts for wanted languages only."""
    # MRCONSO columns: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
    col_names = ["cui", "lat", "ts", "lui", "stt", "sui", "ispref",
                  "aui", "saui", "scui", "sdui", "sab", "tty", "code",
                  "str", "srl", "suppress", "cvf"]
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter="|")
        for row in reader:
            if len(row) < 15:
                continue
            lang = row[1]
            if lang not in _WANTED_LANGUAGES:
                continue
            suppress = row[16] if len(row) > 16 else ""
            if suppress == "O":  # obsolete
                continue
            entry = dict(zip(col_names, row))
            entry["lang"] = lang
            yield entry


def _stream_mrsty(path: Path) -> Iterator[tuple[str, str]]:
    """Stream MRSTY.RRF and yield (CUI, TUI) pairs."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                yield parts[0], parts[1]


def _resolve_entity_type(stys: list[str]) -> str:
    for sty in stys:
        if sty in _STY_TO_ENTITY_TYPE:
            return _STY_TO_ENTITY_TYPE[sty]
    return "Unknown"
