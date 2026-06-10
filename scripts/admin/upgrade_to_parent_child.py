"""
One-time migration: backfill `parent_text` on existing Qdrant chunk points.

Two-pass strategy
-----------------
Pass 1 — Source-file re-parse (high quality, zero LLM cost)
  For each known source type we re-parse the original file to get the full,
  un-truncated section text and write it as parent_text.

  Supported source types
  ~~~~~~~~~~~~~~~~~~~~~~
  dailymed:{set_id}   → data/raw/fda_dailymed/{set_id}.xml  (FDA SPL XML)
  ema:{product_name}  → data/raw/ema_smpc/*.pdf             (Docling PDF parse)
  huatuo_{id}         → data/raw/huatuo/huatuo_lite.jsonl
  germed_{hash}       → data/raw/germed/GERNERMED_dataset.json

Pass 2 — Chunk-concatenation fallback
  Any source_id that could not be re-parsed (e.g. pubmed, no local file) gets
  a parent_text built by concatenating all child chunk texts sorted by page /
  char_offset.  Old chunks were stored with text[:1000] so this is less
  accurate, but still better than a single 300-token child.

Usage
-----
    python scripts/admin/upgrade_to_parent_child.py
    python scripts/admin/upgrade_to_parent_child.py --dry-run
    python scripts/admin/upgrade_to_parent_child.py --collection my_collection
    python scripts/admin/upgrade_to_parent_child.py --data-root /custom/data/raw
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(_SRC))

import medgraphia.logger as _log_module  # noqa: E402

class _Q:
    def info(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass

_log_module.get_logger = lambda _: _Q()

from medgraphia.config import get_settings          # noqa: E402
from medgraphia.vector.qdrant_store import QdrantStore  # noqa: E402

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
# source_id → section_path → full section content
SectionMap = dict[str, dict[str, str]]


# ===========================================================================
# Source-file parsers (re-parse only, no NER / embedding)
# ===========================================================================

def _parse_dailymed_xml(xml_path: Path) -> dict[str, str]:
    """Return {section_title: full_content} for one FDA SPL XML file."""
    ns = "urn:hl7-org:v3"
    root = ET.parse(xml_path).getroot()
    sections: dict[str, str] = {}
    for section_el in root.findall(f".//{{{ns}}}section"):
        title_node = section_el.find(f"./{{{ns}}}title")
        section_title = (
            title_node.text.strip()
            if title_node is not None and title_node.text
            else ""
        )
        text_node = section_el.find(f"./{{{ns}}}text")
        if text_node is not None:
            content = "".join(text_node.itertext()).strip()
            if section_title and content:
                # A parent section and a child section may share the same title;
                # keep the longer version (child sections are always more complete).
                if len(content) > len(sections.get(section_title, "")):
                    sections[section_title] = content
    return sections


def _build_dailymed_map(data_root: Path) -> SectionMap:
    """Parse all cached FDA DailyMed XMLs → {source_id → {section_path → text}}."""
    fda_dir = data_root / "fda_dailymed"
    if not fda_dir.is_dir():
        return {}

    result: SectionMap = {}
    xmls = list(fda_dir.glob("*.xml"))
    print(f"  [dailymed] Parsing {len(xmls)} XML files …", flush=True)
    for xml_path in xmls:
        set_id = xml_path.stem
        source_id = f"dailymed:{set_id}"
        try:
            result[source_id] = _parse_dailymed_xml(xml_path)
        except Exception as exc:
            print(f"    WARN: failed to parse {xml_path.name}: {exc}", flush=True)
    print(f"  [dailymed] Loaded {len(result)} documents.", flush=True)
    return result


def _build_ema_map(data_root: Path) -> SectionMap:
    """Parse all EMA SmPC PDFs with Docling → {source_id → {section_path → text}}."""
    ema_dir = data_root / "ema_smpc"
    if not ema_dir.is_dir():
        return {}

    pdfs = list(ema_dir.glob("*.pdf"))
    if not pdfs:
        return {}

    try:
        from medgraphia.ingestion.parsers.docling_parser import DoclingParser
        from medgraphia.domain import Language, SourceMeta
        from datetime import datetime, UTC
    except Exception as exc:
        print(f"  [ema] Docling not available, skipping: {exc}", flush=True)
        return {}

    parser = DoclingParser()
    result: SectionMap = {}
    print(f"  [ema] Parsing {len(pdfs)} PDF files with Docling …", flush=True)
    for pdf_path in pdfs:
        product_name = pdf_path.stem.lower().replace(" ", "_")
        source_id = f"ema:{product_name}"
        try:
            src = SourceMeta(
                source_id=source_id,
                source_title=pdf_path.stem,
                retrieved_at=datetime.now(UTC),
            )
            doc = parser.parse(str(pdf_path), src, language=Language.EN)
            result[source_id] = {s.section_path: s.content for s in doc.sections if s.content}
        except Exception as exc:
            print(f"    WARN: failed to parse {pdf_path.name}: {exc}", flush=True)
    print(f"  [ema] Loaded {len(result)} documents.", flush=True)
    return result


def _build_huatuo_map(data_root: Path) -> SectionMap:
    """Load Huatuo QA → {source_id → {section_path → text}}."""
    import json
    jsonl = data_root / "huatuo" / "huatuo_lite.jsonl"
    if not jsonl.exists():
        return {}

    result: SectionMap = {}
    print(f"  [huatuo] Scanning {jsonl.name} …", flush=True)
    count = 0
    with open(jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            doc_id = f"huatuo_{data.get('id', i)}"
            result[doc_id] = {
                "Question": data.get("question", ""),
                "Answer": data.get("answer", ""),
                "Related Diseases": data.get("related_diseases", ""),
            }
            count += 1
    print(f"  [huatuo] Loaded {count} documents.", flush=True)
    return result


def _build_germed_map(data_root: Path) -> SectionMap:
    """Load GERNERMED → {source_id → {section_path → text}}."""
    import json, hashlib
    json_file = data_root / "germed" / "GERNERMED_dataset.json"
    if not json_file.exists():
        return {}

    result: SectionMap = {}
    print(f"  [germed] Scanning {json_file.name} …", flush=True)
    with open(json_file, encoding="utf-8") as f:
        records = json.load(f)

    for rec in records:
        de_text = rec.get("de", "")
        en_text = rec.get("en", "")
        if not de_text and not en_text:
            continue
        text_hash = hashlib.md5(de_text.encode()).hexdigest()[:12]
        doc_id = f"germed_{text_hash}"
        result[doc_id] = {
            "de": de_text,
            "en": en_text,
        }
    print(f"  [germed] Loaded {len(result)} documents.", flush=True)
    return result


# ===========================================================================
# Fallback: build parent_text from (possibly truncated) Qdrant chunk texts
# ===========================================================================

def _concat_chunks(chunks: list[dict[str, Any]]) -> str:
    """Merge child chunk texts sorted by page / char_offset."""
    def _key(c: dict) -> tuple:
        page = c.get("page")
        offset = c.get("char_offset")
        return (page is None, page or 0, offset is None, offset or 0)

    ordered = sorted(chunks, key=_key)
    return "\n\n".join(c["text"] for c in ordered if c.get("text"))


# ===========================================================================
# Main
# ===========================================================================

async def _scroll_all(client: Any, collection: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = None
    print(f"Scrolling collection '{collection}' …", flush=True)
    while True:
        result, next_offset = await client.scroll(
            collection_name=collection,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in result:
            payload = dict(point.payload or {})
            payload["_point_id"] = point.id
            records.append(payload)
        if next_offset is None:
            break
        offset = next_offset
    print(f"  → {len(records)} points loaded.", flush=True)
    return records


async def main(collection: str, dry_run: bool, batch_size: int, data_root: Path) -> None:
    store = QdrantStore()
    cfg = get_settings()
    col = collection or cfg.qdrant_collection_chunks

    # ------------------------------------------------------------------
    # 1. Load Qdrant points
    # ------------------------------------------------------------------
    records = await _scroll_all(store._client, col)
    if not records:
        print("Collection is empty — nothing to do.")
        return

    # ------------------------------------------------------------------
    # 2. Build source maps by re-parsing local files (Pass 1)
    # ------------------------------------------------------------------
    print(f"\nBuilding section maps from source files under '{data_root}' …", flush=True)
    source_maps: SectionMap = {}
    source_maps.update(_build_dailymed_map(data_root))
    source_maps.update(_build_ema_map(data_root))
    source_maps.update(_build_huatuo_map(data_root))
    source_maps.update(_build_germed_map(data_root))
    print(f"  → Re-parsed {len(source_maps)} total source documents.\n", flush=True)

    # ------------------------------------------------------------------
    # 3. Group Qdrant points by (source_id, section_path)
    # ------------------------------------------------------------------
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (rec.get("source_id") or "", rec.get("section_path") or "")
        groups[key].append(rec)

    # ------------------------------------------------------------------
    # 4. Build updates
    # ------------------------------------------------------------------
    updates: list[tuple[str, dict[str, Any]]] = []
    skipped = 0
    fallback_count = 0
    reparsed_count = 0

    for (source_id, section_path), chunks in groups.items():
        # --- Try re-parsed source first (Pass 1) ---
        parent_text = ""
        sections_for_source = source_maps.get(source_id, {})
        if sections_for_source and section_path:
            parent_text = sections_for_source.get(section_path, "")

        # --- Fall back to chunk concatenation (Pass 2) ---
        if not parent_text:
            parent_text = _concat_chunks(chunks)
            if parent_text:
                fallback_count += len(chunks)
        else:
            reparsed_count += len(chunks)

        if not parent_text:
            continue

        for chunk in chunks:
            point_id = str(chunk["_point_id"])
            if (chunk.get("parent_text") or "") == parent_text:
                skipped += 1
                continue
            updates.append((point_id, {"parent_text": parent_text}))

    print(
        f"Summary:\n"
        f"  Re-parsed (full text from source file) : {reparsed_count} points\n"
        f"  Fallback  (chunk concat, may be partial): {fallback_count} points\n"
        f"  Already up-to-date, skipped            : {skipped} points\n"
        f"  Total to update                        : {len(updates)} points",
        flush=True,
    )

    if not updates:
        print("\nNothing to update — done.")
        return

    # ------------------------------------------------------------------
    # 5. Apply or dry-run
    # ------------------------------------------------------------------
    if dry_run:
        print("\n[DRY RUN] First 3 updates:")
        for point_id, patch in updates[:3]:
            preview = patch["parent_text"][:300].replace("\n", " ")
            print(f"  point={point_id}\n  parent_text[:300]={preview!r}\n")
        print(f"[DRY RUN] Would update {len(updates)} points. No changes written.")
        return

    print(f"\nWriting {len(updates)} updates in batches of {batch_size} …", flush=True)
    total = await store.set_payload_batch(col, updates, batch_size=batch_size)
    print(f"Done. {total} points updated with parent_text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill parent_text on Qdrant chunk points.")
    parser.add_argument("--collection", default="", help="Override collection name from settings.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    parser.add_argument("--batch-size", type=int, default=100, help="Points per Qdrant request.")
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).parent.parent.parent / "data" / "raw"),
        help="Path to data/raw directory containing fda_dailymed/, ema_smpc/, etc.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.collection, args.dry_run, args.batch_size, Path(args.data_root)))
