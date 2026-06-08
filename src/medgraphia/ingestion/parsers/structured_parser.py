from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from medgraphia.domain import Language, ParsedSection, RawDocument, SourceMeta
from medgraphia.logger import get_logger

logger = get_logger(__name__)


class StructuredParser:
    """
    Industrial-grade parser for medical datasets (JSON, JSONL).
    Transforms diverse formats into system-standard RawDocument objects.
    """

    def parse_huatuo(self, file_path: Path) -> Iterable[RawDocument]:
        """
        Parses Huatuo-Lite (ZH) JSONL.
        Each line is a QA pair: {"id":..., "question":..., "answer":..., "label":...}
        """
        logger.info("parse_huatuo_start", path=str(file_path))
        count = 0
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                doc_id = f"huatuo_{data.get('id', count)}"
                question = data.get("question", "")
                answer = data.get("answer", "")
                label = data.get("label", "General")
                related = data.get("related_diseases", "")

                content = f"Question: {question}\n\nAnswer: {answer}"

                source = SourceMeta(
                    source_id=doc_id,
                    source_title=f"Huatuo QA: {label}",
                    language=Language.ZH,
                    source_url="https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite",
                )

                sections = [
                    ParsedSection(section_path="Question", title="Question", content=question),
                    ParsedSection(section_path="Answer", title="Answer", content=answer),
                ]

                if related:
                    sections.append(
                        ParsedSection(
                            section_path="Related Diseases",
                            title="Related Diseases",
                            content=related,
                        )
                    )

                yield RawDocument(
                    doc_id=doc_id,
                    source=source,
                    language=Language.ZH,
                    title=question[:100],
                    abstract=content[:500],
                    full_text=content,
                    sections=sections,
                    format="text",
                )
                count += 1
        logger.info("parse_huatuo_complete", total=count)

    def parse_germed(self, file_path: Path) -> Iterable[RawDocument]:
        """
        Parses GERNERMED (DE) JSON array.
        Format: [{"de": "...", "en": "...", "annotations": [...]}, ...]
        """
        logger.info("parse_germed_start", path=str(file_path))
        with open(file_path, encoding="utf-8") as f:
            data_list = json.load(f)

        count = 0
        for item in data_list:
            de_text = item.get("de", "")
            en_text = item.get("en", "")

            # Use hash for stable ID
            content_hash = hashlib.md5(de_text.encode()).hexdigest()[:12]
            doc_id = f"germed_{content_hash}"

            source = SourceMeta(
                source_id=doc_id,
                source_title="GERNERMED Clinical Case",
                language=Language.DE,
                source_url="https://github.com/hpi-dhc/GERNERMED",
            )

            # We treat the EN translation as a separate section for context
            sections = [
                ParsedSection(
                    section_path="Clinical Description", title="German Text", content=de_text
                )
            ]
            if en_text:
                sections.append(
                    ParsedSection(
                        section_path="English Translation",
                        title="English Translation",
                        content=en_text,
                    )
                )

            yield RawDocument(
                doc_id=doc_id,
                source=source,
                language=Language.DE,
                title=en_text[:100] if en_text else de_text[:100],
                abstract=de_text,
                full_text=de_text,
                sections=sections,
                format="text",
            )
            count += 1
        logger.info("parse_germed_complete", total=count)

    def load_pubmed_batch(self, dir_path: Path) -> Iterable[RawDocument]:
        """
        Loads already-parsed PubMed JSON files from a directory.
        """
        logger.info("load_pubmed_batch_start", path=str(dir_path))
        count = 0
        for file in dir_path.glob("*.json"):
            try:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)
                    # PubMed files saved by fetch_pubmed.py are RawDocument models
                    yield RawDocument.model_validate(data)
                    count += 1
            except Exception as exc:
                logger.warning("load_pubmed_failed", file=file.name, error=str(exc))
        logger.info("load_pubmed_batch_complete", total=count)
