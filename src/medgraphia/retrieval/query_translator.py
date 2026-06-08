"""
Multilingual query translator for cross-language retrieval.

Root cause this solves
----------------------
BGE-M3 hybrid search combines a dense vector (semantic) with a sparse SPLADE-style
vector (lexical).  When a user submits a Chinese query like "肾衰竭" and the corpus
contains German chunks about "Nierenversagen", the dense vectors are close (BGE-M3
is multilingual), but the sparse token-hash vectors share zero overlap.  The Qdrant
RRF fusion of dense+sparse therefore systematically under-ranks cross-language hits.

Fix
---
Before the vector retrieval step, translate the query into every other supported
language.  The retrieval pipeline then runs one language-filtered Qdrant search per
language using the per-language translation, guaranteeing fair representation in the
candidate pool regardless of the source language.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from medgraphia.domain.base import Language
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED: list[Language] = [Language.EN, Language.ZH, Language.DE]


@dataclass
class TranslatedQuery:
    original: str
    source_language: Language
    translations: dict[Language, str] = field(default_factory=dict)

    def all_queries(self) -> dict[Language, str]:
        """Source query + all translations, keyed by Language."""
        result: dict[Language, str] = {self.source_language: self.original}
        result.update(self.translations)
        return result


class QueryTranslator:
    """
    Translates a medical query into all supported languages via DSPy + LLM.

    Reuses the 'rewriter' LM task (Qwen2.5 or equivalent) so no additional
    provider configuration is needed.  Translations run in parallel via
    asyncio.to_thread so the event loop is not blocked during LLM network IO.

    Falls back to the original query for any language where translation fails,
    ensuring the retrieval pipeline always produces a usable candidate pool.
    """

    @classmethod
    def from_settings(cls) -> QueryTranslator:
        return cls()

    async def translate(
        self,
        query: str,
        source_language: Language,
        target_languages: list[Language] | None = None,
    ) -> TranslatedQuery:
        """
        Translate *query* into each of *target_languages* in parallel.

        Args:
            query:            Source query text.
            source_language:  Detected language of the input query.
            target_languages: Languages to produce. Defaults to all supported
                              languages except *source_language*.

        Returns:
            TranslatedQuery with the original query and all translations.
        """
        if target_languages is None:
            target_languages = [lg for lg in _SUPPORTED if lg != source_language]

        if not target_languages:
            return TranslatedQuery(original=query, source_language=source_language)

        coros = [self._translate_one(query, source_language, tgt) for tgt in target_languages]
        results = await asyncio.gather(*coros, return_exceptions=True)

        translations: dict[Language, str] = {}
        for lang, res in zip(target_languages, results):
            if isinstance(res, Exception):
                logger.warning(
                    "query_translation_failed",
                    target_lang=lang.value,
                    error=str(res),
                )
                translations[lang] = query  # graceful fallback: use original
            else:
                translations[lang] = str(res)

        return TranslatedQuery(
            original=query,
            source_language=source_language,
            translations=translations,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _translate_one(
        self,
        query: str,
        source_language: Language,
        target_language: Language,
    ) -> str:
        src_name = source_language.full_name
        tgt_name = target_language.full_name

        def _sync() -> str:
            import dspy

            from medgraphia.llm.dspy_setup import get_lm
            from medgraphia.programs.translator import get_translator

            lm = get_lm("rewriter")

            with dspy.context(lm=lm):
                program = get_translator()
                pred = program(
                    source_text=query,
                    source_lang=src_name,
                    target_lang=tgt_name,
                )
            return pred.translated_text.strip()

        # Run synchronous DSPy/LLM call in a thread so the event loop stays free
        translated = await asyncio.to_thread(_sync)
        logger.info(
            "query_translated",
            src=source_language.value,
            tgt=target_language.value,
            original=query,
            translated=translated,
        )
        return translated
