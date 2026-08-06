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

Translation backend
--------------------
Uses a local NLLB-200 model (facebook/nllb-200-distilled-600M) instead of a cloud
LLM: translation is a narrow, mechanical task that a purpose-built seq2seq model
handles in tens of milliseconds on GPU, versus seconds for a chat-LLM round trip
over the network — same reasoning as using BERT/GLiNER for NER instead of an LLM.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from medgraphia.domain.base import Language
from medgraphia.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED: list[Language] = [Language.EN, Language.ZH, Language.DE]

# NLLB-200 uses FLORES-200 language codes, not ISO 639-1.
_NLLB_LANG_CODES: dict[Language, str] = {
    Language.EN: "eng_Latn",
    Language.ZH: "zho_Hans",
    Language.DE: "deu_Latn",
}


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
    Translates a medical query into all supported languages via a local NLLB-200
    model. Translations for different target languages run concurrently via
    asyncio.to_thread so the event loop is not blocked during GPU inference.

    Falls back to the original query for any language where translation fails,
    ensuring the retrieval pipeline always produces a usable candidate pool.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from medgraphia.config import get_settings

        self._model_name = model_name or get_settings().query_translator_model
        self._model: Any = None  # lazy-loaded
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._load_lock = asyncio.Lock()
        # The fast (Rust) tokenizer and the shared model instance are not safe for
        # concurrent use from multiple threads — "Already borrowed" panics under
        # asyncio.gather's parallel to_thread calls without this.
        self._inference_lock = asyncio.Lock()

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

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        self._device = device

        logger.info("query_translator_loading", model=self._model_name, device=device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        dtype = torch.float16 if device != "cpu" else torch.float32
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name, torch_dtype=dtype).to(device)
        self._model.eval()
        logger.info("query_translator_loaded", model=self._model_name, device=device)

    def _translate_sync(self, query: str, source_language: Language, target_language: Language) -> str:
        import torch

        src_code = _NLLB_LANG_CODES[source_language]
        tgt_code = _NLLB_LANG_CODES[target_language]

        self._tokenizer.src_lang = src_code
        inputs = self._tokenizer(query, return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
                max_new_tokens=256,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    async def _translate_one(
        self,
        query: str,
        source_language: Language,
        target_language: Language,
    ) -> str:
        await self._ensure_loaded()
        async with self._inference_lock:
            translated = await asyncio.to_thread(
                self._translate_sync, query, source_language, target_language
            )
        logger.info(
            "query_translated",
            src=source_language.value,
            tgt=target_language.value,
            original=query,
            translated=translated,
        )
        return translated
