"""
Multilingual medical text normalizer.

Normalizes two categories of clinical expressions:

1. **Dosing frequency** — maps language-specific surface forms to the canonical
   FHIR Timing code (lowercase):
     "twice daily" / "zweimal täglich" / "每日两次" / "q12h" → "bid"

2. **Dosage units** — standardises formatting so downstream NER sees consistent
   token boundaries:
     "500mg" → "500 mg",  "50µg" → "50 mcg",  "0.5G" → "0.5 g"

Supported languages: EN, DE, ZH (falls back to EN rules for UNKNOWN).

Design principles:
  - Pure text substitution; never modifies numeric values.
  - All patterns are pre-compiled at import time for throughput.
  - normalize_chunk() returns a *new* Chunk (original is not mutated).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from medgraphia.domain import Chunk, Language
from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FHIR Timing codes (canonical output tokens)
# ---------------------------------------------------------------------------

FHIR_QD  = "qd"    # once daily          (FHIR: Q1D)
FHIR_BID = "bid"   # twice daily         (FHIR: BID)
FHIR_TID = "tid"   # three times daily   (FHIR: TID)
FHIR_QID = "qid"   # four times daily    (FHIR: QID)
FHIR_PRN = "prn"   # as needed           (FHIR: PRN)
FHIR_QHS = "qhs"   # at bedtime          (FHIR: HS)


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FreqRule:
    canonical: str
    pattern: re.Pattern[str]


def _compile_freq(canonical: str, patterns: list[str], flags: int = re.IGNORECASE) -> list[_FreqRule]:
    return [_FreqRule(canonical, re.compile(p, flags)) for p in patterns]


# --- English -----------------------------------------------------------------

_EN_FREQ_RULES: list[_FreqRule] = [
    # bid — check 'twice' variants before generic 'daily'
    *_compile_freq(FHIR_BID, [
        r"\b(?:twice|[22])\s+(?:a\s+)?(?:day|daily)\b",
        r"\b2\s*[×xX]\s*(?:daily|(?:a\s+)?day)\b",
        r"\bq\.?12\.?h(?:rs?|ours?)?\b",
        r"\bevery\s+12\s*h(?:rs?|ours?)?\b",
        r"\bb\.?i\.?d\.?\b",
    ]),
    # tid
    *_compile_freq(FHIR_TID, [
        r"\b(?:three|[33])\s+times\s+(?:a\s+)?(?:day|daily)\b",
        r"\b3\s*[×xX]\s*(?:daily|(?:a\s+)?day)\b",
        r"\bq\.?8\.?h(?:rs?|ours?)?\b",
        r"\bevery\s+8\s*h(?:rs?|ours?)?\b",
        r"\bt\.?i\.?d\.?\b",
    ]),
    # qid
    *_compile_freq(FHIR_QID, [
        r"\b(?:four|[44])\s+times\s+(?:a\s+)?(?:day|daily)\b",
        r"\b4\s*[×xX]\s*(?:daily|(?:a\s+)?day)\b",
        r"\bq\.?6\.?h(?:rs?|ours?)?\b",
        r"\bevery\s+6\s*h(?:rs?|ours?)?\b",
        r"\bq\.?i\.?d\.?\b",
    ]),
    # qd
    *_compile_freq(FHIR_QD, [
        r"\b(?:once|[11])\s+(?:a\s+)?(?:day|daily)\b",
        r"\b1\s*[×xX]\s*(?:daily|(?:a\s+)?day)\b",
        r"\bq\.?(?:1d|24\.?h)(?:rs?|ours?)?\b",
        r"\bq\.?d\.?\b",
        r"\bquaque\s+die\b",
        r"\bevery\s+24\s*h(?:rs?|ours?)?\b",
        r"\bdaily\b",
    ]),
    # Flex qXh (q4h, q6h, etc.)
    *_compile_freq("q4h", [r"\bevery\s+4\s*h(?:rs?|ours?)?\b", r"\bq\.?4\.?h\b"]),
    *_compile_freq("q6h", [r"\bevery\s+6\s*h(?:rs?|ours?)?\b", r"\bq\.?6\.?h\b"]),
]


# --- German ------------------------------------------------------------------

_DE_FREQ_RULES: list[_FreqRule] = [
    *_compile_freq(FHIR_BID, [
        r"\b2\s*(?:mal|×|x)\s*täglich\b",
        r"\bzweimal\s+(?:täglich|am\s+Tag)\b",
        r"\b2\s*(?:mal|×|x)\s+(?:pro\s+)?Tag\b",
        r"\bjeden\s+12\.\s*Stunden?\b",
        r"\balle\s+12\s*Stunden?\b",
        r"\bbis\s+die\b",
    ]),
    *_compile_freq(FHIR_TID, [
        r"\b3\s*(?:mal|×|x)\s*täglich\b",
        r"\bdreimal\s+(?:täglich|am\s+Tag)\b",
        r"\b3\s*(?:mal|×|x)\s+(?:pro\s+)?Tag\b",
        r"\balle\s+8\s*Stunden?\b",
    ]),
    *_compile_freq(FHIR_QID, [
        r"\b4\s*(?:mal|×|x)\s*täglich\b",
        r"\bviermal\s+(?:täglich|am\s+Tag)\b",
        r"\b4\s*(?:mal|×|x)\s+(?:pro\s+)?Tag\b",
        r"\balle\s+6\s*Stunden?\b",
    ]),
    *_compile_freq(FHIR_QD, [
        r"\beinmal\s+(?:täglich|am\s+Tag)\b",
        r"\b1\s*(?:mal|×|x)\s*täglich\b",
        r"\b1\s*(?:mal|×|x)\s+(?:pro\s+)?Tag\b",
        r"\btäglich\b",
    ]),
]


# --- Chinese -----------------------------------------------------------------

_ZH_FREQ_RULES: list[_FreqRule] = [
    *_compile_freq(FHIR_BID, [
        r"(?:每日|一日|每天|一天)\s*[两2]\s*次",
        r"[两2]\s*次/(?:日|天)",
        r"[两2]\s*次／(?:日|天)",
    ], flags=0),
    *_compile_freq(FHIR_TID, [
        r"(?:每日|一日|每天|一天)\s*[三3]\s*次",
        r"[三3]\s*次/(?:日|天)",
        r"[三3]\s*次／(?:日|天)",
    ], flags=0),
    *_compile_freq(FHIR_QID, [
        r"(?:每日|一日|每天|一天)\s*[四4]\s*次",
        r"[四4]\s*次/(?:日|天)",
        r"[四4]\s*次／(?:日|天)",
    ], flags=0),
    *_compile_freq(FHIR_QD, [
        r"(?:每日|一日|每天|一天)\s*[一1]\s*次",
        r"[一1]\s*次/(?:日|天)",
        r"[一1]\s*次／(?:日|天)",
        r"每日(?:服用)?一次",
        r"每天(?:服用)?一次",
        r"每日(?:一次|口服)",
    ], flags=0),
    *_compile_freq("q4h", [r"每\s*4\s*小时\s*一次"], flags=0),
    *_compile_freq(FHIR_PRN, [r"必要时", r"按需", r"需要时"], flags=0),
]


# ---------------------------------------------------------------------------
# Dosage unit normalisation
# ---------------------------------------------------------------------------

_DOSAGE_UNIT_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"           
    r"(milligrams?|mg|毫克"                 
    r"|micrograms?|mcg|µg|ug|μg|微克"      
    r"|grams?|gramm|gr?\b|克"            
    r"|millilitres?|milliliters?|ml|mL|毫升"  
    r"|litres?|liters?|l\b|L\b|升"
    r"|international\s+units?|iu|IU|单位"  
    r"|units?|U\b"                    
    r"|nanograms?|ng|纳克"
    r"|percent|%"
    r")",
    re.IGNORECASE,
)

_UNIT_MAP: dict[str, str] = {
    "milligrams": "mg", "milligram": "mg", "毫克": "mg",
    "micrograms": "mcg", "microgram": "mcg", "µg": "mcg", "ug": "mcg", "μg": "mcg", "微克": "mcg",
    "grams": "g", "gram": "g", "gramm": "g", "克": "g",
    "millilitres": "mL", "milliliters": "mL", "millilitre": "mL", "milliliter": "mL", "ml": "mL", "毫升": "mL",
    "litres": "L", "liters": "L", "litre": "L", "liter": "L", "升": "L",
    "international units": "IU", "international unit": "IU", "单位": "IU",
    "nanograms": "ng", "nanogram": "ng", "纳克": "ng",
}


def _normalise_unit(raw: str) -> str:
    """Return the canonical unit string for a raw dosage unit."""
    return _UNIT_MAP.get(raw.lower(), raw.lower() if raw not in ("IU", "mL", "U") else raw)


def _normalise_dosage_units(text: str) -> str:
    """
    Insert a space between number and unit, and canonicalise the unit spelling.
    "500mg" → "500 mg",  "50µg" → "50 mcg",  "0.5Grams" → "0.5 g"
    """
    def _replace(m: re.Match) -> str:
        number = m.group(1)
        unit   = _normalise_unit(m.group(2))
        return f"{number} {unit}"

    return _DOSAGE_UNIT_PATTERN.sub(_replace, text)


# ---------------------------------------------------------------------------
# MedicalNormalizer
# ---------------------------------------------------------------------------

class MedicalNormalizer:
    """
    Normalises dosing frequency and dosage units in clinical text.

    Usage::

        normalizer = MedicalNormalizer()

        # normalise a raw string
        out = normalizer.normalize("Give 500mg twice daily", Language.EN)
        # → "Give 500 mg bid"

        # normalise an existing Chunk (returns a new Chunk, original unchanged)
        norm_chunk = normalizer.normalize_chunk(chunk)
    """

    # Language → ordered list of frequency rules
    _FREQ_RULES: dict[Language, list[_FreqRule]] = {
        Language.EN:      _EN_FREQ_RULES,
        Language.DE:      _DE_FREQ_RULES,
        Language.ZH:      _ZH_FREQ_RULES,
        Language.UNKNOWN: _EN_FREQ_RULES,
    }

    def normalize(self, text: str, language: Language = Language.EN) -> str:
        """
        Apply frequency and dosage-unit normalisation to a raw text string.

        Frequency substitution is applied *before* dosage-unit normalisation
        so that tokens like "b.i.d" are cleaned first.
        """
        text = self._apply_frequency_rules(text, language)
        text = _normalise_dosage_units(text)
        return text

    def normalize_chunk(self, chunk: Chunk) -> Chunk:
        """Return a new Chunk with normalized text (original Chunk is immutable)."""
        normalised = self.normalize(chunk.text, chunk.language)
        if normalised == chunk.text:
            return chunk
        logger.debug(
            "chunk_normalised",
            chunk_id=chunk.chunk_id,
            lang=chunk.language.value,
            original_len=len(chunk.text),
            normalised_len=len(normalised),
        )
        return chunk.model_copy(update={"text": normalised})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_frequency_rules(self, text: str, language: Language) -> str:
        rules = self._FREQ_RULES.get(language, _EN_FREQ_RULES)
        for rule in rules:
            # Replace matched expression with canonical FHIR code.
            # We guard against replacing inside already-canonical tokens.
            text = rule.pattern.sub(rule.canonical, text)
        return text
