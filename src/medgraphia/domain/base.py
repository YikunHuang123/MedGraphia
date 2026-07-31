from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    DISEASE = "Disease"
    DRUG = "Drug"
    SYMPTOM = "Symptom"
    GENE = "Gene"
    PROCEDURE = "Procedure"
    ANATOMY = "Anatomy"
    PHYSIOLOGY = "Physiology"
    LIVING_BEING = "LivingBeing"
    UNKNOWN = "Unknown"


class Language(str, Enum):
    EN = "en"
    ZH = "zh"
    DE = "de"
    UNKNOWN = "unknown"

    @property
    def full_name(self) -> str:
        """Return the human-readable name of the language (e.g. 'English', 'Chinese')."""
        mapping = {
            Language.EN: "English",
            Language.ZH: "Chinese",
            Language.DE: "German",
            Language.UNKNOWN: "Unknown",
        }
        return mapping.get(self, "English")

    @classmethod
    def detect(cls, text: str) -> Language:
        """
        Detect language using a hybrid approach:
        1. Fast Unicode check for Chinese.
        2. Industry-standard FastText (via fast-langdetect) for others.
        3. Default to English.
        """
        if not text or len(text.strip()) < 2:
            return cls.EN

        # Step 1: Instant check for Chinese characters (CJK range)
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return cls.ZH

        # Step 2: Use FastText for robust detection
        try:
            from fast_langdetect import detect as ft_detect

            # Note: low_memory parameter is not supported in all versions of fast-langdetect
            result = ft_detect(text)

            # Handle list vs dict return type from different versions
            if isinstance(result, list) and len(result) > 0:
                lang_code = result[0].get("lang", "en").lower()
            elif isinstance(result, dict):
                lang_code = result.get("lang", "en").lower()
            else:
                lang_code = "en"

            if lang_code == "de":
                return cls.DE
            if lang_code == "zh":
                return cls.ZH
            return cls.EN
        except ImportError:
            # Not using structured logger here to avoid circular imports in base domain
            print("WARNING: 'fast-langdetect' library is not installed. Falling back to English.")
            return cls.EN
        except Exception as e:
            print(f"ERROR: Language detection failed: {e}. Falling back to English.")
            return cls.EN


class QueryType(str, Enum):
    CLINICAL_DECISION = "clinical_decision"
    DRUG_INTERACTION = "drug_interaction"
    LITERATURE_MULTIHOP = "literature_multihop"
    CROSS_CORPUS = "cross_corpus"
    PATIENT_FAQ = "patient_faq"
