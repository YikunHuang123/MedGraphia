from __future__ import annotations
from enum import Enum

class EntityType(str, Enum):
    DISEASE = "Disease"
    DRUG = "Drug"
    SYMPTOM = "Symptom"
    GENE = "Gene"
    PROCEDURE = "Procedure"
    UNKNOWN = "Unknown"

class RelationType(str, Enum):
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    INTERACTS_WITH = "INTERACTS_WITH"
    DOSAGE_FOR = "DOSAGE_FOR"
    SYMPTOM_OF = "SYMPTOM_OF"
    COMPLICATION_OF = "COMPLICATION_OF"
    CODED_AS = "CODED_AS"
    CONTRAINDICATED_IN = "CONTRAINDICATED_IN"
    MENTIONED_IN = "MENTIONED_IN"
    FROM_DOC = "FROM_DOC"

class Language(str, Enum):
    EN = "en"
    ZH = "zh"
    DE = "de"
    UNKNOWN = "unknown"

    @classmethod
    def detect(cls, text: str) -> "Language":
        """
        Detect language based on character set heuristics.
        Returns cls.ZH if CJK characters are found, cls.DE for German umlauts,
        otherwise defaults to cls.EN.
        """
        if not text:
            return cls.EN
        
        # Check for Chinese characters (CJK Unified Ideographs)
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return cls.ZH
            
        # Check for German-specific characters
        lower_text = text.lower()
        if any(c in lower_text for c in ['ä', 'ö', 'ü', 'ß']):
            return cls.DE
            
        return cls.EN

class QueryType(str, Enum):
    CLINICAL_DECISION = "clinical_decision"
    DRUG_INTERACTION = "drug_interaction"
    LITERATURE_MULTIHOP = "literature_multihop"
    CROSS_CORPUS = "cross_corpus"
    PATIENT_FAQ = "patient_faq"
