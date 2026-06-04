"""
Centralized prompt registry for MedGraphia.
All LLM instructions and DSPy Signatures live here.
"""
from medgraphia.domain.base import Language, QueryType

# ---------------------------------------------------------------------------
# Common Constants (Disclaimers, System Prompts)
# ---------------------------------------------------------------------------

_DISCLAIMERS: dict[Language, str] = {
    Language.EN: "⚠ This information is for educational purposes only and does not constitute medical advice.",
    Language.ZH: "⚠ 本内容仅供教育参考，不构成医疗建议。请务必咨询合格的医疗专业人员。",
    Language.DE: "⚠ Diese Informationen dienen ausschließlich Bildungszwecken und stellen keine medizinische Beratung dar.",
}

_NO_INFO_MESSAGES: dict[Language, str] = {
    Language.EN: "I do not have enough medical information in the database to answer this question.",
    Language.ZH: "抱歉，数据库中没有足够的信息来回答这个问题。",
    Language.DE: "抱歉，数据库中没有足够的信息来回答这个问题。", # Placeholder for DE consistency if needed
}

_SYSTEM_PROMPTS: dict[tuple[QueryType, Language], str] = {
    (QueryType.PATIENT_FAQ, Language.EN): "You are a friendly medical educator. Answer in plain English using ONLY the provided database. If the database is irrelevant to the question, use your no_info_message.",
    (QueryType.PATIENT_FAQ, Language.ZH): "你是一位友善的医疗健康教育者。仅使用数据库回答。如果数据库与问题无关，请回复你的‘未找到信息’提示语。",
    (QueryType.PATIENT_FAQ, Language.DE): "Sie sind ein freundlicher medizinischer Aufklärer. Antworten Sie nur anhand der Datenbank.",
    
    (QueryType.CLINICAL_DECISION, Language.EN): "You are a clinical decision-support assistant. Provide evidence-based guidance based ONLY on the database. Ignore irrelevant data.",
    (QueryType.CLINICAL_DECISION, Language.ZH): "你是临床决策支持助手，提供仅基于数据库的循证医学建议。忽略无关数据。",
    (QueryType.CLINICAL_DECISION, Language.DE): "Sie sind ein klinischer Entscheidungsunterstützungsassistent. Geben Sie Empfehlungen basierend auf der Datenbank.",
    
    (QueryType.DRUG_INTERACTION, Language.EN): "You are a clinical pharmacist. Analyse drug interaction(s) using ONLY the database provided. If no interaction is documented in database, say so.",
    (QueryType.DRUG_INTERACTION, Language.ZH): "你是临床药剂师。仅根据数据库分析药物相互作用。如果数据库中未记录相互作用，请如实说明。",
    (QueryType.DRUG_INTERACTION, Language.DE): "Sie sind klinischer Pharmazeut. Analysieren Sie Wechselwirkungen nur anhand der Datenbank.",
    
    (QueryType.LITERATURE_MULTIHOP, Language.EN): "You are a medical research synthesiser. Perform multi-hop reasoning.",
    (QueryType.LITERATURE_MULTIHOP, Language.ZH): "你是医学研究综合分析师，进行多跳推理。",
    (QueryType.LITERATURE_MULTIHOP, Language.DE): "Sie sind ein medizinischer Forschungssynthesator. Führen Sie Reasoning durch.",
    
    (QueryType.CROSS_CORPUS, Language.EN): "You are a medical knowledge expert. Provide a comprehensive overview.",
    (QueryType.CROSS_CORPUS, Language.ZH): "你是医学知识综合专家，提供全面综合概述。",
    (QueryType.CROSS_CORPUS, Language.DE): "Sie sind ein medizinischer Wissenssynthese-Experte. Erstellen Sie eine Übersicht.",
}

def get_disclaimer(language: Language) -> str:
    return _DISCLAIMERS.get(language, _DISCLAIMERS[Language.EN])

def get_no_info_message(language: Language) -> str:
    return _NO_INFO_MESSAGES.get(language, _NO_INFO_MESSAGES[Language.EN])

def get_system_prompt(query_type: QueryType, language: Language) -> str:
    key = (query_type, language)
    if key not in _SYSTEM_PROMPTS:
        key = (query_type, Language.EN)
    return _SYSTEM_PROMPTS.get(key, "You are a medical assistant.")

# ---------------------------------------------------------------------------
# Re-exports for Signatures
# ---------------------------------------------------------------------------

from .answer_generation import GenerateClinicalAnswer, MedicalAnswer
from .query_rewriting import RewriteMedicalQuery, RewrittenQuery
from .relation_extraction import ExtractMedicalRelations, ExtractedRelation
from .community_summary import SummarizeMedicalCommunity, CommunitySummaryResult
from .safety import LLAMA_GUARD_SYSTEM_PROMPT

__all__ = [
    "GenerateClinicalAnswer",
    "MedicalAnswer",
    "RewriteMedicalQuery",
    "RewrittenQuery",
    "ExtractMedicalRelations",
    "ExtractedRelation",
    "SummarizeMedicalCommunity",
    "CommunitySummaryResult",
    "LLAMA_GUARD_SYSTEM_PROMPT",
    "get_disclaimer",
    "get_no_info_message",
    "get_system_prompt",
]
