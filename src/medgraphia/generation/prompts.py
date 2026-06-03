"""
DSPy TypedPredictor prompt modules for MedGraphia.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from medgraphia.domain.base import Language, QueryType
from medgraphia.logger import get_logger

logger = get_logger(__name__)


class MedicalAnswer(BaseModel):
    answer: str = Field(..., description="Response text in the requested language")
    citations: list[int] = Field(
        default_factory=list,
        description="1-indexed citation numbers corresponding to [N] markers in the answer",
    )
    disclaimer: str = Field(
        default="",
        description="Mandatory safety disclaimer; required for clinical/drug/multihop scenarios",
    )


_DISCLAIMERS: dict[Language, str] = {
    Language.EN: "⚠ This information is for educational purposes only and does not constitute medical advice.",
    Language.ZH: "⚠ 本内容仅供教育参考，不构成医疗建议。请务必咨询合格的医疗专业人员。",
    Language.DE: "⚠ Diese Informationen dienen ausschließlich Bildungszwecken und stellen keine medizinische Beratung dar.",
}

_REQUIRES_DISCLAIMER: frozenset[QueryType] = frozenset({
    QueryType.CLINICAL_DECISION,
    QueryType.DRUG_INTERACTION,
    QueryType.LITERATURE_MULTIHOP,
})


def get_disclaimer(language: Language) -> str:
    return _DISCLAIMERS.get(language, _DISCLAIMERS[Language.EN])


def requires_disclaimer(query_type: QueryType) -> bool:
    return query_type in _REQUIRES_DISCLAIMER


_NO_INFO_MESSAGES: dict[Language, str] = {
    Language.EN: "I do not have enough medical information in the provided context to answer this question.",
    Language.ZH: "抱歉，提供的参考资料中没有足够的信息来回答这个问题。",
    Language.DE: "Entschuldigung, aber die bereitgestellten Informationen enthalten nicht genügend medizinische Daten.",
}


def get_no_info_message(language: Language) -> str:
    return _NO_INFO_MESSAGES.get(language, _NO_INFO_MESSAGES[Language.EN])


_SYSTEM_PROMPTS: dict[tuple[QueryType, Language], str] = {
    (QueryType.PATIENT_FAQ, Language.EN): "You are a friendly medical educator. Answer in plain English.",
    (QueryType.PATIENT_FAQ, Language.ZH): "你是一位友善的医疗健康教育者，使用通俗易懂的中文回答问题。",
    (QueryType.PATIENT_FAQ, Language.DE): "Sie sind ein freundlicher medizinischer Aufklärer. Antworten Sie auf Deutsch.",
    
    (QueryType.CLINICAL_DECISION, Language.EN): "You are a clinical decision-support assistant. Provide evidence-based guidance.",
    (QueryType.CLINICAL_DECISION, Language.ZH): "你是临床决策支持助手，提供基于循证医学的临床建议。",
    (QueryType.CLINICAL_DECISION, Language.DE): "Sie sind ein klinischer Entscheidungsunterstützungsassistent. Geben Sie klinische Empfehlungen.",
    
    (QueryType.DRUG_INTERACTION, Language.EN): "You are a clinical pharmacist. Analyse drug interaction(s).",
    (QueryType.DRUG_INTERACTION, Language.ZH): "你是专注药物相互作用的临床药剂师。分析药物相互作用。",
    (QueryType.DRUG_INTERACTION, Language.DE): "Sie sind klinischer Pharmazeut. Analysieren Sie die Wechselwirkung(en).",
    
    (QueryType.LITERATURE_MULTIHOP, Language.EN): "You are a medical research synthesiser. Perform multi-hop reasoning.",
    (QueryType.LITERATURE_MULTIHOP, Language.ZH): "你是医学研究综合分析师，进行多跳推理。",
    (QueryType.LITERATURE_MULTIHOP, Language.DE): "Sie sind ein medizinischer Forschungssynthesator. Führen Sie Reasoning durch.",
    
    (QueryType.CROSS_CORPUS, Language.EN): "You are a medical knowledge expert. Provide a comprehensive overview.",
    (QueryType.CROSS_CORPUS, Language.ZH): "你是医学知识综合专家，提供全面综合概述。",
    (QueryType.CROSS_CORPUS, Language.DE): "Sie sind ein medizinischer Wissenssynthese-Experte. Erstellen Sie eine Übersicht.",
}


def get_system_prompt(query_type: QueryType, language: Language) -> str:
    key = (query_type, language)
    if key not in _SYSTEM_PROMPTS:
        key = (query_type, Language.EN)
    return _SYSTEM_PROMPTS.get(key, "You are a medical assistant.")


async def _json_prompt_predict(context: str, question: str, language: Language, query_type: QueryType, gateway: Any) -> MedicalAnswer:
    from medgraphia.llm.gateway import CompletionRequest, _parse_json_safe
    
    lang_names = {Language.EN: "English", Language.ZH: "Chinese", Language.DE: "German"}
    target_lang = lang_names.get(language, "English")
    no_info_msg = get_no_info_message(language)
    
    user_prompt = (
        f"<context>\n{context}\n</context>\n\n"
        f"QUESTION: {question}\n\n"
        f"RULES:\n"
        f"1. Respond ONLY in {target_lang}.\n"
        f"2. Cite context using [N].\n"
        f"3. If no info, say: {no_info_msg}\n"
        f"Output JSON: {{\"answer\": \"...\", \"citations\": [1], \"disclaimer\": \"...\"}}"
    )

    req = CompletionRequest(system_prompt=get_system_prompt(query_type, language), user_prompt=user_prompt, json_mode=True, temperature=0.1)
    resp = await gateway.acomplete(req)
    parsed = _parse_json_safe(resp.text)
    
    if isinstance(parsed, dict):
        return MedicalAnswer(**{k: v for k, v in parsed.items() if k in MedicalAnswer.model_fields})
    return MedicalAnswer(answer=str(parsed), citations=[])


class MedicalPredictor:
    def __init__(self, query_type: QueryType) -> None:
        self._query_type = query_type

    async def predict(self, context: str, question: str, language: Language, gateway: Any | None = None) -> MedicalAnswer:
        if gateway is None: return MedicalAnswer(answer="No gateway.", citations=[])
        return await _json_prompt_predict(context, question, language, self._query_type, gateway)


class PromptRegistry:
    def __init__(self) -> None:
        self._predictors: dict[QueryType, MedicalPredictor] = {}

    def get(self, query_type: QueryType) -> MedicalPredictor:
        if query_type not in self._predictors:
            self._predictors[query_type] = MedicalPredictor(query_type)
        return self._predictors[query_type]
