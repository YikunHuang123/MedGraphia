"""
DSPy TypedPredictor prompt modules for MedGraphia.

Five query scenarios × three languages (EN / ZH / DE):
  PATIENT_FAQ          — lay-language explanation with mandatory citations
  CLINICAL_DECISION    — evidence-based clinical recommendation + disclaimer
  DRUG_INTERACTION     — interaction analysis with severity + mechanism
  LITERATURE_MULTIHOP  — multi-hop reasoning over literature evidence
  CROSS_CORPUS         — global synthesis across the full knowledge base

Output contract (MedicalAnswer):
  answer    : str        — response in the requested language
  citations : list[int]  — 1-indexed [N] numbers referencing context passages
  disclaimer: str        — mandatory for clinical/drug scenarios, empty otherwise

DSPy integration:
  When dspy-ai is installed, each scenario registers a dspy.TypedPredictor
  backed by a dspy.Signature with a typed MedicalAnswer output field.
  dspy.configure(lm=…) is called lazily the first time a gateway is wired.

Fallback path (no dspy-ai):
  A direct JSON-prompt is sent via LiteLLMGateway.  The response is parsed
  with a lenient JSON extractor and coerced into MedicalAnswer.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from medgraphia.domain.base import Language, QueryType
from medgraphia.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared output model
# ---------------------------------------------------------------------------

class MedicalAnswer(BaseModel):
    """
    Structured output produced by every MedicalPredictor.

    citations MUST contain the integer indices of every [N] reference used
    in the answer text.  Downstream citation injection depends on this list.
    """
    answer: str = Field(..., description="Response text in the requested language")
    citations: list[int] = Field(
        default_factory=list,
        description="1-indexed citation numbers corresponding to [N] markers in the answer",
    )
    disclaimer: str = Field(
        default="",
        description="Mandatory safety disclaimer; required for clinical/drug/multihop scenarios",
    )


# ---------------------------------------------------------------------------
# Safety disclaimers (per language)
# ---------------------------------------------------------------------------

_DISCLAIMERS: dict[Language, str] = {
    Language.EN: (
        "⚠ This information is for educational purposes only and does not constitute "
        "medical advice. Always consult a qualified healthcare professional."
    ),
    Language.ZH: (
        "⚠ 本内容仅供教育参考，不构成医疗建议。请务必咨询合格的医疗专业人员。"
    ),
    Language.DE: (
        "⚠ Diese Informationen dienen ausschließlich Bildungszwecken und stellen keine "
        "medizinische Beratung dar. Konsultieren Sie stets einen qualifizierten Arzt."
    ),
}

_REQUIRES_DISCLAIMER: frozenset[QueryType] = frozenset({
    QueryType.CLINICAL_DECISION,
    QueryType.DRUG_INTERACTION,
    QueryType.LITERATURE_MULTIHOP,
})


def get_disclaimer(language: Language) -> str:
    """Return the safety disclaimer for the given language."""
    return _DISCLAIMERS.get(language, _DISCLAIMERS[Language.EN])


def requires_disclaimer(query_type: QueryType) -> bool:
    """True if the query type requires a safety disclaimer."""
    return query_type in _REQUIRES_DISCLAIMER


# ---------------------------------------------------------------------------
# System prompts: (QueryType, Language) → instruction string
#
# Every prompt instructs the LLM to:
#   1. Cite passages using the [N] notation
#   2. Keep the answer in the requested language
#   3. Include a disclaimer when appropriate
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: dict[tuple[QueryType, Language], str] = {
    # ── PATIENT_FAQ ─────────────────────────────────────────────────────────
    (QueryType.PATIENT_FAQ, Language.EN): (
        "You are a friendly medical educator answering patient questions in plain, "
        "easy-to-understand English. "
        "Use the numbered context passages [1], [2], … to support every factual claim. "
        "Cite each claim inline with [N]. Avoid clinical jargon. "
        "Respond in English only."
    ),
    (QueryType.PATIENT_FAQ, Language.ZH): (
        "你是一位友善的医疗健康教育者，使用通俗易懂的中文回答患者问题。"
        "使用编号的参考资料 [1]、[2]…… 支撑每个事实性陈述，每处事实均需标注 [N]。"
        "避免专业医学术语，仅用中文回答。"
    ),
    (QueryType.PATIENT_FAQ, Language.DE): (
        "Sie sind ein freundlicher medizinischer Aufklärer und beantworten Patientenfragen "
        "in einfachem, verständlichem Deutsch. "
        "Stützen Sie sich auf die nummerierten Kontextpassagen [1], [2] … "
        "und zitieren Sie jeden Fakt mit [N]. Vermeiden Sie Fachjargon. "
        "Antworten Sie ausschließlich auf Deutsch."
    ),
    # ── CLINICAL_DECISION ───────────────────────────────────────────────────
    (QueryType.CLINICAL_DECISION, Language.EN): (
        "You are a clinical decision-support assistant. "
        "Provide evidence-based guidance citing every factual statement with [N] "
        "from the numbered context passages. "
        "Use precise clinical terminology. Include a safety disclaimer. "
        "Respond in English only."
    ),
    (QueryType.CLINICAL_DECISION, Language.ZH): (
        "你是临床决策支持助手，提供基于循证医学的临床建议。"
        "每个事实性陈述均须引用编号参考资料 [N]，使用精准临床术语，"
        "并必须附上安全免责声明。仅用中文回答。"
    ),
    (QueryType.CLINICAL_DECISION, Language.DE): (
        "Sie sind ein klinischer Entscheidungsunterstützungsassistent. "
        "Geben Sie evidenzbasierte klinische Empfehlungen und zitieren Sie "
        "jede Aussage mit [N] aus den nummerierten Kontextpassagen. "
        "Verwenden Sie klinische Fachsprache und fügen Sie einen Haftungsausschluss hinzu. "
        "Antworten Sie ausschließlich auf Deutsch."
    ),
    # ── DRUG_INTERACTION ────────────────────────────────────────────────────
    (QueryType.DRUG_INTERACTION, Language.EN): (
        "You are a clinical pharmacist specialising in drug interactions. "
        "Analyse the drug interaction(s), citing evidence with [N] from the numbered passages. "
        "State the severity (mild / moderate / severe / contraindicated), "
        "explain the pharmacodynamic or pharmacokinetic mechanism, "
        "and include a safety disclaimer. Respond in English only."
    ),
    (QueryType.DRUG_INTERACTION, Language.ZH): (
        "你是专注药物相互作用的临床药剂师。"
        "分析药物相互作用，引用编号资料 [N] 提供证据，"
        "注明严重程度（轻度/中度/重度/禁忌），"
        "解释药效学或药代动力学机制，并附安全免责声明。仅用中文回答。"
    ),
    (QueryType.DRUG_INTERACTION, Language.DE): (
        "Sie sind klinischer Pharmazeut mit Spezialisierung auf Arzneimittelwechselwirkungen. "
        "Analysieren Sie die Wechselwirkung(en) unter Zitierung von [N] aus den Passagen. "
        "Geben Sie den Schweregrad an, erklären Sie den Mechanismus "
        "und fügen Sie einen Haftungsausschluss hinzu. Antworten ausschließlich auf Deutsch."
    ),
    # ── LITERATURE_MULTIHOP ─────────────────────────────────────────────────
    (QueryType.LITERATURE_MULTIHOP, Language.EN): (
        "You are a medical research synthesiser. "
        "Perform multi-hop reasoning across the numbered evidence passages [1], [2], … "
        "and cite every reasoning step with [N]. "
        "Distinguish established evidence, emerging evidence, and inference. "
        "Include a disclaimer. Respond in English only."
    ),
    (QueryType.LITERATURE_MULTIHOP, Language.ZH): (
        "你是医学研究综合分析师，对编号的证据段落 [1]、[2]…… 进行多跳推理。"
        "每个推理步骤均标注 [N]，区分已有证据、新兴证据与推断性结论，"
        "并附上免责声明。仅用中文回答。"
    ),
    (QueryType.LITERATURE_MULTIHOP, Language.DE): (
        "Sie sind ein medizinischer Forschungssynthesator. "
        "Führen Sie mehrstufiges Reasoning über die nummerierten Evidenzpassagen durch "
        "und zitieren Sie jeden Schlussfolgerungsschritt with [N]. "
        "Unterscheiden Sie gesicherte von aufkommender Evidenz. "
        "Fügen Sie einen Haftungsausschluss hinzu. Antworten auf Deutsch."
    ),
    # ── CROSS_CORPUS ────────────────────────────────────────────────────────
    (QueryType.CROSS_CORPUS, Language.EN): (
        "You are a medical knowledge synthesis expert. "
        "Provide a comprehensive overview drawing on all relevant numbered passages [1], [2], … "
        "Cite every claim with [N] and organise the answer with clear sections if needed. "
        "Respond in English only."
    ),
    (QueryType.CROSS_CORPUS, Language.ZH): (
        "你是医学知识综合专家，基于所有相关编号段落 [1]、[2]…… 提供全面综合概述。"
        "每项陈述引用 [N]，如有必要使用清晰章节结构。仅用中文回答。"
    ),
    (QueryType.CROSS_CORPUS, Language.DE): (
        "Sie sind ein medizinischer Wissenssynthese-Experte. "
        "Erstellen Sie eine umfassende Übersicht auf Basis aller relevanten Passagen [1], [2] …. "
        "Zitieren Sie jede Aussage mit [N] und strukturieren Sie die Antwort klar. "
        "Antworten ausschließlich auf Deutsch."
    ),
}


def get_system_prompt(query_type: QueryType, language: Language) -> str:
    """Return the system prompt for (query_type, language), falling back to English."""
    key = (query_type, language)
    if key not in _SYSTEM_PROMPTS:
        key = (query_type, Language.EN)
    return _SYSTEM_PROMPTS.get(key, _SYSTEM_PROMPTS[(QueryType.PATIENT_FAQ, Language.EN)])


# ---------------------------------------------------------------------------
# DSPy integration — optional
# ---------------------------------------------------------------------------

_DSPY_AVAILABLE: bool | None = None


def _dspy_imported() -> bool:
    global _DSPY_AVAILABLE
    if _DSPY_AVAILABLE is None:
        try:
            import dspy  # noqa: F401
            _DSPY_AVAILABLE = True
            logger.info("dspy_available", version=getattr(dspy, "__version__", "unknown"))
        except ImportError:
            _DSPY_AVAILABLE = False
            logger.info("dspy_not_installed", fallback="json_prompt_mode")
    return _DSPY_AVAILABLE  # type: ignore[return-value]


# DSPy Signature + TypedPredictor definitions (only defined when DSPy is present)
_DSPY_PREDICTORS: dict[QueryType, Any] = {}


def _build_dspy_predictor(query_type: QueryType) -> Any | None:
    """Lazily build a dspy.TypedPredictor for *query_type*."""
    if not _dspy_imported():
        return None

    if query_type in _DSPY_PREDICTORS:
        return _DSPY_PREDICTORS[query_type]

    try:
        import dspy

        # Scenario-level instruction embedded in the Signature docstring
        _SCENARIO_DOCS: dict[QueryType, str] = {
            QueryType.PATIENT_FAQ: (
                "Answer a patient FAQ in plain non-technical language. "
                "Cite each factual claim with the [N] marker."
            ),
            QueryType.CLINICAL_DECISION: (
                "Provide evidence-based clinical decision support. "
                "Cite every statement with [N]. Include a safety disclaimer."
            ),
            QueryType.DRUG_INTERACTION: (
                "Analyse drug interaction(s). Cite evidence with [N]. "
                "State severity and mechanism. Include a safety disclaimer."
            ),
            QueryType.LITERATURE_MULTIHOP: (
                "Perform multi-hop reasoning across literature evidence. "
                "Cite every reasoning step with [N]. Include a disclaimer."
            ),
            QueryType.CROSS_CORPUS: (
                "Synthesise a global overview from all evidence passages. "
                "Cite every claim with [N]."
            ),
        }
        doc = _SCENARIO_DOCS.get(query_type, "Answer the medical question using the context.")

        # Build a dspy.Signature class for this scenario.
        class _MedicalAnswerOut(BaseModel):
            answer: str
            citations: list[int]
            disclaimer: str = ""

        sig_namespace = {
            "__doc__": doc,
            "__annotations__": {
                "context": str,
                "question": str,
                "language": str,
                "output": _MedicalAnswerOut,
            },
            "context": dspy.InputField(
                desc="Numbered medical context passages in the format: [1] text … [2] text …"
            ),
            "question": dspy.InputField(desc="User's medical question"),
            "language": dspy.InputField(desc="Target response language: en | zh | de"),
            "output": dspy.OutputField(
                desc=(
                    "JSON object with keys: "
                    "answer (string with [N] citations), "
                    "citations (list of int), "
                    "disclaimer (string, mandatory for clinical scenarios)"
                )
            ),
        }
        sig_cls = type(f"_MedSig_{query_type.value}", (dspy.Signature,), sig_namespace)

        predictor = dspy.TypedPredictor(sig_cls)
        _DSPY_PREDICTORS[query_type] = predictor
        logger.info("dspy_predictor_built", query_type=query_type.value)
        return predictor

    except Exception as exc:
        logger.warning("dspy_predictor_build_failed", query_type=query_type.value, error=str(exc))
        return None


_DSPY_LM_CONFIGURED: bool = False


def _configure_dspy_lm(gateway: Any) -> None:
    """Wire a LiteLLMGateway into DSPy's global LM."""
    global _DSPY_LM_CONFIGURED
    if _DSPY_LM_CONFIGURED or not _dspy_imported():
        return
    try:
        import dspy
        from medgraphia.llm.gateway import LLMProvider

        provider: LLMProvider = gateway._provider
        model_name: str = gateway._model_name

        match provider:
            case LLMProvider.OPENAI:
                dspy_model = model_name
            case LLMProvider.ANTHROPIC:
                dspy_model = model_name
            case LLMProvider.DEEPSEEK:
                dspy_model = f"deepseek/{model_name}"
            case LLMProvider.GEMINI:
                dspy_model = f"gemini/{model_name}"
            case LLMProvider.OLLAMA:
                dspy_model = f"ollama/{model_name}"
            case _:
                dspy_model = model_name

        lm = dspy.LM(dspy_model)
        dspy.configure(lm=lm)
        _DSPY_LM_CONFIGURED = True
        logger.info("dspy_lm_configured", model=dspy_model)

    except Exception as exc:
        logger.warning("dspy_lm_configure_failed", error=str(exc))


# ---------------------------------------------------------------------------
# MedicalPredictor — public facade
# ---------------------------------------------------------------------------

class MedicalPredictor:
    """Language-aware predictor for a single query scenario."""

    def __init__(self, query_type: QueryType) -> None:
        self._query_type = query_type
        self._dspy_predictor: Any | None = None

    async def predict(
        self,
        context: str,
        question: str,
        language: Language,
        gateway: Any | None = None,
    ) -> MedicalAnswer:
        if gateway is None:
            return MedicalAnswer(
                answer="Configuration error: no LLM gateway provided.",
                citations=[],
                disclaimer=get_disclaimer(language),
            )

        if self._dspy_predictor is None and _dspy_imported():
            self._dspy_predictor = _build_dspy_predictor(self._query_type)

        if self._dspy_predictor is not None:
            try:
                _configure_dspy_lm(gateway)
                result = self._dspy_predictor(
                    context=context,
                    question=question,
                    language=language.value,
                )
                raw_output = getattr(result, "output", None)
                if raw_output is not None:
                    return _coerce_to_answer(raw_output, self._query_type, language)
            except Exception as exc:
                logger.warning("dspy_predict_failed", error=str(exc))

        return await _json_prompt_predict(
            context=context,
            question=question,
            language=language,
            query_type=self._query_type,
            gateway=gateway,
        )


# ---------------------------------------------------------------------------
# Fallback: direct JSON-prompt path
# ---------------------------------------------------------------------------

async def _json_prompt_predict(
    context: str,
    question: str,
    language: Language,
    query_type: QueryType,
    gateway: Any,
) -> MedicalAnswer:
    """Send a JSON-instructed prompt and parse the response into MedicalAnswer."""
    from medgraphia.llm.gateway import CompletionRequest, _parse_json_safe

    system_prompt = get_system_prompt(query_type, language)
    disc = get_disclaimer(language) if requires_disclaimer(query_type) else ""

    json_template = {
        "answer": f"<your answer in {language.value.upper()} with [N] citations>",
        "citations": [1, 2],
        "disclaimer": disc
    }

    user_prompt = (
        f"CONTEXT PASSAGES:\n{context}\n\n"
        f"QUESTION: {question}\n"
        f"RESPONSE LANGUAGE: {language.value.upper()}\n\n"
        "Respond ONLY with a valid JSON object following this structure:\n"
        f"{json.dumps(json_template, indent=2)}\n\n"
        "Rules:\n"
        "1. Answer strictly based on the context.\n"
        "2. Use [N] for inline citations.\n"
        "3. Output ONLY the JSON."
    )

    req = CompletionRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        temperature=0.1,
    )

    resp = await gateway.acomplete(req)
    if not resp.ok:
        raise RuntimeError(f"LLM API failed: {resp.metadata.get('error')}")

    parsed = _parse_json_safe(resp.text)
    return _coerce_to_answer(parsed, query_type, language)


def _coerce_to_answer(
    raw: Any,
    query_type: QueryType,
    language: Language,
) -> MedicalAnswer:
    if isinstance(raw, MedicalAnswer):
        answer = raw
    elif isinstance(raw, dict):
        answer = MedicalAnswer(**{k: v for k, v in raw.items() if k in MedicalAnswer.model_fields})
    else:
        answer = MedicalAnswer(answer=str(raw), citations=[])

    if requires_disclaimer(query_type) and not answer.disclaimer:
        answer = answer.model_copy(update={"disclaimer": get_disclaimer(language)})

    return answer


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    content = fence.group(1) if fence else text
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(content[start : end + 1])
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Prompt Registry
# ---------------------------------------------------------------------------

class PromptRegistry:
    def __init__(self) -> None:
        self._predictors: dict[QueryType, MedicalPredictor] = {}

    def get(self, query_type: QueryType) -> MedicalPredictor:
        if query_type not in self._predictors:
            self._predictors[query_type] = MedicalPredictor(query_type)
        return self._predictors[query_type]

    def preload_all(self) -> None:
        for qt in QueryType:
            self.get(qt)
