"""
DSPy Teleprompter Optimization Script.
Run this script to compile and optimize the MedGraphia DSPy signatures.
"""
from __future__ import annotations

import json
from pathlib import Path
import dspy
from dspy.teleprompt import BootstrapFewShot
from pydantic import BaseModel

from medgraphia.llm.dspy_setup import get_lm
from medgraphia.config import settings

# Force DSPy to use the LARGE tier model (DeepSeek-Chat) for optimization
lm = get_lm(
    task="default",
    provider_override=settings.llm_large_provider,
    model_override=settings.llm_large_model
)
dspy.settings.configure(lm=lm)

# ---------------------------------------------------------
# 1. Datasets
# ---------------------------------------------------------

# Sample Dataset for Query Rewriting
REWRITE_DATA = [
    dspy.Example(
        history="User: 什么是二甲双胍？\nAssistant: 是一种用于治疗2型糖尿病的药物...",
        latest_message="它有什么副作用？",
        result={"is_standalone": False, "rewritten_query": "二甲双胍有什么副作用？"}
    ).with_inputs("history", "latest_message"),
    
    dspy.Example(
        history="User: 患者有高血压和哮喘。\nAssistant: 了解，已记录。",
        latest_message="能给他开布洛芬吗？",
        result={"is_standalone": False, "rewritten_query": "高血压和哮喘患者可以服用布洛芬吗？"}
    ).with_inputs("history", "latest_message"),

    dspy.Example(
        history="",
        latest_message="头痛应该吃什么药？",
        result={"is_standalone": True, "rewritten_query": "头痛应该吃什么药？"}
    ).with_inputs("history", "latest_message"),
]

# Shared cross-language note appended to every system_instruction so the LLM
# learns to synthesize multilingual context throughout the training set.
_CROSSLANG_ZH = "数据库内容可能包含中文、英文或德文，请综合所有语言的相关内容并用中文回答。"
_CROSSLANG_EN = "Context paragraphs may be in English, Chinese, or German — synthesize all relevant content and respond in English."
_CROSSLANG_DE = "Die Kontextabsätze können auf Englisch, Chinesisch oder Deutsch verfasst sein — fassen Sie alle relevanten Inhalte zusammen und antworten Sie auf Deutsch."

_SYS_ZH = f"你是临床决策支持助手，提供仅基于数据库的循证医学建议。{_CROSSLANG_ZH}"
_SYS_EN = f"You are a clinical decision-support assistant. Provide evidence-based guidance based ONLY on the database. {_CROSSLANG_EN}"
_SYS_DE = f"Sie sind ein klinischer Entscheidungsunterstützungsassistent. Geben Sie Empfehlungen basierend auf der Datenbank. {_CROSSLANG_DE}"

# Sample Dataset for Clinical Answer Generation (High-Quality Adversarial + Positive Set)
ANSWER_DATA = [
    # --- CATEGORY A: AMBIGUOUS PRONOUNS (THE "IT" TRAP) ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 尖锐湿疣的物理治疗包括激光和冷冻。 [2] 子宫粘连可能导致不孕。 [3] 某些口腔溃疡具有复发性。",
        history="No history.",
        question="怎么治疗它？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        result={
            "reasoning": "The query uses an ambiguous pronoun ‘它’ (it) with no chat history to resolve it. Cannot determine which condition the user means. Must request clarification via no_info_message without listing context entities.",
            "answer": "抱歉，数据库中没有足够的信息来回答这个问题。请明确您指的「它」是什么疾病，以便我为您查找治疗建议。",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供参考。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY B: VAGUE COMPARISONS (THE "BETTER WAY" TRAP) ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 胃溃疡的典型表现是餐后痛。 [2] 痔疮患者常伴有局部疼痛或出血。",
        history="No history.",
        question="能有更好的办法吗？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        result={
            "reasoning": "Vague comparison ‘更好的办法’ with no subject and no chat history. Cannot infer which disease or treatment the user means. Must use no_info_message to request clarification.",
            "answer": "抱歉，数据库中没有足够的信息来回答这个问题。请问您是指哪种具体疾病或治疗方案的「更好办法」？",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供教育参考。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY C: CROSS-LANGUAGE SYNTHESIS — ZH QUESTION + DE/EN CONTEXT ---
    # CRITICAL: replaces the old harmful "肾衰竭 → no_info" example.
    # Teaches the model to translate and synthesize foreign-language context
    # instead of treating it as missing information.
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 肾病综合征通常表现为大量蛋白尿、高度水肿。 [2] Nierenversagen bezeichnet den vollständigen oder teilweisen Verlust der Nierenfunktion. Es wird in akutes Nierenversagen (ANV) und chronisches Nierenversagen (CNV) unterteilt. Ursachen sind u.a. Diabetes mellitus, Hypertonie und Glomerulonephritis. [3] Renal failure is characterized by the kidneys’ inability to filter waste products from the blood adequately.",
        history="No history.",
        question="什么是肾衰竭？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        result={
            "reasoning": "The user asks for a definition of ‘肾衰竭’ (kidney failure). Context [2] is in German and [3] is in English — both directly define kidney failure / Nierenversagen. Context [1] is in Chinese and covers related nephrotic syndrome. I MUST synthesize ALL paragraphs regardless of language, translate the German and English content into Chinese, and answer comprehensively. The no_info_message must NOT be triggered here — relevant medical information exists in the context, even though it is in a different language than the question.",
            "answer": "根据数据库资料，肾衰竭（德文：Nierenversagen）是指肾脏功能完全或部分丧失，无法有效过滤血液中的废物 [2][3]。按发病缓急可分为急性肾衰竭（ANV）和慢性肾衰竭（CNV）[2]。常见病因包括糖尿病、高血压和肾小球肾炎 [2]。数据库中还提及肾病综合征，其特征为大量蛋白尿和高度水肿，可能是肾衰竭的相关前期疾病 [1]。",
            "citations": [1, 2, 3],
            "disclaimer": "⚠ 本内容仅供教育参考，不构成医疗建议。请务必咨询合格的医疗专业人员。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY C2: CROSS-LANGUAGE CLINICAL CONTEXT (NOT ENCYCLOPEDIC) ---
    # Teaches the model that clinical differential/case-note context IS valid evidence.
    # The LLM must synthesize from clinical German/English content instead of refusing
    # because it lacks a formal encyclopedic definition.
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 肾虚是中医中的一个概念，指的是肾脏功能的虚弱或不足，与西医的肾衰竭（Nierenversagen）是不同的病理概念。 [2] Das Differential für das akute Nierenversagen umfasst prärenale Ursachen (z.B. Hypovolämie, Herzinsuffizienz), renale Ursachen (z.B. Glomerulonephritis, akute Tubulusnekrose) sowie postrenale Ursachen (z.B. Harnwegsobstruktion). [3] Renal Impairment: reduced kidney function may result from prerenal, intrinsic renal, or postrenal causes. Acute kidney injury is characterized by a rapid decline in glomerular filtration rate. [4] Nierenversagen - Der Patient entwickelte eine oligurische Niereninsuffizienz mit Kreatininanstieg auf 4,2 mg/dl.",
        history="No history.",
        question="什么是肾衰竭？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        result={
            "reasoning": (
                "The user asks about kidney failure (肾衰竭). [1] Chinese: '肾虚' (TCM) differs from kidney failure. "
                "[2] German: clinical differential for acute kidney failure — prerenal, renal, postrenal. "
                "[3] English: renal impairment / acute kidney injury = rapid GFR decline. "
                "[4] German: clinical case — oliguria, creatinine 4.2 mg/dl. "
                "[2][3][4] all discuss kidney failure in clinical context — this IS relevant. "
                "I must NOT refuse because I lack a formal definition. Clinical content must be synthesized."
            ),
            "answer": "根据数据库中的临床资料，肾衰竭（德文：Nierenversagen）是指肾脏功能严重减退，表现为肾小球滤过率（GFR）迅速下降，可出现少尿、肌酐显著升高等症状 [3][4]。临床上，急性肾衰竭可分为三类：肾前性（如低血容量、心力衰竭）、肾性（如肾小球肾炎、急性肾小管坏死）和肾后性（如尿路梗阻）[2]。注意：中医的「肾虚」概念与西医肾衰竭是不同的病理概念 [1]。",
            "citations": [1, 2, 3, 4],
            "disclaimer": "⚠ 本内容基于数据库临床资料，仅供教育参考，不构成医疗建议。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY D: CROSS-LANGUAGE SYNTHESIS — EN QUESTION + ZH/DE CONTEXT ---
    # Teaches the model to synthesize Chinese and German evidence for an English answer.
    dspy.Example(
        system_instruction=_SYS_EN,
        context="[1] 二甲双胍是治疗2型糖尿病的一线药物，通过抑制肝糖输出发挥作用。 [2] Metformin ist ein orales Antidiabetikum der Biguanid-Klasse. Es reduziert die hepatische Glukoseproduktion und verbessert die Insulinsensitivität.",
        history="No history.",
        question="How does metformin work?",
        target_language="English",
        no_info_message="I do not have enough medical information in the database to answer this question.",
        result={
            "reasoning": "The user asks how metformin works. Context [1] is in Chinese and [2] is in German — both describe the mechanism of action directly. I must read and translate both foreign-language paragraphs, synthesize their content, and answer in English. The no_info_message must NOT be used because [1] and [2] directly and fully answer the question.",
            "answer": "Metformin is a first-line oral antidiabetic agent in the biguanide class [2]. It works primarily by inhibiting hepatic glucose production (gluconeogenesis) [1][2] and improving peripheral insulin sensitivity [2]. These combined mechanisms lower blood glucose levels in patients with type 2 diabetes [1].",
            "citations": [1, 2],
            "disclaimer": "⚠ This information is for educational purposes only and does not constitute medical advice."
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY E: AMBIGUOUS ENGLISH PRONOUN (REFUSAL) ---
    dspy.Example(
        system_instruction=_SYS_EN,
        context="[1] Hypertension is high blood pressure. [2] Diabetes involves insulin resistance.",
        history="No history.",
        question="How to cure it fast?",
        target_language="English",
        no_info_message="I do not have enough medical information in the database to answer this question.",
        result={
            "reasoning": "Ambiguous pronoun ‘it’ with no history to resolve the reference. Context has two unrelated conditions. Cannot infer which one the user means. Must refuse without speculating.",
            "answer": "I am sorry, but the database does not contain enough information to answer this question. Please specify which condition you are referring to by ‘it’.",
            "citations": [],
            "disclaimer": "⚠ Clinical guidance only."
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY F: POSITIVE MONOLINGUAL ZH (FAITHFULNESS) ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 糖尿病是一组以高血糖为特征的代谢性疾病。 [2] 糖尿病的典型症状是多饮、多食、多尿和体重减轻。",
        history="No history.",
        question="什么是糖尿病？",
        target_language="Chinese",
        no_info_message="数据库中未提及。",
        result={
            "reasoning": "The query asks for a definition of diabetes. Database [1] provides a direct definition and [2] lists symptoms. Both are in Chinese. Synthesize and cite both faithfully.",
            "answer": "根据数据库信息，糖尿病是一组以高血糖为特征的代谢性疾病 [1]。其典型临床表现包括多饮、多食、多尿以及体重减轻（即「三多一少」）[2]。",
            "citations": [1, 2],
            "disclaimer": "⚠ 本内容仅供教育参考。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 阿司匹林建议餐后服用以减少胃肠刺激。 [2] 布洛芬可用于缓解轻至中度疼痛。",
        history="No history.",
        question="阿司匹林怎么吃？",
        target_language="Chinese",
        no_info_message="数据库中未提及。",
        result={
            "reasoning": "Direct match for Aspirin in [1]. Provide the specific dosage advice with citation from the database.",
            "answer": "根据数据库记录，阿司匹林应当在餐后服用，以减少对胃肠道的刺激 [1]。",
            "citations": [1],
            "disclaimer": "⚠ 请遵循医嘱。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY G: IRRELEVANT CONTEXT (GENUINE NO-INFO CASE) ---
    # no_info_message is correct here because ALL context is administratively irrelevant.
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 门诊时间为周一至周五。 [2] 医院停车场位于北门。",
        history="No history.",
        question="感冒了怎么办？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        result={
            "reasoning": "The user asks about cold treatment. ALL context paragraphs are purely administrative (clinic hours, parking). There is zero medical relevance across every language. This is a genuine no-info case and the no_info_message is appropriate.",
            "answer": "抱歉，数据库中没有关于感冒治疗的相关医疗信息。",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供参考。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),
]

# ---------------------------------------------------------
# 2. Metric Functions (Evaluators)
# ---------------------------------------------------------

def rewrite_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Evaluate if the rewritten query resolved pronouns and matches intent."""
    expected = example.result["rewritten_query"].lower()
    predicted = pred.result.rewritten_query.lower()
    
    score = 0.0
    # Check if they are similar (simple exact match for this MVP metric)
    if expected == predicted:
        score += 1.0
    
    # Penalize if predicted contains pronouns
    pronouns = ["它", "他", "她", "这个", "那个", "it", "he", "she", "this", "that"]
    if any(p in predicted for p in pronouns) and not any(p in expected for p in pronouns):
        score -= 0.5
        
    return max(0.0, score)


def answer_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Evaluate if the answer matches the reference, has correct citations,
    and does NOT incorrectly refuse when cross-language context is available.
    """
    predicted_ans = pred.result.answer
    predicted_cites = pred.result.citations
    expected_cites = example.result["citations"]

    score = 0.0

    # 1. Citation correctness (0.4 points)
    if set(expected_cites) == set(predicted_cites):
        score += 0.4
    elif set(expected_cites).issubset(set(predicted_cites)):
        score += 0.2  # over-citation is a minor penalty

    # 2. [N] markers present in text (0.3 points)
    if predicted_cites:
        if all(f"[{c}]" in predicted_ans for c in predicted_cites):
            score += 0.3
    else:
        score += 0.3  # no citations expected and none produced

    # 3. Content quality (0.3 points)
    is_refusal_case = len(expected_cites) == 0
    refusal_keywords = ["抱歉", "没有足够", "不足", "sorry", "not enough", "cannot identify"]

    if is_refusal_case:
        # Refusal cases: reward concise refusal, penalise hallucination
        if any(kw in predicted_ans for kw in refusal_keywords):
            score += 0.3
    else:
        # Positive (synthesis) cases: reward substantive answers, penalise false refusal.
        # A cross-language case that incorrectly says "抱歉/sorry" loses all content points.
        if any(kw in predicted_ans for kw in refusal_keywords):
            score -= 0.3  # false refusal when context IS relevant
        elif len(predicted_ans) > 50:
            score += 0.3

    return max(0.0, score)

# ---------------------------------------------------------
# 3. Compilation Functions
# ---------------------------------------------------------

def compile_rewriter():
    print("\n--- Compiling Query Rewriter ---")
    from medgraphia.prompts import RewriteMedicalQuery, RewrittenQuery
    
    class RewriterModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.Predict(RewriteMedicalQuery)
            
        def forward(self, history, latest_message):
            return self.prog(history=history, latest_message=latest_message)
            
    module = RewriterModule()
    teleprompter = BootstrapFewShot(metric=rewrite_metric, max_bootstrapped_demos=2, max_labeled_demos=2)
    
    compiled_rewriter = teleprompter.compile(module, trainset=REWRITE_DATA)
    
    out_dir = Path("data/dspy")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rewriter_compiled.json"
    compiled_rewriter.save(str(out_path))
    print(f"Compiled rewriter saved to {out_path}")


def compile_generator():
    print("\n--- Compiling Answer Generator ---")
    from medgraphia.prompts import GenerateClinicalAnswer, MedicalAnswer

    class GeneratorModule(dspy.Module):
        def __init__(self):
            super().__init__()
            # Use ChainOfThought for complex clinical reasoning
            self.prog = dspy.ChainOfThought(GenerateClinicalAnswer)
            
        def forward(self, system_instruction, context, history, question, target_language, no_info_message):
            return self.prog(
                system_instruction=system_instruction,
                context=context,
                history=history,
                question=question,
                target_language=target_language,
                no_info_message=no_info_message
            )

    module = GeneratorModule()
    # Using BootstrapFewShot to find the best prompt configuration
    teleprompter = BootstrapFewShot(
        metric=answer_metric, 
        max_bootstrapped_demos=3, 
        max_labeled_demos=8
    )
    
    compiled_generator = teleprompter.compile(module, trainset=ANSWER_DATA)
    
    out_dir = Path("data/dspy")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generator_compiled.json"
    compiled_generator.save(str(out_path))
    print(f"Compiled generator saved to {out_path}")


if __name__ == "__main__":
    compile_rewriter()
    compile_generator()
