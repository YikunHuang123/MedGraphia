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

# Sample Dataset for Clinical Answer Generation (High-Quality Adversarial Set)
ANSWER_DATA = [
    # --- CATEGORY A: AMBIGUOUS PRONOUNS (THE "IT" TRAP) ---
    dspy.Example(
        system_instruction="你是一位友善的医疗健康教育者。",
        context="[1] 尖锐湿疣的物理治疗包括激光和冷冻。 [2] 子宫粘连可能导致不孕。 [3] 某些口腔溃疡具有复发性。",
        history="No history.",
        question="怎么治疗它？",
        target_language="Chinese",
        no_info_message="抱歉，提供的参考资料中没有足够的信息来回答。请明确您指的“它”是什么疾病。",
        result={
            "reasoning": "用户使用了模糊代词“它”且对话历史为空。上下文[1][2][3]涉及多种完全不相关的疾病。严禁在主语不明的情况下通过‘列举上下文疾病’来猜测用户意图。必须直接说明信息不足并要求澄清，禁止汇总无关噪音。",
            "answer": "抱歉，提供的参考资料中没有足够的信息来回答这个问题。由于您没有明确说明“它”指代哪种疾病，且参考资料涉及多种不相关的医学情况，我无法为您提供准确的建议。请补充具体的疾病或症状名称。",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供教育参考，不构成医疗建议。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY B: VAGUE COMPARISONS (THE "BETTER WAY" TRAP) ---
    dspy.Example(
        system_instruction="你是一位友善的医疗健康教育者。",
        context="[1] 胃溃疡的典型表现是餐后痛。 [2] 痔疮患者常伴有局部疼痛或出血。",
        history="No history.",
        question="能有更好的办法吗？",
        target_language="Chinese",
        no_info_message="抱歉，参考资料中没有足够的信息。请问您是指哪种疾病或治疗的‘更好办法’？",
        result={
            "reasoning": "问题“能有更好的办法吗”属于比较级查询，但在缺乏基准（具体疾病或当前疗法）的情况下无法回答。上下文内容[1][2]与‘办法’无关。应直接拒答，严禁罗列胃溃疡或痔疮的内容作为填充。",
            "answer": "抱歉，提供的参考资料中没有足够的信息来回答。由于缺乏具体背景或疾病名称，我无法判断您指的“更好办法”是针对何种医疗情况。请提供具体疾病名称以便我为您查找相关建议。",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供教育参考。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY C: ENTITY MISMATCH (THE "KIDNEY" CASE) ---
    dspy.Example(
        system_instruction="你是一位友善的医疗健康教育者。",
        context="[1] 肾病综合征通常表现为大量蛋白尿、高度水肿。 [2] 慢性肾炎患者应定期复查肾功能。",
        history="No history.",
        question="什么是肾衰竭？",
        target_language="Chinese",
        no_info_message="抱歉，参考资料中没有关于“肾衰竭”的直接定义。",
        result={
            "reasoning": "问题是针对“肾衰竭”，上下文[1][2]仅提及“肾病综合征”和“慢性肾炎”。虽然领域相关，但非同一实体。应声明无直接定义，仅在用户可能感兴趣的前提下极简提及相关内容，不应长篇大论。",
            "answer": "抱歉，参考资料中没有关于“肾衰竭”的直接定义。资料主要涉及肾病综合征[1]和慢性肾炎[2]的信息。如果您需要了解这些相关疾病，请告知。",
            "citations": [1, 2],
            "disclaimer": "⚠ 本内容仅供教育参考。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY D: CROSS-LINGUAL NOISE ---
    dspy.Example(
        system_instruction="You are a helpful medical assistant.",
        context="[1] Hypertension is high blood pressure. [2] Diabetes involves insulin resistance.",
        history="No history.",
        question="How to cure it fast?",
        target_language="English",
        no_info_message="I'm sorry, I don't have enough context to identify what 'it' refers to.",
        result={
            "reasoning": "The query 'it' is ambiguous without history. Even though context mentions hypertension and diabetes, assuming 'it' refers to one of them is clinically unsafe. Standard refusal required.",
            "answer": "I'm sorry, but the provided context does not have enough information to answer your question. I cannot determine what 'it' refers to. Please specify the condition you are asking about.",
            "citations": [],
            "disclaimer": "⚠ Educational purposes only."
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY E: POSITIVE CASE (FAITHFULNESS) ---
    dspy.Example(
        system_instruction="你是一位友善的医疗健康教育者。",
        context="[1] 阿司匹林建议餐后服用以减少胃肠刺激。 [2] 布洛芬可用于缓解轻至中度疼痛。",
        history="No history.",
        question="阿司匹林怎么吃？",
        target_language="Chinese",
        no_info_message="资料中未提及。",
        result={
            "reasoning": "精准匹配。问题主语明确（阿司匹林），上下文[1]提供了具体服用建议。",
            "answer": "根据资料，建议在餐后服用阿司匹林，以减少对胃肠道的刺激 [1]。",
            "citations": [1],
            "disclaimer": "⚠ 请遵循医嘱。"
        }
    ).with_inputs("system_instruction", "context", "history", "question", "target_language", "no_info_message"),

    # --- CATEGORY F: IRRELEVANT FRAGMENTS ---
    dspy.Example(
        system_instruction="你是一位友善的医疗健康教育者。",
        context="[1] 门诊时间为周一至周五。 [2] 医院停车场位于北门。",
        history="No history.",
        question="感冒了怎么办？",
        target_language="Chinese",
        no_info_message="抱歉，参考资料中没有关于感冒治疗的信息。",
        result={
            "reasoning": "问题是医疗咨询，但上下文[1][2]仅为行政/地理信息，完全无关。禁止利用常识回答感冒方案，必须严格遵守RAG边界。",
            "answer": "抱歉，提供的参考资料中没有关于感冒治疗建议的信息。资料仅涉及门诊时间和停车场说明。建议咨询医生以获取治疗方案。",
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
    """Evaluate if the answer matches the reference and has correct citations."""
    predicted_ans = pred.result.answer
    predicted_cites = pred.result.citations
    
    expected_cites = example.result["citations"]
    
    score = 0.0
    
    # 1. Correctness of citations (weighted)
    if set(expected_cites) == set(predicted_cites):
        score += 0.4
    elif set(expected_cites).issubset(set(predicted_cites)):
        score += 0.2 # Penalize over-citation
        
    # 2. Formatting: Are [N] markers in the text?
    all_markers_present = all(f"[{c}]" in predicted_ans for c in predicted_cites)
    if all_markers_present and predicted_cites:
        score += 0.3
    elif not predicted_cites:
        score += 0.3 # If no citations expected and none found, that's good
        
    # 3. Content Relevance (Simple heuristic for now)
    # If it's a refusal case, check if the refusal message or keywords are there
    if not expected_cites:
        if any(kw in predicted_ans for kw in ["抱歉", "没", "不足", "clear", "sorry"]):
            score += 0.3
    else:
        # If it's a helpful case, check if length is reasonable
        if len(predicted_ans) > 20:
            score += 0.3
            
    return score

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
        max_labeled_demos=3
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
