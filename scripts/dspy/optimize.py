"""
DSPy Teleprompter Optimization Script.
Run this script to compile and optimize the MedGraphia DSPy signatures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

from medgraphia.domain.base import Language
from medgraphia.llm.dspy_setup import get_lm
from medgraphia.prompts.query_rewriting import RewritingJudge

# Initialize LMs
# - default_lm: used for the modules being optimized (the 'student')
# - judge_lm: used for the metrics and evaluation (the 'teacher')
default_lm = get_lm(task="default")
judge_lm = get_lm(task="judge")

dspy.settings.configure(lm=default_lm)


# Shared cross-language note appended to every system_instruction so the LLM
# learns to synthesize multilingual context throughout the training set.
_CROSSLANG_ZH = "数据库内容可能包含中文、英文或德文，请综合所有语言的相关内容并用中文回答。"
_CROSSLANG_EN = "Context paragraphs may be in English, Chinese, or German — synthesize all relevant content and respond in English."
_CROSSLANG_DE = "Die Kontextabsätze können auf Englisch, Chinesisch oder Deutsch verfasst sein — fassen Sie alle relevanten Inhalte zusammen und antworten Sie auf Deutsch."

_SYS_ZH = f"你是临床决策支持助手，提供仅基于数据库的循证医学建议。{_CROSSLANG_ZH}"
_SYS_EN = f"You are a clinical decision-support assistant. Provide evidence-based guidance based ONLY on the database. {_CROSSLANG_EN}"
_SYS_DE = f"Sie sind ein klinischer Entscheidungsunterstützungsassistent. Geben Sie Empfehlungen basierend auf der Datenbank. {_CROSSLANG_DE}"


# ---------------------------------------------------------
# 1. Datasets
# ---------------------------------------------------------

# Sample Dataset for Query Rewriting
REWRITE_DATA = [
    # 1. Long-Distance Coreference & Distractor Entities
    # E=1 (rheumatoid arthritis), I=2 (disease progression outcome) → Total=3 → SMALL
    dspy.Example(
        history="User: 医生，我妈妈上周去医院检查，被诊断出得了类风湿性关节炎，她现在非常焦虑。\nAssistant: 这是一种慢性的自身免疫性疾病，虽然目前无法彻底治愈，但通过正规的药物治疗是可以很好控制病情的。请安抚她的情绪。\nUser: 好的。她最近这几天总是说胃不太舒服，吃完饭就觉得有点泛酸，也不知道是不是心理作用。\nAssistant: 明白。胃部泛酸可能与饮食习惯有关，也可能是情绪焦虑引起的胃肠功能紊乱，另外，如果她已经开始服用某些治疗关节炎的药物（如非甾体抗炎药），也可能会刺激胃黏膜。\nUser: 我们昨天去药店，店员还推荐买了一些碳酸钙片，说老年人最好补补钙。",
        latest_message="那这个病发展到后期会导致患者的手指关节完全变形吗？",
        result={
            "is_standalone": False,
            "rewritten_query": "类风湿性关节炎发展到后期会导致患者的手指关节完全变形吗？",
        },
        expected_tier="SMALL",
    ).with_inputs("history", "latest_message"),
    # 2. Implicit Subject Continuity & High Verbosity
    # E=2 (atorvastatin + liver/alcohol), I=2 (monitoring plan) → Total=4 → MEDIUM
    dspy.Example(
        history="User: 我最近一直在按照心内科医生的嘱咐，每天晚上睡前服用阿托伐他汀钙片来控制我的血脂水平。\nAssistant: 这是一个很好的用药习惯。阿托伐他汀主要用于降低血液中的低密度脂蛋白胆固醇（坏胆固醇），从而减少心血管疾病的风险。夜间服用通常效果更好，因为肝脏合成胆固醇的高峰期在夜间。",
        latest_message="我听说长期吃这种西药对身体的负担特别重，尤其是我平时偶尔还会喝点白酒，想问一下长期吃的话对肝脏功能损害大不大，需要定期去查什么指标？",
        result={
            "is_standalone": False,
            "rewritten_query": "长期服用阿托伐他汀钙片对肝脏功能损害大不大？如果偶尔饮酒，需要定期检查哪些肝功能指标？",
        },
        expected_tier="MEDIUM",
    ).with_inputs("history", "latest_message"),
    # 3. Multi-Entity Disambiguation with Conditional Logic
    # E=3 (hypertension + gout + amlodipine), I=3 (complex drug-disease interaction with complication risk) → Total=6 → LARGE
    dspy.Example(
        history="User: 我目前的身体状况比较复杂，本来就一直患有原发性高血压，每天早上都在吃氨氯地平控压。\nAssistant: 氨氯地平是长效钙通道阻滞剂，是治疗高血压的一线药物，通常耐受性良好。\nUser: 但是屋漏偏逢连夜雨，我上个月去体检，又查出尿酸偏高，确诊了痛风，前天大脚趾还疼得走不了路。",
        latest_message="既然我的情况这么复杂，那它会不会不仅降不了压，反而引起我体内的尿酸水平进一步升高，从而导致急性发作更频繁？",
        result={
            "is_standalone": False,
            "rewritten_query": "原发性高血压合并痛风的患者服用氨氯地平，会不会导致体内尿酸水平进一步升高，从而引起痛风急性发作更频繁？",
        },
        expected_tier="LARGE",
    ).with_inputs("history", "latest_message"),
    # 4. False Pronoun / Substring Trap (Adversarial)
    # E=2 (postmenopausal osteoporosis + vitamin D/supplements), I=2 (supplementation decision) → Total=4 → MEDIUM
    dspy.Example(
        history="User: 针对绝经后女性的严重骨质疏松症，在日常饮食和营养补充方面，究竟有哪些最权威、最不容忽视的医学建议？\nAssistant: 对于绝经后骨质疏松女性，权威建议包括：1. 保证充足的钙摄入（每日1000-1200mg）。2. 充足的蛋白质摄入。3. 适度的负重运动。",
        latest_message="除了你上面提到的这些常规建议之外，我想进一步确认，患者到底需不需要大量补充维他命D以及其它那个昂贵的微量元素保健品呢？",
        result={
            "is_standalone": False,
            "rewritten_query": "绝经后骨质疏松女性患者是否需要大量补充维他命D以及其它微量元素保健品？",
        },
        expected_tier="MEDIUM",
    ).with_inputs("history", "latest_message"),
    # 5. Cross-Lingual Continuity (English Context -> Chinese Query)
    # E=2 (MDD + SSRIs/SNRIs), I=2 (safety/side-effect question) → Total=4 → MEDIUM
    dspy.Example(
        history="User: I was reading my lab report and it says I am diagnosed with Major Depressive Disorder (MDD). What are the standard first-line pharmacological treatments for this?\nAssistant: For Major Depressive Disorder (MDD), first-line pharmacological treatments typically include Selective Serotonin Reuptake Inhibitors (SSRIs) such as sertraline or fluoxetine, and Serotonin-Norepinephrine Reuptake Inhibitors (SNRIs).",
        latest_message="因为我平时还要开车上班，如果我开始吃这些药，这个病或者这些药物会不会严重影响我的驾驶安全性？",
        result={
            "is_standalone": False,
            "rewritten_query": "重度抑郁症(Major Depressive Disorder)患者服用SSRI或SNRI类抗抑郁药物，是否会严重影响驾驶安全性？",
        },
        expected_tier="MEDIUM",
    ).with_inputs("history", "latest_message"),
]

# Sample Dataset for Clinical Answer Generation (High-Quality Adversarial + Positive Set)
ANSWER_DATA = [
    # --- CATEGORY A: AMBIGUOUS PRONOUNS ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 尖锐湿疣的物理治疗包括激光和冷冻。 [2] 子宫粘连可能导致不孕。 [3] 某些口腔溃疡具有复发性。",
        history="No history.",
        question="怎么治疗它？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        rationale="The query uses an ambiguous pronoun ‘它’ (it) with no chat history to resolve it. Cannot determine which condition the user means. Must request clarification via no_info_message without listing context entities.",
        result={
            "answer": "抱歉，数据库中没有足够的信息来回答这个问题。请明确您指的「它」是什么疾病，以便我为您查找治疗建议。",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供参考。",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    # --- CATEGORY B: VAGUE COMPARISONS ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 胃溃疡的典型表现是餐后痛。 [2] 痔疮患者常伴有局部疼痛或出血。",
        history="No history.",
        question="能有更好的办法吗？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        rationale="Vague comparison ‘更好的办法’ with no subject and no chat history. Cannot infer which disease or treatment the user means. Must use no_info_message to request clarification.",
        result={
            "answer": "抱歉，数据库中没有足够的信息来回答这个问题。请问您是指哪种具体疾病或治疗方案的「更好办法」？",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供教育参考。",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    # --- CATEGORY C1: CROSS-LANGUAGE SYNTHESIS — ZH QUESTION + DE/EN CONTEXT ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 肾病综合征通常表现为大量蛋白尿、高度水肿。 [2] Nierenversagen bezeichnet den vollständigen oder teilweisen Verlust der Nierenfunktion. Es wird in akutes Nierenversagen (ANV) and chronisches Nierenversagen (CNV) unterteilt. Ursachen sind u.a. Diabetes mellitus, Hypertonie und Glomerulonephritis. [3] Renal failure is characterized by the kidneys’ inability to filter waste products from the blood adequately.",
        history="No history.",
        question="什么是肾衰竭？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        rationale="The user asks for a definition of ‘肾衰竭’ (kidney failure). Context [2] is in German and [3] is in English — both directly define kidney failure / Nierenversagen. Context [1] is in Chinese and covers related nephrotic syndrome. I MUST synthesize ALL paragraphs regardless of language, translate the German and English content into Chinese, and answer comprehensively. The no_info_message must NOT be triggered here — relevant medical information exists in the context, even though it is in a different language than the question.",
        result={
            "answer": "根据数据库资料，肾衰竭（德文：Nierenversagen）是指肾脏功能完全或部分丧失，无法有效过滤血液中的废物 [2][3]。按发病缓急可分为急性肾衰竭（ANV）和慢性肾衰竭（CNV）[2]。常见病因包括糖尿病、高血压和肾小球肾炎 [2]。数据库中还提及肾病综合征，其特征为大量蛋白尿和高度水肿，可能是肾衰竭的相关前期疾病 [1]。",
            "citations": [1, 2, 3],
            "disclaimer": "⚠ 本内容仅供教育参考，不构成医疗建议。请务必咨询合格的医疗专业人员。",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    # --- CATEGORY C2: CROSS-LANGUAGE CLINICAL CONTEXT (NOT ENCYCLOPEDIC) ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 肾虚是中医中的一个概念，指的是肾脏功能的虚弱或不足，与西医的肾衰竭（Nierenversagen）是不同的病理概念。 [2] Das Differential für das akute Nierenversagen umfasst prärenale Ursachen (z.B. Hypovolämie, Herzinsuffizienz), renale Ursachen (z.B. Glomerulonephritis, akute Tubulusnekrose) sowie postrenale Ursachen (z.B. Harnwegsobstruktion). [3] Renal Impairment: reduced kidney function may result from prerenal, intrinsic renal, or postrenal causes. Acute kidney injury is characterized by a rapid decline in glomerular filtration rate. [4] Nierenversagen - Der Patient entwickelte eine oligurische Niereninsuffizienz mit Kreatininanstieg auf 4,2 mg/dl.",
        history="No history.",
        question="什么是肾衰竭？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        rationale=(
            "The user asks about kidney failure (肾衰竭). [1] Chinese: '肾虚' (TCM) differs from kidney failure. "
            "[2] German: clinical differential for acute kidney failure — prerenal, renal, postrenal. "
            "[3] English: renal impairment / acute kidney injury = rapid GFR decline. "
            "[4] German: clinical case — oliguria, creatinine 4.2 mg/dl. "
            "[2][3][4] all discuss kidney failure in clinical context — this IS relevant. "
            "I must NOT refuse because I lack a formal definition. Clinical content must be synthesized."
        ),
        result={
            "answer": "根据数据库中的临床资料，肾衰竭（德文：Nierenversagen）是指肾脏功能严重减退，表现为肾小球滤过率（GFR）迅速下降，可出现少尿、肌酐显著升高等症状 [3][4]。临床上，急性肾衰竭可分为三类：肾前性（如低血容量、心力竭）、肾性（如肾小球肾炎、急性肾小管坏死）和肾后性（如尿路梗阻）[2]。注意：中医的「肾虚」概念与西医肾衰竭是不同的病理概念 [1]。",
            "citations": [1, 2, 3, 4],
            "disclaimer": "⚠ 本内容基于数据库临床资料，仅供教育参考，不构成医疗建议。",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    # --- CATEGORY D: CROSS-LANGUAGE SYNTHESIS — EN QUESTION + ZH/DE CONTEXT ---
    # Teaches the model to synthesize Chinese and German evidence for an English answer.
    dspy.Example(
        system_instruction=_SYS_EN,
        context="[1] 二甲双胍是治疗2型糖尿病的一线药物，通过抑制肝糖输出发挥作用。 [2] Metformin ist ein orales Antidiabetikum der Biguanid-Klasse. Es reduziert die hepatische Glukoseproduktion und verbessert die Insulinsensitivität.",
        history="No history.",
        question="How does metformin work?",
        target_language="English",
        no_info_message="I do not have enough medical information in the database to answer this question.",
        rationale="The user asks how metformin works. Context [1] is in Chinese and [2] is in German — both describe the mechanism of action directly. I must read and translate both foreign-language paragraphs, synthesize their content, and answer in English. The no_info_message must NOT be used because [1] and [2] directly and fully answer the question.",
        result={
            "answer": "Metformin is a first-line oral antidiabetic agent in the biguanide class [2]. It works primarily by inhibiting hepatic glucose production (gluconeogenesis) [1][2] and improving peripheral insulin sensitivity [2]. These combined mechanisms lower blood glucose levels in patients with type 2 diabetes [1].",
            "citations": [1, 2],
            "disclaimer": "⚠ This information is for educational purposes only and does not constitute medical advice.",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    # --- CATEGORY E: AMBIGUOUS ENGLISH PRONOUN (REFUSAL) ---
    dspy.Example(
        system_instruction=_SYS_EN,
        context="[1] Hypertension is high blood pressure. [2] Diabetes involves insulin resistance.",
        history="No history.",
        question="How to cure it fast?",
        target_language="English",
        no_info_message="I do not have enough medical information in the database to answer this question.",
        rationale="Ambiguous pronoun ‘it’ with no history to resolve the reference. Context has two unrelated conditions. Cannot infer which one the user means. Must refuse without speculating.",
        result={
            "answer": "I am sorry, but the database does not contain enough information to answer this question. Please specify which condition you are referring to by ‘it’.",
            "citations": [],
            "disclaimer": "⚠ Clinical guidance only.",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    # --- CATEGORY F: POSITIVE MONOLINGUAL ZH (FAITHFULNESS) ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 糖尿病是一组以高血糖为特征的代谢性疾病。 [2] 糖尿病的典型症状是多饮、多食、多尿和体重减轻。",
        history="No history.",
        question="什么是糖尿病？",
        target_language="Chinese",
        no_info_message="数据库中未提及。",
        rationale="The query asks for a definition of diabetes. Database [1] provides a direct definition and [2] lists symptoms. Both are in Chinese. Synthesize and cite both faithfully.",
        result={
            "answer": "根据数据库信息，糖尿病是一组以高血糖为特征的代谢性疾病 [1]。其典型临床表现包括多饮、多食、多尿以及体重减轻（即「三多一少」）[2]。",
            "citations": [1, 2],
            "disclaimer": "⚠ 本内容仅供教育参考。",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 阿司匹林建议餐后服用以减少胃肠刺激。 [2] 布洛芬可用于缓解轻至中度疼痛。",
        history="No history.",
        question="阿司匹林怎么吃？",
        target_language="Chinese",
        no_info_message="数据库中未提及。",
        rationale="Direct match for Aspirin in [1]. Provide the specific dosage advice with citation from the database.",
        result={
            "answer": "根据数据库记录，阿司匹林应当在餐后服用，以减少对胃肠道的刺激 [1]。",
            "citations": [1],
            "disclaimer": "⚠ 请遵循医嘱。",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
    # --- CATEGORY G: IRRELEVANT CONTEXT (GENUINE NO-INFO CASE) ---
    dspy.Example(
        system_instruction=_SYS_ZH,
        context="[1] 门诊时间为周一至周五。 [2] 医院停车场位于北门。",
        history="No history.",
        question="感冒了怎么办？",
        target_language="Chinese",
        no_info_message="抱歉，数据库中没有足够的信息来回答这个问题。",
        rationale="The user asks about cold treatment. ALL context paragraphs are purely administrative (clinic hours, parking). There is zero medical relevance across every language. This is a genuine no-info case and the no_info_message is appropriate.",
        result={
            "answer": "抱歉，数据库中没有关于感冒治疗的相关医疗信息。",
            "citations": [],
            "disclaimer": "⚠ 本内容仅供参考。",
        },
    ).with_inputs(
        "system_instruction", "context", "history", "question", "target_language", "no_info_message"
    ),
]

# ---------------------------------------------------------
# 2. Metric Functions (Evaluators)
# ---------------------------------------------------------


def rewrite_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Evaluate both coreference resolution quality and complexity tier accuracy.

    RewriterModule.forward() returns a flat dspy.Prediction with fields:
      rewritten_query, complexity_tier, complexity_reasoning, is_standalone, tier
    (no nested .result — that was the old RewriteMedicalQuery signature).

    Scoring:
      0.7 — rewriting quality (LLM judge, threshold 0.6)
      0.3 — tier classification (exact string match after normalisation)
    Returns 1.0 only when both components pass; 0.0 otherwise.
    """
    expected_query = example.result["rewritten_query"]
    predicted_query = pred.rewritten_query  # flat field, NOT pred.result.rewritten_query

    # --- Rewriting quality (0.7 weight) ---
    rewrite_ok = False
    if expected_query.strip().lower() == predicted_query.strip().lower():
        rewrite_ok = True
    else:
        judge = dspy.Predict(RewritingJudge)
        try:
            with dspy.context(lm=judge_lm):
                result = judge(
                    history=example.history,
                    latest_message=example.latest_message,
                    expected=expected_query,
                    predicted=predicted_query,
                )
            match = re.search(r"[01]\.\d+|\b[01]\b", str(result.score))
            score = float(match.group(0)) if match else float(result.score)
            score = max(0.0, min(1.0, score))
            rewrite_ok = score > 0.6
        except Exception as e:
            print(f"[Warning] LLM Judge failed: {e}")
            rewrite_ok = False

    # --- Tier accuracy (0.3 weight) ---
    # If the example has no expected_tier label, skip this component (give full credit).
    expected_tier = getattr(example, "expected_tier", None)
    if expected_tier is None:
        tier_ok = True
    else:
        tier_ok = pred.complexity_tier.strip().upper() == expected_tier.strip().upper()

    return 1.0 if (rewrite_ok and tier_ok) else 0.0


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
    from medgraphia.llm.dspy_setup import get_lm
    from medgraphia.programs.rewriter import RewriterModule

    # Explicitly use the rewriter-specific LM (e.g., Qwen) as the 'student'
    student_lm = get_lm("rewriter")

    module = RewriterModule()

    # Merge handcrafted + synthetic examples.  Handcrafted come first so they
    # are prioritised as labeled demos; synthetic examples expand the bootstrap pool.
    synthetic = load_synthetic_rewriter_data()
    trainset = REWRITE_DATA + synthetic
    print(
        f"  Trainset: {len(REWRITE_DATA)} handcrafted + {len(synthetic)} synthetic = {len(trainset)} total"
    )

    teleprompter = BootstrapFewShot(
        metric=rewrite_metric,
        max_bootstrapped_demos=7,
        max_labeled_demos=5,
    )

    # Run compilation within the context of the student LM
    with dspy.context(lm=student_lm):
        compiled_rewriter = teleprompter.compile(module, trainset=trainset)

    # Save the compiled module

    # Ensure we save to the root data directory regardless of where the script is run
    base_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = base_dir / "data" / "dspy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rewriter_compiled.json"
    compiled_rewriter.save(str(out_path))
    print(f"Compiled rewriter saved to {out_path}")

    # print("\n--- [DSPy Inspection] Last 10 LLM Interactions ---")
    # dspy.inspect_history(n=10)


def compile_generator():
    print("\n--- Compiling Answer Generator ---")
    from medgraphia.programs.generator import GeneratorModule

    module = GeneratorModule()

    # Merge handcrafted + synthetic examples.  Handcrafted come first so they
    # are prioritised as labeled demos; synthetic examples expand the bootstrap pool.
    synthetic = load_synthetic_generator_data()
    trainset = ANSWER_DATA + synthetic
    print(
        f"  Trainset: {len(ANSWER_DATA)} handcrafted + {len(synthetic)} synthetic = {len(trainset)} total"
    )

    teleprompter = BootstrapFewShot(
        metric=answer_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
    )

    compiled_generator = teleprompter.compile(module, trainset=trainset)

    # Save the compiled module
    base_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = base_dir / "data" / "dspy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generator_compiled.json"
    compiled_generator.save(str(out_path))
    print(f"Compiled generator saved to {out_path}")

    dspy.inspect_history(n=10)


# ---------------------------------------------------------
# 4. Synthetic data loaders
# ---------------------------------------------------------


def load_synthetic_generator_data() -> list[dspy.Example]:
    """
    Load synthetic generator examples from data/dspy/synthetic_generator_data.json.

    Each JSON record must have: context, history, question, answer, citations, answer_language.
    Returns an empty list if the file does not exist — compile_generator() falls back
    to the handcrafted ANSWER_DATA set.
    """
    path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "dspy"
        / "synthetic_generator_data.json"
    )
    if not path.exists():
        print(f"[INFO] No synthetic generator data found at {path} — using handcrafted set only.")
        return []

    _LANG_SYS = {
        Language.ZH.full_name: _SYS_ZH,
        Language.DE.full_name: _SYS_DE,
        Language.EN.full_name: _SYS_EN,
    }
    _NO_INFO = {
        Language.ZH.full_name: "抱歉，数据库中没有足够的信息来回答这个问题。",
        Language.DE.full_name: "Es tut mir leid, die Datenbank enthält nicht genügend Informationen.",
        Language.EN.full_name: "I do not have enough medical information in the database to answer this question.",
    }

    examples: list[dspy.Example] = []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    for d in raw:
        # Match the field name from generate_synthetic_data.py
        lang_name = d.get("target_language", Language.EN.full_name)
        examples.append(
            dspy.Example(
                system_instruction=_LANG_SYS.get(lang_name, _SYS_EN),
                context=d["context"],
                history=d.get("history", "No history."),
                question=d["question"],
                target_language=lang_name,
                no_info_message=_NO_INFO.get(lang_name, _NO_INFO[Language.EN.full_name]),
                rationale="Synthetic example generated by DeepSeek teacher model.",
                result={
                    "answer": d["answer"],
                    "citations": d.get("citations", []),
                    "disclaimer": d.get("disclaimer", ""),
                },
            ).with_inputs(
                "system_instruction",
                "context",
                "history",
                "question",
                "target_language",
                "no_info_message",
            )
        )

    print(f"[INFO] Loaded {len(examples)} synthetic generator examples from {path.name}")
    return examples


def load_synthetic_rewriter_data() -> list[dspy.Example]:
    """
    Load synthetic rewriter examples from data/dspy/synthetic_rewriter_data.json.

    Each JSON record must have: history, latest_message, expected_rewritten_query, expected_tier.
    Returns an empty list if the file does not exist — compile_rewriter() falls back
    to the handcrafted REWRITE_DATA set.
    """
    path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "dspy"
        / "synthetic_rewriter_data.json"
    )
    if not path.exists():
        print(f"[INFO] No synthetic rewriter data found at {path} — using handcrafted set only.")
        return []

    examples: list[dspy.Example] = []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    for d in raw:
        examples.append(
            dspy.Example(
                history=d["history"],
                latest_message=d["latest_message"],
                result={"is_standalone": False, "rewritten_query": d["expected_rewritten_query"]},
                expected_tier=d["expected_tier"],
            ).with_inputs("history", "latest_message")
        )

    print(f"[INFO] Loaded {len(examples)} synthetic rewriter examples from {path.name}")
    return examples


if __name__ == "__main__":
    compile_rewriter()
    compile_generator()
