
import dspy
from medgraphia.llm.dspy_setup import get_lm, get_program
from medgraphia.config import settings

def test_rag():
    # 1. Initialize with LARGE tier model
    lm = get_lm(
        task="default",
        provider_override=settings.llm_large_provider,
        model_override=settings.llm_large_model
    )
    dspy.settings.configure(lm=lm)

    # 2. Load the compiled generator
    generator = get_program("generator")

    # 3. Define the problematic cases provided by the user
    test_cases = [
        {
            "name": "Case 1: Ambiguous Pronoun (It)",
            "context": "[1] 尖锐湿疣的治疗包括激光。 [2] 子宫粘连可能导致不孕。 [3] 无精症需进一步检查。 [4] 复发性口腔溃疡。",
            "question": "怎么治疗它？"
        },
        {
            "name": "Case 2: Vague Comparison (Better Way)",
            "context": "[1] 多囊卵巢综合征表现为月经不调。 [2] 长期耳鸣可能与听力受损有关。 [3] 胃溃疡。 [4] 痔疮疼得厉害。",
            "question": "能有更好的办法吗？"
        }
    ]

    print("\n" + "="*80)
    print("RUNNING OPTIMIZED RAG TEST CASES")
    print("="*80 + "\n")

    with dspy.context(lm=lm):
        for case in test_cases:
            print(f"--- {case['name']} ---")
            print(f"Question: {case['question']}")
            
            try:
                prediction = generator(
                    system_instruction="你是一位友善的医疗健康教育者。",
                    context=case['context'],
                    history="No history.",
                    question=case['question'],
                    target_language="Chinese",
                    no_info_message="抱歉，参考资料中没有足够的信息。"
                )
                
                print(f"Reasoning: {getattr(prediction, 'reasoning', 'N/A')}")
                print(f"Answer: {prediction.result.answer}")
                print(f"Citations: {prediction.result.citations}")
            except Exception as e:
                print(f"Error: {e}")
            print("-" * 40 + "\n")

if __name__ == "__main__":
    test_rag()
