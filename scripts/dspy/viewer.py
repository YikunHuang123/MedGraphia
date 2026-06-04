
import dspy
from medgraphia.llm.dspy_setup import get_lm, get_program
from medgraphia.config import settings

def inspect():
    # 1. Initialize with LARGE tier model (DeepSeek)
    lm = get_lm(
        task="default",
        provider_override=settings.llm_large_provider,
        model_override=settings.llm_large_model
    )
    dspy.settings.configure(lm=lm)

    # 2. Load the compiled generator
    generator = get_program("generator")

    # 3. Simulate the adversarial "it" case
    context = "[1] 尖锐湿疣的治疗方法包括物理治疗和药物治疗。物理治疗如激光、电灼等。"
    question = "怎么治疗它？"

    print("\n" + "="*80)
    print("DSPy PROMPT INSPECTION (COMPILED WITH DEEPSEEK-CHAT)")
    print("="*80 + "\n")

    # Run one prediction to populate history
    with dspy.context(lm=lm):
        try:
            prediction = generator(
                system_instruction="你是临床决策支持助手，提供仅基于数据库的循证医学建议。忽略无关数据。",
                context=case['context'],
                history="No history.",
                question=case['question'],
                target_language="Chinese",
                no_info_message="抱歉，数据库中没有足够的信息。"
            )

            # Print the rendered prompt
            lm.inspect_history(n=1)
        except Exception as e:
            print(f"Error during inspection: {e}")

if __name__ == "__main__":
    inspect()
