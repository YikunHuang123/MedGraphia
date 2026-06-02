"""
MedGraphia End-to-End Test: From Query to Cited Answer.

This script runs the full Phase 6 (Retrieval) + Phase 7 (Generation) pipeline.
"""
import asyncio
import sys
from medgraphia.retrieval.pipeline import RetrievalPipeline
from medgraphia.generation.pipeline import GenerationPipeline
from medgraphia.domain.base import Language
from medgraphia.logger import get_logger

# 禁用冗余日志，只看核心输出
import logging
logging.getLogger("medgraphia").setLevel(logging.INFO)

async def ask_medgraphia(query: str, lang: Language = Language.ZH):
    print(f"\n" + "="*60)
    print(f"用户问题: {query}")
    print(f"目标语言: {lang.value}")
    print("="*60)

    # 1. 初始化两个核心 Pipeline
    retrieval_pl = RetrievalPipeline.from_settings()
    generation_pl = GenerationPipeline.from_settings()

    # 2. 执行检索 (Phase 6)
    print("\n[1/3] 正在检索医学知识库...")
    ret_result = await retrieval_pl.execute(query)
    print(f"  - 意图识别: {ret_result.query_type.value}")
    print(f"  - 召回证据: {len(ret_result.items)} 条 (图谱/向量/社区)")

    # 3. 执行生成 (Phase 7)
    print("\n[2/3] 正在调度模型生成答案...")
    gen_result = await generation_pl.generate(
        question=query,
        query_type=ret_result.query_type,
        retrieved_items=ret_result.items,
        language=lang
    )

    # 4. 展示结果
    print("\n[3/3] 最终结果:")
    print("-" * 40)
    if gen_result.routing:
        print(f"【模型路由】: {gen_result.routing.provider.value} / {gen_result.routing.model_name} ({gen_result.routing.tier.value} 档)")
    
    print(f"\n【医学回答】:\n{gen_result.answer}")
    
    if gen_result.disclaimer:
        print(f"\n【免责声明】: {gen_result.disclaimer}")

    if gen_result.citations:
        print(f"\n【参考来源】:")
        for cit in gen_result.citations:
            print(f"  [{cit.citation_number}] {cit.source_title} - {cit.section_path}")
    
    print(f"\n【总耗时】: {gen_result.latency_ms:.0f}ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    # 你可以修改这里的问题来测试不同的效果
    test_query = "如何治疗2型糖尿病？"
    if len(sys.argv) > 1:
        test_query = sys.argv[1]
    
    asyncio.run(ask_medgraphia(test_query))
