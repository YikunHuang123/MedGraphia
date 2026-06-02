
import json
import re
import random
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.ingestion.normalizer import MedicalNormalizer
from medgraphia.domain import Language

def blind_test():
    normalizer = MedicalNormalizer()
    processed_dir = Path("data/processed")
    
    # 1. 收集所有处理过的真实文本
    all_text = ""
    for json_file in processed_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 提取 full_text 或 sections 内容
            all_text += data.get("full_text", "")
            for s in data.get("sections", []):
                all_text += s.get("content", "")

    # 2. 定义探测正则：寻找包含数字+潜在单位或频率词的句子
    # 探测词：mg, mcg, ml, units, daily, dose, tablets
    pattern = re.compile(r"([^.]*?\d+\s*(?:mg|mcg|ml|unit|tablets|daily|dose)[^.]*\.)", re.IGNORECASE)
    candidates = list(set(pattern.findall(all_text)))
    
    if not candidates:
        print("No candidates found in processed data!")
        return

    # 3. 随机抽取 10 条真实语料
    sample_size = min(10, len(candidates))
    samples = random.sample(candidates, sample_size)

    print("\n" + "="*80)
    print(f"🕵️  BLIND TEST: REAL CLINICAL DATA NORMALIZATION (n={sample_size})")
    print("="*80)
    
    for i, raw in enumerate(samples):
        raw_clean = raw.strip().replace("\n", " ")
        # 仅显示包含关键词的局部片段
        normalized = normalizer.normalize(raw_clean, Language.EN)
        
        # 检查是否有变化
        status = "✨ CHANGED" if normalized != raw_clean else "⚪ NO CHANGE"
        
        print(f"\n[{i+1}] {status}")
        print(f"RAW:  {raw_clean}")
        print(f"NORM: {normalized}")
        print("-" * 40)

if __name__ == "__main__":
    blind_test()
