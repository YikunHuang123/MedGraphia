import asyncio
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from medgraphia.graph.queries import get_all_entities

# ---------------------------------------------------------------------------
# Blacklists
# ---------------------------------------------------------------------------

# Functional / stopwords that should never be standalone entities
_STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "with", "by",
    "at", "on", "is", "it", "its", "be", "as", "was", "are", "were", "been",
    "this", "that", "these", "those", "have", "has", "had", "not", "but",
    "from", "via", "per", "vs", "i", "ii", "iii", "iv", "no",
    # Over-broad medical terms (too generic to be useful graph nodes)
    "patient", "patients", "study", "studies", "result", "results",
    "treatment", "treatments", "report", "case", "cases", "group", "groups",
    "method", "methods", "data", "analysis", "effect", "effects",
    "level", "levels", "type", "types", "age", "dose", "doses",
    "rate", "rates", "risk", "time", "week", "weeks", "month", "months",
    "year", "years", "day", "days", "use", "used", "using",
}

_STOPWORDS_ZH = {
    "的", "了", "在", "是", "和", "有", "与", "对", "或", "为", "等",
    "该", "其", "此", "以", "及", "但", "而", "被", "将", "可", "较",
    "患者", "研究", "分析", "结果", "方法", "数据", "报告", "病例",
    "治疗", "效果", "水平", "类型", "剂量", "风险", "时间",
}

_STOPWORDS_DE = {
    "der", "die", "das", "ein", "eine", "und", "oder", "mit", "von",
    "in", "an", "auf", "für", "bei", "zu", "ist", "sind", "war",
    "patient", "patienten", "studie", "ergebnis", "methode", "daten",
    "behandlung", "gruppe", "typ", "dosis", "risiko", "zeit",
}

ALL_STOPWORDS = _STOPWORDS_EN | _STOPWORDS_ZH | _STOPWORDS_DE

# Regex patterns for structural junk
_RE_PURE_NUMBER = re.compile(r"^\d+(\.\d+)?$")
_RE_DOSAGE = re.compile(r"^\d+[\s]?(mg|ml|g|kg|ug|mcg|mmol|μg|ng|iu|u|%|x|×)\b", re.IGNORECASE)
_RE_PURE_PUNCT = re.compile(r"^[^\w]+$")
_RE_SINGLE_CHAR = re.compile(r"^[a-zA-Z\d]$")
# Unresolved provisional CUI — entity linker never replaced it
_RE_MENTION_CUI = re.compile(r"^MENTION:")

# Type-label mismatch heuristics: words strongly associated with the WRONG type
_TYPE_MISMATCH_HINTS: dict[str, set[str]] = {
    "Drug": {"disease", "syndrome", "disorder", "infection", "cancer", "疾病", "综合征"},
    "Disease": {"tablet", "capsule", "injection", "mg", "drug", "medication", "药片", "胶囊"},
    "Symptom": {"gene", "protein", "receptor", "enzyme", "基因", "蛋白"},
    "Gene": {"pain", "fever", "nausea", "疼痛", "发热", "恶心"},
    "Anatomy": {"disease", "syndrome", "cancer", "疾病", "综合征"},
    "Physiology": {"tablet", "capsule", "drug", "药片"},
    "LivingBeing": {"tablet", "capsule", "syndrome", "药片"},
}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_short(entities: list[dict]) -> list[dict]:
    return [e for e in entities if len(e["label"].strip()) <= 3]


def check_pure_punct(entities: list[dict]) -> list[dict]:
    return [e for e in entities if _RE_PURE_PUNCT.match(e["label"].strip())]


def check_stopwords(entities: list[dict]) -> list[dict]:
    return [e for e in entities if e["label"].strip().lower() in ALL_STOPWORDS]


def check_numbers_and_dosages(entities: list[dict]) -> list[dict]:
    result = []
    for e in entities:
        label = e["label"].strip()
        if _RE_PURE_NUMBER.match(label) or _RE_DOSAGE.match(label):
            result.append(e)
    return result


def check_unresolved_cuis(entities: list[dict]) -> list[dict]:
    return [e for e in entities if _RE_MENTION_CUI.match(e.get("cui", ""))]


def check_type_mismatch(entities: list[dict]) -> list[dict]:
    result = []
    for e in entities:
        hints = _TYPE_MISMATCH_HINTS.get(e["entity_type"], set())
        label_lower = e["label"].lower()
        if any(hint in label_lower for hint in hints):
            result.append(e)
    return result


def check_near_duplicates(entities: list[dict]) -> list[tuple[dict, dict]]:
    """Find pairs where one label is a simple plural/suffix variant of another (same type)."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        by_type[e["entity_type"]].append(e)

    pairs = []
    for etype, group in by_type.items():
        normalized: dict[str, dict] = {}
        for e in group:
            key = e["label"].strip().lower().rstrip("s").rstrip("e")  # naive stem
            if key in normalized:
                pairs.append((normalized[key], e))
            else:
                normalized[key] = e
    return pairs


def sample_by_type(entities: list[dict], n: int = 8) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in entities:
        by_type[e["entity_type"]].append(e)
    return {etype: group[:n] for etype, group in sorted(by_type.items())}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_section(title: str, items: list, formatter):
    print(f"\n{'─'*60}")
    print(f"  {title}  ({len(items)} 个)")
    print(f"{'─'*60}")
    for item in items[:25]:
        print(f"  {formatter(item)}")
    if len(items) > 25:
        print(f"  ... 省略 {len(items) - 25} 个")


async def main():
    print("连接 Neo4j，拉取所有实体...")
    try:
        entities = await get_all_entities()
    except Exception as e:
        print(f"连接失败: {e}")
        return

    if not entities:
        print("数据库中没有实体。")
        return

    total = len(entities)
    print(f"共 {total} 个实体节点\n")

    short        = check_short(entities)
    punct        = check_pure_punct(entities)
    stops        = check_stopwords(entities)
    numbers      = check_numbers_and_dosages(entities)
    unresolved   = check_unresolved_cuis(entities)
    mismatches   = check_type_mismatch(entities)
    near_dups    = check_near_duplicates(entities)

    # Deduplicate across checks for a "total suspect" count
    suspect_cuis = (
        {e["cui"] for e in short} |
        {e["cui"] for e in punct} |
        {e["cui"] for e in stops} |
        {e["cui"] for e in numbers} |
        {e["cui"] for e in unresolved} |
        {e["cui"] for e in mismatches}
    )

    print("=" * 60)
    print("  NER 污染检测报告")
    print("=" * 60)
    print(f"  总实体数        : {total}")
    print(f"  疑似问题实体数  : {len(suspect_cuis)}  ({len(suspect_cuis)/total*100:.1f}%)")
    print(f"  ├─ 极短实体(≤3) : {len(short)}")
    print(f"  ├─ 纯标点       : {len(punct)}")
    print(f"  ├─ 停用词/泛化词: {len(stops)}")
    print(f"  ├─ 纯数字/剂量  : {len(numbers)}")
    print(f"  ├─ CUI未解析    : {len(unresolved)}")
    print(f"  ├─ 类型疑似错配 : {len(mismatches)}")
    print(f"  └─ 近似重复对   : {len(near_dups)}")

    fmt_e = lambda e: f"[{e['entity_type']:10s}] '{e['label']}'  ({e['cui']})"
    fmt_p = lambda p: f"[{p[0]['entity_type']:10s}] '{p[0]['label']}'  <->  '{p[1]['label']}'"

    if short:
        _print_section("极短实体 (≤ 3 字符)", short, fmt_e)
    if punct:
        _print_section("纯标点/边界异常", punct, fmt_e)
    if stops:
        _print_section("停用词 / 过泛化词", stops, fmt_e)
    if numbers:
        _print_section("纯数字 / 剂量片段", numbers, fmt_e)
    if unresolved:
        _print_section("CUI 未解析 (MENTION: 前缀)", unresolved, fmt_e)
    if mismatches:
        _print_section("类型疑似错配", mismatches, fmt_e)
    if near_dups:
        _print_section("近似重复对 (同类型)", near_dups, fmt_p)

    print(f"\n{'─'*60}")
    print("  各类型随机采样 (用于人工快速判断)")
    print(f"{'─'*60}")
    for etype, samples in sample_by_type(entities).items():
        print(f"\n  [{etype}] — 共 {sum(1 for e in entities if e['entity_type']==etype)} 个")
        for e in samples:
            print(f"    '{e['label']}'")


if __name__ == "__main__":
    asyncio.run(main())
