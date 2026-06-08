import asyncio
import sys
import types
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mock structlog and other potential missing deps before importing medgraphia
mock_structlog = types.ModuleType("structlog")
sys.modules["structlog"] = mock_structlog


class MockLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


import medgraphia.logger

medgraphia.logger.get_logger = lambda name: MockLogger()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from medgraphia.data.drugbank import DrugBankConnector
from medgraphia.data.fda_dailymed import FDADailyMedConnector
from medgraphia.data.pubmed import PubMedConnector, PubMedFetchConfig
from medgraphia.domain import Language, RawDocument, SourceMeta
from medgraphia.ingestion.chunker import MedicalChunker
from medgraphia.ingestion.parsers.docling_parser import DoclingParser


# ---------------------------------------------------------------------------
# Helper: Save to JSON
# ---------------------------------------------------------------------------
def save_processed_result(doc: RawDocument, name: str):
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{name}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        # 使用 Pydantic 的 model_dump_json 导出所有结构化信息
        f.write(doc.model_dump_json(indent=2))
    print(f"  ✅ Saved structured JSON to: {file_path}")


async def inspect_pipeline():
    chunker = MedicalChunker(max_tokens=250)

    # 1. PubMed
    print("\n" + "=" * 60)
    print("--- 🔬 [1/4] PUBMED (Structured Abstract) ---")
    try:
        async with PubMedConnector() as pubmed:
            pubmed_docs = await pubmed.fetch(
                PubMedFetchConfig(query="metformin randomized trial", max_results=1)
            )
            if pubmed_docs:
                doc = pubmed_docs[0]
                print(f"Title: {doc.title}")
                print(f"Sections Count: {len(doc.sections)}")
                save_processed_result(doc, "pubmed_metformin")
    except Exception as e:
        print(f"PubMed Error: {e}")

    # 2. FDA DailyMed
    print("\n" + "=" * 60)
    print("--- 💊 [2/4] FDA DAILYMED (Structured XML) ---")
    try:
        async with FDADailyMedConnector() as fda:
            fda_docs = await fda.fetch_by_drug_name("metformin", limit=1)
            if fda_docs:
                doc = fda_docs[0]
                print(f"Title: {doc.title}")
                print(f"Sections Count: {len(doc.sections)}")
                save_processed_result(doc, "fda_metformin")
    except Exception as e:
        print(f"FDA Error: {e}")

    # 3. EMA SmPC (PDF Parsing via Docling)
    print("\n" + "=" * 60)
    print("--- 🇪🇺 [3/4] EMA SmPC (PDF Structural Parsing) ---")
    try:
        local_pdf = Path(
            "data/raw/ema_smpc/Icandra__previously_Vildagliptin___metformin_hydrochloride_Novartis_.pdf"
        )
        if local_pdf.exists():
            parser = DoclingParser()
            source = SourceMeta(source_id="ema_test", source_title="EMA SmPC Test")
            print(f"Parsing local PDF: {local_pdf.name}...")
            doc = parser.parse(str(local_pdf), source, language=Language.EN)
            print(f"Title: {doc.title}")
            print(f"Sections Count: {len(doc.sections)}")
            save_processed_result(doc, "ema_icandra_pdf")
        else:
            print("EMA Test PDF not found.")
    except Exception as e:
        print(f"EMA/Docling Error: {e}")

    # 4. DrugBank
    print("\n" + "=" * 60)
    print("--- 🏦 [4/4] DRUGBANK (External XML) ---")
    try:
        db_path = Path("data/raw/drugbank/drugbank_partial.xml")
        if db_path.exists():
            db_conn = DrugBankConnector(str(db_path))
            drugs = db_conn.fetch_all(limit=1)
            if drugs:
                doc = drugs[0]
                print(f"Drug: {doc.title}")
                print(f"Sections Count: {len(doc.sections)}")
                save_processed_result(doc, "drugbank_lepirudin")
        else:
            print("DrugBank XML not found.")
    except Exception as e:
        print(f"DrugBank Error: {e}")


if __name__ == "__main__":
    asyncio.run(inspect_pipeline())
