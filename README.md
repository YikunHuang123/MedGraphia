# 🧬 MedGraphia

**A GraphRAG-powered multilingual medical knowledge Q&A system** that fuses a Neo4j knowledge graph, three-path hybrid retrieval, and a multi-model LLM strategy to deliver clinically explainable answers with full evidence trails — in Chinese, English, and German.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-0080FF)](https://qdrant.tech/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF6B35)](https://langchain-ai.github.io/langgraph/)
[![Languages](https://img.shields.io/badge/Languages-ZH_|_EN_|_DE-orange)](#-features)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Knowledge Graph Schema](#-knowledge-graph-schema)
- [Tech Stack](#-tech-stack)
- [How It Works](#️-how-it-works)
- [Deployment Modes](#-deployment-modes)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License & Contact](#-license--contact)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 GraphRAG Retrieval | Three-path hybrid: Neo4j subgraph traversal + BGE-M3 dense/sparse vector search + Leiden community summaries — combined via RRF |
| 🌐 Multilingual (ZH / EN / DE) | All surface forms of the same concept ("心肌梗死 / myocardial infarction / Myokardinfarkt") are aligned to a single UMLS CUI via SapBERT-XLMR — cross-lingual retrieval |
| 🔗 Mandatory Evidence Citations | Every answer is traceable to a specific chunk, section path, and versioned source — unanswerable questions are refused rather than fabricated |
| 🏥 Medical NER + Entity Linking | GLiNER-biomed (zero-shot, multilingual) + language-specific fine models (BioBERT EN / ClinicalBERT-CN ZH / GerMedBERT DE) → SapBERT-XLMR linking to UMLS CUI |
| ⚡ Schema-Constrained Relation Extraction | LLM relation extraction limited to a closed medical schema (TREATS, CAUSES, INTERACTS_WITH, DOSAGE_FOR…) — no hallucinated relationship types |
| 🛡️ Safety Guardrails | Llama Guard 4 input/output filtering; faithfulness scoring via RAGAS; hard refusal for unanswerable queries; mandatory disclaimer on patient-facing responses |
| 🔀 Multi-Model LLM Router | Routes by risk level and query complexity: Qwen2.5/DeepSeek (ZH), LeoLM/EuroLLM+BioMistral (DE+medical EN), Claude/GPT-4 (high-risk clinical decisions) |
| 📊 RAGAS Evaluation | Online faithfulness scoring per response; offline golden set (500+ ZH/EN/DE Q&A pairs curated by clinical advisors); CI blocks release on benchmark regression |
| 🔒 Compliance-Aware Design | GDPR / EU AI Act / PIPL design principles; Microsoft Presidio PII/PHI de-identification; LiteLLM unified audit gateway; data residency by market (EU/CN/US) |
| 🪶 Lite Mode | Runs on a 16 GB laptop: Ollama 7B quantized models, Chroma vector DB, UMLS subset via MetamorphoSys, domain-scoped data (100–500 PubMed abstracts + 50 drug labels) |
| 🖥️ Streamlit UI | Chat interface with inline citation viewer, knowledge graph explorer, ingestion pipeline monitor, and admin panel |

---

## 🎬 Architecture

### Build Pipeline (Offline)

```
┌────────────────────────────────── DATA SOURCES ──────────────────────────────────────┐
│  PubMed E-utilities  │  EMA SmPC XML  │  FDA DailyMed  │  DrugBank                  │
│  UMLS Metathesaurus  │  AWMF S3 Guidelines  │  国家卫健委临床路径  │  Internal EHR   │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │  (API / bulk download — no scraping)
         ┌─────────────────────────────────┼──────────────────────────────────┐
         │                                 │                                  │
┌────────▼───────────┐          ┌──────────▼────────────┐          ┌─────────▼────────┐
│  Parse & OCR        │          │ Section-aware Chunking │          │  Multi-lang NER   │
│  Docling  (EN / DE) │─────────▶│ anchor: section →     │─────────▶│  GLiNER-biomed    │
│  MinerU   (ZH)      │          │ paragraph → sentence  │          │  BioBERT (EN)     │
│  Tesseract+PaddleOCR│          │ + FHIR Timing Norm.   │          │  ClinicalBERT-CN  │
└────────────────────┘          └───────────────────────┘          │  GerMedBERT (DE)  │
                                                                    └─────────┬─────────┘
                                                                              │
                    ┌─────────────────────────────────────────────────────────▼──────────┐
                    │  Entity Linking: SapBERT-XLMR + BM25 candidates → UMLS CUI         │
                    │  ZH / EN / DE surface forms → single node (e.g. C0027051 = MI)     │
                    └─────────────────────────────────────┬──────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────▼──────────────────────────────┐
                    │  Schema-guided Relation Extraction (LLM + closed schema)             │
                    │  TREATS · CAUSES · INTERACTS_WITH · DOSAGE_FOR · SYMPTOM_OF         │
                    │  Each edge: evidence_text · source_id · chunk_id · confidence        │
                    └───────────────────────┬─────────────────────────────────────────────┘
                                            │
              ┌─────────────────────────────┼──────────────────────────┐
              │                             │                          │
   ┌──────────▼───────────────┐  ┌──────────▼──────────────┐         │
   │  Leiden Community         │  │  BGE-M3 Embedding        │         │
   │  Detection + LLM          │  │  dense + sparse +        │         │
   │  community summaries      │  │  ColBERT (100+ langs)    │         │
   └──────────┬────────────────┘  └──────────┬──────────────┘         │
              └─────────────────────┬─────────┘                        │
                                    │                                   │
          ┌─────────────────────────▼───────────────────────────────────▼──────────┐
          │                          STORAGE LAYER                                  │
          │   ┌─────────────────────────┐   ┌────────────────┐   ┌──────────────┐  │
          │   │  Neo4j 5.x               │   │  Qdrant        │   │  MinIO       │  │
          │   │  Knowledge Graph         │   │  Vector Store  │   │  + Iceberg   │  │
          │   │  entities · relations    │   │  BGE-M3 3-way  │   │  raw docs    │  │
          │   │  community summaries     │   │  hybrid index  │   │  provenance  │  │
          │   └─────────────────────────┘   └────────────────┘   └──────────────┘  │
          └────────────────────────────────────────────────────────────────────────┘
```

### Query Pipeline (Online)

```
                      User Query  (ZH / EN / DE)
                               │
                               ▼
          ┌────────────────────────────────────────────────────────┐
          │  LangGraph Agent — Query Classification & Routing       │
          │  ① Clinical Decision Aid   ② Drug Interaction          │
          │  ③ Literature Multi-hop    ④ Cross-corpus Global QA    │
          │  ⑤ Patient FAQ (降语 + disclaimer)                     │
          └──────────────────────────┬─────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼────────┐   ┌─────────▼─────────┐  ┌────────▼──────────────┐
    │  Graph Retrieval  │   │  Hybrid Vector     │  │  Community Summary    │
    │  Neo4j 1–2-hop    │   │  BGE-M3 dense +   │  │  Leiden global search │
    │  subgraph from    │   │  sparse on Qdrant  │  │  (comprehensive QA    │
    │  query entity CUI │   │                    │  │   across full corpus) │
    └─────────┬─────────┘   └─────────┬──────────┘  └────────┬──────────────┘
              └──────────────────────┬┘─────────────────────┘
                                     │
                          ┌──────────▼──────────────┐
                          │  RRF Fusion              │
                          │  + bge-reranker-v2-m3    │
                          │    cross-encoder          │
                          └──────────┬───────────────┘
                                     │
          ┌──────────────────────────▼─────────────────────────────┐
          │  LLM Router — risk level + complexity                    │
          │  ZH:  Qwen2.5 / DeepSeek                                │
          │  DE:  LeoLM / EuroLLM + BioMistral                      │
          │  EN:  BioMistral / Me-LLaMA                             │
          │  High-risk clinical:  Claude / GPT-4 (via BAA channel)  │
          │  All calls through LiteLLM audit gateway                 │
          └──────────────────────────┬─────────────────────────────┘
                                     │
          ┌──────────────────────────▼─────────────────────────────┐
          │  Llama Guard 4 — output safety check                    │
          │  RAGAS faithfulness score (low → downgrade / refuse)    │
          │  → Answer + Inline Citations [1][2] + Disclaimer        │
          └──────────────────────────┬─────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │  FastAPI  (port 8058)           │
                     │  Streamlit UI  (port 8501)      │
                     │  Langfuse observability panel   │
                     └────────────────────────────────┘
```

---

## 🗺️ Knowledge Graph Schema

### Node Types

| Label | Description | Primary Source |
|---|---|---|
| `Disease` | Disease / syndrome (SNOMED CT concepts) | UMLS · CMeKG · ICD-10/11 |
| `Drug` | Drug substance (RxNorm / ATC-DDD) | DrugBank · EMA SmPC · FDA DailyMed |
| `Symptom` | Clinical sign or symptom | SNOMED CT · UMLS |
| `Gene` | Gene / protein target | NCBI Gene |
| `Procedure` | Clinical procedure (SNOMED) | UMLS |
| `Chunk` | Source text chunk with `section_path` + `source_version` | PubMed · Guidelines · Labels |
| `Community` | Leiden-detected entity cluster with LLM summary | Computed |

Every node carries a `cui` (UMLS CUI) property as the cross-lingual anchor, plus `lang_labels` for ZH/EN/DE surface forms.

### Relationship Types

| Type | Example triple | Stored properties |
|---|---|---|
| `TREATS` | Metformin → T2DM | confidence · evidence_text · chunk_id |
| `CAUSES` | T2DM → Retinopathy | confidence · evidence_text · chunk_id |
| `INTERACTS_WITH` | Warfarin ↔ Aspirin | severity · mechanism · chunk_id |
| `DOSAGE_FOR` | (Chunk) → Metformin | route · population · chunk_id |
| `SYMPTOM_OF` | Polyuria → T2DM | frequency · chunk_id |
| `COMPLICATION_OF` | Nephropathy → T2DM | prevalence · chunk_id |
| `CODED_AS` | T2DM → ICD-10 E11 | code_system · version |
| `CONTRAINDICATED_IN` | Metformin → Renal Failure | evidence_level · chunk_id |

---

## 🛠 Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Language** | Python 3.12 | |
| **API Framework** | FastAPI 0.115 + Uvicorn | SSE streaming support |
| **Agent Orchestration** | LangGraph (LangChain) | Stateful, branching, retriable query agent |
| **Prompt Optimization** | DSPy | Typed prompt modules per scenario + language |
| **GraphRAG Framework** | LightRAG | 1/100 cost vs Microsoft GraphRAG; community summary borrowed from MS GraphRAG |
| **Graph Database** | Neo4j 5.x | Cypher, APOC, enterprise ACL |
| **Vector Store** | Qdrant (enterprise) · Chroma (lite) | Qdrant: native dense + sparse hybrid |
| **Object Storage** | MinIO + Apache Iceberg | Raw docs, parse artifacts, provenance snapshots |
| **Embedding** | BGE-M3 (BAAI) | Dense + sparse + ColBERT; 100+ languages; 8192 token ctx |
| **Entity NER** | GLiNER-biomed · BioBERT · ClinicalBERT-CN · GerMedBERT | Multi-lang, domain-specialized |
| **Entity Linking** | SapBERT-XLMR + BM25 | Cross-lingual → UMLS CUI |
| **Reranker** | bge-reranker-v2-m3 | Cross-encoder, multilingual |
| **Community Detection** | Leiden algorithm | Graph clustering for global QA |
| **Document Parsing** | Docling (EN/DE) · MinerU (ZH) | Section-aware; table / formula extraction |
| **OCR** | Tesseract 5 + PaddleOCR | Fallback for scanned PDFs and images |
| **LLM (ZH)** | Qwen2.5 / DeepSeek | Self-hosted via vLLM/Ollama, or API |
| **LLM (DE)** | LeoLM / EuroLLM + BioMistral | German-native; medical pre-training |
| **LLM (EN medical)** | BioMistral / Me-LLaMA | PubMed pre-trained |
| **LLM (high-risk)** | Claude / GPT-4 | Via Azure OpenAI EU data residency or Anthropic |
| **LLM Gateway** | LiteLLM (self-hosted) | Unified audit log, cost tracking, routing |
| **Safety** | Llama Guard 4 / NeMo Guardrails | Input + output filtering |
| **Observability** | Langfuse (self-hosted) | GDPR-safe; prompt/token/latency/cost tracing |
| **Evaluation** | RAGAS | Faithfulness · Answer Relevance · Context Precision/Recall |
| **Auth (lite)** | API Key (X-API-Key) | Invite-token flow |
| **Auth (enterprise)** | Keycloak SSO + OPA | OIDC + row-level ACL by clinical role |
| **Pipeline Orchestration** | Prefect 3 / Airflow 2 | Incremental build DAG |
| **UI** | Streamlit | Chat · KG explorer · pipeline monitor · admin |
| **Containerization** | Docker + Docker Compose | Multi-target Dockerfiles |

---

## ⚙️ How It Works

### 1 — Offline Build Pipeline

When you run `scripts/build_graph.py` (or trigger the Prefect DAG), data flows through six stages:

**Stage 1 — Fetch & Parse**

Data is pulled exclusively from authorized sources (APIs and official bulk downloads — no web scraping):

- **EN/DE PDFs** (EMA SmPC, AWMF guidelines, PubMed full-text): processed by **Docling**, which preserves table structure, formula layout, and figure captions.
- **ZH PDFs** (国家卫健委 clinical pathways, CNKI): processed by **MinerU**, optimized for Chinese double-column academic layouts.
- **Scanned documents**: **Tesseract 5 + PaddleOCR** fallback, language-auto-detected at paragraph level.

**Stage 2 — Section-aware Chunking + Normalization**

Text is split following the document's structural hierarchy (`section → paragraph → sentence`), not by a fixed token count. Each chunk carries a `section_path` metadata tag (e.g. `"Dosing > Pediatric > Renal adjustment"`) to enable provenance tracing. A domain normalizer unifies dose expressions across all three languages (e.g. `"bid"`, `"2 x täglich"`, `"每日两次"`, `"q 12 h"` → FHIR Timing object).

**Stage 3 — Multi-language NER**

A two-stage pipeline extracts medical entities:

1. **Coarse pass**: GLiNER-biomed performs zero-shot, language-agnostic entity detection to enumerate candidate spans.
2. **Fine pass**: A language-specific model refines each candidate — BioBERT/PubMedBERT (EN), ClinicalBERT-CN with BiLSTM+CRF (ZH), GerMedBERT (DE).

This hybrid approach is dramatically cheaper than asking an LLM to perform NER directly over the full corpus.

**Stage 4 — Entity Linking to UMLS CUI**

Each recognized mention is resolved to a UMLS CUI via SapBERT-XLMR (trained on UMLS synonym pairs with contrastive learning):

1. BM25 retrieves the top-50 UMLS concept candidates.
2. SapBERT cross-encoder reranks them by semantic similarity.
3. Candidates below a confidence threshold are flagged for human review.

This is the cross-lingual backbone: "心肌梗死", "myocardial infarction", and "Myokardinfarkt" resolve to the same CUI (`C0027051`) and therefore share graph edges and vector neighbors.

**Stage 5 — Relation Extraction + Graph Construction**

An LLM (Qwen2.5-7B in lite mode, or a stronger model in enterprise) extracts relations within each chunk, restricted to a closed schema. Free-form relationship types are rejected. Each produced edge stores `evidence_text`, `source_id`, `chunk_id`, `confidence`, and `extracted_by_model_version` — enabling full audit trails. Leiden community detection then clusters the resulting graph, and a second LLM pass generates a natural-language summary for each community (used by the community retriever at query time).

**Stage 6 — Embedding**

All text chunks are embedded with **BGE-M3**, which simultaneously produces three representations per chunk: **dense** (semantic), **sparse** (BM25-style), and **ColBERT** (multi-vector). Entity nodes are separately embedded with SapBERT for entity-level similarity. Both are stored in Qdrant.

---

### 2 — Online Query Pipeline

When a user submits a query, the system runs the following steps:

**Step 1 — Query Classification & Language Routing**

The **LangGraph agent** classifies the incoming query into one of five strategies and detects the query language. Cross-lingual retrieval is always enabled: a Chinese question can retrieve German or English evidence if it matches via CUI.

**Step 2 — Three-Path Retrieval (parallel)**

| Path | Mechanism | Best for |
|---|---|---|
| **Graph traversal** | NER + entity linking on the query → Neo4j 1–2-hop subgraph expansion | Drug interactions, clinical relationships, structured fact lookup |
| **Hybrid vector search** | BGE-M3 dense + sparse hybrid on Qdrant, merged via RRF | Semantic similarity, paraphrase, rare terminology |
| **Community summary** | Semantic search over Leiden community summaries | "What are all comorbidities of T2DM?" — global, cross-corpus synthesis |

**Step 3 — RRF Fusion + Neural Reranking**

Results from all three paths are merged with Reciprocal Rank Fusion, then passed through **bge-reranker-v2-m3** (multilingual cross-encoder) to produce a final ranked candidate list.

**Step 4 — LLM Router**

The router selects the generation model based on query risk level and complexity:

| Risk | Complexity | Selected model |
|---|---|---|
| Patient FAQ | Low | Qwen2.5-7B / DeepSeek (self-hosted or API) |
| Drug interaction, dosing | Medium | BioMistral / LeoLM |
| Clinical decision support | High | Claude 3.5 / GPT-4 via EU data-residency channel |

**Step 5 — Safety Check + Citation**

Llama Guard 4 screens both input and output. RAGAS faithfulness is scored online; answers below threshold are either downgraded to a more cautious response or refused. Every answer includes numbered citations `[1][2]` linked to the exact source chunk, section path, and document version.

---

## 🚀 Deployment Modes

MedGraphia supports two deployment configurations that share the same codebase. Switch between them by selecting the Docker Compose file and the corresponding `.env` template.

| | Enterprise Mode | Lite Mode |
|---|---|---|
| **Target** | Production server / cloud | 16 GB Mac / PC (M1/M2 or RTX 3060+) |
| **Data scope** | Full multi-domain corpus | Single domain (e.g. T2DM), 100–500 abstracts + 50 drug labels |
| **UMLS** | Full Metathesaurus | MetamorphoSys subset (SNOMED CT + RxNorm + MeSH, EN+ZH only) |
| **Vector DB** | Qdrant (dense + sparse hybrid) | Qdrant or Chroma |
| **Neo4j memory** | 16 GB+ page cache | 1–2 GB page cache (< 100K nodes / 500K edges) |
| **LLM** | vLLM/SGLang self-hosted 70B + cloud routed | Ollama 7B 4-bit GGUF or DeepSeek/Qwen API |
| **Auth** | Keycloak SSO + OPA role-based ACL | API key invite flow |
| **Observability** | Langfuse + Prometheus + Grafana | Langfuse only |
| **Compose file** | `docker-compose.yml` | `docker-compose.lite.yml` |

> **Architecture parity**: Lite mode preserves the full code path — graph retrieval, hybrid vector, community summaries, LLM router, guardrails, citations. Upgrading to enterprise requires only a larger dataset, more memory, and switching compose files.

---

## 📦 Installation

### Option A — Enterprise Mode (Docker Compose)

Requires Docker Engine + Compose v2. Recommended RAM: 32 GB+.

**1. Clone the repository**

```bash
git clone https://github.com/YikunHuang123/MedGraphia.git
cd MedGraphia
```

**2. Create your environment file**

```bash
cp .env.example .env
```

Minimum required settings:

```bash
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-neo4j-password

# Vector store
VECTOR_STORE=qdrant
QDRANT_URL=http://qdrant:6333

# Embedding (self-hosted BGE-M3 via text-embeddings-inference, or API)
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BASE_URL=http://embedding:8080

# LLM (pick a provider)
LLM_PROVIDER=deepseek          # deepseek | openai | anthropic | local
DEEPSEEK_API_KEY=sk-...

# Safety guardrails
GUARDRAILS_ENABLED=true
LLAMA_GUARD_MODEL=meta-llama/Llama-Guard-4-8B

# Observability
LANGFUSE_HOST=http://langfuse:3000
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# Auth
AUTH_MODE=apikey               # apikey | keycloak
ADMIN_BOOTSTRAP_KEY=your-admin-key
```

**3. Start all services**

```bash
docker compose up --build
```

This starts: `neo4j`, `qdrant`, `minio`, `api` (port **8058**), `worker`, `ui` (port **8501**), `langfuse`.

**4. Bootstrap the knowledge graph**

```bash
# Fetch and index a domain-specific dataset (adjust flags as needed)
docker compose exec worker python scripts/build_graph.py \
  --domain cardiovascular \
  --pubmed-query "cardiovascular drug interactions" \
  --pubmed-limit 500 \
  --include-drugbank \
  --include-ema-smpc

# Monitor progress in the Streamlit admin panel
# http://localhost:8501
```

**5. Open the UI**

Navigate to `http://localhost:8501`. The interactive API docs are at `http://localhost:8058/docs`.

---

### Option B — Lite Mode (Low-spec Device)

Runs on a 16 GB M1/M2 Mac or a Windows laptop with a mid-range GPU.

**1. Clone and configure**

```bash
git clone https://github.com/YikunHuang123/MedGraphia.git
cd MedGraphia
cp .env.lite.example .env
```

Key differences from enterprise `.env`:

```bash
# Use Ollama for local LLM inference (zero API cost after setup)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b            # 4-bit GGUF, ~4 GB VRAM
LLM_BASE_URL=http://host.docker.internal:11434

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text

# Chroma as zero-infra vector store alternative
VECTOR_STORE=chroma             # or qdrant

# Reduce Neo4j memory footprint
NEO4J_PAGE_CACHE=1G
NEO4J_HEAP_INITIAL=512M
NEO4J_HEAP_MAX=1G

# Domain restriction (keeps graph < 100K nodes)
DEFAULT_DOMAIN=t2dm             # Builds only the T2DM sub-graph
```

**2. Pull Ollama models**

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

**3. Start the lite stack**

```bash
docker compose -f docker-compose.lite.yml up --build
```

**4. Bootstrap with a narrow dataset**

```bash
# Fetch 200 T2DM abstracts + 30 drug labels — completes in minutes
docker compose -f docker-compose.lite.yml exec worker python scripts/build_graph.py \
  --domain t2dm \
  --pubmed-limit 200 \
  --drug-limit 30
```

---

### Option C — Local Development (without Docker)

**Prerequisites:** Python 3.12, Conda/uv, running Neo4j 5.x, Qdrant or Chroma, Ollama (optional).

```bash
git clone https://github.com/YikunHuang123/MedGraphia.git
cd MedGraphia

# Create environment
conda create -n medgraphia python=3.12 -y
conda activate medgraphia
pip install -e ".[dev]"

# Configure
cp .env.lite.example .env
# Edit .env with your local service addresses

# Start API server
uvicorn medgraphia.api:create_app --factory --host 0.0.0.0 --port 8058 --reload

# Start pipeline worker (separate terminal)
prefect worker start --pool default-agent-pool

# Start Streamlit UI (separate terminal)
streamlit run src/medgraphia/ui/streamlit_app.py
```

---

## 💡 Usage

### Build pipeline status

The pipeline runs asynchronously. Check progress in the Streamlit admin panel or via the API:

```bash
curl http://localhost:8058/admin/pipeline/status \
  -H "X-API-Key: your-admin-key"
```

```json
{
  "stage": "relation_extraction",
  "documents_processed": 312,
  "documents_total": 500,
  "nodes_created": 48203,
  "edges_created": 187441,
  "progress": 0.62
}
```

### Ask a clinical question (blocking)

```bash
curl -X POST http://localhost:8058/chat \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the drug interactions between metformin and contrast agents in T2DM patients?",
    "session_id": "sess_abc123",
    "language": "en"
  }'
```

```json
{
  "session_id": "sess_abc123",
  "content": "Metformin should be withheld before and 48 hours after iodinated contrast administration due to risk of contrast-induced nephropathy leading to lactic acidosis [1][2]. The ADA 2024 guidelines recommend...",
  "citations": [
    {
      "citation_number": 1,
      "source_title": "EMA SmPC — Metformin hydrochloride",
      "source_version": "EMA/SmPC/2024-01",
      "section_path": "4.4 Special warnings > Renal function",
      "content_snippet": "Metformin must be discontinued at the time of, or prior to, the imaging procedure..."
    },
    {
      "citation_number": 2,
      "source_title": "ADA Standards of Care 2024",
      "section_path": "Section 11 > Chronic Kidney Disease",
      "content_snippet": "Hold metformin before procedures using iodinated contrast media..."
    }
  ],
  "retrieval_paths_used": ["graph_traversal", "hybrid_vector"],
  "model_used": "BioMistral-7B",
  "faithfulness_score": 0.94
}
```

### Streaming response (SSE)

```bash
curl -N -X POST http://localhost:8058/chat/stream \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "T2DM 患者二甲双胍的剂量调整原则？", "session_id": "sess_abc123"}'
```

```
data: {"delta": "根据中国2型糖尿病防治指南"}
data: {"delta": "（2024年版），二甲双胍的起始剂量"}
data: {"delta": "通常为500 mg，每日2次..."}
data: {"done": true, "citations": [...], "disclaimer": "本回答仅供参考，不构成诊疗建议，请遵医嘱。"}
```

### Query the knowledge graph directly

```bash
# Look up a drug's interactions by CUI or name
curl "http://localhost:8058/graph/entity?name=metformin&lang=en&hops=2" \
  -H "X-API-Key: your-api-key"
```

```json
{
  "entity": {"cui": "C0025598", "label": "Metformin", "type": "Drug"},
  "subgraph": {
    "nodes": 14,
    "edges": 31,
    "relations": [
      {"type": "TREATS", "target": {"cui": "C0011860", "label": "T2DM"}},
      {"type": "INTERACTS_WITH", "target": {"cui": "C0009413", "label": "Contrast Media"}, "severity": "major"},
      {"type": "CONTRAINDICATED_IN", "target": {"cui": "C0035222", "label": "Renal Failure"}}
    ]
  }
}
```

---

## ⚙️ Configuration

All settings are loaded from `.env` via Pydantic Settings. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DEPLOYMENT_MODE` | `lite` | `enterprise` or `lite` — controls defaults and component selection |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_PAGE_CACHE` | `4G` | Neo4j page cache; set to `1G` in lite mode |
| `VECTOR_STORE` | `qdrant` | `qdrant` or `chroma` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant service URL |
| `LLM_PROVIDER` | `deepseek` | `deepseek` \| `openai` \| `anthropic` \| `ollama` \| `local` |
| `LLM_MODEL` | `deepseek-chat` | Model name / path |
| `LLM_BASE_URL` | _(provider default)_ | Custom inference endpoint (vLLM / Ollama / LiteLLM) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model; `nomic-embed-text` for Ollama |
| `GRAPHRAG_FRAMEWORK` | `lightrag` | `lightrag` — LightRAG is the default; see notes on MS GraphRAG cost |
| `COMMUNITY_SUMMARY_LLM` | _(same as LLM_MODEL)_ | Model used for Leiden community summary generation |
| `GUARDRAILS_ENABLED` | `true` | Toggle Llama Guard 4 safety filtering |
| `RAGAS_FAITHFULNESS_THRESHOLD` | `0.75` | Answers scoring below this are downgraded or refused |
| `AUTH_MODE` | `apikey` | `apikey` (lite) or `keycloak` (enterprise) |
| `ADMIN_BOOTSTRAP_KEY` | _(required)_ | Initial admin API key |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse self-hosted URL |
| `DEFAULT_DOMAIN` | `general` | Domain filter for lite mode (e.g. `t2dm`, `cardiovascular`) |
| `PUBMED_MAX_RESULTS` | `500` | Cap on PubMed abstracts fetched per domain |
| `DRUG_LABEL_LIMIT` | `50` | Cap on drug label documents per domain |
| `PII_DEIDENTIFY` | `true` | Run Microsoft Presidio de-identification on all ingested text |

---

## 🗂 Project Structure

```
MedGraphia/
├── docker-compose.yml              # Enterprise stack: neo4j, qdrant, minio, api, worker, ui, langfuse
├── docker-compose.lite.yml         # Lite stack: neo4j, qdrant/chroma, api, ui
├── docker/
│   ├── Dockerfile.api              # FastAPI + Uvicorn (multi-stage)
│   ├── Dockerfile.worker           # Pipeline worker (Prefect agent)
│   └── Dockerfile.ui               # Streamlit UI
├── .env.example                    # Enterprise env template
├── .env.lite.example               # Lite-mode env template
├── pyproject.toml
├── scripts/
│   ├── fetch_pubmed.py             # Pull PubMed subset via E-utilities API (no scraping)
│   ├── fetch_ema_smpc.py           # Download EMA SmPC XML bulk
│   ├── fetch_fda_dailymed.py       # FDA DailyMed REST download
│   ├── load_umls_subset.py         # Load MetamorphoSys UMLS export into Neo4j
│   └── build_graph.py              # Bootstrap: orchestrate full offline pipeline
│
└── src/medgraphia/
    ├── config.py                   # Pydantic Settings — all env vars, deployment mode switch
    ├── domain.py                   # Domain models: Entity, Relation, Chunk, Community, Session, Message
    │
    ├── data/                       # Authorized data source connectors
    │   ├── pubmed.py               # PubMed E-utilities API (NCBI — compliant, versioned)
    │   ├── ema_smpc.py             # EMA SmPC XML bulk downloader
    │   ├── fda_dailymed.py         # FDA DailyMed REST API
    │   ├── drugbank.py             # DrugBank connector (academic / commercial license)
    │   └── umls.py                 # UMLS Metathesaurus loader (MetamorphoSys subset)
    │
    ├── ingestion/                  # Offline build pipeline
    │   ├── pipeline.py             # Prefect / Airflow DAG — stages orchestration
    │   ├── parsers/
    │   │   ├── docling_parser.py   # Docling: EN/DE medical PDF (tables, formulas, figures)
    │   │   ├── mineru_parser.py    # MinerU: ZH academic PDF (double-column, formula)
    │   │   └── ocr_parser.py       # Tesseract 5 + PaddleOCR fallback for scanned docs
    │   ├── chunker.py              # Section-aware anchor chunking — not fixed-size 512
    │   ├── normalizer.py           # Dose/unit normalization → FHIR Timing object
    │   ├── ner/
    │   │   ├── gliner_ner.py       # GLiNER-biomed zero-shot multilingual coarse NER
    │   │   ├── biobert_ner.py      # BioBERT / PubMedBERT fine NER (English)
    │   │   ├── clinicalbert_cn.py  # ClinicalBERT-CN BiLSTM+CRF (Chinese EMR)
    │   │   └── germedbert_ner.py   # GerMedBERT / bert-base-german-clinical (German)
    │   ├── entity_linker.py        # SapBERT-XLMR + BM25 → UMLS CUI cross-lingual alignment
    │   ├── relation_extractor.py   # LLM schema-guided RE (closed relation type set)
    │   ├── community_builder.py    # Leiden algorithm + LLM community summary generation
    │   └── embedder.py             # BGE-M3: dense + sparse + ColBERT (100+ languages)
    │
    ├── graph/                      # Knowledge graph layer (Neo4j)
    │   ├── client.py               # Neo4j 5.x async driver + connection pool
    │   ├── schema.py               # Node labels, relationship types, property constraints
    │   └── queries.py              # Cypher library: subgraph expansion, path search, CUI lookup
    │
    ├── vector/                     # Vector store (pluggable backend)
    │   ├── base.py                 # Abstract VectorStoreBase interface
    │   ├── qdrant_store.py         # Qdrant: dense + sparse hybrid (enterprise & lite)
    │   └── chroma_store.py         # Chroma: zero-infra fallback for lite mode
    │
    ├── retrieval/                  # Online query pipeline (three-path hybrid)
    │   ├── router.py               # LangGraph: classify query → select retrieval strategy
    │   ├── graph_retriever.py      # Neo4j 1–2-hop subgraph from entity CUIs in query
    │   ├── vector_retriever.py     # BGE-M3 dense + sparse hybrid search on Qdrant
    │   ├── community_retriever.py  # Leiden community summary search (global QA)
    │   ├── reranker.py             # bge-reranker-v2-m3 cross-encoder
    │   └── fusion.py               # Reciprocal Rank Fusion (RRF) across all three paths
    │
    ├── generation/                 # LLM generation + safety layer
    │   ├── llm_router.py           # Route by risk level + complexity → model selector
    │   ├── providers/
    │   │   ├── openai_provider.py  # Azure OpenAI (EU data residency / US BAA)
    │   │   ├── anthropic_provider.py
    │   │   ├── deepseek_provider.py # ZH: DeepSeek / Qwen-Turbo API
    │   │   └── local_provider.py   # vLLM / SGLang / Ollama self-hosted
    │   ├── litellm_gateway.py      # LiteLLM unified gateway: audit log, cost, routing
    │   ├── guardrails.py           # Llama Guard 4 input/output safety + RAGAS online scoring
    │   ├── citation.py             # Inline citation injection → provenance to chunk_id + section
    │   └── prompts.py              # DSPy typed prompt modules (per scenario × language)
    │
    ├── api/                        # FastAPI application
    │   ├── __init__.py             # App factory with lifespan management
    │   ├── schemas.py              # Pydantic request / response DTOs
    │   ├── deps.py                 # FastAPI Depends: auth, session, rate limit
    │   ├── middleware.py           # Audit logging, GDPR-safe request tracing
    │   ├── chat.py                 # POST /chat (blocking) + POST /chat/stream (SSE)
    │   ├── knowledge.py            # GET /graph/entity, GET /graph/subgraph, GET /graph/community
    │   ├── admin.py                # Pipeline trigger, model config, user management
    │   ├── health.py               # GET /health/live  &  GET /health/ready
    │   └── auth.py                 # API key auth (lite) / Keycloak OIDC (enterprise)
    │
    ├── observability/              # Monitoring, tracing, evaluation
    │   ├── langfuse_client.py      # Langfuse self-hosted: trace LLM calls, cost, user feedback
    │   ├── eval.py                 # RAGAS: faithfulness, answer relevance, context precision/recall
    │   └── metrics.py              # Prometheus metrics endpoint (/metrics)
    │
    ├── ui/
    │   └── streamlit_app.py        # Chat · KG graph explorer · pipeline monitor · admin panel
    │
    └── tests/
        ├── test_parsers.py         # Document parsing and chunking tests
        ├── test_ner_pipeline.py    # NER + entity linking accuracy tests
        ├── test_entity_linking.py  # Cross-lingual CUI alignment tests
        ├── test_retrieval.py       # Three-path retrieval + fusion tests
        └── test_api.py             # FastAPI endpoint integration tests
```

---

## 🔮 Roadmap

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome.

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** — follow the existing code style (Ruff / Black formatting).

3. **Add or update tests** for any new behaviour:
   ```bash
   pytest -v tests/
   ```

4. **Commit** with a descriptive message:
   ```bash
   git commit -m "feat: add cross-lingual entity search endpoint"
   ```

5. **Open a Pull Request** against `main`. Include:
   - A description of the problem solved or feature added
   - Which language(s) (ZH / EN / DE) are affected or tested
   - Any relevant changes to `.env.example` or `docker-compose.yml`

**Reporting bugs:** Please open a GitHub Issue with the label `bug`, your Python version, deployment mode (enterprise / lite), and a minimal reproduction snippet.

---

## 📄 License & Contact

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Author:** Yikun Huang
**Email:** q1945948369@gmail.com
**GitHub:** [@YikunHuang123](https://github.com/YikunHuang123)

> Built as a production-oriented medical GraphRAG system — covering multilingual NLP pipelines, knowledge graph construction, three-path hybrid retrieval, and compliance-aware LLM generation across Chinese, English, and German.
