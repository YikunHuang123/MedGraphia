# 🧬 MedGraphia

**A GraphRAG-powered multilingual medical knowledge Q&A system** that fuses a Neo4j knowledge graph, three-path hybrid retrieval, and a multi-model LLM strategy to deliver clinically explainable answers with full evidence trails — in English, Chinese, and German.

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

| Feature                                                         | Description                                                                                                                                                                                                                                                                                   |
|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🧠 Multi-Route GraphRAG Retrieval                               | Three-path hybrid: Neo4j subgraph traversal + Hybrid retrieval combining BGE-M3 dense/sparse embeddings indexed in Qdrant + Leiden algorithm community summaries — combined via RRF                                                                                                          |
| 🌐 Multilingual (ZH / EN / DE) align                            | All surface forms of the same concept ("心肌梗死 / myocardial infarction / Myokardinfarkt") are aligned to a single CUI (MeSH ID) via SapBERT-XLMR for graph retrieval. At query time, **multilingual expansion** (Step 0.5) translates the query into all three corpus languages via `QueryTranslator` and runs parallel per-language Qdrant searches with quota-based merging — eliminating the hybrid-search lexical bias where BGE-M3 sparse vectors have zero token overlap across languages (e.g., a Chinese "肾衰竭" query would otherwise miss German "Nierenversagen" chunks despite semantic proximity in the dense space). |
| 🔬️ DSPy-driven Prompt Optimization                    | Leveraging **compiled DSPy programs** to enforce a "zero-assumption principle". Optimization strategies include:<br>• **Adversarial Few-Shot Bootstrapping**: Tuning via high-quality ambiguous/negative samples.<br>• **Explicit Reasoning Chain (CoT) Optimization**: Hardcoding medical logic into reasoning traces to forbid speculative answering.<br>• **Minimalist Response Enforcement**: Training the model to avoid summarizing irrelevant context noise.<br>• **Semantic Mismatch Detection**: Fine-tuning the boundary between grounded knowledge and retrieval noise.<br>• **Cross-Language Synthesis Training**: Few-shot demos explicitly teach the LLM to translate and synthesize foreign-language context paragraphs (DE/EN in a ZH answer) rather than treating them as missing information; updated metric penalises false refusals on positive cross-language cases. Clinical context (differential diagnoses, case notes) is treated as valid evidence, not refused for lacking an encyclopedic definition. |
| ⏳ Long-Short Term Memory System                                 | **Short-Term:** LLM-based Contextual Query Rewriting resolves pronouns across recent chat turns into standalone queries. **Long-Term:** Async Neo4j updates build cross-session user profiles with an **exponential time-decay algorithm** graph edge, enabling language-agnostic personalization. |
| 🏥 Two-Stage Cascade NER & Entity Linking                       | GLiNER zero-shot multilingual coarse pass + language-specific BERT precision pass (d4data/biomedical-ner-all EN / bert-base-chinese-medical-ner ZH / configurable DE) → SapBERT-XLMR linking to CUI (MeSH ID)                                                                                 |
| ⚡ Schema-Constrained LLM Relation Extraction                    | LLM relation extraction limited to a closed medical schema (TREATS, CAUSES, INTERACTS_WITH, DOSAGE_FOR…) — no hallucinated relationship types                                                                                                                  |
| 🏗️ Section-aware Chunking                                      | Text is split based on structural hierarchy (Section → Sub-section → Paragraph) rather than fixed token counts. Each chunk carries a metadata section_path, ensuring contextual grounding during retrieval.                                                                                   |
| 👁️ Multilingual PDF files support (Multi-Engine Parsing & OCR) | Hybrid pipeline using Docling (EN/DE) and MinerU (ZH) for structural layout analysis (tables/formulas). Integrated Tesseract 5 + PaddleOCR fallback for scanned medical records.                                                                               |
| 🔀 Multi-Model LLM Router                                       | Automatically divide user problems into three levels according to the complexity, and call different llm models                                                                                                                   |
| 🔗 Mandatory Evidence Citations                                 | Every answer is traceable to a specific chunk, section path, and versioned source — unanswerable questions are refused rather than fabricated                                                                                  |
| 🛡️ Safety Guardrails                                           | Two-stage proactive defense: Llama-Guard 3 input filtering (pre-retrieval) + output moderation (post-generation); aligned with S1-S14 safety categories; mandatory medical disclaimers and automatic model provisioning. |
| 📊 RAGAS Evaluation                                             | Standardized evaluation framework using RAGAS; support for automated synthetic medical testset generation with reasoning evolution; offline evaluation of RAG pipeline metrics (Faithfulness, Relevance, Precision, Recall) |

---

## 🎬 Architecture

### Build Pipeline (Offline)

```
┌────────────────────────────────── DATA SOURCES ──────────────────────────────────────┐
│ [EN] PubMed, FDA DailyMed  │ [ZH] Chinese Medical QA  │ [DE] German Medical Data     │
│ [EN/DE] EMA SmPC XML       │ [Anchor] MeSH Multilingual Descriptor Index             │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │  (API / bulk download — no scraping)
         ┌─────────────────────────────────┼──────────────────────────────────┐
         │                                 │                                  │
┌────────▼────────────┐          ┌──────────▼────────────┐          ┌─────────▼─────────┐
│  Parse & OCR        │          │ Section-aware Chunking│          │  Multi-lang NER   │
│  Docling  (EN / DE) │─────────▶│ anchor: section →     │─────────▶│  GLiNER-biomed    │
│  MinerU   (ZH)      │          │ paragraph → sentence  │          │  BioBERT (EN)     │
│  Tesseract+PaddleOCR│          │ + FHIR Timing Norm.   │          │  ClinicalBERT-CN  │
└─────────────────────┘          └───────────────────────┘          │  GerMedBERT (DE)  │
                                                                    └─────────┬─────────┘
                                                                              │
                    ┌─────────────────────────────────────────────────────────▼──────────┐
                    │  Entity Linking: SapBERT-XLMR + BM25 candidates → MeSH ID          │
                    │  ZH / EN / DE surface forms → MeSH ID (e.g. D009203 = MI)          │
                    └─────────────────────────────────────┬──────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────▼────────────────────────────────┐
                    │  Schema-guided Relation Extraction (LLM + closed schema)             │
                    │  TREATS · CAUSES · INTERACTS_WITH · DOSAGE_FOR · SYMPTOM_OF          │
                    │  Each edge: evidence_text · source_id · chunk_id · confidence        │
                    └────────────────────────┬─────────────────────────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────┐
              │                              │                          │
   ┌──────────▼────────────────┐  ┌──────────▼───────────────┐          │
   │  Leiden Community         │  │  BGE-M3 Embedding        │          │
   │  Detection + LLM          │  │  dense + sparse          │          │
   │  community summaries      │  │                          │          │
   └──────────┬────────────────┘  └──────────┬──────────────┘           │
              └─────────────────────┬─────────┘                         │
                                    │                                   │
          ┌─────────────────────────▼───────────────────────────────────▼──────────┐
          │                          STORAGE LAYER                                 │
          │   ┌─────────────────────────┐   ┌────────────────┐   ┌──────────────┐  │
          │   │  Neo4j 5.x              │   │  Qdrant        │   │  MinIO       │  │
          │   │  Knowledge Graph        │   │  Vector Store  │   │              │  │
          │   │  entities · relations   │   │  dense + sparse│   │  raw docs    │  │
          │   │  community summaries    │   │  hybrid index  │   │  provenance  │  │
          │   └─────────────────────────┘   └────────────────┘   └──────────────┘  │
          └────────────────────────────────────────────────────────────────────────┘
```

### Query Pipeline 

```
               User Query + Conversation History (Long-Short Memory)
                                     │
                                     ▼
          ┌────────────────────────────────────────────────────────┐
          │  Llama-Guard Input Filter (Proactive Defense)          │
          │  (Checks S1-S14 violations before retrieval)           │
          └──────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
          ┌────────────────────────────────────────────────────────┐
          │  Query Rewriter — context-aware query condensation     │
          │  (resolves coreference/ellipsis via chat history)      │
          └──────────────────────────┬─────────────────────────────┘
                                     │
          ┌──────────────────────────▼─────────────────────────────┐
          │  Multilingual Query Expansion                          │
          │  QueryTranslator: query → ZH / EN / DE translations    │
          │  → Per-language Qdrant quota searches → merged pool    │
          │  (eliminates sparse-vector lexical bias across langs)  │
          └──────────────────────────┬─────────────────────────────┘
                                     │
          ┌──────────────────────────│─────────────────────────────┐
          │                          ▼                             │
          │  LangGraph Router — Intent & Entity Mapping            │
          │  ① Query NER & linking to MeSH CUI                    │
          │  ② Intent classification into 5 QueryTypes            │
          │  ③ Retrieval Plan generation                          │
          └──────────────────────────┬─────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐  ┌────────▼──────────────┐
    │  Graph Retrieval  │   │  Hybrid Vector    │  │  Community Summary    │
    │  Neo4j 1–2-hop    │   │  BGE-M3 dense +   │  │  Global search over   │
    │  subgraph from    │   │  sparse on Qdrant │  │  Leiden communities   │
    │  query entity CUI │   │                   │  │  (Multi-hop/Overview) │
    └─────────┬─────────┘   └────────┬──────────┘  └────────┬──────────────┘
              └──────────────────────┼──────────────────────┘
                                     │
                          ┌──────────▼───────────────┐
                          │  RRF Fusion (Reciprocal) │
                          │  + bge-reranker-v2-m3    │
                          │    multilingual cross    │
                          └──────────┬───────────────┘
                                     │
          ┌──────────────────────────▼─────────────────────────────┐
          │  LLM Router — tier-based model selection               │
          │  - Tier: SMALL (FAQ) / MEDIUM (Inter.) / LARGE (Decis.)│
          │  - Infrastructure: LiteLLM + Pydantic AI Gateway       │
          └──────────────────────────┬─────────────────────────────┘
                                     │
          ┌──────────────────────────▼─────────────────────────────┐
          │  Generation Pipeline — cited answer construction       │
          │  - Context-aware Pydantic-typed prompts                │
          │  - Automated inline [N] citation injection             │
          │  - Medical disclaimer & evidence provenance            │
          └──────────────────────────┬─────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │  Post-Processing & Persistence │
                     │  - Save interaction to Neo4j   │
                     │  - Async User Interest Update  │
                     │    (Long Memory)               │
                     └───────────────┬────────────────┘
                                     │
                     ┌───────────────▼────────────────┐
                     │  FastAPI / Streamlit UI         │
                     │  Langfuse Tracing & Logs        │
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

Every node carries a `cui` (MeSH ID) property as the cross-lingual anchor, plus `lang_labels` for ZH/EN/DE surface forms.

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

| Layer | Technology | Notes                                                                 |
|---|---|-----------------------------------------------------------------------|
| **Language** | Python 3.12 |                                                                       |
| **API Framework** | FastAPI + Uvicorn | SSE streaming support                                                 |
| **Agent Orchestration** | LangGraph (LangChain) | Stateful, branching, retriable query agent                            |
| **Prompt Optimization** | DSPy | Optimized via **Adversarial Few-Shot Bootstrapping**; enforces clinical rigor through pre-compiled reasoning traces |
| **GraphRAG Framework** | LightRAG | less cost vs Microsoft GraphRAG |
| **Graph Database** | Neo4j 5.x | Nodes and relationships                                              |
| **Vector Store** | Qdrant | Native dense + sparse hybrid    |
| **Object Storage** | MinIO + Apache Iceberg | Raw docs, parse artifacts, provenance snapshots                       |
| **Embedding** | BGE-M3 (BAAI) | Dense + sparse + ColBERT; 100+ languages                |
| **Entity NER** | GLiNER (`urchade/gliner_mediumv2.1`) · `d4data/biomedical-ner-all` (EN) · `bert-base-chinese-medical-ner` (ZH) · configurable DE | Multi-lang, domain-specialized; all language models unified in `bert_ner.py` |
| **Entity Linking** | SapBERT-XLMR + BM25 | Cross-lingual → MeSH ID                                               |
| **Reranker** | bge-reranker-v2-m3 | Cross-encoder, multilingual                                           |
| **Community Detection** | Leiden algorithm | Graph clustering for global QA                                        |
| **Document Parsing** | Docling (EN/DE) · MinerU (ZH) | Section-aware; table / formula extraction                             |
| **OCR** | Tesseract 5 + PaddleOCR | Fallback for scanned PDFs and images                                  |
| **LLM (ZH)** | Qwen2.5 / DeepSeek | Self-hosted via vLLM/Ollama, or API                                   |
| **LLM (DE)** | LeoLM / EuroLLM + BioMistral | German-native; medical pre-training                                   |
| **LLM (EN medical)** | BioMistral / Me-LLaMA | PubMed pre-trained                                                    |
| **LLM (high-risk)** | Claude / GPT-4 | Via Azure OpenAI EU data residency or Anthropic                       |
| **LLM Gateway** | LiteLLM (self-hosted) | Unified audit log, cost tracking, routing                             |
| **Safety** | Llama-Guard-3-1B / NeMo Guardrails | Input + output filtering; S1-S14 policy |
| **Observability** | Langfuse (self-hosted) | GDPR-safe; prompt/token/latency/cost tracing                          |
| **Evaluation** | RAGAS | Faithfulness · Answer Relevance · Context Precision/Recall · **Synthetic Testset Generation** |
| **Auth** | API Key / Keycloak SSO | Configurable strategy: `none`, `apikey`, or `oidc`                    |
| **Pipeline Orchestration** | Prefect 3 / Airflow 2 | Incremental build DAG                                                 |
| **UI** | Streamlit | Chat · KG explorer · pipeline monitor · admin                         |
| **Containerization** | Docker + Docker Compose | Multi-target Dockerfiles                                              |

---

## ⚙️ How It Works

### 1 — Offline Build Pipeline

When you run `scripts/pipeline/build_graph.py` (or trigger the Prefect DAG), data flows through eight stages:

**Stage 1 — Fetch**

Data is pulled exclusively from authorized sources (APIs and official bulk downloads — no web scraping): PubMed abstracts via E-utilities, FDA DailyMed drug labels via REST API, EMA SmPC local PDFs, and DrugBank XML (if configured).

**Stage 2 — Parse**

Raw documents are parsed into structured form:

- **EN/DE PDFs** (EMA SmPC, AWMF guidelines, PubMed full-text): processed by **Docling**, which preserves table structure, formula layout, and figure captions.
- **ZH PDFs** (国家卫健委 clinical pathways, CNKI): processed by **MinerU**, optimized for Chinese double-column academic layouts.
- **Scanned documents**: **Tesseract 5 + PaddleOCR** fallback, language-auto-detected at paragraph level.
- **Structured data** (JSON/JSONL medical QA datasets): processed by `structured_parser.py`.

**Stage 3 — Section-aware Chunking + Normalization**

Text is split following the document's structural hierarchy (`section → paragraph → sentence`), not by a fixed token count. Each chunk carries a `section_path` metadata tag (e.g. `"Dosing > Pediatric > Renal adjustment"`) to enable provenance tracing. A domain normalizer unifies dose expressions across all three languages (e.g. `"bid"`, `"2 x täglich"`, `"每日两次"`, `"q 12 h"` → FHIR Timing object).

**Stage 4 — Multi-language NER**

A two-stage pipeline extracts medical entities:

1. **Coarse pass**: GLiNER (`urchade/gliner_mediumv2.1`) performs zero-shot, language-agnostic entity detection to enumerate candidate spans.
2. **Fine pass** (optional): A language-specific BERT model refines each candidate — `d4data/biomedical-ner-all` (EN), `bert-base-chinese-medical-ner` (ZH); German model is configurable via `NER_BERT_DE_MODEL` (disabled by default). All language variants are implemented in the unified `bert_ner.py` module.

This hybrid approach is dramatically cheaper than asking an LLM to perform NER directly over the full corpus.

**Stage 5 — Entity Linking to MeSH ID**

Each recognized mention is resolved to a MeSH ID via SapBERT-XLMR (trained on UMLS synonym pairs with contrastive learning):

1. BM25 retrieves the top-50 UMLS concept candidates.
2. SapBERT cross-encoder reranks them by semantic similarity.
3. Candidates below a confidence threshold are flagged for human review.

This is the cross-lingual backbone: "心肌梗死", "myocardial infarction", and "Myokardinfarkt" resolve to the same CUI (`C0027051`) and therefore share graph edges and vector neighbors.

**Stage 6 — Relation Extraction**

An LLM (configured via `LLM_MODEL`) extracts relations within each chunk, restricted to a closed schema. Free-form relationship types are rejected. Each produced edge stores `evidence_text`, `source_id`, `chunk_id`, `confidence`, and `extracted_by_model_version` — enabling full audit trails. Extracted relations are written to Neo4j immediately.

**Stage 7 — Embedding**

All text chunks are embedded with **BGE-M3**, which simultaneously produces **dense** (semantic) and **sparse** (BM25-style) representations per chunk. Both are stored in Qdrant under the `medgraphia_chunks` collection.

**Stage 8 — Community Detection**

Leiden community detection clusters the entity graph, and a second LLM pass generates a natural-language summary for each community (used by the community retriever at query time). Summaries are written to Neo4j `Community` nodes.

---

### 2 — Online Query Pipeline

When a user submits a query, the system runs the following steps:

**Step 1 — Query Classification & Language Routing**

The **LangGraph agent** classifies the incoming query into one of five strategies and detects the query language. Cross-lingual retrieval is always enabled: a Chinese question can retrieve German or English evidence if it matches via CUI.

**Step 1.5 — Multilingual Query Expansion**

When `MULTILINGUAL_RETRIEVAL_ENABLED=true` (default), the `QueryTranslator` translates the rewritten query into all three corpus languages (ZH / EN / DE) in parallel before vector retrieval. The vector retriever then runs one Qdrant search per language — filtered to that language's chunks — with a per-language quota (default `MULTILINGUAL_PER_LANG_QUOTA=7`), plus an unfiltered pass. Results are de-duplicated and score-merged into a single candidate pool.

This solves a structural problem in BGE-M3 hybrid search: the sparse (BM25-style) component assigns `abs(hash(token)) % 2³¹` to each token, so tokens from different languages for the same concept (e.g. `肾衰竭` vs `Nierenversagen`) have **zero overlap** in the sparse space. Without expansion, a Chinese query would systematically under-rank German chunks regardless of their dense-vector semantic proximity.

**Step 2 — Three-Path Retrieval (parallel)**

| Path | Mechanism | Best for |
|---|---|---|
| **Graph traversal** | NER + entity linking on the query → Neo4j 1–2-hop subgraph expansion | Drug interactions, clinical relationships, structured fact lookup |
| **Hybrid vector search** | BGE-M3 dense + sparse hybrid on Qdrant; with multilingual expansion, runs separate per-language quota searches and merges candidates | Semantic similarity, paraphrase, cross-language retrieval |
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

**Step 5 — Clinical Rigor, Safety & Citation**

This stage ensures the final response adheres to medical standards:
1. **Adversarial Gatekeeping**: Every query is processed by a **compiled DSPy program** using pre-tuned reasoning traces. This enforces a "zero-assumption principle" to intercept ambiguous queries or irrelevant context.
2. **Safety Filtering**: Llama-Guard 3 performs proactive input filtering (S1-S14 categories) and post-generation output moderation.
3. **Evidence Citation**: Answers include numbered citations `[1][2]` linked to the exact source chunk, section path, and document version. 

Pipeline quality is verified via offline RAGAS evaluation cycles.


---

## 🚀 Deployment Modes

MedGraphia supports two deployment configurations that share the same codebase. Switch between them by selecting the Docker Compose file and the corresponding `.env` template.

| | Enterprise Mode | Lite Mode |
|---|---|---|
| **Target** | Production server / cloud | 16 GB Mac / PC (M1/M2 or RTX 3060+) |
| **Data scope** | Full multi-domain corpus | Single domain (e.g. T2DM), 100–500 abstracts + 50 drug labels |
| **UMLS** | Full Metathesaurus | MetamorphoSys subset (SNOMED CT + RxNorm + MeSH, EN+ZH only) |
| **Vector DB** | Qdrant (dense + sparse hybrid) | Qdrant (dense + sparse hybrid, reduced memory footprint) |
| **Neo4j memory** | 16 GB+ page cache | 1–2 GB page cache (< 100K nodes / 500K edges) |
| **LLM** | vLLM/SGLang self-hosted 70B + cloud routed | Ollama 7B 4-bit GGUF or DeepSeek/Qwen API |
| **Auth** | Keycloak SSO + OPA role-based ACL | API key invite flow |
| **Observability** | Langfuse + Prometheus + Grafana | Langfuse only |
| **Compose file** | `docker-compose.yml` | `docker-compose.lite.yml` |

> **Architecture parity**: Lite mode preserves the full code path — graph retrieval, hybrid vector, community summaries, LLM router, citations. Upgrading to enterprise requires only a larger dataset, more memory, and switching compose files.

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
LLAMA_GUARD_PROVIDER=ollama
LLAMA_GUARD_MODEL=llama-guard3:1b

# Observability
LANGFUSE_HOST=http://langfuse:3000
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

# Auth
AUTH_STRATEGY=apikey           # none | apikey | oidc (keycloak)
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
docker compose exec worker python scripts/pipeline/build_graph.py \
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

# Qdrant for vector storage
VECTOR_STORE=qdrant

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
docker compose -f docker-compose.lite.yml exec worker python scripts/pipeline/build_graph.py \
  --domain t2dm \
  --pubmed-limit 200 \
  --drug-limit 30
```

---

### Option C — Local Development (without Docker)

**Prerequisites:** Python 3.12, Conda/uv, running Neo4j 5.x, Qdrant (native dense + sparse hybrid), Ollama (optional).

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
  "model_used": "BioMistral-7B"
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
| **Architecture** | | |
| `STORAGE_BACKEND` | `local` | `local` (disk) or `s3` (MinIO/AWS S3) |
| `AUTH_STRATEGY` | `apikey` | `none`, `apikey`, or `oidc` (Keycloak); env var: `AUTH_STRATEGY` |
| **Neo4j** | | |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_PAGE_CACHE` | `1G` | Page cache size (e.g., `1G` for laptop, `32G` for server) |
| **Vector Store** | | |
| `VECTOR_STORE` | `qdrant` | Only `qdrant` is supported |
| **LLM & Embedding** | | |
| `LLM_PROVIDER` | `groq` | `openai` \| `anthropic` \| `deepseek` \| `gemini` \| `groq` \| `ollama` \| `local` |
| `EMBEDDING_PROVIDER`| `ollama` | `huggingface` \| `openai` \| `ollama` |
| **Observability** | | |
| `TRACING_ENABLED` | `false` | Enable Langfuse tracing |
| `METRICS_ENABLED` | `false` | Enable Prometheus metrics |
| **Compliance** | | |
| `PII_DEIDENTIFY` | `false` | Run Microsoft Presidio PHI de-identification |
| `GUARDRAILS_ENABLED`| `true` | Enable Llama-Guard safety checks |
| `LLAMA_GUARD_PROVIDER`| `ollama` | Provider for the safety model |
| `LLAMA_GUARD_MODEL`| `llama-guard3:1b`| Model name for the safety guardrails |
| **Multilingual Retrieval** | | |
| `MULTILINGUAL_RETRIEVAL_ENABLED` | `true` | Translate query into ZH/EN/DE before vector search to eliminate hybrid-search lexical bias across corpus languages |
| `MULTILINGUAL_PER_LANG_QUOTA` | `7` | Max chunks retrieved per language in per-language quota mode; total candidate pool = `3 × quota + unfiltered_pass` |

---

## 🗂 Project Structure

```
MedGraphia/
├── docker-compose.yml              # Enterprise stack: neo4j, qdrant, minio, api, worker, ui, langfuse
├── docker-compose.lite.yml         # Lite stack: neo4j, qdrant, api, ui
├── docker/
│   ├── Dockerfile.api              # FastAPI + Uvicorn (multi-stage)
│   ├── Dockerfile.worker           # Pipeline worker (Prefect agent)
│   └── Dockerfile.ui               # Streamlit UI
├── .env.example                    # Enterprise env template
├── .env.lite.example               # Lite-mode env template
├── pyproject.toml
├── scripts/
│   ├── dspy/                   # DSPy optimization and inspection tools
│   │   ├── optimize.py         # Adversarial tuning pipeline
│   │   ├── viewer.py          # Prompt & reasoning trace inspection
│   │   └── test_rigor.py       # Clinical robustness validation
│   ├── data_fetchers/
│   │   ├── fetch_pubmed.py         # Pull PubMed subset via E-utilities API (no scraping)
│   │   ├── fetch_ema_smpc.py       # Download EMA SmPC XML bulk
│   │   ├── fetch_fda_dailymed.py   # FDA DailyMed REST download
│   │   ├── fetch_chinese_qa.py     # Fetch Chinese medical QA datasets (Huatuo etc.)
│   │   ├── fetch_germed.py         # Fetch German medical data
│   │   └── import_mesh.py          # Load MeSH Descriptor Index into local data dir
│   ├── pipeline/
│   │   ├── build_graph.py          # Bootstrap: thin CLI wrapper for ingestion/pipeline.py
│   │   ├── embed_entities.py       # Standalone entity embedding task
│   │   ├── ingest_multilingual.py  # Ingest multilingual corpus
│   │   ├── retrieval.py            # Ad-hoc retrieval test script
│   │   ├── ask_llm.py              # One-shot LLM query helper
│   │   └── test_api.py             # API smoke-test script
│   ├── admin/
│   │   ├── check_neo4j.py          # Verify Neo4j connectivity and schema
│   │   ├── count_neo4j_nodes.py    # Report node/edge counts by label
│   │   ├── inspect_data.py         # Inspect parsed document data
│   │   ├── reset_databases.py      # Wipe Neo4j + Qdrant for a fresh run
│   │   └── setup_chat_storage.py   # Create Neo4j chat history indexes
│   └── evaluation/
│       ├── blind_test_normalizer.py # Evaluate dose/unit normalizer accuracy
│       └── eval_ner_linking.py      # Evaluate NER + entity linking pipeline
│
└── src/medgraphia/
    ├── config.py                   # Pydantic Settings — all env vars, deployment mode switch
    ├── knowledge_base.py           # Domain query / drug seed definitions
    ├── logger.py                   # Structured logging setup
    │
    ├── domain/                     # Domain model package
    │   ├── base.py                 # Core types: Entity, Relation, Language, QueryType
    │   ├── document.py             # RawDocument, ParsedSection, Chunk, SourceMeta
    │   ├── medical.py              # Medical entity hierarchy
    │   ├── chat.py                 # Session, Message, Citation models
    │   └── community.py            # Community node model
    │
    ├── data/                       # Authorized data source connectors
    │   ├── dspy/                   # Compiled DSPy programs (optimized reasoning traces)
    │   ├── pubmed.py               # PubMed E-utilities API (NCBI — compliant, versioned)
    │   ├── ema_smpc.py             # EMA SmPC XML bulk downloader
    │   ├── fda_dailymed.py         # FDA DailyMed REST API
    │   ├── drugbank.py             # DrugBank connector (academic / commercial license)
    │   └── mesh.py                 # MeSH Descriptor Index loader (automatic download)
    │
    ├── ingestion/                  # Offline build pipeline
    │   ├── pipeline.py             # Prefect flow + 8 task stages (fetch→parse→chunk→ner→link→extract→embed→community)
    │   ├── parsers/
    │   │   ├── docling_parser.py   # Docling: EN/DE medical PDF (tables, formulas, figures)
    │   │   ├── mineru_parser.py    # MinerU: ZH academic PDF (double-column, formula)
    │   │   ├── ocr_parser.py       # Tesseract 5 + PaddleOCR fallback for scanned docs
    │   │   └── structured_parser.py # JSON/JSONL medical QA datasets (Huatuo etc.)
    │   ├── chunker.py              # Section-aware anchor chunking — not fixed-size 512
    │   ├── normalizer.py           # Dose/unit normalization → FHIR Timing object
    │   ├── ner/
    │   │   ├── gliner_ner.py       # GLiNER zero-shot multilingual coarse NER
    │   │   ├── bert_ner.py         # Unified BERT precision pass: EN / ZH / DE in one module
    │   │   ├── pipeline.py         # MedicalNERPipeline: combines GLiNER + BERT, deduplicates spans
    │   │   └── _types.py           # Internal MentionSpan type
    │   ├── entity_linker.py        # SapBERT-XLMR + BM25 → MeSH ID cross-lingual alignment
    │   ├── relation_extractor.py   # LLM schema-guided RE (closed relation type set)
    │   ├── community_builder.py    # Leiden algorithm + LLM community summary generation
    │   └── embedder.py             # BGE-M3: dense + sparse (100+ languages) → Qdrant
    │
    ├── graph/                      # Knowledge graph layer (Neo4j)
    │   ├── client.py               # Neo4j 5.x async driver + connection pool
    │   ├── schema.py               # Node labels, relationship types, property constraints
    │   └── queries.py              # Cypher library: subgraph expansion, path search, CUI lookup
    │
    ├── vector/                     # Vector store
    │   ├── base.py                 # Abstract VectorStoreBase interface
    │   └── qdrant_store.py         # Qdrant: dense + sparse hybrid (enterprise & lite)
    │
    ├── llm/                        # LLM client layer
    │   ├── gateway.py              # LiteLLMGateway: unified multi-provider interface (OpenAI / Anthropic / DeepSeek / Gemini / Groq / Ollama)
    │   └── client.py               # pydantic-ai model factory for structured LLM output
    │
    ├── retrieval/                  # Online query pipeline (three-path hybrid)
    │   ├── pipeline.py             # RetrievalPipeline: orchestrates all retrieval steps
    │   ├── router.py               # Query classification → retrieval strategy selection
    │   ├── rewriter.py             # QueryRewriter: condense history into standalone query
    │   ├── query_translator.py     # QueryTranslator: translate query into ZH/EN/DE for per-language quota retrieval (Step 0.5)
    │   ├── query_ner.py            # NER on incoming query for entity-based graph lookup
    │   ├── graph_retriever.py      # Neo4j 1–2-hop subgraph from entity CUIs in query
    │   ├── vector_retriever.py     # BGE-M3 dense + sparse hybrid search on Qdrant; retrieve_multilingual() for per-language quota search
    │   ├── community_retriever.py  # Leiden community summary search (global QA)
    │   ├── reranker.py             # bge-reranker-v2-m3 cross-encoder
    │   └── fusion.py               # Reciprocal Rank Fusion (RRF) across all three paths
    │
    ├── generation/                 # LLM generation layer
    │   ├── pipeline.py             # GenerationPipeline: context prep → routing → LLM → citations
    │   ├── llm_router.py           # Route by query type / language → SMALL / MEDIUM / LARGE tier
    │   ├── citation.py             # Inline citation injection → provenance to chunk_id + section
    │   └── prompts.py              # Pydantic-typed prompt modules (per scenario × language)
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
    ├── observability/              # Monitoring & tracing
    │   └── langfuse_client.py      # Langfuse self-hosted: trace LLM calls, cost, user feedback
    │
    ├── ui/
    │   ├── streamlit_app.py        # Streamlit entry point
    │   ├── api_client.py           # HTTP client for the FastAPI backend
    │   ├── components/             # Reusable UI components (chat_history, citations, graph_viz, styles)
    │   └── pages/
    │       ├── 1_Chat.py           # Chat interface with inline citation viewer
    │       ├── 2_Graph_Explorer.py # Knowledge graph explorer
    │       ├── 3_Dashboard.py      # Ingestion pipeline monitor
    │       └── 4_Admin.py          # Admin panel
    │
    └── tests/
        ├── test_parsers.py         # Document parsing tests
        ├── test_chunker.py         # Chunking + normalization tests
        ├── test_ner.py             # NER pipeline (GLiNER + BERT) accuracy tests
        ├── test_entity_linker.py   # Cross-lingual CUI alignment tests
        ├── test_relation_extractor.py # Schema-guided relation extraction tests
        ├── test_community_builder.py  # Leiden + community summary tests
        └── test_llm_gateway.py    # LiteLLMGateway integration tests
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
