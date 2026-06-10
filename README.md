# 🧬 MedGraphia

A multilingual medical QA system powered by a Knowledge-Graph based Graph-RAG architecture. It
achieves deep cross-lingual alignment (English, Chinese, German) via the MeSH ontology. By
constructing long-short-term memory through time-decay graph edges and integrating multi-tier LLM
intelligent routing with DSPy-optimized prompt management for rigorous reasoning, it builds a safe,
traceable clinical AI brain.

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
- [LLM Core Functional Capabilities](#-llm-core-functional-capabilities)
- [Architecture](#-architecture)
- [Knowledge Graph Schema](#-knowledge-graph-schema)
- [Tech Stack](#-tech-stack)
- [How It Works](#️-how-it-works)
- [Deployment Modes](#-deployment-modes)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License & Contact](#-license--contact)

---

## ✨ Features

| Feature                                                         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🧠 Advanced GraphRAG Engine | A deep implementation of GraphRAG principles: <br>• **Local Search**: NER-driven subgraph traversal (1-2 hops) for precise clinical triples.<br>• **Global Search**: **Leiden algorithm** community detection + LLM-generated hierarchical summaries for cross-corpus synthesis.<br>• **Hybrid RRF**: Merging graph traversal, dense/sparse vector search, and community insights via Reciprocal Rank Fusion.<br>• **Semantic Glue**: All data is anchored to MeSH CUIs, enabling the graph to act as a cross-lingual and cross-document relational bridge.                                                                                                                                |
| 🌐 Multilingual (ZH / EN / DE) align                            | All surface forms of the same concept ("心肌梗死 / myocardial infarction / Myokardinfarkt") are aligned to a single CUI (MeSH ID) via SapBERT-XLMR for graph retrieval. At query time, **multilingual expansion** (Step 0.5) translates the query into all three corpus languages via `QueryTranslator` and runs parallel per-language Qdrant searches with quota-based merging                                                                                                                                                                                                                                                                                                                |
| 🔬️ DSPy-driven Prompt Optimization                    | Use **DSPy** to manage and optimize prompts. Optimization strategies include:<br>• **Automated Prompt Compilation**: Using `BootstrapFewShot` and `MIPROv2` to automatically select and inject the best reasoning traces (CoT) into the prompt.<br>• **Synthetic Data Factory**: Built-in pipeline to reverse-engineer high-quality, multilingual QA pairs from grounded graph chunks.<br>• **Adversarial Tuning**: Defending against false pronouns and hallucinated knowledge via explicitly negative training examples.<br>• **Clinical Tiering**: The Rewriter is trained to simultaneously condense queries and classify their clinical complexity (SMALL/MEDIUM/LARGE) for the LLM Router. |
| ⏳ Long-Short Term Memory System                                 | • **Short-Term:** LLM-based Contextual Query Rewriting resolves pronouns across recent chat turns into standalone queries. <br>• **Long-Term:** Async Neo4j updates build cross-session user profiles with an **exponential time-decay algorithm** graph edge, enabling language-agnostic personalization.                                                                                                                                                                                                                                                                                                                                                                                 |
| 🏥 Two-Stage Cascade NER & Entity Linking                       | GLiNER zero-shot multilingual coarse pass + language-specific BERT precision pass (biomedical-ner-all EN, DE / bert-base-chinese-medical-ner ZH) → SapBERT-XLMR linking to CUI (MeSH ID)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ⚡ Schema-Constrained LLM Relation Extraction                    | LLM relation extraction limited to a closed medical schema (TREATS, CAUSES, INTERACTS_WITH, DOSAGE_FOR…) — no hallucinated relationship types                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 🏗️ Section-aware Chunking                                      | Text is split based on structural hierarchy (Section → Sub-section → Paragraph) rather than fixed token counts. Each chunk carries a metadata section_path, ensuring contextual grounding during retrieval.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 🔀 Multi-Model LLM Router                                       | Automatically divide user problems into three levels according to the complexity, and call different llm models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 👁️ Multilingual PDF files support (Multi-Engine Parsing & OCR) | Hybrid pipeline using Docling (EN/DE) and MinerU (ZH) for structural layout analysis (tables/formulas). Integrated Tesseract 5 + PaddleOCR fallback for scanned medical records.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 🔗 Mandatory Evidence Citations                                 | Every answer is traceable to a specific chunk, section path, and versioned source — unanswerable questions are refused rather than fabricated                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 🛡️ Safety Guardrails                                           | Two-stage proactive defense: Llama-Guard 3 input filtering (pre-retrieval) + output moderation (post-generation); aligned with S1-S14 safety categories; mandatory medical disclaimers and automatic model provisioning.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 📊 RAGAS Evaluation                                             | Standardized evaluation framework using RAGAS; support for automated synthetic medical testset generation with reasoning evolution; offline evaluation of RAG pipeline metrics (Faithfulness, Relevance, Precision, Recall)                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ⚡ Redis-Backed NER Result Cache                                | Query-side NER + Entity Linking results (GLiNER → SapBERT-XLMR → MeSH CUI) are persisted in Redis. Repeated or concurrently-identical queries skip BERT inference entirely, cutting routing latency from **300–500 ms → 5 ms**.  |
| 🔄 Arq Pipeline Task Queue                                      | The offline build pipeline is dispatched as a durable **Arq** task (Redis-backed) executed by a dedicated worker process. |

---

### 🚀 Technical Deep Dives

#### **1. Cross-Lingual Ontology Alignment**
The system achieves deep cross-lingual alignment (ZH, EN, DE) through a unified MeSH ontology and a parallel multi-language retrieval mechanism:
*   **Offline Entity Unification:** During ingestion, a two-stage Cascade NER pipeline (GLiNER + language-specific BERT) extracts medical entities. The `EntityLinker` then utilizes **SapBERT-XLMR** to map diverse surface forms (e.g., *"心肌梗死"*, *"myocardial infarction"*, *"Myokardinfarkt"*) to a single, globally unique **MeSH CUI** (Concept Unique Identifier). The Neo4j Knowledge Graph stores edges based on these language-agnostic CUIs, allowing relations extracted from German or English documents to be directly connected to Chinese nodes.
*   **Online Parallel Retrieval:** During the query phase, the `QueryTranslator` asynchronously translates the user's input into all supported corpus languages. The pipeline then executes parallel hybrid vector searches (dense + sparse via BGE-M3) on Qdrant, enforcing a per-language retrieval quota. This ensures that sparse vector matching (which relies on exact token overlap) functions correctly across language boundaries before merging the candidates via RRF.

#### **2. Long-Short Term Memory System**
MedGraphia manages context through a dual-layer memory architecture:
*   **Short-Term Memory:** A dedicated `Rewriter` module (DSPy-optimized) resolves coreference and ellipsis from a **5-message sliding window** (approx. 2.5 conversation turns). This ensures that history is condensed into a standalone search query to maintain high retrieval precision without overwhelming the LLM with raw transcripts.
*   **Long-Term Memory:** User-entity interactions are persisted in Neo4j via `INTERESTED_IN` relationships. The system implements a **Recurrent Interaction Decay** logic: every time a user interacts with a clinical entity, the existing interest weight is discounted by a factor (default **0.9**) before adding a unit increment ($W_{new} = W_{old} \times 0.9 + 1.0$). During retrieval, this allows the system to prioritize a user's long-term chronic history (which accumulates weight over time) while naturally deprioritizing transient symptoms.

#### **3. DSPy-driven Logic Self-Evolution**
Instead of static prompts, MedGraphia uses **DSPy programs** that evolve through automated data-driven compilation:
*   **Synthetic Data Factory:** Using a high-capability Teacher model, the system reverse-engineers high-quality, multilingual QA pairs from grounded graph chunks. This "Reverse-RAG" approach ensures that training examples are anchored in real database evidence. 
*   **Teacher-Guided Bayesian Optimization (MIPROv2):** For complex modules like the Query Rewriter (which handles both semantic resolution and routing difficulty scoring) for the small model, MedGraphia employs the advanced **MIPROv2** algorithm. 
    *   **The Teacher model (large)** analyzes the training set and brainstorms multiple variations of instructional prompts.
    *   **The Student model (small)** executes these variations on a validation set.
    *   **The Optimizer** uses a Bayesian search algorithm (Tree-structured Parzen Estimator) driven by a **Continuous Multiplicative Metric** (which penalizes routing errors but softly rewards reasoning traces). Over multiple trials, it discovers the exact linguistic instructions and few-shot examples that maximize the small local model's clinical accuracy, effectively bridging the capability gap between a 7B model and flagship LLMs.
    *   **The Metric** — Performance is judged by a three-part composite score: 85% rewrite quality (continuous 0–1 score from an LLM judge), 15% reasoning trace completeness, and a multiplicative routing gate that caps the total at 30% when the clinical complexity tier is misclassified. This design prevents gaming any single dimension while preserving a stable gradient signal across all Bayesian trials.
    *   **The Result** — Starting from a baseline of **60%** on the held-out validation set, MIPROv2 converged to **76.7%** after 18 trials — demonstrating that a compact local model can match the routing precision of a flagship LLM when given the right few-shot demonstrations and instructions.

#### **4. Hierarchical Multi-Model Routing**
To optimize for latency and cost, the system employs a tiered routing logic orchestrated via **Pydantic-AI**:
*   **Complexity Rubric (E+I Score):** During the rewriting phase, the `Rewriter` calculates a **Complexity Score** by summing two dimensions: **Entities** (E, 1-3 based on count) and **Intent** (I, 1-3 based on depth/ambiguity). 
*   **Tiered Dispatch:** Queries are routed based on the total score: **SMALL** (≤3), **MEDIUM** (4), or **LARGE** (≥5). Simple administrative or single-entity questions are handled by local, low-latency models, while complex multi-hop clinical decisions are dispatched to flagship cloud models, ensuring safety and precision only where needed.

#### **5. Hybrid Global-Local Retrieval**
The system fuses three distinct retrieval strategies to ensure comprehensive coverage:
*   **Local Graph Traversal:** 1-2 hop expansion from query CUIs in Neo4j to find structured clinical facts (TREATS, CAUSES, etc.).
*   **Global Community Summarization:** Semantic search over **Leiden-detected entity communities** in Neo4j to answer broad, cross-corpus overview questions.
*   **Hybrid Vector Search:** Parallel BGE-M3 dense and sparse indexing on Qdrant. Results are merged via **Reciprocal Rank Fusion (RRF)** and prioritized by a cross-encoder reranker, ensuring both semantic depth and lexical precision (e.g., drug dosages).

#### **6. Parent-Child Document Retrieval (Small Chunk Search, Large Chunk Recall)**
Medical guidelines and FDA labels are inherently long and context-dependent. Fixed-size chunking often slices critical evidence (e.g., adverse reactions, dosage instructions) mid-sentence, leading to severe context truncation. MedGraphia solves this using a dynamic **Parent-Child Retrieval** architecture:
*   **Small Chunk Search (Vector/Graph):** Documents are split into dense, highly semantic sub-chunks (max 300 tokens) to guarantee precise vector matching and high retrieval precision.
*   **Large Chunk Recall (LLM Generation):** At query time, retrieved child chunks automatically pull their full `parent_text` (the complete clinical section) via Qdrant payloads before being injected into the LLM context window.
*   **Impact:** This ensures the generator LLM sees the complete clinical picture without re-triggering expensive database queries, completely eliminating the "hard-cutoff" hallucination problem.

---

## 🤖 LLM Core Functional Capabilities

MedGraphia leverages Large Language Models across the entire data lifecycle, from offline knowledge construction to real-time clinical reasoning.

| Capability                     | Module | Tech Stack | Prompt Management |
|:-------------------------------|:---|:---|:---|
| **Contextual Query Rewriting** | `Rewriter` | `dspy.Predict` | **DSPy Optimized**: Resolves coreference and ellipsis (e.g., "What are its side effects?") into standalone queries using conversation history. |
| **Clinical Answer Generation** | `Generator` | `dspy.ChainOfThought` | **DSPy Optimized**: Synthesizes multilingual context into evidence-based answers with mandatory inline [N] citations and medical disclaimers. |
| **Relation Extraction**        | `Extractor` | `dspy.Predict` | **DSPy Optimized**: Extracts high-fidelity medical triples (Disease → TREATS → Drug) from raw text, constrained to a closed ontology schema. |
| **Community Summarization**    | `Summarizer` | `dspy.Predict` | **DSPy Optimized**: Generates hierarchical summaries for Leiden-detected entity clusters to support global, cross-corpus thematic queries. |
| **Multilingual Translation**   | `Translator` | `dspy.Predict` | **DSPy Optimized**: Parallel translation of queries into ZH, EN, and DE to eliminate lexical bias in sparse/hybrid retrieval paths. |
| **Intelligent Intent Routing** | `Router` | `pydantic-ai` | **Typed Agent**: Classifies queries into 5 intent tiers (e.g., FAQ vs. Multi-hop) and generates a structured retrieval plan. |
| **Proactive Safety Guarding**  | `Guard` | `Llama-Guard 3` | **Direct Inference**: Performs two-stage (input/output) safety filtering against S1-S14 hazard categories. |

---

## 🎬 Architecture

### Build Pipeline (Offline)

```
┌────────────────────────────────── DATA SOURCES ──────────────────────────────────────┐
│ [EN] PubMed, FDA, DrugBank │ [ZH] Medical QA (Huatuo) │ [DE] Clin. Cases (GERNERMED) │
│ [EN] EMA SmPC (Local PDFs) │ [Anchor] MeSH Multilingual Descriptor Index             │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │  (API / bulk download)
         ┌─────────────────────────────────┼──────────────────────────────────┐
         │                                 │                                  │
┌────────▼────────────┐          ┌──────────▼────────────┐          ┌─────────▼─────────┐
│  Parse & OCR        │          │ Section-aware Chunking│          │  Multi-lang NER:  │
│  Docling  (EN / DE) │─────────▶│ anchor: section →     │─────────▶│  GLiNER + BERT    │
│  MinerU   (ZH)      │          │ paragraph → sentence  │          │                   │
│  Tesseract+PaddleOCR│          │ + FHIR Timing Norm.   │          └─────────┬─────────┘
└─────────────────────┘          └───────────────────────┘                    │
                                                                              │
                    ┌─────────────────────────────────────────────────────────▼──────────┐
                    │  Entity Linking: SapBERT-XLMR + BM25 candidates → MeSH ID          │
                    │  ZH / EN / DE surface forms → MeSH ID (e.g. D009203 = MI)          │
                    └─────────────────────────────────────┬──────────────────────────────┘
                                                          │
                    ┌─────────────────────────────────────▼────────────────────────────────┐
                    │  Schema-guided Relation Extraction (LLM + DSPy)                      │
                    │  TREATS · CAUSES · INTERACTS_WITH · DOSAGE_FOR · SYMPTOM_OF · etc.   │
                    │  Each edge: evidence_text · source_id · chunk_id · confidence        │
                    └────────────────────────┬─────────────────────────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────┐
              │                              │                          │
   ┌──────────▼────────────────┐  ┌──────────▼───────────────┐          │
   │  Leiden Community         │  │  BGE-M3 Embedding        │          │
   │  Detection + LLM          │  │  dense + sparse          │          │
   │  community summaries      │  │                          │          │
   └──────────┬────────────────┘  └───────────┬──────────────┘          │
              └─────────────────────┬─────────┘                         │
                                    │                                   │
          ┌─────────────────────────▼───────────────────────────────────▼──────────┐
          │                          STORAGE LAYER                                 │
          │   ┌─────────────────────────┐   ┌────────────────┐   ┌──────────────┐  │
          │   │  Neo4j 5.x              │   │  Qdrant        │   │  Local Disk  │  │
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
          │  Parallel Multilingual Query Expansion                 │
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
    │  Graph Retrieval  │   │  Hybrid Vector:   │  │  Community Summary    │
    │  Neo4j 1–2-hop    │   │  (BGE-M3)         │  │  Global search over   │
    │  subgraph from    │   │  dense + sparse   │  │  Leiden communities   │
    │  query entity CUI │   │  on Qdrant        │  │  (Multi-hop/Overview) │
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
          │  - Infrastructure: LiteLLM + LangGraph                 │
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

#### 1. Medical Entities (MeSH-Anchored)
Core knowledge nodes. All medical entities are uniquely identified by their **MeSH CUI**.

| Label | Description | Key Properties |
|---|---|---|
| `Disease` | Clinical conditions / syndromes | `cui`, `label`, `lang_zh`, `lang_de`, `confidence` |
| `Drug` | Substances / pharmaceutical products | `cui`, `label`, `lang_zh`, `lang_de`, `confidence` |
| `Symptom` | Signs and symptoms | `cui`, `label`, `lang_zh`, `lang_de`, `confidence` |
| `Gene` | Genes / proteins / markers | `cui`, `label`, `lang_zh`, `lang_de`, `confidence` |
| `Procedure` | Diagnostic or therapeutic procedures | `cui`, `label`, `lang_zh`, `lang_de`, `confidence` |

#### 2. Knowledge & Application Structure
Nodes supporting the RAG workflow and system metadata.

| Label | Description | Key Properties |
|---|---|---|
| `Document` | Source document metadata | `doc_id`, `title`, `source_id`, `format`, `language` |
| `Chunk` | Section-aware text fragment | `chunk_id`, `text`, `section_path`, `token_count` |
| `Community` | Leiden-detected entity cluster | `community_id`, `summary`, `size` |
| `User` | Application user profile | `id` (user_id) |
| `ChatSession` | Metadata for a chat thread | `session_id`, `user_id`, `language`, `domain` |
| `ChatMessage` | Individual message content | `message_id`, `role`, `content`, `model_used` |
| `ApiKey` | Authentication keys | `key_hash`, `prefix`, `role`, `active` |
| `PipelineStatus`| Progress of build tasks | `domain`, `stage`, `progress`, `error` |

---

### Relationship Types

#### 1. Semantic Relations (LLM Extracted)
Directed edges between two **Medical Entities**, derived from text evidence.

| Type | Example Triple (Source → Target) | Stored Properties |
|---|---|---|
| `TREATS` | Metformin → T2DM | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |
| `CAUSES` | T2DM → Retinopathy | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |
| `INTERACTS_WITH` | Warfarin → Aspirin | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |
| `SYMPTOM_OF` | Polyuria → T2DM | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |
| `COMPLICATION_OF` | Nephropathy → T2DM | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |
| `CONTRAINDICATED_IN`| Metformin → Renal Failure | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |
| `DOSAGE_FOR` | Tablets → Aspirin | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |
| `CODED_AS` | T2DM → E11 (ICD-10) | `confidence`, `evidence_text`, `chunk_id`, `source_id`, `extracted_by` |

#### 2. Structural & System Relations

| Type | Source → Target | Description |
|---|---|---|
| `MENTIONED_IN` | Entity → Chunk | Indicates an entity mention within a specific text fragment |
| `FROM_DOC` | Chunk → Document | Links a chunk back to its parent source document |
| `MEMBER_OF` | Entity → Community | Assigns an entity to a Leiden-detected community cluster |
| `HAS_MESSAGE` | ChatSession → ChatMessage| History chain for a specific conversation session |
| `INTERESTED_IN` | User → Entity | Tracks user-specific interests with decaying `weight` |

---

## 🛠 Tech Stack

| Layer | Technology                                                                                                                                    | Notes                                                                                                                                                                |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Language** | Python 3.12                                                                                                                                   |                                                                                                                                                                      |
| **API Framework** | FastAPI + Uvicorn                                                                                                                             | SSE streaming support                                                                                                                                                |
| **Agent Orchestration** | LangGraph (LangChain)                                                                                                                         | Stateful, branching, retriable query agent                                                                                                                           |
| **Prompt Optimization** | DSPy                                                                                                                                          | Optimized via **Adversarial Few-Shot Bootstrapping**; enforces clinical rigor through pre-compiled reasoning traces                                                  |
| **GraphRAG Framework** | Custom Implementation                                                                                                                         | Fuses **Leiden community summarization** (Global), **Entity-centric traversal** (Local), and **MeSH-anchored semantic alignment** for complex cross-corpus reasoning. |
| **Graph Database** | Neo4j 5.x                                                                                                                                     | Nodes and relationships                                                                                                                                              |
| **Vector Store** | Qdrant                                                                                                                                        | Native dense + sparse hybrid                                                                                                                                         |
| **Embedding** | BGE-M3 (BAAI)                                                                                                                                 | Dense + sparse                                                                                                                                                       |
| **Entity NER** | GLiNER (`gliner_mediumv2.1`) · `biomedical-ner-all` (EN / DE) · `bert-base-chinese-medical-ner` (ZH) | Multi-lang, domain-specialized                                                                                                                                       |
| **Entity Linking** | SapBERT-XLMR + BM25                                                                                                                           | Cross-lingual → MeSH ID                                                                                                                                              |
| **Relation Extraction** | Schema-guided LLM                                                                                                                       | Zero-shot extraction constrained to closed medical ontology                                                                                                          |
| **Reranker** | bge-reranker-v2-m3                                                                                                                            | Cross-encoder, multilingual                                                                                                                                          |
| **Community Detection** | Leiden algorithm                                                                                                                              | Graph clustering for global QA                                                                                                                                       |
| **Document Parsing** | Docling (EN/DE) · MinerU (ZH)                                                                                                                 | Section-aware; table / formula extraction                                                                                                                            |
| **OCR** | Tesseract 5 + PaddleOCR                                                                                                                       | Fallback for scanned PDFs and images                                                                                                                                 |
| **LLM Inference** | LiteLLM Gateway                                                                                                                               | Unified API layer. Supports **DeepSeek**, **OpenAI**, **Anthropic**, **Groq**, and **Ollama**.                                                                       |
| **Safety** | Llama-Guard-3-1B                                                                                                                              | Input + output filtering; S1-S14 policy                                                                                                                              |
| **Observability** | Langfuse (self-hosted)                                                                                                                        | prompt/token/latency/cost tracing                                                                                                                                    |
| **Evaluation** | RAGAS                                                                                                                                         | Faithfulness · Answer Relevance · Context Precision/Recall · **Synthetic Testset Generation**                                                                        |
| **Auth** | API Key                                                                                                                                       | Simple and secure key-based access control                                                                                                                           |
| **Pipeline Orchestration** | Prefect 3                                                                                                                           | Prefect for complex DAG orchestration                                                              |
| **Cache & Task Queue** | **Redis 7** + Arq                                                                                                                             | NER result cache + Redis-backed async worker queue                                                                                                                   |
| **UI** | Streamlit                                                                                                                                     | Chat · KG explorer · pipeline monitor · admin                                                                                                                        |
| **Containerization** | Docker + Docker Compose                                                                                                                       | Multi-target Dockerfiles                                                                                                                                             |

---

## ⚙️ How It Works

### 1 — Offline Build Pipeline

When you run `scripts/pipeline/build_graph.py` (or trigger the Prefect DAG), data flows through eight stages:

**Stage 1 — Fetch**

Data is ingested from a multilingual corpus across three languages:
- **English (EN)**: 
    - **PubMed**: Large-scale clinical abstracts (fetched via E-utilities or loaded from `data/raw/pubmed`).
    - **FDA DailyMed**: Structured US drug labels (REST API).
    - **EMA SmPC**: European drug labels (local PDFs in `data/raw/ema_smpc`).
    - **DrugBank**: High-quality pharmacological data (XML).
- **Chinese (ZH)**: 
    - **Huatuo-Lite**: Large-scale medical QA dataset (`data/raw/huatuo`).
- **German (DE)**: 
    - **GERNERMED**: Specialized German clinical case data (`data/raw/germed`).

**Stage 2 — Parse**

Documents are converted into a unified `RawDocument` schema using language-optimized engines:
- **Multilingual PDFs (EN/DE)**: Handled by **Docling** to preserve complex tables and section paths.
- **Chinese Academic PDFs**: Optimized via **MinerU** for double-column layouts and formula extraction.
- **Structured Datasets (ZH/DE/EN)**: The `StructuredParser` handles JSON/JSONL formats (Huatuo, GERNERMED, PubMed-preparsed) to ensure high-fidelity knowledge extraction.
- **Scanned Artifacts**: Fallback to **OCR (Tesseract 5 + PaddleOCR)** for image-only medical records.

**Stage 3 — Section-aware Chunking + Normalization**

Text is split following the document's structural hierarchy (`section → paragraph → sentence`). Each chunk carries a `section_path` metadata tag. A domain normalizer (`MedicalNormalizer`) unifies clinical expressions across languages:
- **Dosing**: `"每日两次"`, `"bid"`, `"twice daily"` → `"bid"` (**FHIR Timing code**).
- **Units**: `"500mg"` → `"500 mg"`, `"10μg"` → `"10 mcg"`.

**Stage 4 — Multi-language NER**

A two-stage pipeline extracts medical entities (`EntityType.DISEASE`, `EntityType.DRUG`, etc.):
1. **Coarse pass**: GLiNER (`gliner_mediumv2.1`) performs zero-shot multilingual entity detection across EN, ZH, and DE.
2. **Fine pass**: Language-specific BERT models refine candidate spans for higher precision.
    - **English**: `biomedical-ner-all`.
    - **Chinese**: `bert-base-chinese-medical-ner`.
    - **German**: Multilingual coarse pass only (fine pass model configurable).

**Stage 5 — Entity Linking to MeSH ID**

Provisional mentions are resolved to global **MeSH CUIs**:
1. **BM25 retrieval**: Finds top-K lexical candidates from the MeSH index.
2. **SapBERT-XLMR**: Cross-lingual semantic re-ranking to find the best CUI match.
3. **Graph Write**: Linked entities and `MENTIONED_IN` relationships are written to Neo4j.

**Stage 6 — Relation Extraction**

A **DSPy-powered extractor** identifies relationships between linked entities using the `DEFAULT_LLM_MODEL`. Extraction is strictly constrained to the system's `RelationType` schema:
- `TREATS`, `CAUSES`, `INTERACTS_WITH`, `SYMPTOM_OF`, `COMPLICATION_OF`, `CONTRAINDICATED_IN`, `DOSAGE_FOR`, `CODED_AS`.
- Every relation includes `evidence_text` and `confidence` score.

**Stage 7 — Embedding**

Chunks are embedded using **BGE-M3**, producing both **dense** (semantic) and **sparse** (lexical) vectors. These are indexed in Qdrant for hybrid retrieval.

**Stage 8 — Community Detection**

The **Leiden algorithm** clusters the entity graph based on relationship topology. An LLM then generates hierarchical summaries for each community, which are stored in Neo4j to support global "overview" queries.

---

### 2 — Online Query Pipeline

When a user submits a query, the system executes a multi-stage pipeline orchestrated by the API and the `RetrievalPipeline`:

**Step 0 — Proactive Defense (Safety Check)**
Before any processing, **Llama-Guard 3** inspects the input query and conversation history. Violations of S1-S14 safety categories (e.g., self-harm, illegal acts) trigger an immediate safe refusal.

**Step 1 — Contextual Rewriting (Short-Term Memory)**
If conversation history exists, the `QueryRewriter` resolves pronouns and ellipsis (e.g., "What are its side effects?" → "What are the side effects of Metformin?"), and create an overview of historical chat records in a fixed window, to create a standalone search query.

**Step 2 — Intent Routing & Multilingual Expansion**
- **NER & Linking**: The `QueryRouter` runs a multilingual NER pass on the query to identify medical entities and link them to MeSH CUIs. Results are transparently cached in Redis (TTL 1 h) via `route_async()`; cache hits skip BERT/SapBERT inference and reduce routing latency from **~2550 ms to < 5 ms**.
- **Intent Classification**: The query is classified into one of five categories: `CLINICAL_DECISION`, `DRUG_INTERACTION`, `LITERATURE_MULTIHOP`, `CROSS_CORPUS`, or `PATIENT_FAQ`.
- **Multilingual expansion**: To eliminate lexical bias in sparse retrieval, the `QueryTranslator` translates the query into all three corpus languages (ZH / EN / DE) in parallel.

**Step 3 — Three-Path Retrieval & Fusion**
The system executes a concurrent retrieval plan:
1. **Graph Traversal**: 1-2 hop expansion from query CUIs in Neo4j to find structured clinical facts.
2. **Hybrid Vector Search**: Parallel per-language Qdrant searches using BGE-M3 (dense + sparse) with quota-based merging.
3. **Community Summary**: Semantic search over Leiden-detected community summaries for global insights.

**Reciprocal Rank Fusion (RRF)** merges these paths, followed by a **bge-reranker-v2-m3** cross-encoder pass to select the top-20 most relevant context passages.

**Step 4 — Generation & Clinical Rigor**
- **LLM Routing**: The `LLMRouter` selects the optimal model tier (`SMALL`, `MEDIUM`, or `LARGE`) based on query complexity and language preference.
- **DSPy Prompt**: The generation is handled by a **compiled DSPy program** that enforces reasoning chains (CoT) and clinical rigor through pre-tuned few-shot demonstrations.
- **Output Moderation**: A final Llama-Guard check ensures the generated answer is safe before delivery.

**Step 5 — Persistence & Long-Term Memory**
- **Evidence Citation**: The system injects numbered citations `[1][2]` that map exactly to the source chunks and section paths.
- **Async Interest Update**: User interactions are recorded in Neo4j by creating or updating `INTERESTED_IN` relationships with an **exponential time-decay algorithm**, building a longitudinal user profile for language-agnostic personalization.


---

## 🚀 Deployment Modes

MedGraphia supports two deployment configurations that share the same codebase. Switch between them by selecting the Docker Compose file and the corresponding `.env` template.

| | Enterprise Mode | Lite Mode |
|---|---|---|
| **Target** | Production server / cloud | 16 GB Mac / PC (M1/M2 or RTX 3060+) |
| **Data scope** | Full multi-domain corpus | Single domain (e.g. T2DM), 100–500 abstracts + 50 drug labels |
| **Knowledge Base** | MeSH (Full Multilingual Index) | MeSH (Selected Descriptors) |
| **Vector DB** | Qdrant (dense + sparse hybrid) | Qdrant (dense + sparse hybrid, reduced memory footprint) |
| **Neo4j memory** | 16 GB+ page cache | 1–2 GB page cache (< 100K nodes / 500K edges) |
| **LLM** | vLLM/SGLang self-hosted 70B + cloud routed | Ollama 7B 4-bit GGUF or DeepSeek/Qwen API |
| **Auth** | Keycloak SSO + OPA role-based ACL | API key invite flow |
| **Observability** | Langfuse (GDPR-safe tracing) | Langfuse only |
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
DEFAULT_LLM_PROVIDER=deepseek          # deepseek | openai | anthropic | local
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

This starts: `neo4j`, `qdrant`, `api` (port **8058**), `worker`, `ui` (port **8501**), `langfuse`.

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
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=qwen2.5:7b            # 4-bit GGUF, ~4 GB VRAM
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
arq medgraphia.worker.WorkerSettings

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
  "model_used": "deepseek/deepseek-chat"
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

### 🧬 DSPy Prompt Optimization (Self-Improvement)

MedGraphia allows you to continuously improve the clinical reasoning of the system by generating synthetic training data and compiling prompts via DSPy.

**Step 1: Generate Synthetic Data**

Use the Teacher model (e.g., DeepSeek) to reverse-engineer high-quality, multilingual QA pairs from your grounded graph chunks.

```bash
# Generate 50 multilingual grounded QA pairs
python scripts/dspy/generate_synthetic_data.py 50
```

**Step 2: Compile & Optimize Signatures**

Run the DSPy Teleprompter to evaluate the synthetic data and bootstrap the best Chain-of-Thought (CoT) reasoning traces into compiled JSON files.

```bash
# Compiles Rewriter and Generator modules using BootstrapFewShot / MIPROv2
python scripts/dspy/optimize.py
```

*Note: The compiled JSON files are automatically saved to `data/dspy/` and injected into the pipeline at runtime to guide the Student model's clinical reasoning.*

---

## 📊 Evaluation

MedGraphia includes a comprehensive evaluation suite using RAGAS for quality measurement and automated testset generation.

> **Note**: Evaluation scripts require `OPENAI_API_KEY` to be set in your environment, as they use GPT-4o-mini as a "judge" model.

### 1. Synthetic Testset Generation

Generate high-quality medical QA pairs from your processed data:

```bash
python scripts/evaluation/generate_testset.py \
  --data-dir data/processed \
  --test-size 50 \
  --docs-limit 10 \
  --output data/evaluation/synthetic_testset.csv
```

**Parameters:**
- `--data-dir`: Directory containing processed JSON files.
- `--test-size`: Number of QA pairs to generate.
- `--docs-limit`: Number of source documents to load.
- `--output`: Output CSV path.
- `--append`: Append to existing file instead of overwriting.
- `--max-workers`: Number of parallel RAGAS workers (default: 4).

### 2. RAG Pipeline Metrics

Evaluate the full RAG pipeline (Faithfulness, Relevance, Precision, Recall) using the generated testset:

```bash
python scripts/evaluation/eval_rag_metrics.py \
  --input data/evaluation/synthetic_testset.csv \
  --output eval_results.csv
```

**🏆 Performance Leap (Parent-Child Optimization & DSPy MIPROv2)**

By implementing the **Parent-Child Retrieval architecture** (expanding LLM context boundaries) combined with **DSPy prompt tuning** and fuzzy semantic deduplication, the system's baseline local 7B model achieved a significant leap in clinical QA rigor:

*   **Faithfulness**: Surged from `0.714` to **`0.849`** (Responses are strictly grounded in medical evidence without hallucinations).
*   **Answer Relevancy**: Improved from `0.698` to **`0.882`** (Direct, highly professional answers to complex medical queries).
*   **Context Precision**: Jumped from `0.574` to **`0.832`** (Highly relevant documents are ranked at the top of the context).
*   **Context Recall**: Increased from `0.555` to **`0.699`** (Better capture of long-tail medical facts via parent-text expansion).

---

## ⚙️ Configuration

All settings are loaded from `.env` via Pydantic Settings. Key variables:

| Variable | Default | Description |
|---|---|---|
| **Architecture** | | |
| `AUTH_STRATEGY` | `apikey` | Authentication strategy: `none` or `apikey` (Default) |
| **Neo4j** | | |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_PAGE_CACHE` | `1G` | Page cache size (e.g., `1G` for laptop, `32G` for server) |
| **Vector Store** | | |
| `VECTOR_STORE` | `qdrant` | Only `qdrant` is supported |
| **LLM & Embedding** | | |
| `DEFAULT_LLM_PROVIDER` | `groq` | `openai` \| `anthropic` \| `deepseek` \| `gemini` \| `groq` \| `ollama` \| `local` |
| `EMBEDDING_PROVIDER`| `ollama` | `huggingface` \| `openai` \| `ollama` |
| **Observability** | | |
| `TRACING_ENABLED` | `false` | Enable Langfuse tracing |
| **Compliance** | | |
| `GUARDRAILS_ENABLED`| `true` | Enable Llama-Guard safety checks |
| `LLAMA_GUARD_PROVIDER`| `ollama` | Provider for the safety model |
| `LLAMA_GUARD_MODEL`| `llama-guard3:1b`| Model name for the safety guardrails |
| **Cache & Task Queue** | |
| `REDIS_URL` | *(unset)* | Redis connection string. When set, enables NER result caching and Arq task-queue dispatch (e.g. `redis://redis:6379/0`). Omit to run without Redis. |
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
├── data/
│   └── dspy/                   # Compiled DSPy programs (optimized reasoning traces)
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
│   │   ├── clean_short_chunks.py   # Utility to remove low-info chunks
│   │   ├── embed_entities.py       # Standalone entity embedding task
│   │   ├── extract_relations_only.py # Schema-guided RE only mode
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
│       ├── generate_testset.py      # Automated synthetic testset generation (RAGAS)
│       ├── eval_rag_metrics.py      # RAG pipeline evaluation (Faithfulness, etc.)
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
    │   ├── normalizer.py           # Dose/unit normalization → FHIR Timing code
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
    │   ├── gateway.py              # LiteLLMGateway: unified multi-provider interface
    │   ├── client.py               # pydantic-ai model factory for structured LLM output
    │   └── dspy_setup.py           # DSPy infrastructure: LM configuration & task routing
    │
    ├── programs/                   # DSPy Programs: Encapsulated business logic
    │   ├── rewriter.py             # Context-aware query rewrite module
    │   ├── generator.py            # Clinical answer generation module (CoT)
    │   ├── extractor.py            # Relation extraction module
    │   ├── summarizer.py           # Community summarization module
    │   └── translator.py           # Medical terminology translation module
    │
    ├── retrieval/                  # Online query pipeline (three-path hybrid)
    │   ├── pipeline.py             # RetrievalPipeline: orchestrates all retrieval steps
    │   ├── router.py               # Query classification → retrieval strategy selection
    │   ├── rewriter.py             # QueryRewriter: condense history into standalone query
    │   ├── query_translator.py     # QueryTranslator: translate query into ZH/EN/DE
    │   ├── query_ner.py            # NER on incoming query for entity-based graph lookup
    │   ├── graph_retriever.py      # Neo4j 1–2-hop subgraph from entity CUIs in query
    │   ├── vector_retriever.py     # BGE-M3 dense + sparse hybrid search on Qdrant
    │   ├── community_retriever.py  # Leiden community summary search (global QA)
    │   ├── reranker.py             # bge-reranker-v2-m3 cross-encoder
    │   └── fusion.py               # Reciprocal Rank Fusion (RRF) across all three paths
    │
    ├── generation/                 # LLM generation layer
    │   ├── pipeline.py             # GenerationPipeline: context prep → routing → LLM → citations
    │   ├── llm_router.py           # Route by query type / language → SMALL / MEDIUM / LARGE tier
    │   ├── citation.py             # Inline citation injection → provenance
    │   └── guard.py                # Llama-Guard 3 safety filtering logic
    │
    ├── prompts/                    # Pydantic-typed prompt modules (DSPy signatures)
    │   ├── answer_generation.py
    │   ├── community_summary.py
    │   ├── query_rewriting.py
    │   ├── relation_extraction.py
    │   └── safety.py
    │
    ├── api/                        # FastAPI application
    │   ├── __init__.py             # App factory with lifespan management
    │   ├── schemas.py              # Pydantic request / response DTOs
    │   ├── deps.py                 # FastAPI Depends: auth, session, rate limit
    │   ├── middleware.py           # Audit logging, GDPR-safe request tracing
    │   ├── chat.py                 # POST /chat (blocking) + POST /chat/stream (SSE)
    │   ├── knowledge.py            # GET /graph/entity, GET /graph/entity/search, GET /graph/stats
    │   ├── admin.py                # Pipeline trigger, model config, user management
    │   ├── health.py               # GET /health/live  &  GET /health/ready
    │   └── auth.py                 # API key auth (lite) / Keycloak OIDC (enterprise)
    │
    ├── cache/                      # Redis-backed caching layer
    │   ├── redis_client.py         # Async Redis singleton with graceful no-op fallback
    │   └── ner_cache.py            # NER + Entity Linking result cache (key: ner:{lang}:{sha256[:16]})
    │
    ├── worker/                     # Arq async task-queue worker
    │   ├── __init__.py             # WorkerSettings — start with `arq medgraphia.worker.WorkerSettings`
    │   ├── tasks.py                # task_build_pipeline: full offline pipeline as a durable Arq task
    │   └── arq_client.py           # Arq pool singleton for enqueueing from the API process
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

- [ ] **Adaptive Context Injection**: Dynamically adjust context window based on LLM's uncertainty.

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

> Moving beyond simple hybrid search, MedGraphia bridges the semantic gap across Chinese, English,
  and German medical data through deep ontology alignment. It ensures clinical rigor via DSPy prompt
  optimization and proactive Llama-Guard safety checks. Equipped with a dual long-short-term memory
  system featuring time-decay attention, intelligent query-complexity LLM routing, and an integrated
  RAGAS evaluation suite, it is designed from the ground up for safe, scalable, and personalized
  healthcare AI.