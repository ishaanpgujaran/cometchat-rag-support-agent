# Aster & Row — RAG Customer Support Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-232%20passed-brightgreen.svg)]()
[![Evaluation](https://img.shields.io/badge/eval%20suite-25%2F25%20(100%25)-success.svg)]()

A production-grade, deterministic Retrieval-Augmented Generation (RAG) customer support agent built from first principles for **Aster & Row** — a premium outdoor and lifestyle gear brand specializing in technical bags, drinkware, and travel accessories.

Built without heavy, opaque agent frameworks (no LangChain, no LlamaIndex), this system implements a **deterministic reliability and security boundary around an LLM**. It guarantees strict lifecycle precedence, zero raw-database disclosure, hard boundary routing, real-time citation validation, and automated conflict resolution.

---

## Demo

![Demo Walkthrough](docs/demo.gif)

> **Demo Walkthrough Video Highlights:**
> 1. **Knowledge-base Question with Citation:** Querying return windows and refund policies correctly retrieves and cites `01-returns-policy-current.md#Standard return window` without hallucination.
> 2. **Real-time Order Lookup:** Querying order `ORD-1007` invokes `lookup_order`, sanitizes sensitive fields, and presents transit status via UPS without exposing raw JSON or internal tags.
> 3. **Multi-Turn Conversation:** Asking *"Do you ship to Canada?"* followed by *"What about duties?"* maintains conversational topic context across turns.
> 4. **Safe Refusal & Human Handoff:** Inquiries requesting cancellations, refunds, address changes, or out-of-scope vegan material guarantees cleanly escalate to human support (`human_handoff = True`).
> 5. **Automated Evaluation Suite:** Running `python -m evaluation.run_eval` demonstrates a **25/25 (100%) pass rate** across all visible and edge evaluation suites.

*(Video file also available in repo: [`docs/demo.mp4`](docs/demo.mp4))*

---

## Technology Stack & Architectural Decisions

| Layer | Technology Chosen | Rationale & Architectural Design |
| :--- | :--- | :--- |
| **LLM Engine** | **Google Gemini Flash-class** (Primary: `gemini-3.7-flash`; Fallbacks: `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite` via official `google-genai` SDK) | Ultra-fast time-to-first-token, strong instruction compliance for strict JSON/markdown constraints, and generous token limits. Configurable via `.env`. |
| **Embeddings** | **`BAAI/bge-small-en-v1.5`** (Local via `sentence-transformers`) | State-of-the-art 384-dimensional dense semantic representation running locally on CPU/GPU without third-party API latency or rate limits. |
| **Sparse Retrieval** | **BM25 (`rank_bm25`)** | Exact-keyword and SKU/order/policy term matching to complement dense semantic embeddings. |
| **Hybrid RAG** | **Dense + Sparse Hybrid Ranker** with Metadata Scoring | Normalized weighted combination of cosine similarity and BM25 scores, augmented with document metadata boosts for active/official policy chunks and severe penalties for superseded drafts. |
| **Order Tool & Safety** | **Deterministic Whitelist DTO (`SafeOrderResult`)** | Tool execution never exposes raw database rows. Customer PII and internal fields (`internal_notes`, `fraud_risk_score`, `warehouse_id`) are stripped before LLM exposure. |
| **Agent Architecture** | **Custom Deterministic State Machine** | Full control over the orchestration lifecycle, eliminating framework abstractions, unpredictable prompt injections, and hidden execution loops. |
| **User Interfaces** | **Streamlit Web UI** + **Typer / Rich CLI** | Interactive Web UI with live observability trace inspection sidebar, alongside a terminal CLI with rich panel rendering and `--debug` execution tracing. |
| **Storage & Sessions** | **In-Memory Thread-Safe Session Store** | Ephemeral, structured conversation state storing message histories, extracted order context, topic tracking, and execution traces. |

---

## Architectural Tradeoffs & Decision Matrix

To guarantee strict compliance, privacy, deterministic safety, and sub-second response latency, every architectural component was selected after evaluating alternative approaches.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               ARCHITECTURAL DECISION MATRIX                             │
├──────────────────────┬──────────────────────────────┬───────────────────────────────────┤
│ Component            │ Chosen Approach              │ Rejected Alternatives             │
├──────────────────────┼──────────────────────────────┼───────────────────────────────────┤
│ Embedding Model      │ BAAI/bge-small-en-v1.5       │ all-MiniLM-L6-v2, OpenAI/Gemini   │
│ Retrieval Strategy   │ Hybrid (Dense + BM25 + Meta) │ Pure Vector DB, Pure Sparse BM25  │
│ Orchestration Core   │ Custom Pure-Python Engine    │ LangChain, LlamaIndex, CrewAI     │
│ Intent Routing       │ Deterministic Regex Cascade  │ LLM-as-a-Router, Semantic Router  │
│ Tool Data Protection │ Whitelist DTO (SafeOrder)    │ Raw Database JSON in Context      │
│ Conflict Resolution  │ Lifecycle Filter + Registry  │ LLM Prompt Reconciliation         │
│ Safety Guardrails    │ Deterministic Regex Validator│ Secondary LLM Guardrail Call      │
└──────────────────────┴──────────────────────────────┴───────────────────────────────────┘
```

### 1. Embedding Model: `BAAI/bge-small-en-v1.5` vs. Alternatives

- **Chosen:** `BAAI/bge-small-en-v1.5` (Local 384-dimensional dense vectors via `sentence-transformers`).
- **Alternatives Evaluated:** `sentence-transformers/all-MiniLM-L6-v2`, OpenAI `text-embedding-3-small`, Google Gemini Cloud Embeddings.
- **Tradeoff Justification:**
  - **Why BGE-small:** Ranked top-tier on the Massive Text Embedding Benchmark (MTEB) for retrieval and passage ranking. Its asymmetric query-passage scoring excels at matching conversational user questions with formal policy headings. Local execution guarantees 0ms network latency, zero per-token embedding costs, and offline deterministic embeddings.
  - **Why `all-MiniLM-L6-v2` was rejected:** An older architecture with noticeably lower semantic discriminability on nuanced policy differences (e.g. differentiating between standard 30-day return windows vs. TrailPlus 45-day exceptions).
  - **Why Cloud Embeddings were rejected:** Adding an external API hop for every retrieval turn introduces failure points, latency overhead (200–500ms), and consumes API rate limits (critical when operating under strict free-tier quotas).

### 2. Retrieval Architecture: Hybrid (Dense + BM25 + Precedence) vs. Pure Vector DB

- **Chosen:** In-Memory NumPy Cosine Similarity + BM25 (`rank_bm25`) + Metadata Precedence Ranker.
- **Alternatives Evaluated:** Pure Vector Database (ChromaDB / Pinecone / Qdrant), Pure Sparse Search (Elasticsearch / BM25 only).
- **Tradeoff Justification:**
  - **Why Hybrid with Metadata:** Dense retrieval captures semantic intent (e.g., *"How long do I have to send it back?"* $\rightarrow$ `01-returns-policy-current.md`), while BM25 guarantees exact keyword recall for specific order codes, SKU names, and exact policy terms (e.g., *"Breeze Tumbler"*, *"TrailPlus"*). Our metadata scoring layer directly applies business logic boosts (`audience=customer`, `status=active`) and heavy penalties (`status=superseded`, `audience=internal`) before context synthesis.
  - **Why Pure Vector DB was rejected:** Vector-only search frequently misses exact alphanumeric matches (order numbers, SKU names) and cannot enforce document lifecycle status without complex post-query filtering. External vector databases also introduce unnecessary infrastructure overhead for a curated 14-document knowledge base.
  - **Why Pure BM25 was rejected:** Incapable of handling paraphrased queries or synonym matching where customers use colloquial language.

### 3. Orchestration: Custom Deterministic Engine vs. Agent Frameworks

- **Chosen:** Lightweight, custom Python orchestrator (`app/agent/orchestrator.py`).
- **Alternatives Evaluated:** LangChain, LlamaIndex, CrewAI, AutoGen.
- **Tradeoff Justification:**
  - **Why Custom Orchestrator:** 100% transparent execution paths, exact control over prompt structures, predictable zero-overhead execution, and seamless integration with native `google-genai` SDK `genai_types.Content` objects. Critical for preserving internal model attributes like `thought_signature` on multi-turn function calls.
  - **Why Frameworks were rejected:** Heavy agent frameworks introduce deep abstraction layers that obscure prompt payloads, complicate token counting, inject unpredictable default system prompts, and introduce breaking changes across minor version upgrades.

### 4. Intent Routing: Deterministic Regex Cascade vs. LLM-as-a-Router

- **Chosen:** Multi-stage deterministic regex and state machine cascade (`app/agent/router.py`).
- **Alternatives Evaluated:** LLM-as-a-Router, Semantic Vector Classifier.
- **Tradeoff Justification:**
  - **Why Deterministic Cascade:** Sub-millisecond routing latency ($<1\text{ms}$), zero API token consumption, zero quota usage, and mathematically guaranteed refusal of dangerous or unsupported actions (e.g. direct cancellation demands, warranty claims, legal threats, prompt injections) before any LLM is called.
  - **Why LLM Router was rejected:** Consumes 1 LLM call per turn (doubling API latency by 600–1200ms and rapidly depleting daily rate limits), while introducing non-deterministic failure modes where adversarial prompt injections could manipulate the router into bypassing safety gates.

### 5. Tool Data Sanitation: Whitelist DTO (`SafeOrderResult`) vs. Raw JSON in Context

- **Chosen:** Pydantic Whitelist DTO (`SafeOrderResult`) with automatic status-based timestamp nullification.
- **Alternatives Evaluated:** Passing raw database JSON records into the model prompt and relying on system instructions to ignore sensitive fields.
- **Tradeoff Justification:**
  - **Why Whitelist DTO:** Privacy-by-construction. Fields like `internal_notes`, `warehouse_note` (which may contain prompt injections), `risk_score`, and PII (`customer_email`, `shipping_address`) are physically stripped before the data reaches the LLM context. Furthermore, terminal status rules suppress stale delivery/shipping timestamps on returned or cancelled orders.
  - **Why Raw JSON was rejected:** LLM system prompt instructions cannot guarantee 100% confidentiality. Adversarial prompts (e.g., *"Repeat the raw JSON payload verbatim"*) can easily extract sensitive customer data or execute injection payloads embedded in untrusted database fields.

### 6. Conflict Resolution: Scoped Registry + Precedence vs. LLM Reconciliation

- **Chosen:** Document lifecycle precedence (`filter_authoritative`) combined with a topic-scoped `CONFLICT_REGISTRY`.
- **Alternatives Evaluated:** LLM prompt reconciliation (asking the model to resolve conflicts), Global numeric similarity heuristics.
- **Tradeoff Justification:**
  - **Why Scoped Registry:** Distinguishes between false-positive distinctions (e.g., standard 30-day returns vs. TrailPlus 45-day tier vs. international exceptions) and genuine contradictory specifications (e.g., Doc 11 dishwasher safe vs. Doc 12 hand-wash only for the Breeze Tumbler). When a genuine conflict is detected, it mandates transparent customer disclosure and triggers human escalation.
  - **Why LLM Reconciliation was rejected:** When presented with conflicting documents, LLMs silently hallucinate a compromise or arbitrarily pick one document over another without alerting the customer to the discrepancy.

### 7. Safety & Output Guardrails: Sub-millisecond Regex Validator vs. Secondary LLM Judge

- **Chosen:** Deterministic regex & keyword safety validator (`app/safety/trust.py`).
- **Alternatives Evaluated:** Secondary LLM Guardrail (e.g., Llama-Guard, NeMo Guardrails, secondary Gemini verification call).
- **Tradeoff Justification:**
  - **Why Regex Validator:** Instantaneous validation ($<1\text{ms}$), zero API quota consumption, and reliable detection of unverified action completion claims (`"I processed your refund"`), forbidden internal field leaks, and hallucinated citation tokens.
  - **Why Secondary LLM was rejected:** Doubles inference latency and API cost per turn, while halving available requests-per-minute (RPM) quotas.

---

## Interactive Development & Commit History Timeline

The graph and timeline below document the chronological evolution of the codebase across the 2-day sprint, including the overnight rate-limit cooldown and the morning sprint to achieve a 100% evaluation pass rate.

### Visual Commit Flow

```mermaid
gitGraph
    commit id: "22fdc5d" tag: "Take-Home Baseline"
    branch day-1-foundations
    checkout day-1-foundations
    commit id: "5512c5d" msg: "Scaffold modular architecture"
    commit id: "ca07551" msg: "KB ingestion & hybrid retrieval"
    commit id: "e61f653" msg: "Safe order tool & DTO"
    commit id: "a739d2b" msg: "Session store & orchestrator"
    commit id: "10ab173" msg: "Structured tracing & observability"
    commit id: "5a77b8b" msg: "Streamlit UI & Rich CLI"
    commit id: "bd13514" msg: "Gemini Flash engine integration"
    branch rate-limit-pause
    checkout rate-limit-pause
    commit id: "RPD-PAUSE" msg: "⏸️ Rate limit pause (RPD exhausted)"
    checkout day-1-foundations
    merge rate-limit-pause id: "RESUME" msg: "⚡ Resumed work (Day 2)"
    branch day-2-quality-flywheel
    checkout day-2-quality-flywheel
    commit id: "ee15238" msg: "Supersession precedence filter"
    commit id: "95943f0" msg: "25-case eval & model rotation"
    commit id: "ddce753" msg: "TrailPlus supplemental retrieval"
    commit id: "1b6ada2" msg: "Baseline eval recorded (52%)"
    commit id: "8637a8e" msg: "Reasoning & intent hardening"
    commit id: "0abc0a2" msg: "🏆 100% eval pass milestone"
    commit id: "f574253" msg: "Model selector & UI docs"
    commit id: "36cdaa8" msg: "Safety & citation cleanup"
    commit id: "21eb320" msg: "Policy router guards & polish"
    checkout main
    merge day-2-quality-flywheel id: "PRODUCTION" tag: "25/25 PASS"
```

### Chronological Milestone Timeline

```mermaid
timeline
    title Aster & Row Support Agent Development Sprint
    section Day 1 — Foundations & Core Architecture (Aug 22)
        18:31 : Commit 5512c5d : Initialized modular repository structure and configuration
        19:04 : Commit ca07551 : Markdown chunk parser, BGE-small embeddings, BM25 indexing
        19:28 : Commit e61f653 : Deterministic order lookup tool with SafeOrderResult DTO
        19:55 : Commit a739d2b : In-memory session context store and core orchestration loop
        20:31 : Commit 10ab173 : Privacy-aware structured JSON logging and trace data models
        21:02 : Commit 5a77b8b : Streamlit Web UI and Typer Rich terminal CLI interfaces
        21:32 : Commit bd13514 : Upgraded to Gemini Flash engine with genai_types.Content
        21:35 : ⏸️ Daily Rate Limit Cooldown : Daily Gemini API free-tier RPD exhausted; paused work overnight
    section Day 2 — Quality Flywheel & 100% Eval Pass (Aug 23)
        12:00 : ⚡ Resumed Work : Resumed 30m prior to first commit; diagnosed failure modes
        12:32 : Commit ee15238 : Filtered superseded docs (Bug 1) and fixed precedence scoring
        13:51 : Commit 95943f0 : Expanded eval benchmark to 25 cases & built model rotation engine
        14:26 : Commit ddce753 : Added supplemental KB retrieval for TrailPlus order policies (Bug 6)
        14:49 : Commit 1b6ada2 : Captured early baseline benchmark report: 13/25 passed (52%)
        15:35 : Commit 8637a8e : Hardened order intent detection and Gemini reasoning prompt
        17:28 : Commit 0abc0a2 : 🏆 All 25/25 evaluation cases passing (100% Pass Rate)
        17:42 : Commit f574253 : Added live model selection dropdown to Web UI with error handling
        18:22 : Commit 36cdaa8 : Fixed 'internal' keyword validator false-positive (Bug 8) and citations
        20:28 : Commit 21eb320 : Protected policy inquiries against context bleed (Bug 9) and UI polish
```

### Commit Milestones & What Happened Before Each Commit

| Commit | Timestamp | Stage & Context | What Was Happening & Resolved |
| :--- | :--- | :--- | :--- |
| `5512c5d` | Aug 22, 18:31 | **Scaffold** | Initialized the modular package structure (`app/agent`, `app/retrieval`, `app/orders`, `app/safety`, `app/policy`, `app/session`, `app/observability`). |
| `ca07551` | Aug 22, 19:04 | **Retrieval** | Built Markdown document parser with YAML frontmatter extraction, BM25 indexing, and local BGE-small dense embeddings. |
| `e61f653` | Aug 22, 19:28 | **Tool Safety** | Implemented `lookup_order` with Pydantic whitelist DTO (`SafeOrderResult`) and case/whitespace normalization (`ORD-XXXX`). |
| `a739d2b` | Aug 22, 19:55 | **Orchestration** | Implemented `SessionStore` with thread-locking and the main multi-turn agent execution loop. |
| `10ab173` | Aug 22, 20:31 | **Observability** | Designed privacy-safe structured JSON tracing recording timestamps, candidates, scores, tool calls, and safety flags. |
| `5a77b8b` | Aug 22, 21:02 | **UI / CLI** | Built interactive Rich terminal CLI and Streamlit Web UI with debug trace inspectability. |
| `bd13514` | Aug 22, 21:32 | **Model Engine** | Integrated official Google GenAI SDK with structured `genai_types.Content` multi-turn message handling. |
| *RPD Pause* | *Aug 22, 21:35* | **⏸️ Rate Limit Pause** | **Exhausted free-tier Requests-Per-Day (RPD) during continuous interactive testing. Work was paused overnight to allow daily quota refresh.** |
| *Resume* | *Aug 23, 12:00* | **⚡ Resumed Work** | **Resumed work 30 minutes prior to first commit. Diagnosed early baseline failure modes and formulated the model rotation strategy.** |
| `ee15238` | Aug 23, 12:32 | **Bug 1 & 2 Fix** | Resolved superseded document conflict errors and removed brittle numeric fallback heuristics. |
| `95943f0` | Aug 23, 13:51 | **Eval & Rotation** | Expanded evaluation suite to 25 diverse scenarios; implemented automated round-robin model rotation across `GEMINI_EVAL_MODELS` to prevent rate-limiting. |
| `ddce753` | Aug 23, 14:26 | **Bug 6 Fix** | Added supplemental KB retrieval on `ORDER_LOOKUP` routes for TrailPlus membership queries. |
| `1b6ada2` | Aug 23, 14:49 | **Baseline Record** | Formatted and committed the official early baseline run benchmark: **13 / 25 passed (52%)**. |
| `8637a8e` | Aug 23, 15:35 | **Reasoning Prompt** | Refined system prompts for strict citation anchoring, preserved `thought_signature`, and hardened intent parsing. |
| `0abc0a2` | Aug 23, 17:28 | **🏆 100% Pass** | Resolved remaining edge cases in prompt injection and terminal timestamp suppression. **Achieved 25/25 (100%) evaluation pass rate.** |
| `f574253` | Aug 23, 17:42 | **Model Switcher** | Added interactive model selector dropdown in Streamlit UI with automated graceful fallback handling. |
| `36cdaa8` | Aug 23, 18:22 | **Bug 8 & 10 Fix** | Whitelisted legitimate descriptive uses of `"internal document"` and sanitized hallucinated order citation artifacts. |
| `21eb320` | Aug 23, 20:28 | **Bug 9 & UI Polish** | Guarded router continuation against context bleed, refined Streamlit UI aesthetics with starter prompt chips, and polished Rich CLI. |

---

## Architecture Flow

```
                      ┌─────────────────────────┐
                      │    Incoming Customer    │
                      │         Message         │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │   Deterministic Intent  │
                      │      Router Engine      │
                      └───────┬───┬───┬───┬─────┘
                              │   │   │   │
        ┌─────────────────────┘   │   │   └──────────────────────┐
        │                         │   │                          │
        ▼                         ▼   ▼                          ▼
┌───────────────┐       ┌───────────────┐ ┌───────────────┐ ┌────────────────┐
│ KNOWLEDGE_    │       │ ORDER_LOOKUP  │ │ NEEDS_ORDER_  │ │ UNSAFE_OR_     │
│ LOOKUP        │       │               │ │ ID            │ │ UNSUPPORTED    │
└───────┬───────┘       └───────┬───────┘ └───────┬───────┘ └────────┬───────┘
        │                       │                 │                  │
        ▼                       ▼                 │                  │
┌───────────────┐       ┌───────────────┐         │                  │
│ Hybrid Vector │       │ Safe Order    │         │                  │
│ & BM25 Search │       │ Whitelist DTO │         │                  │
└───────┬───────┘       └───────┬───────┘         │                  │
        │                       │                 │                  │
        ▼                       ▼                 │                  │
┌───────────────┐       ┌───────────────┐         │                  │
│ Active Status │       │ Optional KB   │         │                  │
│ & Precedence  │       │ Context Boost │         │                  │
└───────┬───────┘       └───────┬───────┘         │                  │
        │                       │                 │                  │
        ▼                       ▼                 │                  │
┌───────────────┐       ┌───────────────┐         │                  │
│ Conflict Group│       │ Dynamic Tool  │         │                  │
│ & Deduplication       │ Citations     │         │                  │
└───────┬───────┘       └───────┬───────┘         │                  │
        │                       │                 │                  │
        └───────────────┬───────┘                 │                  │
                        ▼                         │                  │
        ┌───────────────────────────────┐         │                  │
        │  Gemini Flash Model Engine    │         │                  │
        │  (Strict Evidence & Anchors)  │         │                  │
        └───────────────┬───────────────┘         │                  │
                        │                         │                  │
                        ▼                         │                  │
        ┌───────────────────────────────┐         │                  │
        │  Post-Generation Trust &      │         │                  │
        │  Citation Validation Layer    │         │                  │
        └───────────────┬───────────────┘         │                  │
                        │                         │                  │
                        ├─────────────────────────┴──────────────────┘
                        ▼
        ┌───────────────────────────────┐
        │        AgentResponse          │
        │ • Verified Answer Text        │
        │ • Deduplicated Real Citations │
        │ • Deterministic Handoff Flag  │
        └───────────────────────────────┘
```

---

## Setup and Run Instructions (Clean Clone)

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.10+ supported)
- Google Gemini API Key ([Get an API key from Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Clone & Install
```bash
# Clone repository
git clone https://github.com/your-username/cometchat-rag-support-agent.git
cd cometchat-rag-support-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the `.env.example` template and add your Gemini API key:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
# Required
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional Model Overrides (defaults shown)
GEMINI_MODEL=gemini-3.7-flash
GEMINI_EVAL_MODELS=gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## Running the Agent

### Terminal CLI Mode
```bash
# Interactive conversation CLI
python3 -m app.cli

# Interactive CLI with real-time debug trace output
python3 -m app.cli --debug
```

### Streamlit Web UI Mode
```bash
# Launch browser interface
streamlit run app/web.py
```
Open your browser at `http://localhost:8501`. The Web UI includes:
- Live conversation chat interface with starter suggestion prompt cards.
- Sidebar with **Debug Trace** toggle to inspect real-time execution traces (route decisions, retrieved candidate scores, tool executions, and safety validation flags).
- Session management with one-click **New Conversation** reset.

---

## Running Automated Tests & Evaluations

### Run pytest Test Suite (Unit + Integration + Regression)
```bash
pytest tests/
```
*Current test suite: **232 passed, 0 failed** in ~12 seconds.*

### Run Evaluation Suite (25 Test Cases)
```bash
python -m evaluation.run_eval
```
*The evaluation suite rotates across models defined in `GEMINI_EVAL_MODELS` to respect free-tier RPM/RPD limits. Expected runtime: ~6–7 minutes. Results and breakdown are saved to `evaluation/results.json`.*

---

## Evaluation Results: Baseline vs. Final

The evaluation benchmark contains 25 rigorous test cases spanning edge cases across 10 evaluation categories (including visible baseline cases, multi-turn state drift, prompt injections, superseded policy precedence, and privacy leaks).

| Evaluation Category | Baseline (Early Prototype) | Final Implementation | Improvement |
| :--- | :---: | :---: | :---: |
| **Retrieval Accuracy** | 3 / 7 (43%) | **7 / 7 (100%)** | +57% |
| **Groundedness & Factuality** | 4 / 7 (57%) | **7 / 7 (100%)** | +43% |
| **Tool Execution** | 6 / 9 (67%) | **9 / 9 (100%)** | +33% |
| **Tool Argument Precision** | 2 / 3 (67%) | **3 / 3 (100%)** | +33% |
| **Privacy & Data Redaction** | 1 / 1 (100%) | **1 / 1 (100%)** | 100% (Maintained) |
| **Multi-turn Context Tracking**| 0 / 2 (0%) | **2 / 2 (100%)** | +100% |
| **Safety & Threat Escalation**| 2 / 2 (100%) | **2 / 2 (100%)** | 100% (Maintained) |
| **Abstention & Out-of-Scope** | 4 / 6 (67%) | **6 / 6 (100%)** | +33% |
| **Citation Precision** | 5 / 11 (45%) | **11 / 11 (100%)** | +55% |
| **Source Conflict Handling** | 0 / 1 (0%) | **1 / 1 (100%)** | +100% |
| **Total Test Suite** | **13 / 25 (52%)** | **25 / 25 (100%)** | **+48% (Perfect Score)** |

---

## Bug Diary: High-Impact Failures & Fixes

All 10 failure modes discovered during development and manual exploration are indexed below and documented in full detail in [docs/BUG_DIARY.md](docs/BUG_DIARY.md). Here are four critical representative entries:

- **Bug 1:** Superseded Policy Document Treated as Active Conflict
- **Bug 2:** False-Positive Conflict Detections Across Distinct Document Contexts *(Discovered beyond visible case wording)*
- **Bug 3:** Gemini SDK Function Calling `thought_signature` Stripping
- **Bug 4:** False Completed-Action Redaction for Valid Order Statuses *(Discovered beyond visible case wording)*
- **Bug 5:** Over-broad `NEEDS_ORDER_ID` Router Signal Hijacking Policy Inquiries
- **Bug 6:** ORDER_LOOKUP Omitted Supplemental KB Retrieval for Policy Queries
- **Bug 7:** Stale Historical Delivery Timestamps Leaked on Terminal Returned Orders
- **Bug 8:** "internal" Word False Positive in Forbidden-Field Validator *(Discovered beyond visible case wording)*
- **Bug 9:** Router Context Bleed — Order Session ID Reused for Unrelated Queries *(Discovered beyond visible case wording)*
- **Bug 10:** LLM Hallucinating Citation Headings for Order Tool Responses *(Discovered beyond visible case wording)*

---

### Bug Deep-Dive 1: Superseded Policy Document Treated as Active Conflict
- **Reproduction**: When evaluating standard return window policy queries, the retrieval layer pulled both `01-returns-policy-current.md` (30 days, `status: active`) and `02-returns-policy-superseded-2024.md` (60 days, `status: superseded`).
- **Root Cause**: The conflict detector treated any pair of conflicting values as an active dispute without respecting document lifecycle status (`superseded` vs `active`).
- **Fix**: Implemented `_is_superseded_pair()` and `filter_authoritative()` in `app/policy/scoring.py` and `app/policy/conflict.py`. If one document in a pair contains `status: superseded` or is explicitly referenced by `supersedes: RET-2024-01`, it is strictly filtered out and receives a heavy rank penalty (-0.60).
- **Regression Test**: [`tests/regression/test_superseded_conflict.py`](tests/regression/test_superseded_conflict.py)

---

### Bug Deep-Dive 2: Router Context Bleed on Generic Continuation Keywords
- **Reproduction**: After looking up an order (e.g. `ORD-1003`), asking an unrelated knowledge inquiry such as *"How long does standard shipping take within the US?"* routed to `ORDER_LOOKUP` and attempted to reuse `ORD-1003`.
- **Root Cause**: `ORDER_CONTINUATION_SIGNALS` in `app/agent/router.py` matched the bare keyword `"shipping"` anywhere in the string, erroneously classifying a general policy inquiry as an order follow-up.
- **Fix**: Replaced generic single-word patterns with explicit order-referencing phrases (`"it"`, `"that"`, `"my package"`, `"when will it arrive"`). Added `_POLICY_INQUIRY_PATTERNS` and guarded Signal 4 with `not is_policy_inquiry`.
- **Regression Test**: [`tests/regression/test_router_no_context_bleed.py`](tests/regression/test_router_no_context_bleed.py)

---

### Bug Deep-Dive 3: Hallucinated Citations for Order Tool Records
- **Reproduction**: For order tracking lookups (`"Where is ORD-1007?"`), the LLM hallucinated fictitious document citation anchors like `[05-domestic-shipping.md#Order Lookups]` or `[01-returns-policy-current.md#ORD-1007]`.
- **Root Cause**: The orchestrator asked the LLM to generate citation tags for all answers, including responses sourced purely from tool execution results.
- **Fix**: 
  1. Orchestrator deterministically appends `f"Order record: {order_id}"` to `AgentResponse.citations` upon successful tool execution without relying on LLM generation.
  2. Dynamically injected available document citation anchors into the prompt system instructions (`CITATION CONSTRAINT`).
  3. Response validator in `app/safety/trust.py` strips any hallucinated citation tokens without leaving bracket artifacts.
- **Regression Test**: [`tests/regression/test_citation_artifact_cleanup.py`](tests/regression/test_citation_artifact_cleanup.py)

---

### Bug Deep-Dive 4: "internal" Word False Positive in Safety Validator
- **Reproduction**: A safe response explaining *"The migration note is an internal document and not authoritative policy"* was flagged as an internal data leak, triggering redacting and unnecessary handoff.
- **Root Cause**: `_FORBIDDEN_PATTERNS` in `app/safety/trust.py` searched for the bare word `\binternal\b`, triggering on natural English descriptive adjectives.
- **Fix**: Narrowed the regex to data disclosure contexts (`r'\binternal\s+(note|notes|field|fields|data|record|score|tag|flag)\b'`) and added an explicit whitelist for safe descriptive phrases (`"internal document"`, `"internal migration"`).
- **Regression Test**: [`tests/regression/test_internal_word_false_positive.py`](tests/regression/test_internal_word_false_positive.py)

---

## Known Limitations & Production Readiness Roadmap

1. **Session Store Persistence**:
   - *Current*: Ephemeral in-memory store (`Dict[str, Session]`), lost on server restart.
   - *Production Roadmap*: Migrate to Redis with TTL expiration or PostgreSQL JSONB backed with connection pooling.
2. **Customer Authentication & Authorization**:
   - *Current*: Possession of an Order ID (`ORD-XXXX`) permits lookup.
   - *Production Roadmap*: Integrate OAuth2 / OpenID Connect to verify customer JWT session tokens and ensure orders belong to the authenticated user ID.
3. **Asynchronous Vector Database**:
   - *Current*: In-memory NumPy cosine similarity index computed on startup.
   - *Production Roadmap*: Replace with Qdrant, Milvus, or pgvector for sub-millisecond retrieval across tens of thousands of dynamic product and policy documents.
4. **Action Execution & Webhooks**:
   - *Current*: The agent strictly abstains from executing transactional modifications (cancellations, refunds, address edits).
   - *Production Roadmap*: Implement authenticated two-factor confirmation workflows with Stripe / Shopify webhooks for idempotent action fulfillment.

---

## AI Coding Tools Used

### Tools Used
- **Google Antigravity**: Primary agentic AI coding assistant used for end-to-end multi-file architecture implementation, test-driven refactoring, running sandboxed terminal verification commands, and automating regression test suite generation.
- **Claude (Anthropic)**: Used during early system design for prompt engineering strategy, threat modeling against prompt injection vectors, and initial evaluation benchmark structuring.

### Example of an Incomplete / Erroneous AI-Generated Suggestion
- **Context**: Designing multi-turn context continuation for `app/agent/router.py`.
- **The AI Suggestion**: The AI tool proposed adding a broad list of single-word regex keywords to `ORDER_CONTINUATION_SIGNALS`:
  ```python
  # Erroneous AI Suggestion:
  ORDER_CONTINUATION_SIGNALS = [
      r"\b(it|that|the order|my order|my package)\b",
      r"\b(arrive|arrival|delivery|deliver|tracking|track|shipped|shipping|carrier|status|update)\b",
  ]
  ```
- **Why It Was Erroneous**: Adding bare keywords like `"shipping"`, `"delivery"`, and `"status"` caused a critical context bleed regression. When a customer completed an order lookup and next asked a general knowledge question like *"How long does standard shipping take within the US?"*, the router matched the word `"shipping"`, mistakenly treated it as a continuation of the previous order, and routed to `ORDER_LOOKUP`. Because order lookup bypasses knowledge base retrieval, the LLM received no shipping policy documents and failed with *"I do not have enough information in the documentation..."*.
- **The Resolution**: We rejected bare single-word matching. We rewrote `ORDER_CONTINUATION_SIGNALS` to require explicit pronoun or order references (`"it"`, `"that"`, `"my package"`, `"when will it arrive"`), defined `_POLICY_INQUIRY_PATTERNS`, and added a strict `and not is_policy_inquiry` guard to prevent policy inquiries from ever being hijacked by prior session context.
