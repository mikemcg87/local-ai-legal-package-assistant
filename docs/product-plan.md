# Product Plan

## Product Thesis

Medical Bundle RAG is a local medico-legal bundle review workbench for expert witnesses. It should help an expert move through a large case bundle faster while preserving source traceability and professional judgment.

This is not a general PDF chatbot. The product should optimise for:

- reliable page-level evidence;
- fast factual retrieval;
- chronology construction;
- source text review before reuse;
- no cloud dependency for client material;
- minimal cognitive overhead for an expert witness.

## Primary User

The primary user is an expert witness preparing or reviewing a medico-legal report. They already know the clinical domain. They need help finding and organising facts, not legal or medical opinions from the model.

Typical tasks:

- find when a symptom, diagnosis, treatment, scan, test, or referral first appears;
- reconstruct a chronology of consultations, investigations, procedures, prescriptions, and omissions;
- check whether a fact appears in the bundle before relying on it;
- copy relevant source excerpts into a report workflow;
- preserve page references for later verification.

## Product Principles

1. **Local first.** Case PDFs, extracted text, embeddings, prompts, and outputs stay on the user machine.
2. **Evidence before prose.** Retrieved source text and page references are more important than fluent summaries.
3. **Model-pluggable.** The app should work with Ollama-compatible local models without hardcoding one model family.
4. **Eval-driven improvement.** Retrieval and answer quality should be measured against private case-specific question sets.
5. **Human-in-the-loop.** The system should support review, correction, and export rather than pretending to produce final expert opinion.
6. **Guided local setup.** Users should be helped toward models and extraction settings that fit their own machine.

## Current State

The prototype already supports:

- PDF upload and local case storage;
- Docling OCR or Granite-Docling VLM extraction;
- chunking and ChromaDB storage;
- BM25 plus semantic hybrid retrieval;
- optional cross-encoder reranking;
- Ollama-based local summarisation;
- a basic timeline view.

## Main Product Gaps

### 1. Citation Reliability

Page attribution is currently approximate because chunks are derived from exported markdown and then mapped back to page positions. Medico-legal use needs page references that are trustworthy enough for report drafting.

Target:

- each chunk should carry page provenance directly from the extraction item where possible;
- multi-page chunks should represent page spans;
- source links should open the correct page;
- evals should check expected-page hit rates.

### 2. Retrieval Quality

Current retrieval is useful but broad. The product needs task-specific retrieval modes:

- exact date search;
- date range search;
- provider search;
- treatment / medication search;
- investigation / imaging / lab search;
- chronology search;
- omissions or "not documented" checks.

### 3. Timeline Trust

The current timeline is regex and keyword based. It should become a reviewable artefact:

- event confidence;
- duplicate grouping controls;
- source page links;
- review status;
- manual correction or exclusion;
- export to CSV/Markdown.

### 4. Expert Workflow

The UI should feel like a bundle review desk:

- source-first result cards;
- split pane with PDF/source text;
- pinned findings;
- saved questions;
- timeline and search side by side;
- export selected findings into report notes.

### 5. Data Boundary

Real case bundles exist locally and must remain private. Public work should use synthetic examples only.

### 6. Onboarding

The user should not need to understand local model sizing before getting value. Setup should explain:

- whether Ollama is reachable;
- whether a chat model and embedding model are installed;
- approximate machine fit based on local memory;
- which installed model is a sensible first choice;
- whether OCR or VLM extraction is the better default for the machine.

## Model Strategy

Do not hardcode the product around a single model. The core pipeline should expose:

- embedding model;
- reranker model;
- chat/summarisation model;
- extraction mode.

Initial local candidates can include Ollama-compatible models such as `gpt-oss`, Gemma-family models, Qwen-family models, and Mistral-family models. Model selection should be based on private eval results, not benchmarks alone.

## First Hardening Milestones

### Milestone 1: Private Eval Harness

Add a private eval format for real cases:

- question;
- expected source pages;
- required terms or facts;
- optional max page;
- retrieval settings.

The eval harness should run locally without committing case data.

### Milestone 2: Provenance Refactor

Refactor ingestion so chunks retain direct page/page-span provenance from extraction items. Add tests with synthetic chunks.

### Milestone 3: Guided Onboarding

Add a setup page that detects local model readiness, estimates model fit, and gives practical extraction guidance without hardcoding one required model.

### Milestone 4: Retrieval Workbench

Expose retrieval diagnostics in the UI:

- semantic score;
- BM25 score;
- rerank score;
- page;
- matched dates;
- source snippet.

### Milestone 5: Reviewable Timeline

Turn timeline output into a reviewable, editable, exportable chronology.

### Milestone 6: Public-Safe Demo

Create a synthetic medico-legal bundle or small generated text fixture so the public repo can demonstrate the workflow without real client material.
