# Medical Bundle RAG Tool - PRD

**Project Name:** Bundle Facts Retrieval Tool (working title)
**User:** Expert witness (medical negligence cases)
**Date Created:** 2025-11-30
**Status:** Prototype Phase

---

## 1. Problem Statement

Expert witness spends significant time manually searching through large medical negligence bundles (10-36MB PDFs) to find specific facts, dates, treatments, test results, and clinical notes while writing expert reports.

**Current Workflow:**
1. Opens bundle PDF + law firm instructions + report template (3 windows)
2. Manually searches/scrolls through bundle for relevant facts
3. Copy-pastes relevant medical notes into report
4. Writes analysis/opinion based on extracted facts

**Pain Point:** Steps 2-3 are time-consuming. A typical bundle might require 20-30 searches across the document.

---

## 2. Solution Overview

A local AI tool that allows the expert witness to:
- Ask free-form questions about the bundle ("What was the initial treatment for X?")
- Get back: A summary + the original text chunks + source references
- Copy results directly into his report

**Key Principle:** Make this faster than manual searching, while preserving source traceability.

---

## 3. Requirements

### 3.1 Functional Requirements

**FR1: Bundle Processing**
- Accept medical negligence bundles (10-36MB PDFs, potentially larger)
- Extract structured content with Docling
- Preserve metadata (page numbers, section headers, chronological order)
- Create searchable vectorstore (ChromaDB)

**FR2: Question & Retrieval**
- Accept free-form natural language questions
- Retrieve top 5 relevant chunks from bundle
- Summarize retrieved content into 2-4 sentence summary
- Return both summary + original chunks + source references
- Fast responses (target: <10 seconds per query)

**FR3: Source Traceability**
- Return page numbers for all retrieved content
- Allow expert witness to cite sources in report
- Maintain chain of evidence (what was retrieved, where from)

**FR4: User Interface**
- Simple, no frills
- Question input box
- Display summary + chunks + references
- Copy-to-clipboard functionality
- No complex workflows

### 3.2 Non-Functional Requirements

**NFR1: Performance**
- Query response time: <10 seconds on target hardware
- Bundle processing: <5 minutes for 36MB PDF
- Memory efficient (16GB target RAM)

**NFR2: Reliability**
- Handles bundles up to 50MB without crashing
- Graceful error handling for corrupt PDFs
- Local operation (no external API dependencies)

**NFR3: Usability**
- Minimal setup required
- Works with Windows + optional Linux dual-boot
- No training required for expert witness

**NFR4: Security**
- All data stays local (no cloud, no external APIs)
- No data persistence beyond current case
- Case files remain confidential

---

## 4. Technical Architecture

### 4.1 Core Components

```
USER INTERFACE (Web):
├── [1. Upload Bundle (PDF)]
├── [2. Paste/Upload Instructions (System Prompt)]
└── [3. Chat Interface]

PROCESSING PIPELINE (Backend - Python):
├── Docling → Extract & Chunk (preserve metadata)
├── Embedding → Sentence-transformers (all-MiniLM-L6-v2)
└── ChromaDB → Vector Store (persistent per case)

QUERY PIPELINE (Per Question):
├── Retrieve → Top 5 chunks + metadata
├── Context Assembly → Instructions + Chunks
├── LLM Summarization → Local LLM (Ollama/LM Studio)
└── Output → Summary + Chunks + References

LOCAL STORAGE:
/cases/
├── case_001_smith_vs_clinic/
│   ├── original_bundle.pdf
│   ├── system_prompt.txt
│   ├── chroma_db/ (vectorstore)
│   └── chat_history.json (optional)
└── case_002_jones_vs_hospital/
    └── ...
```

### 4.2 Technology Stack

- **PDF Processing:** Docling
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB
- **LLM Runtime:** Ollama + LM Studio
- **LLM Model:** Mistral 7B or Qwen 7B (retrieval + summarization)
- **Backend:** Python 3.11+ (FastAPI or Flask)
- **Frontend:** Web UI (HTML/CSS/JavaScript - no complex framework needed)
- **OS:** Windows primary, Linux optional

### 4.3 System Prompt / Instructions Integration

**How Instructions Are Used:**

Law firm instructions (e.g., "Focus on duty of care under Bolam test") are pasted into the tool and become part of the system context for the LLM.

When answering questions, the LLM is prompted like:

```
You are an expert assistant helping review medical negligence bundles.

CASE INSTRUCTIONS FROM LAW FIRM:
[Expert's instructions pasted here - e.g., "Focus on whether treatment
met standard of care under Bolam test. Identify gaps in documentation."]

BUNDLE FACTS (Retrieved from PDF):
[Top 5 relevant chunks from the case bundle]

USER QUESTION:
"Was the treatment protocol appropriate for this diagnosis?"

Provide a summary referencing the retrieved facts and the case instructions.
```

This ensures:
- ✅ Answers are aligned with law firm's specific requirements
- ✅ Expert witness doesn't have to manually reference instructions
- ✅ Consistency across all answers in the case
- ✅ Instructions can be case-specific

**Instructions Stay Local:**
- Pasted into case folder as `system_prompt.txt`
- Never sent to external APIs
- Deleted when case folder is deleted

---

### 4.4 Hardware Requirements

**Target Machine Specs:**
- **CPU:** Modern multi-core (i5/i7 or equivalent)
- **RAM:** 16GB (sufficient for Mistral 7B + ChromaDB)
- **Storage:** 256GB+ SSD (M.2 preferred)
- **GPU:** Optional (not required for retrieval tasks)
- **Budget:** $1000-1500

---

## 5. User Workflow (Happy Path)

1. **Case Setup (one-time per case):**
   - Expert witness receives bundle + instructions from law firm
   - Opens tool web interface
   - Clicks "Upload Bundle" → selects medical negligence PDF
   - Pastes/uploads instructions (law firm's specific requirements)
   - Tool processes: Extracts, chunks, embeds, creates vectorstore (~3-5 min)
   - Shows: "Ready to analyze"

2. **Case Analysis (Chat Interface):**
   - Expert witness opens report template + tool side-by-side
   - In chat, types question: "What was the diagnosis on admission?"
   - Tool returns:
     - **Summary:** 2-4 sentence synthesis
     - **Relevant chunks:** Original text from bundle
     - **Page references:** Where information came from
   - Expert copies/edits summary into report
   - Asks next question in chat (maintains context)
   - Repeats for ~20-30 questions throughout case analysis

3. **Case Closure:**
   - Report complete
   - Expert witness can delete case folder (bundle + vectorstore) or archive
   - No data persistence beyond case completion

---

## 6. Output Format (To Be Refined)

**Chat Interface Response:**

```
USER: "What was the initial treatment protocol?"

ASSISTANT:

📋 SUMMARY
Patient presented with acute symptoms on 15th March. Initial treatment
included pain management and antibiotics. Blood cultures were ordered
for bacterial confirmation.

📄 RELEVANT SECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Page 12 - Clinical Notes, Admission Entry]
"Patient admitted 15/03/2023 at 14:30. Presenting complaint: chest pain.
Initial observations: BP 145/90, HR 102. Pain rated 8/10."

[Page 14 - Treatment Orders]
"Prescribed: Paracetamol 500mg QID, Ibuprofen 400mg TDS,
Penicillin G 1.2M units IV QID pending culture results."

[Page 18 - Laboratory]
"Blood cultures x2 ordered 15/03/2023. Troponin pending."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 SOURCES
• Page 12 (Admission notes)
• Page 14 (Treatment orders)
• Page 18 (Lab orders)

[COPY SUMMARY]  [COPY ALL]
```

**Key Features:**
- Easily copyable summary at top
- Original chunks preserved below (what he'd normally copy-paste)
- Page references for citation
- Copy buttons for convenience

*(Exact formatting TBD through iteration with expert witness)*

---

## 7. MVP Scope

**Phase 1 (Prototype on M4 Mac - Backend + CLI):**
- ✅ Docling extraction working
- ✅ ChromaDB vectorstore creation & storage
- ✅ Basic retrieval (top 5 chunks + metadata)
- ✅ LLM summarization (with system prompt context)
- ✅ Command-line interface for testing with real bundle

**Phase 2 (Web UI - User Facing):**
- ✅ Web interface (Flask or FastAPI)
  - Upload bundle endpoint
  - Upload/paste system prompt endpoint
  - Chat interface
- ✅ Output formatting (summary + chunks + references)
- ✅ Copy-to-clipboard functionality
- ✅ Case folder management

**Phase 3 (Refinement & Testing):**
- ✅ User testing with real bundle (iterate output format)
- ✅ Error handling (corrupt PDFs, large files, etc.)
- ✅ Performance optimization
- ✅ Local storage structure finalization

**Phase 4 (Deployment Ready):**
- ✅ Windows compatibility testing
- ✅ Linux dual-boot compatibility (optional)
- ✅ Standalone executable packaging
- ✅ Setup & user documentation for target machine

---

## 8. Out of Scope (For Now)

- ❌ Historical case analysis/research
- ❌ Multi-bundle case comparison
- ❌ Automated expert opinion generation
- ❌ Report template integration
- ❌ Collaborative features
- ❌ Cloud storage or sync
- ❌ Complex citation formats

---

## 9. Success Criteria

1. **Expert witness can ask questions faster than manual PDF search**
   - Retrieval + summary in <10 seconds
   - Faster than scrolling through 36MB PDF

2. **Output is trustworthy and citable**
   - Source references accurate
   - Can be copied directly into report
   - No hallucinations or false information

3. **Setup is simple enough for non-technical user**
   - "Load bundle" → "Ask questions" workflow
   - Minimal technical knowledge required

4. **Works reliably on target hardware**
   - No crashes on 36-50MB bundles
   - Handles varied PDF structures (medical, legal, etc.)

---

## 10. Next Steps

1. **Week 1:** Build prototype on M4 Mac
   - Docling extraction + chunking working
   - ChromaDB vectorstore creation
   - LLM summarization functional
   - Command-line interface

2. **Week 2:** User testing
   - Test with real bundle expert witness has
   - Gather feedback on output format
   - Identify pain points

3. **Week 3:** Refine & iterate
   - Improve summarization quality
   - Build simple web UI
   - Finalize output format

4. **Week 4+:** Polish & document
   - Performance optimization
   - Deployment guide for Windows machine
   - Documentation for setup

---

## 11. Open Questions (To Resolve Through Iteration)

**UI/UX:**
- [ ] Exact format for source references (page # only vs. section + subsection)?
- [ ] Should tool maintain chat history per case for reference?
- [ ] How to display very long chunks (truncate + "show more")?
- [ ] Should summaries be editable directly in chat, or only after copying?

**Functionality:**
- [ ] Multi-question context: Does Q2 use context from Q1, or independent retrieval?
- [ ] Should system prompt be updatable mid-case?
- [ ] How many chunks to retrieve per question (currently 5 - is that right)?

**Workflow:**
- [ ] Should case folder be auto-created on bundle upload, or manual entry?
- [ ] Should tool suggest related follow-up questions based on summary?
- [ ] Delete vs. Archive strategy for closed cases?

---

**Owner:** Project maintainer
**Last Updated:** 2025-11-30
