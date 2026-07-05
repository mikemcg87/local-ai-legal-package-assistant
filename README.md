# Medical Bundle RAG

A local RAG (Retrieval-Augmented Generation) tool for searching and querying large medical/legal PDF bundles. Built for expert witnesses who need to efficiently find specific clinical notes, dates, treatments, and documentation within case bundles.

## Features

- **100% Local** - All processing happens on your machine. No data leaves your computer.
- **Clinical Timeline** - Auto-generates chronological timeline of all dated events (consultations, tests, procedures, admissions)
- **Hybrid Search** - Combines semantic (meaning-based) and keyword (BM25) search for accurate retrieval
- **Cross-Encoder Re-ranking** - Uses ms-marco-MiniLM for precision refinement after initial retrieval
- **VLM Extraction** - Optional Granite-Docling Vision Language Model for complex document layouts
- **OCR Support** - Automatically OCRs scanned pages using macOS Vision framework (forced full-page OCR)
- **Source Citations** - Every answer includes clickable page references that open the PDF at that location
- **Password-Protected PDFs** - Handles encrypted PDF bundles
- **Web Interface** - Simple drag-and-drop UI for uploading and querying

## How It Works

1. **Ingest**: Upload a PDF bundle. The system extracts text from each page (using OCR for scanned pages), splits into chunks, and creates embeddings.

2. **Index**: Chunks are stored in ChromaDB with vector embeddings (via Ollama's nomic-embed-text) and a BM25 keyword index.

3. **Query**: When you ask a question, the system:
   - Searches both semantic (meaning) and keyword (exact match) indexes
   - Combines results using reciprocal rank fusion
   - Sends top chunks to an LLM (Ollama) to generate a summarized answer
   - Returns the answer with source page references

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) with models:
  - `mistral:7b-instruct` (or another chat model like `deepseek-r1:32b`)
  - `mxbai-embed-large` (for embeddings - 1024 dimensions)
  - `ms-marco-minilm-l6-v2` (for cross-encoder re-ranking)
- macOS (for OCR via Vision framework)
- Optional: [Docling](https://github.com/DS4SD/docling) for VLM extraction

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd medical-bundle-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Pull required Ollama models
ollama pull mistral:7b-instruct
ollama pull mxbai-embed-large
ollama pull ms-marco-minilm-l6-v2
```

## Quick Start

### Web Interface (Recommended)

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the web app
python run.py
```

Open http://localhost:5000 in your browser.

Use **Local Setup** from the home page first if you need to check whether Ollama,
the installed models, and the extraction settings are a sensible fit for the
machine.

At minimum, local review needs one retrieval model and one chat model. The setup
screen checks installed Ollama models, shows LM Studio as an OpenAI-compatible
local runtime option, and gives copyable commands such as:

```bash
ollama pull mxbai-embed-large
ollama run gemma4:e4b
ollama run gpt-oss:20b
ollama run qwen3:30b
```

1. Click "Upload New Bundle"
2. Drag and drop your PDF (enter password if encrypted)
3. Choose extraction mode:
   - **Docling OCR** (faster, good for typed text)
   - **Granite-Docling VLM** (slower, better for complex layouts)
4. Wait for processing to complete (~3 hours for 500 pages with VLM)
5. View the auto-generated **Timeline** with all dated clinical events
6. Or use **Chat** to ask specific questions!

### Command Line

```bash
# Ingest a PDF
bundle-rag ingest "/path/to/bundle.pdf" --case my_case --password "optional"

# Query interactively
bundle-rag query --case my_case

# List cases
bundle-rag cases

# Run the public retrieval eval suite (synthetic fixture)
bundle-rag eval evals/public/suite.json --output reports/public-baseline.json

# Private suites live in evals/private/ (gitignored, never committed)
bundle-rag eval evals/private/suite.json --output reports/private/report.json
```

## Run the demo with the synthetic bundle

The repo ships a fully synthetic 10-page medico-legal bundle (fictional
patient, clinicians, and institutions — see
`scripts/generate_synthetic_bundle.py`) so the whole pipeline can be
exercised without any real case material:

```bash
# 1. Pull the embedding model
ollama pull mxbai-embed-large

# 2. Ingest the synthetic bundle
bundle-rag ingest evals/public/fixtures/synthetic-bundle.pdf --case synthetic-demo

# 3. Run the retrieval eval baseline
bundle-rag eval evals/public/suite.json --output reports/public-baseline.json
```

The suite includes absent-fact questions (facts deliberately missing from
the bundle) to check that retrieval does not manufacture support for them.
Current baseline on the synthetic case: page-hit rate 1.0, mean page MRR
0.43, required-term coverage 1.0. The modest MRR is a known limitation of
chunk-start page attribution and is the current improvement target — see
`docs/eval-plan.md` for the eval-first working rule.

## Configuration

Edit `src/bundle_rag/config.py` to adjust:

```python
LLM_MODEL = "mistral:7b-instruct"  # Ollama model for responses
CHUNK_SIZE = 1500            # Characters per chunk
CHUNK_OVERLAP = 300          # Overlap between chunks
TOP_K = 10                   # Number of chunks to retrieve
```

## Web UI Options

- **Instructions**: Custom system prompt to guide the AI's responses (e.g., "Focus on dates and clinical findings")
- **Model**: Select which Ollama model to use
- **Max Page**: Exclude pages beyond a certain number (useful for filtering boilerplate sections)

## Project Structure

```
medical-bundle-rag/
├── src/bundle_rag/
│   ├── app.py          # Flask web application
│   ├── cli.py          # Command-line interface
│   ├── config.py       # Configuration settings
│   ├── ingest.py       # PDF extraction + OCR/VLM
│   ├── vectorstore.py  # ChromaDB + hybrid search
│   ├── retrieval.py    # Query + LLM integration
│   ├── timeline.py     # Timeline extraction from chunks
│   └── templates/
│       ├── index.html      # Landing page
│       ├── chat.html       # Q&A interface
│       └── timeline.html   # Timeline view
├── cases/              # Stored case data (gitignored)
│   └── {case_name}/
│       ├── chunks.json     # Extracted text chunks
│       ├── timeline.json   # Generated timeline
│       └── chroma/         # Vector database
├── uploads/            # Temporary upload storage (gitignored)
├── run.py              # Simple launcher script
└── pyproject.toml      # Project dependencies
```

## Clinical Timeline Feature

The timeline view automatically extracts and organizes all dated clinical events from your medical bundle:

- **Auto-extraction**: Parses dates in multiple formats (DD/MM/YYYY, "23rd December 2016", etc.)
- **Event classification**: Categorizes events as consultations, tests, procedures, admissions, discharges, or prescriptions
- **Provider extraction**: Identifies doctor names, hospitals, and clinics
- **Smart filtering**: Filter by date range, event type, or provider name
- **Page references**: Click any event to jump to the source page in the PDF
- **Date validation**: Filters impossible dates and boilerplate text

**To access**: Click the "📅 Timeline" button in any case view.

## Tips for Best Results

1. **Be specific** - "What did Dr Smith document on 23rd December 2016?" works better than "What happened?"

2. **Use the max page filter** - Medical bundles often have boilerplate after the clinical notes. Set max page to exclude these.

3. **Check the source chunks** - Click "Show chunks" to see exactly what text was retrieved. This helps refine your questions.

4. **Try different phrasings** - If you don't get the expected results, rephrase using terms likely to appear in the documents.

## Troubleshooting

**"Ollama not running"** - Start Ollama: `ollama serve`

**Slow first query** - The embedding model loads on first use. Subsequent queries are faster.

**Missing pages** - Some pages may be scanned images. The OCR should handle these, but quality varies.

**Database locked error** - Stop any running instances and restart the app.

## License

MIT
