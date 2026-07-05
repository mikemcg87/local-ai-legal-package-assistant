"""Retrieval and LLM summarization via Ollama."""

from pathlib import Path

import ollama
from rich.console import Console
from sentence_transformers import CrossEncoder

from bundle_rag.config import LLM_MODEL, TOP_K
from bundle_rag.vectorstore import HybridVectorStore

console = Console()

# Cross-encoder for re-ranking (loaded lazily)
_cross_encoder = None
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder() -> CrossEncoder:
    """Lazy load cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        console.print(f"[yellow]Loading cross-encoder model...[/yellow]")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Re-rank chunks using cross-encoder for better relevance."""
    if not chunks:
        return chunks
    
    encoder = get_cross_encoder()
    
    # Create query-chunk pairs
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # Get cross-encoder scores
    scores = encoder.predict(pairs)
    
    # Add rerank scores and sort
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
    
    # Sort by rerank score
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked[:top_k]


SYSTEM_PROMPT = """You are an assistant helping an expert witness review medical negligence bundles.
Your role is like a paralegal or secretary - you retrieve and summarize facts from the bundle.
You do NOT provide legal or medical opinions.

When answering:
- Summarize the relevant facts from the retrieved chunks in 2-4 sentences
- Be precise and factual
- Reference page numbers when available
- If the retrieved chunks don't contain relevant information, say so clearly
- Never fabricate information not present in the chunks"""


def format_chunks_for_context(chunks: list[dict]) -> str:
    """Format retrieved chunks for LLM context."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page")
        page_str = f"Page {page}" if page and page > 0 else "Page unknown"
        section = chunk.get("section") or ""
        
        formatted.append(f"[Chunk {i} - {page_str}]\n{chunk['text']}")
    
    return "\n\n".join(formatted)


def query_bundle(
    case_dir: Path,
    question: str,
    system_prompt: str | None = None,
    model: str = LLM_MODEL,
    top_k: int = TOP_K,
    max_page: int | None = None,
    semantic_weight: float = 0.5,
    bm25_weight: float = 0.5,
    use_reranking: bool = True,
) -> dict:
    """
    Query the bundle and get a summarized answer.
    
    Returns:
        {
            "summary": "LLM-generated summary",
            "chunks": [list of retrieved chunks with metadata],
            "model": "model used"
        }
    """
    store = HybridVectorStore(case_dir)
    
    if store.count() == 0:
        raise ValueError("Vectorstore is empty. Run ingest and build first.")
    
    # Retrieve more chunks initially for re-ranking
    fetch_k = top_k * 3 if use_reranking else top_k
    
    # Retrieve relevant chunks using hybrid search
    chunks = store.query(
        question, 
        top_k=fetch_k, 
        max_page=max_page,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
    )
    
    # Re-rank with cross-encoder for better precision
    if use_reranking and len(chunks) > top_k:
        console.print(f"[yellow]Re-ranking {len(chunks)} chunks...[/yellow]")
        chunks = rerank_chunks(question, chunks, top_k)
    
    # Build prompt
    base_prompt = system_prompt or SYSTEM_PROMPT
    context = format_chunks_for_context(chunks)
    
    full_prompt = f"""{base_prompt}

RETRIEVED CONTENT FROM BUNDLE:
{context}

USER QUESTION:
{question}

Provide a concise summary based on the retrieved content above."""

    # Call Ollama
    console.print(f"[yellow]Querying {model}...[/yellow]")
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
    )
    
    summary = response["message"]["content"]
    
    return {
        "summary": summary,
        "chunks": chunks,
        "model": model,
    }


def list_models() -> list[str]:
    """List available Ollama models."""
    try:
        result = ollama.list()
        # Handle both old dict format and new object format
        if hasattr(result, 'models'):
            return [m.model for m in result.models]
        elif isinstance(result, dict) and "models" in result:
            return [m.get("name") or m.get("model") for m in result["models"]]
        return []
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        return []
