"""ChromaDB vectorstore with hybrid BM25 + semantic search."""

import json
import re
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
import ollama
from rank_bm25 import BM25Okapi
from rich.console import Console

from bundle_rag.config import TOP_K

console = Console()

# Use Ollama's mxbai-embed-large model (1024d, higher quality)
EMBEDDING_MODEL = "mxbai-embed-large"

# Boilerplate phrases that indicate non-clinical content
BOILERPLATE_PHRASES = [
    "statement of truth", "expert's declaration", "overriding duty to the court",
    "professional indemnity insurance", "criminal sanctions", "perjury",
    "concurrent evidence of experts", "joint statement declaration",
    "meetings between experts", "instructing lawyers", "adversely affect the validity",
    "court-directed experts", "annex a", "annex b", "annex c", "pursuant to",
    "protocol for", "guidelines for the preparation", "expert witness",
    "in accordance with the rules", "the court directs", "upon hearing counsel",
    "time limits", "interlocutory proceedings", "listed for trial",
    "sequential exchange", "date fixed for trial", "reviewed by the master",
    "extension of time", "order dated", "party alleging default",
    "the action shall be", "within 14 days", "within 28 days",
]

# Date patterns for extraction
DATE_PATTERNS = [
    r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b',  # 23/12/2016, 23-12-2016
    r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{2,4})\b',  # 23rd December 2016
    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})\b',  # December 23, 2016
]


def extract_dates(text: str) -> list[str]:
    """Extract date strings from text."""
    dates = []
    text_lower = text.lower()
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            dates.append('-'.join(str(m) for m in match))
    return dates


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r'\b\w+\b', text.lower())


class HybridVectorStore:
    """Vector store with hybrid BM25 + semantic search."""
    
    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.db_path = self.case_dir / "chroma_db"
        self.bm25_path = self.case_dir / "bm25_index.pkl"
        self._client = None
        self._collection = None
        self._bm25 = None
        self._corpus = None
        self._chunk_ids = None
    
    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="chunks",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
    
    @property
    def bm25(self) -> BM25Okapi:
        """Load BM25 index from disk."""
        if self._bm25 is None and self.bm25_path.exists():
            with open(self.bm25_path, 'rb') as f:
                data = pickle.load(f)
                self._bm25 = data['bm25']
                self._corpus = data['corpus']
                self._chunk_ids = data['chunk_ids']
        return self._bm25
    
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding using Ollama's nomic-embed-text."""
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        return response['embeddings'][0]
    
    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        console.print(f"[yellow]Generating embeddings with {EMBEDDING_MODEL}...[/yellow]")
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend(response['embeddings'])
            console.print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        
        return embeddings
    
    def add_chunks(self, chunks: list[dict]) -> int:
        """Add chunks to both vector store and BM25 index."""
        if not chunks:
            return 0
        
        texts = [c["text"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Extract dates from each chunk for metadata
        metadatas = []
        for c in chunks:
            dates = extract_dates(c["text"])
            metadatas.append({
                "page": c.get("page") or -1,
                "section": c.get("section") or "",
                "dates": ",".join(dates) if dates else "",
            })
        
        # Build vector embeddings
        embeddings = self._get_embeddings_batch(texts)
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        console.print(f"[green]Added {len(chunks)} chunks to vector store[/green]")
        
        # Build BM25 index
        console.print("[yellow]Building BM25 keyword index...[/yellow]")
        tokenized_corpus = [tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._corpus = texts
        self._chunk_ids = ids
        
        # Save BM25 index
        with open(self.bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': self._bm25,
                'corpus': self._corpus,
                'chunk_ids': self._chunk_ids,
            }, f)
        console.print(f"[green]Built BM25 index for {len(chunks)} chunks[/green]")
        
        return len(chunks)
    
    def _is_boilerplate(self, text: str) -> bool:
        """Check if chunk is likely legal/template boilerplate."""
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in BOILERPLATE_PHRASES)
    
    def query(
        self, 
        query_text: str, 
        top_k: int = TOP_K, 
        filter_boilerplate: bool = True, 
        max_page: int | None = None,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[dict]:
        """
        Hybrid query combining semantic search and BM25.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            filter_boilerplate: Whether to filter out legal boilerplate
            max_page: Maximum page number to include
            semantic_weight: Weight for semantic similarity (0-1)
            bm25_weight: Weight for BM25 keyword matching (0-1)
        """
        fetch_count = top_k * 4  # Fetch more to allow for filtering and fusion
        
        # --- Semantic Search ---
        query_embedding = self._get_embedding(query_text)
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"],
        )
        
        # Build semantic scores dict
        semantic_scores = {}
        for i, chunk_id in enumerate(semantic_results["ids"][0]):
            # Convert distance to similarity (cosine distance to similarity)
            similarity = 1 - semantic_results["distances"][0][i]
            semantic_scores[chunk_id] = {
                "score": similarity,
                "text": semantic_results["documents"][0][i],
                "metadata": semantic_results["metadatas"][0][i],
            }
        
        # --- BM25 Search ---
        bm25_scores = {}
        if self.bm25 is not None:
            tokenized_query = tokenize(query_text)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores
            max_score = max(scores) if max(scores) > 0 else 1
            for i, score in enumerate(scores):
                chunk_id = self._chunk_ids[i]
                bm25_scores[chunk_id] = score / max_score
        
        # --- Combine Results (Reciprocal Rank Fusion style) ---
        all_chunk_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        combined_results = []
        for chunk_id in all_chunk_ids:
            sem_score = semantic_scores.get(chunk_id, {}).get("score", 0)
            bm25_score = bm25_scores.get(chunk_id, 0)
            
            # Get chunk data
            if chunk_id in semantic_scores:
                text = semantic_scores[chunk_id]["text"]
                metadata = semantic_scores[chunk_id]["metadata"]
            else:
                # Fetch from collection if only in BM25 results
                idx = self._chunk_ids.index(chunk_id)
                text = self._corpus[idx]
                result = self.collection.get(ids=[chunk_id], include=["metadatas"])
                metadata = result["metadatas"][0] if result["metadatas"] else {}
            
            # Apply filters
            if filter_boilerplate and self._is_boilerplate(text):
                continue
            
            page = metadata.get("page", -1)
            if max_page and page and page > max_page:
                continue
            
            # Combined score
            combined_score = (semantic_weight * sem_score) + (bm25_weight * bm25_score)
            
            combined_results.append({
                "id": chunk_id,
                "text": text,
                "page": page,
                "section": metadata.get("section", ""),
                "dates": metadata.get("dates", ""),
                "semantic_score": sem_score,
                "bm25_score": bm25_score,
                "score": combined_score,
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results[:top_k]
    
    def count(self) -> int:
        return self.collection.count()
    
    def delete(self) -> bool:
        """Delete the vectorstore and BM25 index."""
        import shutil
        deleted = False
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
            deleted = True
        if self.bm25_path.exists():
            self.bm25_path.unlink()
            deleted = True
        return deleted


# Alias for backwards compatibility
VectorStore = HybridVectorStore


def build_vectorstore(case_dir: Path) -> HybridVectorStore:
    """Build vectorstore from ingested chunks."""
    case_dir = Path(case_dir)
    chunks_file = case_dir / "chunks.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError("No chunks found. Run ingest first.")
    
    with open(chunks_file) as f:
        data = json.load(f)
    
    store = HybridVectorStore(case_dir)
    store.add_chunks(data["chunks"])
    return store
