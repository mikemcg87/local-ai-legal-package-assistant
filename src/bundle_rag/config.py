"""Configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CASES_DIR = PROJECT_ROOT / "cases"

LLM_MODEL = "mistral:7b-instruct"
CHUNK_SIZE = 1500  # Larger chunks to keep page content together
CHUNK_OVERLAP = 300
TOP_K = 10  # Retrieve more chunks for complete coverage

# Hybrid search weights
SEMANTIC_WEIGHT = 0.5
BM25_WEIGHT = 0.5
