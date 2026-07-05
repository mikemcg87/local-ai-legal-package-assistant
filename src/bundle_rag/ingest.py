"""PDF ingestion via Docling with forced OCR or VLM for scanned pages."""

import json
import tempfile
from pathlib import Path

import pikepdf
from rich.console import Console

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrMacOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel import vlm_model_specs

from bundle_rag.config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()


def decrypt_pdf(pdf_path: Path, password: str) -> Path:
    """Decrypt a password-protected PDF, return path to decrypted temp file."""
    console.print("[yellow]Decrypting PDF...[/yellow]")
    
    with pikepdf.open(pdf_path, password=password) as pdf:
        temp_dir = Path(tempfile.gettempdir())
        decrypted_path = temp_dir / f"decrypted_{pdf_path.name}"
        pdf.save(decrypted_path)
    
    console.print("[green]PDF decrypted successfully[/green]")
    return decrypted_path


def is_encrypted(pdf_path: Path) -> bool:
    """Check if PDF is password-protected."""
    try:
        with pikepdf.open(pdf_path) as pdf:
            return False
    except pikepdf.PasswordError:
        return True


import re

# Patterns that indicate semantic boundaries in medical documents
SECTION_PATTERNS = [
    r'\n#{1,3}\s+',  # Markdown headers
    r'\n[A-Z][A-Z\s]{10,}(?:\n|:)',  # ALL CAPS HEADERS
    r'\n(?:HISTORY|EXAMINATION|DIAGNOSIS|TREATMENT|ASSESSMENT|PLAN|IMPRESSION|FINDINGS|MEDICATIONS|ALLERGIES|VITALS|LABS|IMAGING|CONSULTATION|DISCHARGE|ADMISSION|PROGRESS NOTE|CLINICAL NOTE)[:\s]',
    r'\n\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s*[-:]',  # Date entries like "23/12/2016 -"
    r'\n\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?[\s-]',  # Time entries
]

PARAGRAPH_BREAK = r'\n\s*\n'


def semantic_chunk_text(text: str, max_size: int = CHUNK_SIZE, min_size: int = 200) -> list[str]:
    """
    Split text into semantic chunks based on:
    - Section headers
    - Paragraph breaks
    - Date/time boundaries (common in medical notes)
    
    Falls back to size-based splitting if chunks are too large.
    """
    if len(text) <= max_size:
        return [text] if text.strip() else []
    
    # First, split on major section boundaries
    combined_pattern = '|'.join(f'({p})' for p in SECTION_PATTERNS)
    
    # Split but keep the delimiters
    parts = re.split(f'({combined_pattern})', text, flags=re.IGNORECASE)
    
    # Recombine parts with their headers
    sections = []
    current = ""
    for part in parts:
        if part is None:
            continue
        if not current:
            current = part
        elif len(current) + len(part) <= max_size:
            current += part
        else:
            if current.strip():
                sections.append(current.strip())
            current = part
    if current.strip():
        sections.append(current.strip())
    
    # Further split large sections on paragraph breaks
    chunks = []
    for section in sections:
        if len(section) <= max_size:
            if len(section) >= min_size:
                chunks.append(section)
            elif chunks:
                # Merge small chunks with previous
                if len(chunks[-1]) + len(section) <= max_size:
                    chunks[-1] += "\n\n" + section
                else:
                    chunks.append(section)
            else:
                chunks.append(section)
        else:
            # Split on paragraphs
            paragraphs = re.split(PARAGRAPH_BREAK, section)
            current_chunk = ""
            for para in paragraphs:
                if not para.strip():
                    continue
                if len(current_chunk) + len(para) + 2 <= max_size:
                    current_chunk += ("\n\n" if current_chunk else "") + para
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # If single paragraph is too large, split by sentences
                    if len(para) > max_size:
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        current_chunk = ""
                        for sent in sentences:
                            if len(current_chunk) + len(sent) + 1 <= max_size:
                                current_chunk += (" " if current_chunk else "") + sent
                            else:
                                if current_chunk.strip():
                                    chunks.append(current_chunk.strip())
                                current_chunk = sent
                    else:
                        current_chunk = para
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    
    return chunks


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into semantic chunks (wrapper for backwards compatibility)."""
    return semantic_chunk_text(text, max_size=size)


def extract_pdf(pdf_path: Path, use_vlm: bool = False) -> list[dict]:
    """Extract content from PDF using Docling with forced OCR or VLM.
    
    Args:
        pdf_path: Path to the PDF file
        use_vlm: If True, use Granite-Docling VLM (better for complex layouts).
                 If False, use standard OCR pipeline (faster for typed text).
    """
    if use_vlm:
        console.print("[yellow]Extracting PDF with Granite-Docling VLM (MLX)...[/yellow]")
        console.print("[dim]This provides better structure understanding but is slower.[/dim]")
        
        # Use Granite-Docling VLM with MLX acceleration for Apple Silicon
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
        )
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )
    else:
        console.print("[yellow]Extracting PDF with Docling (forced OCR)...[/yellow]")
        
        # Configure pipeline with forced OCR
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Use macOS native OCR with forced full-page OCR
        ocr_options = OcrMacOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
    
    console.print("[yellow]Converting document (this may take a while for large PDFs)...[/yellow]")
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Build page mapping from document items
    page_char_ranges = {}  # Maps character position to page number
    current_pos = 0
    
    for item, _level in doc.iterate_items():
        page_num = 1
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    page_num = prov.page_no
                    break
        
        text = ""
        if hasattr(item, 'text') and item.text:
            text = item.text
        
        if text:
            page_char_ranges[current_pos] = page_num
            current_pos += len(text) + 1  # +1 for newline
    
    # Export full markdown
    full_text = doc.export_to_markdown()
    console.print(f"[yellow]Extracted {len(full_text)} characters of text[/yellow]")
    
    if len(full_text.strip()) < 50:
        console.print("[red]Warning: Very little text extracted![/red]")
        return []
    
    # Chunk the full text
    all_chunks = []
    chunk_idx = 0
    text_chunks = chunk_text(full_text)
    
    for chunk_content in text_chunks:
        # Estimate page number (rough approximation)
        chunk_start = full_text.find(chunk_content[:100]) if len(chunk_content) >= 100 else full_text.find(chunk_content)
        page_num = 1
        for pos, page in sorted(page_char_ranges.items()):
            if pos <= chunk_start:
                page_num = page
            else:
                break
        
        all_chunks.append({
            "text": chunk_content,
            "page": page_num,
            "section": None,
            "idx": chunk_idx,
        })
        chunk_idx += 1
    
    console.print(f"[green]Created {len(all_chunks)} chunks[/green]")
    return all_chunks


def ingest_bundle(pdf_path: Path, case_dir: Path, password: str | None = None, use_vlm: bool = False) -> dict:
    """Ingest PDF into case directory, return metadata.
    
    Args:
        pdf_path: Path to the PDF bundle
        case_dir: Directory to store case data
        password: PDF password if encrypted
        use_vlm: Use Granite-Docling VLM instead of OCR (slower but better for complex layouts)
    """
    pdf_path = Path(pdf_path)
    case_dir = Path(case_dir)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle encrypted PDFs
    working_pdf = pdf_path
    viewable_pdf = case_dir / "bundle.pdf"
    
    if is_encrypted(pdf_path):
        if not password:
            raise ValueError("PDF is password-protected. Provide --password option.")
        working_pdf = decrypt_pdf(pdf_path, password)
        import shutil
        shutil.copy(working_pdf, viewable_pdf)
        console.print(f"[green]Saved decrypted PDF for viewing[/green]")
    else:
        import shutil
        shutil.copy(pdf_path, viewable_pdf)
    
    chunks = extract_pdf(working_pdf, use_vlm=use_vlm)
    
    metadata = {
        "source": pdf_path.name,
        "source_path": str(viewable_pdf.absolute()),
        "chunks": len(chunks),
        "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
    }
    
    # Save chunks
    with open(case_dir / "chunks.json", "w") as f:
        json.dump({"metadata": metadata, "chunks": chunks}, f)
    
    console.print(f"[green]Extracted {len(chunks)} chunks from {metadata['size_mb']}MB PDF[/green]")
    return {"metadata": metadata, "chunks": chunks}
