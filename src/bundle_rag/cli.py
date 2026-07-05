"""CLI interface for bundle-rag."""

import shutil
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bundle_rag.config import CASES_DIR, LLM_MODEL
from bundle_rag.ingest import ingest_bundle
from bundle_rag.vectorstore import build_vectorstore, HybridVectorStore
from bundle_rag.retrieval import query_bundle, list_models
from bundle_rag.evaluation import run_suite, write_report

console = Console()


@click.group()
def main():
    """Bundle RAG - Medical bundle retrieval tool."""
    pass


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--case", "-c", required=True, help="Case name/identifier")
@click.option("--password", "-p", default=None, help="PDF password if encrypted")
@click.option("--vlm", is_flag=True, help="Use Granite-Docling VLM (slower but better for complex layouts/tables)")
def ingest(pdf_path: str, case: str, password: str | None, vlm: bool):
    """Ingest a PDF bundle into a case."""
    case_dir = CASES_DIR / case
    
    if case_dir.exists():
        if not click.confirm(f"Case '{case}' exists. Overwrite?"):
            return
        shutil.rmtree(case_dir)
    
    mode = "VLM (Granite-Docling)" if vlm else "OCR (forced full-page)"
    console.print(f"\n[bold]Ingesting bundle into case: {case}[/bold]")
    console.print(f"[dim]Mode: {mode}[/dim]\n")
    
    # Step 1: Extract and chunk
    result = ingest_bundle(Path(pdf_path), case_dir, password=password, use_vlm=vlm)
    
    # Step 2: Build vectorstore
    console.print()
    build_vectorstore(case_dir)
    
    console.print(f"\n[bold green]Done![/bold green] Case '{case}' ready for queries.")
    console.print(f"  Chunks: {result['metadata']['chunks']}")
    console.print(f"  Location: {case_dir}")


@main.command()
@click.option("--case", "-c", required=True, help="Case name")
@click.option("--model", "-m", default=LLM_MODEL, help="Ollama model to use")
@click.option("--top-k", "-k", default=5, help="Number of chunks to retrieve")
def query(case: str, model: str, top_k: int):
    """Interactive query session for a case."""
    case_dir = CASES_DIR / case
    
    if not case_dir.exists():
        console.print(f"[red]Case '{case}' not found. Run ingest first.[/red]")
        return
    
    console.print(f"\n[bold]Querying case: {case}[/bold]")
    console.print(f"Model: {model} | Top-K: {top_k}")
    console.print("Type 'quit' to exit, 'chunks' to show last chunks\n")
    
    last_result = None
    
    while True:
        try:
            question = console.input("[bold cyan]Question:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "chunks" and last_result:
            show_chunks(last_result["chunks"])
            continue
        
        try:
            result = query_bundle(case_dir, question, model=model, top_k=top_k)
            last_result = result
            
            # Show summary
            console.print()
            console.print(Panel(result["summary"], title="Summary", border_style="green"))
            
            # Show sources
            pages = [c["page"] for c in result["chunks"] if c.get("page") and c["page"] > 0]
            if pages:
                console.print(f"[dim]Sources: Pages {', '.join(map(str, sorted(set(pages))))}[/dim]")
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


def show_chunks(chunks: list[dict]):
    """Display retrieved chunks."""
    console.print()
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page")
        page_str = f"Page {page}" if page and page > 0 else "Unknown"
        score = chunk.get("score", 0)
        
        console.print(Panel(
            chunk["text"],
            title=f"[{i}] {page_str} (score: {score:.3f})",
            border_style="blue",
        ))
    console.print()


@main.command()
def cases():
    """List all cases."""
    if not CASES_DIR.exists():
        console.print("[yellow]No cases found.[/yellow]")
        return
    
    table = Table(title="Cases")
    table.add_column("Name")
    table.add_column("Chunks")
    table.add_column("Source")
    
    for case_dir in sorted(CASES_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        
        store = HybridVectorStore(case_dir)
        count = store.count() if (case_dir / "chroma_db").exists() else 0
        
        import json
        meta_file = case_dir / "chunks.json"
        source = "-"
        if meta_file.exists():
            with open(meta_file) as f:
                data = json.load(f)
                source = data.get("metadata", {}).get("source", "-")
        
        table.add_row(case_dir.name, str(count), source)
    
    console.print(table)


@main.command()
@click.option("--case", "-c", required=True, help="Case name to delete")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete(case: str, yes: bool):
    """Delete a case and all its data."""
    case_dir = CASES_DIR / case
    
    if not case_dir.exists():
        console.print(f"[red]Case '{case}' not found.[/red]")
        return
    
    if not yes and not click.confirm(f"Delete case '{case}' and all data?"):
        return
    
    shutil.rmtree(case_dir)
    console.print(f"[green]Deleted case '{case}'[/green]")


@main.command()
def models():
    """List available Ollama models."""
    available = list_models()
    
    if not available:
        console.print("[yellow]No models found. Is Ollama running?[/yellow]")
        return
    
    console.print("[bold]Available Ollama models:[/bold]")
    for m in available:
        console.print(f"  - {m}")


@main.command("eval")
@click.argument("suite_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Write JSON report to this path")
def eval_suite(suite_path: str, output: str | None):
    """Run a private retrieval eval suite against a local case."""
    report = run_suite(Path(suite_path))
    metrics = report["metrics"]

    console.print(f"\n[bold]Eval suite:[/bold] {suite_path}")
    console.print(f"Case: {report['case']}")
    console.print(f"Questions: {report['question_count']}")
    console.print()
    console.print(f"Page hit rate:          {metrics['page_hit_rate']:.2%}")
    console.print(f"Mean page MRR:          {metrics['mean_page_mrr']:.3f}")
    console.print(f"Required term coverage: {metrics['mean_required_term_coverage']:.2%}")

    table = Table(title="Question Results")
    table.add_column("ID")
    table.add_column("Page hit")
    table.add_column("MRR")
    table.add_column("Term coverage")
    table.add_column("Retrieved pages")

    for item in report["questions"]:
        table.add_row(
            item["id"],
            "yes" if item["page_hit"] else "no",
            f"{item['page_mrr']:.3f}",
            f"{item['required_term_coverage']:.0%}" if item["total_required_terms"] else "-",
            ", ".join(str(p) for p in item["retrieved_pages"][:8]) or "-",
        )
    console.print(table)

    if output:
        path = write_report(report, Path(output))
        console.print(f"\n[green]Wrote report:[/green] {path}")


if __name__ == "__main__":
    main()
