"""Flask web application for Bundle RAG."""

import subprocess
import sys
import time
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import ollama

from bundle_rag.config import CASES_DIR, LLM_MODEL
from bundle_rag.ingest import ingest_bundle, is_encrypted
from bundle_rag.vectorstore import build_vectorstore
from bundle_rag.retrieval import query_bundle, list_models
from bundle_rag.onboarding import get_setup_status

app = Flask(__name__, 
    template_folder=Path(__file__).parent / "templates",
    static_folder=Path(__file__).parent / "static"
)
app.secret_key = "bundle-rag-secret-key-change-in-production"

UPLOAD_FOLDER = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

processing_status = {}


def ensure_ollama_running():
    """Check if Ollama is running, try to start if not."""
    try:
        ollama.list()
        return True
    except Exception:
        # Try to start Ollama
        try:
            if sys.platform == "win32":
                subprocess.Popen(["ollama", "serve"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # Mac and Linux - use ollama serve
                subprocess.Popen(["ollama", "serve"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for it to start
            for _ in range(10):
                time.sleep(1)
                try:
                    ollama.list()
                    return True
                except:
                    pass
        except Exception as e:
            print(f"Could not start Ollama: {e}")
        return False


@app.route("/")
def index():
    """Landing page - upload PDF."""
    ollama_ok = ensure_ollama_running()
    models = list_models() if ollama_ok else []
    setup_status = get_setup_status()
    cases = []
    if CASES_DIR.exists():
        cases = [d.name for d in CASES_DIR.iterdir() if d.is_dir()]
    return render_template("index.html", ollama_ok=ollama_ok, models=models, cases=cases, setup_status=setup_status)


@app.route("/setup")
def setup_page():
    """Local setup and model-fit onboarding."""
    return render_template("setup.html", setup_status=get_setup_status())


@app.route("/api/setup-status")
def setup_status_api():
    """Return local setup and model-fit status."""
    return jsonify(get_setup_status())


@app.route("/check-password", methods=["POST"])
def check_password():
    """Check if uploaded PDF needs a password."""
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Save temporarily
    filename = secure_filename(file.filename)
    temp_path = UPLOAD_FOLDER / filename
    file.save(temp_path)
    
    needs_password = is_encrypted(temp_path)
    
    return jsonify({
        "filename": filename,
        "needs_password": needs_password,
        "path": str(temp_path)
    })


@app.route("/process", methods=["POST"])
def process_bundle():
    """Process the uploaded PDF bundle."""
    data = request.json
    pdf_path = Path(data.get("path"))
    case_name = data.get("case_name", pdf_path.stem)
    password = data.get("password")
    instructions = data.get("instructions", "")
    use_vlm = data.get("use_vlm", False)
    
    # Sanitize case name
    case_name = secure_filename(case_name)
    case_dir = CASES_DIR / case_name
    
    # Store processing status
    processing_status[case_name] = {"status": "processing", "progress": "Starting..."}
    
    def process_async():
        try:
            mode_text = "Granite-Docling VLM" if use_vlm else "Docling OCR"
            processing_status[case_name]["progress"] = f"Extracting PDF with {mode_text}..."
            result = ingest_bundle(pdf_path, case_dir, password=password, use_vlm=use_vlm)
            
            processing_status[case_name]["progress"] = "Building vector index..."
            build_vectorstore(case_dir)
            
            # Save instructions if provided
            if instructions.strip():
                with open(case_dir / "instructions.txt", "w") as f:
                    f.write(instructions)
            
            processing_status[case_name] = {
                "status": "complete",
                "chunks": result["metadata"]["chunks"],
                "case_name": case_name
            }
        except Exception as e:
            processing_status[case_name] = {"status": "error", "error": str(e)}
    
    thread = threading.Thread(target=process_async)
    thread.start()
    
    return jsonify({"case_name": case_name, "status": "processing"})


@app.route("/status/<case_name>")
def get_status(case_name):
    """Get processing status for a case."""
    return jsonify(processing_status.get(case_name, {"status": "unknown"}))


@app.route("/chat/<case_name>")
def chat_page(case_name):
    """Chat interface for a case."""
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return "Case not found", 404
    
    # Load instructions if they exist
    instructions = ""
    instructions_file = case_dir / "instructions.txt"
    if instructions_file.exists():
        instructions = instructions_file.read_text()
    
    # Load PDF path from metadata
    pdf_path = ""
    chunks_file = case_dir / "chunks.json"
    if chunks_file.exists():
        import json
        with open(chunks_file) as f:
            data = json.load(f)
            pdf_path = data.get("metadata", {}).get("source_path", "")
    
    models = list_models()
    return render_template("chat.html", 
        case_name=case_name, 
        instructions=instructions,
        models=models,
        current_model=LLM_MODEL,
        pdf_path=pdf_path
    )


@app.route("/query", methods=["POST"])
def query():
    """Query the bundle."""
    data = request.json
    case_name = data.get("case_name")
    question = data.get("question")
    model = data.get("model", LLM_MODEL)
    instructions = data.get("instructions", "")
    max_page = data.get("max_page")  # Optional page limit
    if max_page:
        try:
            max_page = int(max_page)
        except:
            max_page = None
    
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return jsonify({"error": "Case not found"}), 404
    
    # Build system prompt with instructions
    system_prompt = None
    if instructions.strip():
        system_prompt = f"""You are an assistant helping an expert witness review medical negligence bundles.
Your role is like a paralegal or secretary - you retrieve and summarize facts from the bundle.
You do NOT provide legal or medical opinions.

CASE INSTRUCTIONS FROM LAW FIRM:
{instructions}

When answering:
- Summarize the relevant facts from the retrieved chunks in 2-4 sentences
- Be precise and factual
- Reference page numbers when available
- If the retrieved chunks don't contain relevant information, say so clearly
- Never fabricate information not present in the chunks"""
    
    try:
        result = query_bundle(case_dir, question, system_prompt=system_prompt, model=model, max_page=max_page)
        
        return jsonify({
            "summary": result["summary"],
            "chunks": result["chunks"],
            "model": result["model"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save-instructions", methods=["POST"])
def save_instructions():
    """Save updated instructions for a case."""
    data = request.json
    case_name = data.get("case_name")
    instructions = data.get("instructions", "")
    
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return jsonify({"error": "Case not found"}), 404
    
    with open(case_dir / "instructions.txt", "w") as f:
        f.write(instructions)
    
    return jsonify({"status": "saved"})


@app.route("/pdf/<case_name>")
def serve_pdf(case_name):
    """Serve the original PDF for a case."""
    import json
    case_dir = CASES_DIR / case_name
    chunks_file = case_dir / "chunks.json"
    
    if not chunks_file.exists():
        return "PDF not found", 404
    
    with open(chunks_file) as f:
        data = json.load(f)
    
    pdf_path = data.get("metadata", {}).get("source_path")
    if not pdf_path or not Path(pdf_path).exists():
        return "PDF file not found", 404
    
    return send_file(pdf_path, mimetype='application/pdf')


@app.route("/delete-case/<case_name>", methods=["POST"])
def delete_case(case_name):
    """Delete a case and all its data."""
    import shutil
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return jsonify({"error": "Case not found"}), 404
    
    shutil.rmtree(case_dir)
    return jsonify({"status": "deleted"})


@app.route("/case-info/<case_name>")
def case_info(case_name):
    """Get info about a case."""
    import json
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return jsonify({"error": "Case not found"}), 404

    info = {"name": case_name}

    chunks_file = case_dir / "chunks.json"
    if chunks_file.exists():
        with open(chunks_file) as f:
            data = json.load(f)
            info["metadata"] = data.get("metadata", {})
            info["chunk_count"] = len(data.get("chunks", []))

    instructions_file = case_dir / "instructions.txt"
    info["has_instructions"] = instructions_file.exists()

    return jsonify(info)


@app.route("/timeline/<case_name>")
def timeline_page(case_name):
    """Timeline view for a case."""
    import json
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return "Case not found", 404

    timeline_file = case_dir / "timeline.json"

    # Generate timeline if it doesn't exist
    if not timeline_file.exists():
        from bundle_rag.timeline import build_timeline
        try:
            timeline = build_timeline(case_dir)
        except Exception as e:
            return f"Error generating timeline: {str(e)}", 500
    else:
        with open(timeline_file) as f:
            timeline = json.load(f)

    return render_template("timeline.html",
        case_name=case_name,
        timeline=timeline
    )


@app.route("/api/timeline/<case_name>")
def timeline_api(case_name):
    """API endpoint for timeline data."""
    import json
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return jsonify({"error": "Case not found"}), 404

    timeline_file = case_dir / "timeline.json"
    if not timeline_file.exists():
        return jsonify({"error": "Timeline not generated yet"}), 404

    with open(timeline_file) as f:
        timeline = json.load(f)

    return jsonify(timeline)


@app.route("/api/regenerate-timeline/<case_name>", methods=["POST"])
def regenerate_timeline(case_name):
    """Force timeline regeneration."""
    case_dir = CASES_DIR / case_name
    if not case_dir.exists():
        return jsonify({"error": "Case not found"}), 404

    from bundle_rag.timeline import build_timeline
    try:
        timeline = build_timeline(case_dir)
        return jsonify({"status": "regenerated", "event_count": timeline["event_count"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_app(host="127.0.0.1", port=5000, open_browser=True):
    """Run the Flask app."""
    ensure_ollama_running()
    
    if open_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_app()
