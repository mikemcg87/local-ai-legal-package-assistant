"""Local setup checks and model-fit guidance."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import ollama

from bundle_rag.config import LLM_MODEL


EMBEDDING_NAME_MARKERS = (
    "embed",
    "embedding",
    "mxbai",
    "nomic",
    "bge",
    "minilm",
    "ms-marco",
)


@dataclass(frozen=True)
class InstalledModel:
    name: str
    size_gb: float | None = None
    modified_at: str | None = None

    @property
    def is_embedding(self) -> bool:
        lowered = self.name.lower()
        return any(marker in lowered for marker in EMBEDDING_NAME_MARKERS)


@dataclass(frozen=True)
class ModelRecommendation:
    name: str
    role: str
    command: str
    fit: str
    reason: str


@dataclass(frozen=True)
class RuntimeOption:
    name: str
    status: str
    label: str
    endpoint: str
    reason: str
    next_step: str


def get_system_memory_gb() -> float | None:
    """Return approximate physical RAM in GB."""
    system = platform.system().lower()

    if system == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip()) / (1024**3)
        except Exception:
            return None

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024**3)
    except Exception:
        return None


def classify_machine(memory_gb: float | None) -> dict[str, str]:
    """Classify the machine for local LLM use."""
    if memory_gb is None:
        return {
            "tier": "unknown",
            "label": "Unknown memory",
            "guidance": "Use smaller quantized models first and confirm responsiveness with evals.",
            "default_extraction": "OCR",
        }

    if memory_gb < 12:
        return {
            "tier": "constrained",
            "label": "Constrained local machine",
            "guidance": "Start with small quantized chat models and keep VLM extraction off unless needed.",
            "default_extraction": "OCR",
        }

    if memory_gb < 24:
        return {
            "tier": "balanced",
            "label": "Balanced local machine",
            "guidance": "Use 7B-9B class quantized chat models first; try larger models only after evals justify the latency.",
            "default_extraction": "OCR",
        }

    return {
        "tier": "capable",
        "label": "Capable local machine",
        "guidance": "You can test larger local chat models and VLM extraction, but choose by case eval results.",
        "default_extraction": "VLM when layout quality matters",
    }


def recommended_model_stack(memory_gb: float | None) -> list[ModelRecommendation]:
    """Return practical install suggestions for the detected machine tier."""
    machine = classify_machine(memory_gb)
    tier = machine["tier"]

    recommendations = [
        ModelRecommendation(
            name="mxbai-embed-large",
            role="retrieval",
            command="ollama pull mxbai-embed-large",
            fit="All machines",
            reason="Embedding model used for local semantic retrieval.",
        )
    ]

    if tier == "constrained":
        recommendations.append(
            ModelRecommendation(
                name="gemma4:e2b",
                role="chat",
                command="ollama run gemma4:e2b",
                fit="Small first choice",
                reason="Start with Gemma 4's edge model before trying larger reasoning models.",
            )
        )
    elif tier == "balanced":
        recommendations.extend(
            [
                ModelRecommendation(
                    name="gemma4:e4b",
                    role="chat",
                    command="ollama run gemma4:e4b",
                    fit="Balanced local candidate",
                    reason="Newer Gemma 4 edge model with strong local reasoning and multimodal support.",
                ),
                ModelRecommendation(
                    name="gpt-oss:20b",
                    role="chat",
                    command="ollama run gpt-oss:20b",
                    fit="Reasoning candidate",
                    reason="Worth testing on evals if latency is acceptable on this machine.",
                ),
            ]
        )
    else:
        recommendations.extend(
            [
                ModelRecommendation(
                    name="gpt-oss:20b",
                    role="chat",
                    command="ollama run gpt-oss:20b",
                    fit="Reasoning candidate",
                    reason="Strong local reasoning candidate to compare against faster models.",
                ),
                ModelRecommendation(
                    name="gemma4:26b",
                    role="chat",
                    command="ollama run gemma4:26b",
                    fit="Workstation MoE candidate",
                    reason="Newer Gemma 4 MoE model with 256K context; test if latency is acceptable.",
                ),
                ModelRecommendation(
                    name="qwen3:30b",
                    role="chat",
                    command="ollama run qwen3:30b",
                    fit="Alternative reasoning candidate",
                    reason="Useful comparison model if Gemma 4 or gpt-oss underperform on private evals.",
                ),
            ]
        )

    return recommendations


def check_lm_studio(endpoint: str = "http://127.0.0.1:1234/v1/models") -> bool:
    """Return whether LM Studio's OpenAI-compatible local server is reachable."""
    try:
        with urlopen(endpoint, timeout=0.35) as response:
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def runtime_options(ollama_ok: bool, lm_studio_ok: bool) -> list[RuntimeOption]:
    """Describe local runtime options for setup guidance."""
    return [
        RuntimeOption(
            name="Ollama",
            status="ready" if ollama_ok else "not_running",
            label="Ready" if ollama_ok else "Not running",
            endpoint="http://127.0.0.1:11434",
            reason="Currently supported by chat, retrieval, and model detection in this app.",
            next_step="Use the suggested Ollama install commands, then refresh setup.",
        ),
        RuntimeOption(
            name="LM Studio",
            status="ready" if lm_studio_ok else "available",
            label="Server detected" if lm_studio_ok else "OpenAI-compatible option",
            endpoint="http://127.0.0.1:1234/v1",
            reason="Useful for downloading and serving local GGUF models through an OpenAI-compatible API.",
            next_step="Start LM Studio's local server; chat integration is the next runtime adapter to add.",
        ),
    ]


def _model_from_ollama_item(item: Any) -> InstalledModel:
    name = getattr(item, "model", None) or getattr(item, "name", None)
    if not name and isinstance(item, dict):
        name = item.get("model") or item.get("name")

    size = getattr(item, "size", None)
    modified_at = getattr(item, "modified_at", None)
    if isinstance(item, dict):
        size = item.get("size", size)
        modified_at = item.get("modified_at", modified_at)

    size_gb = None
    if size:
        try:
            size_gb = float(size) / (1024**3)
        except (TypeError, ValueError):
            size_gb = None

    return InstalledModel(
        name=str(name or "unknown"),
        size_gb=size_gb,
        modified_at=str(modified_at) if modified_at else None,
    )


def list_installed_models() -> list[InstalledModel]:
    """Return installed Ollama models with lightweight metadata."""
    result = ollama.list()
    items = getattr(result, "models", None)
    if items is None and isinstance(result, dict):
        items = result.get("models", [])
    return [_model_from_ollama_item(item) for item in items or []]


def estimate_fit(model: InstalledModel, memory_gb: float | None) -> dict[str, str | float | None]:
    """Estimate whether an installed model is a sensible fit for this machine."""
    if model.is_embedding:
        return {
            "status": "support",
            "label": "retrieval support",
            "reason": "Embedding or reranking model, not the chat model used to answer questions.",
            "estimated_memory_gb": None,
        }

    if memory_gb is None or model.size_gb is None:
        return {
            "status": "unknown",
            "label": "unknown fit",
            "reason": "Model or machine memory size is unavailable.",
            "estimated_memory_gb": None,
        }

    estimated_memory_gb = max(4.0, model.size_gb * 1.6)
    if estimated_memory_gb <= memory_gb * 0.55:
        status = "good"
        label = "good fit"
        reason = "Likely enough headroom for local questioning while the app is running."
    elif estimated_memory_gb <= memory_gb * 0.75:
        status = "borderline"
        label = "borderline"
        reason = "May work, but expect slower responses or memory pressure on large bundles."
    else:
        status = "too_large"
        label = "likely too large"
        reason = "Choose a smaller or more heavily quantized model first."

    return {
        "status": status,
        "label": label,
        "reason": reason,
        "estimated_memory_gb": round(estimated_memory_gb, 1),
    }


def choose_chat_model(models: list[dict[str, Any]], current_model: str = LLM_MODEL) -> str | None:
    """Choose the best currently installed chat model candidate."""
    chat_models = [m for m in models if m["fit"]["status"] != "support"]
    if not chat_models:
        return None

    current_matches = [m for m in chat_models if m["name"] == current_model]
    if current_matches and current_matches[0]["fit"]["status"] in {"good", "unknown"}:
        return current_model

    fit_rank = {"good": 0, "unknown": 1, "borderline": 2, "too_large": 3}
    sorted_models = sorted(
        chat_models,
        key=lambda m: (
            fit_rank.get(m["fit"]["status"], 9),
            -(m["size_gb"] or 0),
            m["name"],
        ),
    )
    return sorted_models[0]["name"]


def get_setup_status() -> dict[str, Any]:
    """Build a setup report for onboarding and diagnostics."""
    memory_gb = get_system_memory_gb()
    machine = classify_machine(memory_gb)

    try:
        installed = list_installed_models()
        ollama_ok = True
        ollama_error = None
    except Exception as exc:
        installed = []
        ollama_ok = False
        ollama_error = str(exc)
    lm_studio_ok = check_lm_studio()

    model_reports = []
    for model in installed:
        model_reports.append(
            {
                "name": model.name,
                "size_gb": round(model.size_gb, 1) if model.size_gb is not None else None,
                "modified_at": model.modified_at,
                "is_embedding": model.is_embedding,
                "fit": estimate_fit(model, memory_gb),
            }
        )

    embedding_models = [m for m in model_reports if m["is_embedding"]]
    chat_models = [m for m in model_reports if not m["is_embedding"]]
    recommended_chat_model = choose_chat_model(model_reports)
    recommendations = recommended_model_stack(memory_gb)

    ready = bool(ollama_ok and chat_models and embedding_models)

    return {
        "ready": ready,
        "ollama_ok": ollama_ok,
        "ollama_error": ollama_error,
        "lm_studio_ok": lm_studio_ok,
        "memory_gb": round(memory_gb, 1) if memory_gb is not None else None,
        "machine": machine,
        "models": model_reports,
        "chat_model_count": len(chat_models),
        "embedding_model_count": len(embedding_models),
        "recommended_chat_model": recommended_chat_model,
        "recommended_stack": [
            {
                "name": item.name,
                "role": item.role,
                "command": item.command,
                "fit": item.fit,
                "reason": item.reason,
            }
            for item in recommendations
        ],
        "current_default_model": LLM_MODEL,
        "runtime_options": [
            {
                "name": item.name,
                "status": item.status,
                "label": item.label,
                "endpoint": item.endpoint,
                "reason": item.reason,
                "next_step": item.next_step,
            }
            for item in runtime_options(ollama_ok, lm_studio_ok)
        ],
        "checks": [
            {
                "id": "ollama",
                "label": "Ollama running",
                "ok": ollama_ok,
                "detail": "Local model server is reachable." if ollama_ok else ollama_error,
            },
            {
                "id": "chat_model",
                "label": "Chat model installed",
                "ok": bool(chat_models),
                "detail": f"{len(chat_models)} chat candidate(s) found." if chat_models else "Install at least one local chat model.",
            },
            {
                "id": "embedding_model",
                "label": "Embedding model installed",
                "ok": bool(embedding_models),
                "detail": f"{len(embedding_models)} retrieval support model(s) found." if embedding_models else "Install an embedding model before ingesting bundles.",
            },
        ],
    }
