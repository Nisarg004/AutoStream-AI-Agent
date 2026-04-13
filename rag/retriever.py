from __future__ import annotations
import json
import logging
import os
import warnings

# Suppress all HF/transformers noise before any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

warnings.filterwarnings("ignore")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("filelock").setLevel(logging.ERROR)

import numpy as np
import faiss

# Suppress stderr from transformers load reports
import transformers
transformers.logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")

# Lazy-loaded globals
_model: SentenceTransformer | None = None
_index: faiss.IndexFlatL2 | None = None
_chunks: list[str] = []


def _load_knowledge() -> list[str]:
    """Flatten knowledge.json into retrievable text chunks."""
    with open(_KB_PATH, "r") as f:
        kb = json.load(f)

    chunks = []
    for plan in kb.get("plans", []):
        features = ", ".join(plan.get("features", []))
        chunks.append(
            f"{plan['name']}: {plan['price']}, {plan['videos']}, "
            f"{plan['resolution']} resolution. Features: {features}."
        )
    for policy in kb.get("policies", []):
        chunks.append(f"{policy['topic']}: {policy['content']}")
    return chunks


def _build_index(chunks: list[str], model: SentenceTransformer) -> faiss.IndexFlatL2:
    embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def _ensure_loaded() -> None:
    global _model, _index, _chunks
    if _index is not None:
        return
    import sys, io
    _stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        _model = SentenceTransformer(_MODEL_NAME)
    finally:
        sys.stderr = _stderr
    _chunks = _load_knowledge()
    _index = _build_index(_chunks, _model)


def retrieve(query: str, top_k: int = 3) -> str:
    """Return the top-k most relevant knowledge chunks for a query."""
    _ensure_loaded()
    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index.search(query_vec, top_k)
    results = [_chunks[i] for i in indices[0] if i < len(_chunks)]
    return "\n".join(results)
