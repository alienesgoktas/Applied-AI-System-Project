"""The single integrated RAG entry point: a natural-language query -> results.

Both the CLI (``src/main.py``) and the web UI (``app.py``) call
``recommend_from_query``, so LLM profile-parsing, retrieval, confidence scoring,
and the grounded explanation always run through one place — and all logging lives
here. The pipeline never raises on an LLM failure: parsing and explanation both
degrade to deterministic paths, so the returned result is always well-formed.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from src.confidence import score_confidence
from src.llm import generate_explanation, parse_profile
from src.recommender import BALANCED, ScoringStrategy, recommend_songs


def backend_label(result: Dict) -> str:
    """The provider that actually served this result: 'local'/'anthropic'/'offline'.

    Returns 'offline' whenever no LLM path ran (deterministic fallback), regardless
    of which backend was configured. Shared by the CLI and web UIs so the badge
    logic lives in one place.
    """
    provider = result["backend"].split(":")[0]
    llm_active = result["used_llm"] or result["profile_source"] == "llm"
    return provider if llm_active else "offline"


_LOG_DIR = "logs"
_configured = False


def get_logger() -> logging.Logger:
    """Configure the ``recommender`` logger once: stderr + best-effort file."""
    global _configured
    logger = logging.getLogger("recommender")
    if not _configured:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        stream = logging.StreamHandler()  # stderr
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(_LOG_DIR, "app.log"), encoding="utf-8"
            )
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            pass  # file logging is best-effort; stderr still works
        _configured = True
    return logger


def recommend_from_query(
    query: str,
    songs: List[Dict],
    k: int = 5,
    *,
    backend=None,
    strategy: ScoringStrategy = BALANCED,
) -> Dict:
    """Run the full RAG pipeline and return a structured result dict.

    Keys: ``query, profile, profile_source, results, confidence, explanation,
    used_llm, backend``. ``results`` are ``(song, score, reasons)`` tuples from the
    retriever (the authoritative, deterministic record the caller renders).
    """
    log = get_logger()
    backend_name = getattr(backend, "name", "offline") if backend is not None else "offline"
    log.info("query=%r backend=%s k=%d strategy=%s", query, backend_name, k, strategy.name)

    profile, profile_source = parse_profile(query, songs, backend=backend)
    log.info("profile_source=%s profile=%s", profile_source, profile)

    results = recommend_songs(profile, songs, k=k, strategy=strategy)
    conf = score_confidence(results, strategy)
    log.info(
        "results=%d confidence=%s strong_matches=%d",
        len(results),
        conf["confidence"],
        conf["strong_matches"],
    )

    explanation, used_llm = generate_explanation(query, results, backend=backend)
    log.info("explanation_used_llm=%s", used_llm)

    return {
        "query": query,
        "profile": profile,
        "profile_source": profile_source,
        "results": results,
        "confidence": conf,
        "explanation": explanation,
        "used_llm": used_llm,
        "backend": backend_name,
    }
