"""The single integrated RAG entry point: a natural-language query -> results.

Both the CLI (``src/main.py``) and the web UI (``app.py``) call
``recommend_from_query``, so LLM profile-parsing, retrieval, confidence scoring,
and the grounded explanation always run through one place — and all logging lives
here. The pipeline never raises on an LLM failure: parsing and explanation both
degrade to deterministic paths, so the returned result is always well-formed.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional

from src.confidence import score_confidence
from src.llm import generate_explanation, parse_profile
from src.recommender import (
    BALANCED,
    ScoringStrategy,
    load_genre_notes,
    recommend_songs,
    retrieve_notes,
    score_detail,
)

_GENRE_NOTES_PATH = "data/genre_notes.csv"
_genre_notes_cache: Optional[Dict[str, str]] = None


def _genre_notes() -> Dict[str, str]:
    """Lazy-load the RAG second source once; degrade to {} if the file is absent OR
    malformed. The notes are pure enrichment, so a bad/edited-away file must never crash
    a recommendation — a missing column (KeyError), bad CSV (csv.Error), or non-UTF-8
    bytes (UnicodeDecodeError) all fall back to no notes, same as a missing file."""
    global _genre_notes_cache
    if _genre_notes_cache is None:
        try:
            _genre_notes_cache = load_genre_notes(_GENRE_NOTES_PATH)
        except (OSError, KeyError, csv.Error, UnicodeDecodeError, ValueError):
            _genre_notes_cache = {}
    return _genre_notes_cache


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


def recommend_for_profile(
    query: str,
    profile: Dict,
    songs: List[Dict],
    k: int = 5,
    *,
    backend=None,
    strategy: ScoringStrategy = BALANCED,
    max_per_artist: Optional[int] = None,
    profile_source: str = "given",
) -> Dict:
    """Retrieve + explain from an ALREADY-PARSED profile (skips NL parsing).

    Shared by ``recommend_from_query`` and the conversational-refinement path,
    which mutates a profile between turns. Returns the same result-dict shape.
    """
    log = get_logger()
    backend_name = getattr(backend, "name", "offline") if backend is not None else "offline"

    results = recommend_songs(profile, songs, k=k, strategy=strategy, max_per_artist=max_per_artist)
    breakdowns = [score_detail(profile, song, strategy)[1] for song, _score, _reason in results]
    conf = score_confidence(results, strategy)
    # RAG second source: genre-knowledge notes for the retrieved genres, fed to the generator.
    notes = retrieve_notes(results, _genre_notes())
    explanation, used_llm = generate_explanation(query, results, backend=backend, notes=notes)
    log.info(
        "for_profile query=%r source=%s results=%d confidence=%s used_llm=%s notes=%d",
        query, profile_source, len(results), conf["confidence"], used_llm, len(notes),
    )
    return {
        "query": query,
        "profile": profile,
        "profile_source": profile_source,
        "results": results,
        "breakdowns": breakdowns,
        "confidence": conf,
        "notes": notes,
        "explanation": explanation,
        "used_llm": used_llm,
        "backend": backend_name,
    }


def recommend_from_query(
    query: str,
    songs: List[Dict],
    k: int = 5,
    *,
    backend=None,
    strategy: ScoringStrategy = BALANCED,
    blocked_genres: Optional[List[str]] = None,
    max_per_artist: Optional[int] = None,
) -> Dict:
    """Run the full RAG pipeline (NL parse -> retrieve -> explain) and return a
    structured result dict. Keys: ``query, profile, profile_source, results,
    breakdowns, confidence, explanation, used_llm, backend``.
    """
    log = get_logger()
    backend_name = getattr(backend, "name", "offline") if backend is not None else "offline"
    log.info("query=%r backend=%s k=%d strategy=%s", query, backend_name, k, strategy.name)

    profile, profile_source = parse_profile(query, songs, backend=backend)
    if blocked_genres:  # merge UI-selected blocks with any the parser found
        merged = set(profile.get("blocked_genres") or []) | {str(g).lower() for g in blocked_genres}
        profile = {**profile, "blocked_genres": sorted(merged)}

    return recommend_for_profile(
        query, profile, songs, k=k, backend=backend, strategy=strategy,
        max_per_artist=max_per_artist, profile_source=profile_source,
    )
